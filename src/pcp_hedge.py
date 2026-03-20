# src/pcp_hedge.py
"""
Put-Call Parity Module — Two modes:

  1. ARBITRAGE DETECTION
     C + PV(X) = P + S  must hold in efficient markets.
     When it breaks, a risk-free profit exists:

       C + PV(X) < P + S  →  "LHS cheap"
         Action: Buy Call + Invest PV(X) + Sell Put + Short Spot

       C + PV(X) > P + S  →  "LHS expensive"
         Action: Sell Call + Borrow PV(X) + Buy Put + Long Spot

  2. SYNTHETIC HEDGING
     Spot BUY  → Synthetic Short = Buy Put  + Sell Call
     Spot SELL → Synthetic Long  = Buy Call + Sell Put

  3. COMBINED: hedge_with_arb_check()
     Runs arbitrage check first, then builds the hedge — so the bot can
     capture mispricing profit AND protect the spot position simultaneously.

MT5 option symbol format (broker-dependent):
  EURUSD-C-1.1000-20250620
  EURUSD-P-1.1000-20250620
"""

import math
import MetaTrader5 as mt5
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from datetime import datetime

logger = logging.getLogger("scalping")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class OptionContract:
    symbol: str
    option_type: str      # 'call' or 'put'
    strike: float
    expiry: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class ParityResult:
    spot: float
    strike: float
    call_price: float
    put_price: float
    pv_strike: float          # PV(X) = X * e^(-rT)
    lhs: float                # C + PV(X)
    rhs: float                # P + S
    parity_diff: float        # lhs - rhs   (+ = LHS expensive, - = LHS cheap)
    parity_holds: bool
    risk_free_rate: float
    time_to_expiry: float     # years
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class ArbitrageSignal:
    """
    Describes a detected put-call parity violation and the trades to capture it.

    direction:
      'buy_lhs'  → LHS cheap  → Buy Call + Invest PV(X) + Sell Put + Short Spot
      'sell_lhs' → LHS expensive → Sell Call + Borrow PV(X) + Buy Put + Long Spot

    legs: list of (symbol, action, description) tuples
    is_executable: True only if net_profit >= min_profit_threshold
    """
    underlying: str
    direction: str
    parity: ParityResult
    call_contract: OptionContract
    put_contract: OptionContract
    legs: List[Tuple[str, str, str]]
    gross_profit: float
    transaction_cost_estimate: float
    net_profit: float
    is_executable: bool
    execution_results: dict = field(default_factory=dict)


@dataclass
class SyntheticHedge:
    direction: str            # 'synthetic_long' or 'synthetic_short'
    call_contract: OptionContract
    put_contract: OptionContract
    call_action: str
    put_action: str
    net_cost: float
    parity: ParityResult
    hedge_ratio: float        # lots for option legs


# ============================================================================
# OPTION SYMBOL HELPERS
# ============================================================================

def build_option_symbol(underlying: str, option_type: str, strike: float, expiry: str) -> str:
    """
    Build the MT5 option symbol string.
    IMPORTANT: adjust this format to match your broker's naming convention.

    Common formats:
      EURUSD-C-1.1000-20250620
      EURUSD.C.11000.20250620
      #EURUSD_C_1100_062025
    """
    otype = "C" if option_type == "call" else "P"
    return f"{underlying}-{otype}-{strike:.4f}-{expiry}"


def _round_to_nearest_strike(spot: float, underlying: str) -> float:
    if "JPY" in underlying:
        increment = 0.50
    elif underlying in ("XAUUSD", "GOLD"):
        increment = 5.0
    else:
        increment = 0.0050
    return round(round(spot / increment) * increment, 6)


def _fetch_option_contract(symbol: str, option_type: str, strike: float) -> Optional[OptionContract]:
    if not mt5.symbol_select(symbol, True):
        logger.debug(f"Could not select option: {symbol}")
        return None
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or (tick.bid == 0.0 and tick.ask == 0.0):
        logger.debug(f"No tick data for option: {symbol}")
        return None
    info = mt5.symbol_info(symbol)
    expiry_str = ""
    if info and hasattr(info, "expiration_time") and info.expiration_time:
        expiry_str = datetime.fromtimestamp(info.expiration_time).strftime("%Y%m%d")
    return OptionContract(
        symbol=symbol, option_type=option_type, strike=strike,
        expiry=expiry_str, bid=tick.bid, ask=tick.ask,
    )


def find_atm_options(
    underlying: str,
    expiry: str,
    strike_override: Optional[float] = None,
) -> Tuple[Optional[OptionContract], Optional[OptionContract]]:
    """Fetch nearest ATM call and put from MT5."""
    tick = mt5.symbol_info_tick(underlying)
    if tick is None:
        logger.error(f"Cannot get spot for {underlying}")
        return None, None
    spot   = (tick.bid + tick.ask) / 2.0
    strike = strike_override if strike_override is not None else _round_to_nearest_strike(spot, underlying)
    call   = _fetch_option_contract(build_option_symbol(underlying, "call", strike, expiry), "call", strike)
    put    = _fetch_option_contract(build_option_symbol(underlying, "put",  strike, expiry), "put",  strike)
    if call is None or put is None:
        logger.warning(
            f"Options unavailable: {underlying} K={strike} expiry={expiry}. "
            "Check build_option_symbol() for your broker's format."
        )
    return call, put


# ============================================================================
# CORE PARITY CALCULATION
# ============================================================================

def compute_parity(
    spot: float,
    call: OptionContract,
    put: OptionContract,
    risk_free_rate: float = 0.05,
    days_to_expiry: int = 30,
    tolerance: float = 0.0005,
) -> ParityResult:
    """
    Evaluate C + PV(X) = P + S

    PV(X) = X * e^(-rT)  (continuous compounding)

    parity_diff = LHS - RHS
      > 0 → LHS expensive (sell call, buy put/spot)
      < 0 → LHS cheap     (buy call, sell put/spot)
      ≈ 0 → parity holds, no free lunch
    """
    T    = days_to_expiry / 365.0
    X    = call.strike
    pv_x = X * math.exp(-risk_free_rate * T)

    lhs  = call.mid + pv_x
    rhs  = put.mid  + spot
    diff = lhs - rhs

    return ParityResult(
        spot=spot, strike=X,
        call_price=call.mid, put_price=put.mid, pv_strike=pv_x,
        lhs=lhs, rhs=rhs, parity_diff=diff,
        parity_holds=abs(diff) <= tolerance,
        risk_free_rate=risk_free_rate, time_to_expiry=T,
    )


# ============================================================================
# 1. ARBITRAGE DETECTION
# ============================================================================

def detect_arbitrage(
    underlying: str,
    expiry: str,
    risk_free_rate: float = 0.05,
    days_to_expiry: int = 30,
    min_profit_threshold: float = 0.0003,
    tolerance: float = 0.0005,
    strike_override: Optional[float] = None,
) -> Optional[ArbitrageSignal]:
    """
    Detect put-call parity violations and build an arbitrage signal.

    When C + PV(X) ≠ P + S, two sides of the equation are mispriced.
    We buy the cheap side and sell the expensive side to lock in
    a riskless profit that converges to zero at expiry.

    Violation A — LHS cheap  (diff < 0,  C + PV(X) < P + S):
      Buy  Call   — long the cheap derivative
      Invest PV(X) at risk-free rate (holds X at expiry to exercise)
      Sell Put    — short the expensive synthetic
      Short Spot  — short the expensive cash position

    Violation B — LHS expensive  (diff > 0,  C + PV(X) > P + S):
      Sell Call   — short the expensive derivative
      Borrow PV(X) at risk-free rate
      Buy  Put    — long the cheap synthetic
      Long Spot   — long the cheap cash position

    Transaction costs are estimated as half-spread per leg (entry + exit).
    Signal is marked executable only if net_profit >= min_profit_threshold.

    Args:
        underlying:           MT5 symbol, e.g. 'EURUSD'
        expiry:               Option expiry 'YYYYMMDD'
        risk_free_rate:       Annual rate for PV(X) = X*e^(-rT)
        days_to_expiry:       Calendar days to expiry
        min_profit_threshold: Net profit floor to mark signal as executable
        tolerance:            Parity tolerance (violations below this ignored)
        strike_override:      Force a specific strike

    Returns:
        ArbitrageSignal if a violation exists, None if parity holds.
    """
    tick = mt5.symbol_info_tick(underlying)
    if tick is None:
        logger.error(f"[ARB] Cannot get spot for {underlying}")
        return None

    spot        = (tick.bid + tick.ask) / 2.0
    spot_spread = tick.ask - tick.bid

    call, put = find_atm_options(underlying, expiry, strike_override)
    if call is None or put is None:
        return None

    parity = compute_parity(spot, call, put, risk_free_rate, days_to_expiry, tolerance)

    if parity.parity_holds:
        logger.info(
            f"[ARB] {underlying} parity holds — diff={parity.parity_diff:+.5f} "
            f"(±{tolerance} tolerance). No signal."
        )
        return None

    # Transaction cost: half-spread per option leg + spot leg, entry and exit
    transaction_cost = (call.spread / 2.0 + put.spread / 2.0 + spot_spread / 2.0) * 2
    gross_profit     = abs(parity.parity_diff)
    net_profit       = gross_profit - transaction_cost
    is_executable    = net_profit >= min_profit_threshold

    if parity.parity_diff < 0:
        # LHS CHEAP → buy lhs (call + bond), sell rhs (put + spot)
        direction = "buy_lhs"
        legs = [
            (call.symbol,  "buy",    "Buy cheap call (LHS underpriced)"),
            ("BOND",       "invest", f"Invest PV(X)={parity.pv_strike:.5f} at r={risk_free_rate}"),
            (put.symbol,   "sell",   "Sell expensive put (RHS overpriced)"),
            (underlying,   "sell",   "Short spot (RHS overpriced)"),
        ]
    else:
        # LHS EXPENSIVE → sell lhs (call + bond), buy rhs (put + spot)
        direction = "sell_lhs"
        legs = [
            (call.symbol,  "sell",   "Sell expensive call (LHS overpriced)"),
            ("BOND",       "borrow", f"Borrow PV(X)={parity.pv_strike:.5f} at r={risk_free_rate}"),
            (put.symbol,   "buy",    "Buy cheap put (RHS underpriced)"),
            (underlying,   "buy",    "Long spot (RHS underpriced)"),
        ]

    signal = ArbitrageSignal(
        underlying=underlying, direction=direction,
        parity=parity, call_contract=call, put_contract=put,
        legs=legs, gross_profit=gross_profit,
        transaction_cost_estimate=transaction_cost,
        net_profit=net_profit, is_executable=is_executable,
    )

    status = "EXECUTABLE ✓" if is_executable else f"below threshold ({min_profit_threshold:.5f})"
    logger.warning(
        f"[ARB SIGNAL] {underlying} | {direction.upper()} | {status} | "
        f"diff={parity.parity_diff:+.5f} | gross={gross_profit:.5f} | "
        f"costs≈{transaction_cost:.5f} | net={net_profit:+.5f}"
    )
    for sym, action, reason in legs:
        logger.info(f"  Leg: {action.upper():8s} {sym:35s} — {reason}")

    return signal


def execute_arbitrage(signal: ArbitrageSignal, lots: float) -> dict:
    """
    Execute the tradeable legs of an arbitrage signal on MT5.

    The BOND leg ('invest'/'borrow') cannot be placed via MT5 — handle
    that separately through your broker's money market. All option and
    spot legs are placed here.

    Partial execution is logged but not rolled back automatically —
    monitor the execution_results dict and manage open legs manually
    if any leg fails.

    Args:
        signal: ArbitrageSignal from detect_arbitrage()
        lots:   Lot size per leg

    Returns:
        dict keyed by symbol → MT5 order result (or status string for BOND)
    """
    if not signal.is_executable:
        logger.warning(
            f"[ARB] Signal not executable (net={signal.net_profit:.5f}). Skipping."
        )
        return {}

    results    = {}
    all_ok     = True
    open_legs  = []  # track successfully opened legs for partial rollback logging

    for sym, action, reason in signal.legs:

        if sym == "BOND":
            logger.info(f"  [ARB] BOND leg — handle manually: {action} PV(X)={signal.parity.pv_strike:.5f}")
            results["BOND"] = {"action": action, "status": "manual_required"}
            continue

        is_option = sym in (signal.call_contract.symbol, signal.put_contract.symbol)
        mt5_action = action if action in ("buy", "sell") else ("buy" if action == "long" else "sell")

        if is_option:
            contract = signal.call_contract if sym == signal.call_contract.symbol else signal.put_contract
            result   = _place_option_order(contract, mt5_action, lots)
        else:
            result = _place_spot_order(sym, mt5_action, lots)

        results[sym] = result

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            logger.error(f"  [ARB] Leg FAILED: {mt5_action.upper()} {sym} — code={code}")
            all_ok = False
        else:
            logger.info(f"  [ARB] Leg OK: {mt5_action.upper()} {sym} @ {result.price:.5f}")
            open_legs.append((sym, mt5_action))

    if not all_ok:
        logger.error(
            f"[ARB] PARTIAL execution — {len(open_legs)} leg(s) open: {open_legs}. "
            "Review and close orphaned legs manually."
        )

    signal.execution_results = results
    logger.info(f"[ARB] Execution {'SUCCESS' if all_ok else 'PARTIAL'}")
    return results


def scan_arbitrage_opportunities(
    symbols: List[str],
    expiry: str,
    risk_free_rate: float = 0.05,
    days_to_expiry: int = 30,
    min_profit_threshold: float = 0.0003,
    auto_execute: bool = False,
    lots: float = 0.01,
) -> List[ArbitrageSignal]:
    """
    Scan a list of underlyings for parity violations in a single pass.

    Args:
        symbols:      e.g. ['EURUSD', 'GBPUSD', 'USDJPY']
        auto_execute: Automatically execute executable signals if True
        lots:         Lot size used when auto_execute=True

    Returns:
        List of ArbitrageSignal objects (only violations are included)
    """
    signals = []
    logger.info(f"[ARB SCAN] Scanning {len(symbols)} symbol(s)...")

    for sym in symbols:
        sig = detect_arbitrage(sym, expiry, risk_free_rate, days_to_expiry, min_profit_threshold)
        if sig is not None:
            signals.append(sig)
            if auto_execute and sig.is_executable:
                logger.info(f"[ARB SCAN] Auto-executing on {sym}...")
                execute_arbitrage(sig, lots)

    executable = [s for s in signals if s.is_executable]
    logger.info(
        f"[ARB SCAN] Done — {len(signals)} violation(s), "
        f"{len(executable)} executable after costs."
    )
    return signals


# ============================================================================
# 2. SYNTHETIC HEDGING
# ============================================================================

def build_synthetic_hedge(
    trade_direction: str,
    underlying: str,
    expiry: str,
    trade_lots: float,
    risk_free_rate: float = 0.05,
    days_to_expiry: int = 30,
    strike_override: Optional[float] = None,
    hedge_ratio: float = 1.0,
) -> Optional[SyntheticHedge]:
    """
    Build a synthetic hedge using put-call parity.

    Spot BUY  → Synthetic Short (Buy Put + Sell Call)
                replicates: P + S offsets long exposure
    Spot SELL → Synthetic Long  (Buy Call + Sell Put)
                replicates: C + PV(X) offsets short exposure

    If parity is violated at hedge construction time, the mismatch is
    logged — consider running detect_arbitrage() first via hedge_with_arb_check().
    """
    tick = mt5.symbol_info_tick(underlying)
    if tick is None:
        logger.error(f"[HEDGE] Cannot get spot for {underlying}")
        return None

    spot = (tick.bid + tick.ask) / 2.0
    call, put = find_atm_options(underlying, expiry, strike_override)
    if call is None or put is None:
        logger.error(f"[HEDGE] Options unavailable — hedge cannot be built for {underlying}.")
        return None

    parity = compute_parity(spot, call, put, risk_free_rate, days_to_expiry)

    if not parity.parity_holds:
        logger.warning(
            f"[HEDGE] Parity violated at hedge time (diff={parity.parity_diff:+.5f}). "
            "Use hedge_with_arb_check() to also capture this mispricing."
        )
    else:
        logger.info(
            f"[HEDGE] Parity holds (diff={parity.parity_diff:+.5f}). "
            f"C+PV(X)={parity.lhs:.5f} ≈ P+S={parity.rhs:.5f}"
        )

    hedge_lots = round(trade_lots * hedge_ratio, 2)

    if trade_direction == "buy":
        direction   = "synthetic_short"
        call_action = "sell"
        put_action  = "buy"
        net_cost    = put.ask - call.bid
    else:
        direction   = "synthetic_long"
        call_action = "buy"
        put_action  = "sell"
        net_cost    = call.ask - put.bid

    hedge = SyntheticHedge(
        direction=direction,
        call_contract=call, put_contract=put,
        call_action=call_action, put_action=put_action,
        net_cost=net_cost, parity=parity, hedge_ratio=hedge_lots,
    )

    logger.info(
        f"[HEDGE] {direction.upper()} | "
        f"{call_action.upper()} {call.symbol} @ {call.mid:.5f} | "
        f"{put_action.upper()} {put.symbol} @ {put.mid:.5f} | "
        f"net_cost={net_cost:+.5f} | lots={hedge_lots}"
    )
    return hedge


def execute_synthetic_hedge(hedge: SyntheticHedge, lots: float) -> dict:
    """Execute both option legs of the synthetic hedge on MT5."""
    results = {"call_result": None, "put_result": None, "success": False}

    call_result = _place_option_order(hedge.call_contract, hedge.call_action, lots)
    results["call_result"] = call_result

    if call_result is None or call_result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"[HEDGE] Call leg failed ({call_result.retcode if call_result else 'None'}). "
            "Aborting put leg — spot position is UNHEDGED."
        )
        return results

    put_result = _place_option_order(hedge.put_contract, hedge.put_action, lots)
    results["put_result"] = put_result

    if put_result is None or put_result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"[HEDGE] Put leg failed. PARTIAL HEDGE — call leg open, close manually if needed."
        )
        return results

    results["success"] = True
    logger.info(
        f"[HEDGE] Both legs OK — "
        f"Call @ {call_result.price:.5f} | Put @ {put_result.price:.5f}"
    )
    return results


# ============================================================================
# 3. COMBINED: ARBITRAGE-AWARE HEDGE
# ============================================================================

def hedge_with_arb_check(
    trade_direction: str,
    underlying: str,
    expiry: str,
    trade_lots: float,
    risk_free_rate: float = 0.05,
    days_to_expiry: int = 30,
    hedge_ratio: float = 1.0,
    min_profit_threshold: float = 0.0003,
    auto_execute_arb: bool = False,
) -> dict:
    """
    Combined entry point: detect arbitrage FIRST, then build the hedge.

    Workflow:
      1. detect_arbitrage() — check if parity is violated
      2. If executable and auto_execute_arb=True → fire arb trades
         (capturing free profit from mispricing)
      3. build_synthetic_hedge() — always runs, protects the spot position

    This lets the bot do both jobs simultaneously:
      - Extract riskless profit if parity is broken
      - Hedge spot exposure regardless

    Args:
        auto_execute_arb: Set True to fire arb trades automatically

    Returns:
        {
          'arb_signal':  ArbitrageSignal or None,
          'arb_results': execution results dict or {},
          'hedge':       SyntheticHedge or None,
        }
    """
    output = {"arb_signal": None, "arb_results": {}, "hedge": None}

    # Step 1 — Arbitrage check
    arb = detect_arbitrage(
        underlying, expiry, risk_free_rate, days_to_expiry, min_profit_threshold
    )
    output["arb_signal"] = arb

    if arb is not None:
        if arb.is_executable:
            logger.warning(
                f"[PCP] Arbitrage on {underlying}! net={arb.net_profit:.5f} | "
                f"auto_execute={auto_execute_arb}"
            )
            if auto_execute_arb:
                output["arb_results"] = execute_arbitrage(arb, trade_lots)
        else:
            logger.info(
                f"[PCP] Parity violation on {underlying} but below cost threshold "
                f"(net={arb.net_profit:.5f}). Logged only."
            )

    # Step 2 — Build hedge (always)
    output["hedge"] = build_synthetic_hedge(
        trade_direction, underlying, expiry, trade_lots,
        risk_free_rate, days_to_expiry, hedge_ratio=hedge_ratio,
    )

    return output


# ============================================================================
# MONITORING UTILITY
# ============================================================================

def check_parity_for_symbol(
    underlying: str,
    expiry: str,
    risk_free_rate: float = 0.05,
    days_to_expiry: int = 30,
) -> Optional[ParityResult]:
    """Standalone parity check — no trades placed."""
    tick = mt5.symbol_info_tick(underlying)
    if tick is None:
        return None
    spot  = (tick.bid + tick.ask) / 2.0
    call, put = find_atm_options(underlying, expiry)
    if call is None or put is None:
        return None
    result = compute_parity(spot, call, put, risk_free_rate, days_to_expiry)
    logger.info(
        f"[PARITY] {underlying} | S={spot:.5f} K={result.strike:.5f} | "
        f"C={result.call_price:.5f} P={result.put_price:.5f} PV(X)={result.pv_strike:.5f} | "
        f"LHS={result.lhs:.5f} RHS={result.rhs:.5f} diff={result.parity_diff:+.5f} | "
        f"holds={result.parity_holds}"
    )
    return result


# ============================================================================
# LOW-LEVEL ORDER HELPERS
# ============================================================================

def _place_option_order(contract: OptionContract, action: str, lots: float):
    tick = mt5.symbol_info_tick(contract.symbol)
    if tick is None:
        logger.error(f"No tick for option {contract.symbol}")
        return None
    order_type = mt5.ORDER_TYPE_BUY  if action == "buy"  else mt5.ORDER_TYPE_SELL
    price      = tick.ask            if action == "buy"  else tick.bid
    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL, "symbol": contract.symbol,
        "volume": float(lots), "type": order_type, "price": float(price),
        "deviation": 50, "magic": 999888, "comment": f"pcp-{action}",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_RETURN,
    })
    if result is None:
        logger.error(f"order_send None for option {contract.symbol}")
    elif result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"  ✓ {action.upper()} {contract.symbol} @ {result.price:.5f}")
    else:
        logger.error(f"  ✗ {action.upper()} {contract.symbol} — {result.retcode}: {result.comment}")
    return result


def _place_spot_order(symbol: str, action: str, lots: float):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"No tick for spot {symbol}")
        return None
    order_type = mt5.ORDER_TYPE_BUY  if action == "buy"  else mt5.ORDER_TYPE_SELL
    price      = tick.ask            if action == "buy"  else tick.bid
    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
        "volume": float(lots), "type": order_type, "price": float(price),
        "deviation": 50, "magic": 999889, "comment": f"pcp-arb-spot-{action}",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_RETURN,
    })
    if result is None:
        logger.error(f"order_send None for spot {symbol}")
    elif result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"  ✓ {action.upper()} {symbol} (spot) @ {result.price:.5f}")
    else:
        logger.error(f"  ✗ {action.upper()} {symbol} (spot) — {result.retcode}: {result.comment}")
    return result