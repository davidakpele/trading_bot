# src/pcp_hedge.py
"""
Put-Call Parity Synthetic Hedging Module
Formula: C + PV(X) = P + S
=> C - P = S - PV(X)

Synthetic positions derived from parity:
  Synthetic Long  (hedge a SELL) = Buy Call + Sell Put  [replicates holding asset]
  Synthetic Short (hedge a BUY)  = Buy Put  + Sell Call [replicates shorting asset]

MT5 options symbols convention (broker-dependent), e.g.:
  EURUSD-C-1.1000-20250620  (Call, strike 1.1000, expiry 20250620)
  EURUSD-P-1.1000-20250620  (Put,  strike 1.1000, expiry 20250620)
"""

import math
import MetaTrader5 as mt5
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("scalping")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class OptionContract:
    symbol: str          # Full MT5 symbol string
    option_type: str     # 'call' or 'put'
    strike: float
    expiry: str          # e.g. '20250620'
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class ParityResult:
    spot: float
    strike: float
    call_price: float
    put_price: float
    pv_strike: float          # PV(X) = X * e^(-r*T)
    lhs: float                # C + PV(X)
    rhs: float                # P + S
    parity_diff: float        # lhs - rhs  (0 = perfect parity)
    parity_holds: bool        # True if |diff| < tolerance
    risk_free_rate: float
    time_to_expiry: float     # in years


@dataclass
class SyntheticHedge:
    direction: str            # 'synthetic_long' or 'synthetic_short'
    call_contract: OptionContract
    put_contract: OptionContract
    call_action: str          # 'buy' or 'sell'
    put_action: str           # 'buy' or 'sell'
    net_cost: float           # positive = debit, negative = credit
    parity: ParityResult
    hedge_ratio: float        # position lots for the hedge


# ============================================================================
# OPTION SYMBOL DISCOVERY
# ============================================================================

def build_option_symbol(underlying: str, option_type: str, strike: float, expiry: str) -> str:
    """
    Build the MT5 option symbol string.
    Adjust this to match your broker's naming convention.

    Common formats:
      - EURUSD-C-1.1000-20250620
      - EURUSD.C.11000.20250620
      - #EURUSD_C_1100_062025
    """
    strike_str = f"{strike:.4f}"
    otype = "C" if option_type == "call" else "P"
    return f"{underlying}-{otype}-{strike_str}-{expiry}"


def find_atm_options(
    underlying: str,
    expiry: str,
    strike_override: Optional[float] = None,
    num_strikes: int = 3
) -> Tuple[Optional[OptionContract], Optional[OptionContract]]:
    """
    Find the nearest ATM call and put contracts from MT5 for a given underlying.

    Args:
        underlying:      e.g. 'EURUSD'
        expiry:          e.g. '20250620'
        strike_override: use a specific strike instead of ATM
        num_strikes:     how many strikes around ATM to search (if broker lists them)

    Returns:
        (call_contract, put_contract) or (None, None) on failure
    """
    tick = mt5.symbol_info_tick(underlying)
    if tick is None:
        logger.error(f"Cannot get spot price for {underlying}")
        return None, None

    spot = (tick.bid + tick.ask) / 2.0
    strike = strike_override if strike_override else _round_to_nearest_strike(spot, underlying)

    call_sym = build_option_symbol(underlying, "call", strike, expiry)
    put_sym  = build_option_symbol(underlying, "put",  strike, expiry)

    call = _fetch_option_contract(call_sym, "call", strike)
    put  = _fetch_option_contract(put_sym,  "put",  strike)

    if call is None or put is None:
        logger.warning(
            f"Could not fetch options for {underlying} strike={strike} expiry={expiry}. "
            f"Tried: {call_sym}, {put_sym}. Check broker symbol format."
        )

    return call, put


def _fetch_option_contract(symbol: str, option_type: str, strike: float) -> Optional[OptionContract]:
    """Fetch a single option contract from MT5."""
    # Ensure symbol is selected in Market Watch
    if not mt5.symbol_select(symbol, True):
        logger.debug(f"Could not select option symbol: {symbol}")
        return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None or (tick.bid == 0.0 and tick.ask == 0.0):
        logger.debug(f"No tick data for option: {symbol}")
        return None

    info = mt5.symbol_info(symbol)
    expiry_str = ""
    if info and hasattr(info, "expiration_time") and info.expiration_time:
        from datetime import datetime
        expiry_str = datetime.fromtimestamp(info.expiration_time).strftime("%Y%m%d")

    return OptionContract(
        symbol=symbol,
        option_type=option_type,
        strike=strike,
        expiry=expiry_str,
        bid=tick.bid,
        ask=tick.ask,
    )


def _round_to_nearest_strike(spot: float, underlying: str) -> float:
    """
    Round spot to nearest standard strike increment.
    Forex majors typically use 0.0050 or 0.0100 increments.
    """
    if "JPY" in underlying:
        increment = 0.50
    elif underlying in ("XAUUSD", "GOLD"):
        increment = 5.0
    else:
        increment = 0.0050  # standard forex major increment

    return round(round(spot / increment) * increment, 6)


# ============================================================================
# PUT-CALL PARITY CALCULATION
# ============================================================================

def compute_parity(
    spot: float,
    call: OptionContract,
    put: OptionContract,
    risk_free_rate: float = 0.05,
    days_to_expiry: int = 30,
    tolerance: float = 0.0005
) -> ParityResult:
    """
    Compute Put-Call Parity: C + PV(X) = P + S

    Args:
        spot:            Current underlying price
        call:            Call option contract
        put:             Put option contract
        risk_free_rate:  Annual risk-free rate (e.g. 0.05 for 5%)
        days_to_expiry:  Calendar days until expiry
        tolerance:       Max |diff| to consider parity holding

    Returns:
        ParityResult with full breakdown
    """
    T = days_to_expiry / 365.0
    X = call.strike  # Both call and put share the same strike
    pv_x = X * math.exp(-risk_free_rate * T)  # PV(X) = X * e^(-rT)

    C = call.mid
    P = put.mid

    lhs = C + pv_x      # Left side:  C + PV(X)
    rhs = P + spot      # Right side: P + S

    diff = lhs - rhs

    return ParityResult(
        spot=spot,
        strike=X,
        call_price=C,
        put_price=P,
        pv_strike=pv_x,
        lhs=lhs,
        rhs=rhs,
        parity_diff=diff,
        parity_holds=abs(diff) <= tolerance,
        risk_free_rate=risk_free_rate,
        time_to_expiry=T,
    )


# ============================================================================
# SYNTHETIC HEDGE CONSTRUCTION
# ============================================================================

def build_synthetic_hedge(
    trade_direction: str,        # 'buy' or 'sell' (the SPOT trade being hedged)
    underlying: str,
    expiry: str,
    trade_lots: float,
    risk_free_rate: float = 0.05,
    days_to_expiry: int = 30,
    strike_override: Optional[float] = None,
    hedge_ratio: float = 1.0,    # fraction of position to hedge (0.0 – 1.0)
) -> Optional[SyntheticHedge]:
    """
    Build a synthetic hedge using put-call parity.

    Spot BUY  → hedge with Synthetic Short (Buy Put + Sell Call)
    Spot SELL → hedge with Synthetic Long  (Buy Call + Sell Put)

    Args:
        trade_direction: 'buy' or 'sell' for the spot position
        underlying:      MT5 symbol, e.g. 'EURUSD'
        expiry:          Option expiry string, e.g. '20250620'
        trade_lots:      Lot size of the spot trade
        risk_free_rate:  Annual risk-free rate
        days_to_expiry:  Calendar days to option expiry
        strike_override: Force a specific strike
        hedge_ratio:     Fraction of spot position to hedge (default: full hedge)

    Returns:
        SyntheticHedge dataclass, or None if options are unavailable
    """
    tick = mt5.symbol_info_tick(underlying)
    if tick is None:
        logger.error(f"Cannot get spot tick for {underlying}")
        return None

    spot = (tick.bid + tick.ask) / 2.0

    call, put = find_atm_options(underlying, expiry, strike_override)
    if call is None or put is None:
        logger.error(
            f"Options unavailable for {underlying} expiry={expiry}. "
            "Hedge cannot be constructed."
        )
        return None

    parity = compute_parity(spot, call, put, risk_free_rate, days_to_expiry)

    if not parity.parity_holds:
        logger.warning(
            f"Put-Call Parity deviation detected: diff={parity.parity_diff:.5f}. "
            f"Hedge may be imperfect. C={parity.call_price:.5f} P={parity.put_price:.5f} "
            f"PV(X)={parity.pv_strike:.5f} S={spot:.5f}"
        )
    else:
        logger.info(
            f"Parity holds (diff={parity.parity_diff:.5f}). "
            f"C+PV(X)={parity.lhs:.5f} ≈ P+S={parity.rhs:.5f}"
        )

    hedge_lots = round(trade_lots * hedge_ratio, 2)

    if trade_direction == "buy":
        # Spot BUY → Synthetic Short: Buy Put + Sell Call
        direction = "synthetic_short"
        call_action = "sell"
        put_action  = "buy"
        # Net cost: pay put premium, receive call premium
        net_cost = put.ask - call.bid
    else:
        # Spot SELL → Synthetic Long: Buy Call + Sell Put
        direction = "synthetic_long"
        call_action = "buy"
        put_action  = "sell"
        # Net cost: pay call premium, receive put premium
        net_cost = call.ask - put.bid

    hedge = SyntheticHedge(
        direction=direction,
        call_contract=call,
        put_contract=put,
        call_action=call_action,
        put_action=put_action,
        net_cost=net_cost,
        parity=parity,
        hedge_ratio=hedge_lots,
    )

    logger.info(
        f"Synthetic hedge built: {direction.upper()} | "
        f"{call_action.upper()} {call.symbol} @ {call.mid:.5f} | "
        f"{put_action.upper()} {put.symbol} @ {put.mid:.5f} | "
        f"Net cost: {net_cost:+.5f} | Hedge lots: {hedge_lots}"
    )

    return hedge


# ============================================================================
# HEDGE EXECUTION
# ============================================================================

def execute_synthetic_hedge(hedge: SyntheticHedge, lots: float) -> dict:
    """
    Execute both legs of the synthetic hedge on MT5.

    Args:
        hedge: SyntheticHedge from build_synthetic_hedge()
        lots:  Lot size for the option legs

    Returns:
        dict with 'call_result' and 'put_result'
    """
    results = {"call_result": None, "put_result": None, "success": False}

    # --- Execute CALL leg ---
    call_result = _place_option_order(
        hedge.call_contract, hedge.call_action, lots
    )
    results["call_result"] = call_result

    if call_result is None or call_result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"Call leg failed: {call_result.retcode if call_result else 'None'}. "
            "Aborting hedge — put leg will NOT be placed."
        )
        return results

    # --- Execute PUT leg ---
    put_result = _place_option_order(
        hedge.put_contract, hedge.put_action, lots
    )
    results["put_result"] = put_result

    if put_result is None or put_result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"Put leg failed: {put_result.retcode if put_result else 'None'}. "
            "WARNING: Call leg is open — partial hedge. Consider closing call leg manually."
        )
        return results

    results["success"] = True
    logger.info(
        f"Both hedge legs executed successfully. "
        f"Call @ {call_result.price:.5f} | Put @ {put_result.price:.5f}"
    )
    return results


def _place_option_order(contract: OptionContract, action: str, lots: float):
    """Place a single option order (buy or sell) on MT5."""
    tick = mt5.symbol_info_tick(contract.symbol)
    if tick is None:
        logger.error(f"Cannot get tick for option: {contract.symbol}")
        return None

    if action == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       contract.symbol,
        "volume":       float(lots),
        "type":         order_type,
        "price":        float(price),
        "deviation":    50,
        "magic":        999888,
        "comment":      f"pcp-hedge-{action}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    if result is None:
        logger.error(f"order_send returned None for {contract.symbol}")
        return None

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"  ✓ {action.upper()} {contract.symbol} @ {result.price:.5f}")
    else:
        logger.error(f"  ✗ {action.upper()} {contract.symbol} failed: {result.retcode} - {result.comment}")

    return result


# ============================================================================
# PARITY MONITORING UTILITY
# ============================================================================

def check_parity_for_symbol(
    underlying: str,
    expiry: str,
    risk_free_rate: float = 0.05,
    days_to_expiry: int = 30,
) -> Optional[ParityResult]:
    """
    Standalone parity check — useful for monitoring and logging.
    Does not open any trades.
    """
    tick = mt5.symbol_info_tick(underlying)
    if tick is None:
        logger.error(f"Cannot get spot for {underlying}")
        return None

    spot = (tick.bid + tick.ask) / 2.0
    call, put = find_atm_options(underlying, expiry)

    if call is None or put is None:
        logger.warning(f"Options not available for parity check: {underlying}")
        return None

    result = compute_parity(spot, call, put, risk_free_rate, days_to_expiry)

    logger.info(
        f"[PARITY CHECK] {underlying} | S={spot:.5f} | K={result.strike:.5f} | "
        f"C={result.call_price:.5f} | P={result.put_price:.5f} | "
        f"PV(X)={result.pv_strike:.5f} | "
        f"C+PV(X)={result.lhs:.5f} | P+S={result.rhs:.5f} | "
        f"diff={result.parity_diff:+.5f} | holds={result.parity_holds}"
    )
    return result