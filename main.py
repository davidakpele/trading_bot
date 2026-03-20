# main.py
import argparse
import os
import sys
import MetaTrader5 as mt5
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.generate_synthetic import main as gen_main
from src.train import train_model
from src.live_bot import run_live
from src.monitor import TradingMonitor
from src.pcp_hedge import (
    check_parity_for_symbol,
    scan_arbitrage_opportunities,
    detect_arbitrage,
)


def run_monitor_with_path(symbol=None, refresh_interval=5, mt5_path=None):
    if mt5_path:
        if not mt5.initialize(mt5_path):
            print(f"Failed to initialize MT5: {mt5_path}"); return
    else:
        if not mt5.initialize():
            print("Failed to initialize MT5"); return
    print("MT5 initialized")
    TradingMonitor(symbol=symbol).run_monitor(refresh_interval=refresh_interval)
    mt5.shutdown()


def run_parity_check(symbol, expiry, risk_free_rate, days_to_expiry, mt5_path=None):
    """Print a full parity breakdown — no trades placed."""
    if mt5_path:
        if not mt5.initialize(mt5_path): print("MT5 init failed"); return
    else:
        if not mt5.initialize(): print("MT5 init failed"); return

    result = check_parity_for_symbol(symbol, expiry, risk_free_rate, days_to_expiry)
    if result:
        print("\n" + "=" * 65)
        print("  PUT-CALL PARITY  —  C + PV(X) = P + S")
        print("=" * 65)
        print(f"  Underlying  : {symbol}")
        print(f"  Spot   (S)  : {result.spot:.5f}")
        print(f"  Strike (X)  : {result.strike:.5f}")
        print(f"  PV(X)       : {result.pv_strike:.5f}  [X·e^(-rT), r={result.risk_free_rate}, T={result.time_to_expiry:.4f}yr]")
        print(f"  Call   (C)  : {result.call_price:.5f}")
        print(f"  Put    (P)  : {result.put_price:.5f}")
        print(f"  LHS C+PV(X) : {result.lhs:.5f}")
        print(f"  RHS P+S     : {result.rhs:.5f}")
        print(f"  Difference  : {result.parity_diff:+.5f}")
        print(f"  Parity holds: {'YES ✓' if result.parity_holds else 'NO  ✗  (violation detected)'}")
        print("=" * 65)
    else:
        print("Parity check failed — options may not be available.")
    mt5.shutdown()


def run_arb_scan(symbols, expiry, risk_free_rate, days_to_expiry,
                 min_profit, mt5_path=None):
    """Scan a list of symbols for executable arbitrage opportunities."""
    if mt5_path:
        if not mt5.initialize(mt5_path): print("MT5 init failed"); return
    else:
        if not mt5.initialize(): print("MT5 init failed"); return

    print(f"\nScanning {len(symbols)} symbol(s) for arbitrage...\n")
    signals = scan_arbitrage_opportunities(
        symbols=symbols,
        expiry=expiry,
        risk_free_rate=risk_free_rate,
        days_to_expiry=days_to_expiry,
        min_profit_threshold=min_profit,
        auto_execute=False,
    )

    if not signals:
        print("No parity violations found. Markets look efficient.")
    else:
        print(f"\n{'='*65}")
        print(f"  {len(signals)} violation(s) found")
        print(f"{'='*65}")
        for sig in signals:
            tag = "EXECUTABLE ✓" if sig.is_executable else "below threshold"
            print(f"\n  {sig.underlying} | {sig.direction.upper()} | {tag}")
            print(f"  diff={sig.parity.parity_diff:+.5f} | gross={sig.gross_profit:.5f} | "
                  f"costs≈{sig.transaction_cost_estimate:.5f} | net={sig.net_profit:+.5f}")
            print("  Legs:")
            for sym, action, reason in sig.legs:
                print(f"    {action.upper():8s} {sym:35s}  {reason}")
        print(f"\n  Executable: {sum(1 for s in signals if s.is_executable)}/{len(signals)}")
        print(f"{'='*65}")

    mt5.shutdown()


def run():
    parser = argparse.ArgumentParser(description="Scalping Bot — PCP Arbitrage + Hedging")

    parser.add_argument("action", choices=[
        "gen-data", "train", "live", "monitor", "parity-check", "arb-scan"
    ])

    # General
    parser.add_argument("--csv",           default="data/scalping_large_dataset.csv")
    parser.add_argument("--symbol",        default="EURUSD")
    parser.add_argument("--lots",          type=float, default=0.01)
    parser.add_argument("--sl",            type=int,   default=8)
    parser.add_argument("--tp",            type=int,   default=12)
    parser.add_argument("--minutes",       type=int,   default=3000)
    parser.add_argument("--mt5-path",      default=None)
    parser.add_argument("--interval",      type=int,   default=5)
    parser.add_argument("--poll-interval", type=int,   default=10)

    # PCP / arbitrage
    parser.add_argument("--no-hedge",         action="store_true",
                        help="Disable synthetic hedge")
    parser.add_argument("--no-arb",           action="store_true",
                        help="Disable arbitrage detection scan")
    parser.add_argument("--auto-execute-arb", action="store_true",
                        help="Auto-execute arbitrage trades when found")
    parser.add_argument("--hedge-ratio",      type=float, default=1.0)
    parser.add_argument("--risk-free-rate",   type=float, default=0.05)
    parser.add_argument("--option-days",      type=int,   default=30)
    parser.add_argument("--option-expiry",    type=str,   default=None,
                        help="Option expiry YYYYMMDD (auto if omitted)")
    parser.add_argument("--min-arb-profit",   type=float, default=0.0003,
                        help="Min net profit to treat arb as executable")
    parser.add_argument("--arb-symbols",      type=str,   default=None,
                        help="Comma-separated symbols for arb scan, e.g. EURUSD,GBPUSD")

    args = parser.parse_args()

    # Resolve expiry
    if args.option_expiry is None:
        from datetime import datetime, timedelta
        expiry = (datetime.now() + timedelta(days=args.option_days)).strftime("%Y%m%d")
    else:
        expiry = args.option_expiry

    # Resolve arb symbol list
    arb_symbols = (
        [s.strip() for s in args.arb_symbols.split(",")]
        if args.arb_symbols else [args.symbol]
    )

    # ── Actions ──────────────────────────────────────────────────────────

    if args.action == "gen-data":
        gen_main(out_path=args.csv, minutes_per_symbol=args.minutes)

    elif args.action == "train":
        train_model(args.csv)

    elif args.action == "live":
        print(f"  Hedge        : {'OFF' if args.no_hedge else 'ON'}")
        print(f"  Arb detect   : {'OFF' if args.no_arb else 'ON'}")
        print(f"  Auto-exec arb: {args.auto_execute_arb}")
        print(f"  Expiry       : {expiry}")
        run_live(
            symbol=args.symbol,
            lots=args.lots,
            sl_pips=args.sl,
            tp_pips=args.tp,
            poll_interval=args.poll_interval,
            path=args.mt5_path,
            use_pcp_hedge=not args.no_hedge,
            use_arb_detection=not args.no_arb,
            auto_execute_arb=args.auto_execute_arb,
            hedge_ratio=args.hedge_ratio,
            risk_free_rate=args.risk_free_rate,
            option_days_to_expiry=args.option_days,
            option_expiry=expiry,
            min_arb_profit=args.min_arb_profit,
            arb_scan_symbols=arb_symbols,
        )

    elif args.action == "monitor":
        run_monitor_with_path(args.symbol, args.interval, args.mt5_path)

    elif args.action == "parity-check":
        run_parity_check(args.symbol, expiry, args.risk_free_rate,
                         args.option_days, args.mt5_path)

    elif args.action == "arb-scan":
        run_arb_scan(arb_symbols, expiry, args.risk_free_rate,
                     args.option_days, args.min_arb_profit, args.mt5_path)


if __name__ == "__main__":
    run()