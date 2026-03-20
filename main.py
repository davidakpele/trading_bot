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
from src.pcp_hedge import check_parity_for_symbol


def run_monitor_with_path(symbol=None, refresh_interval=5, mt5_path=None):
    """Wrapper function to handle MT5 path for monitor"""
    if mt5_path:
        if not mt5.initialize(mt5_path):
            print(f"Failed to initialize MT5 with path: {mt5_path}")
            return
    else:
        if not mt5.initialize():
            print("Failed to initialize MT5")
            return

    print("MT5 initialized successfully")
    monitor = TradingMonitor(symbol=symbol)
    monitor.run_monitor(refresh_interval=refresh_interval)
    mt5.shutdown()


def run_parity_check(symbol, expiry, risk_free_rate, days_to_expiry, mt5_path=None):
    """Standalone parity check action."""
    if mt5_path:
        if not mt5.initialize(mt5_path):
            print(f"Failed to initialize MT5: {mt5_path}")
            return
    else:
        if not mt5.initialize():
            print("Failed to initialize MT5")
            return

    print(f"Running Put-Call Parity check for {symbol}...")
    result = check_parity_for_symbol(symbol, expiry, risk_free_rate, days_to_expiry)

    if result:
        print("\n" + "=" * 60)
        print("PUT-CALL PARITY RESULT")
        print("=" * 60)
        print(f"  Formula  : C + PV(X) = P + S")
        print(f"  Spot (S) : {result.spot:.5f}")
        print(f"  Strike(X): {result.strike:.5f}")
        print(f"  PV(X)    : {result.pv_strike:.5f}  [X*e^(-rT), r={result.risk_free_rate}, T={result.time_to_expiry:.4f}yr]")
        print(f"  Call (C) : {result.call_price:.5f}")
        print(f"  Put  (P) : {result.put_price:.5f}")
        print(f"  LHS C+PV(X): {result.lhs:.5f}")
        print(f"  RHS P+S    : {result.rhs:.5f}")
        print(f"  Difference : {result.parity_diff:+.5f}")
        print(f"  Parity holds: {'YES ✓' if result.parity_holds else 'NO  ✗ (arbitrage opportunity)'}")
        print("=" * 60)
    else:
        print("Parity check failed — options may not be available for this symbol.")

    mt5.shutdown()


def run():
    parser = argparse.ArgumentParser(description="Scalping Trading Bot with PCP Hedging")

    parser.add_argument(
        "action",
        choices=["gen-data", "train", "live", "monitor", "parity-check"],
        help="Action: gen-data | train | live | monitor | parity-check"
    )

    # General
    parser.add_argument("--csv",      default="data/scalping_large_dataset.csv")
    parser.add_argument("--symbol",   default="EURUSD")
    parser.add_argument("--lots",     type=float, default=0.01)
    parser.add_argument("--sl",       type=int, default=8,  help="Stop loss in pips")
    parser.add_argument("--tp",       type=int, default=12, help="Take profit in pips")
    parser.add_argument("--minutes",  type=int, default=3000)
    parser.add_argument("--mt5-path", default=None)
    parser.add_argument("--interval", type=int, default=5,  help="Monitor refresh interval (s)")
    parser.add_argument("--poll-interval", type=int, default=10, help="Live bot poll interval (s)")

    # PCP hedge flags
    parser.add_argument(
        "--no-hedge",
        action="store_true",
        help="Disable PCP synthetic hedge (hedge is ON by default)"
    )
    parser.add_argument(
        "--hedge-ratio",
        type=float, default=1.0,
        help="Fraction of spot position to hedge via options (0.0–1.0, default 1.0)"
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float, default=0.05,
        help="Annual risk-free rate used in PV(X)=X*e^(-rT) (default 0.05)"
    )
    parser.add_argument(
        "--option-days",
        type=int, default=30,
        help="Days to option expiry for PV calculation (default 30)"
    )
    parser.add_argument(
        "--option-expiry",
        type=str, default=None,
        help="Option expiry date YYYYMMDD (auto-computed from --option-days if omitted)"
    )

    args = parser.parse_args()

    # ── Actions ──────────────────────────────────────────────────────────────

    if args.action == "gen-data":
        print("Generating synthetic data...")
        gen_main(out_path=args.csv, minutes_per_symbol=args.minutes)

    elif args.action == "train":
        print("Training model...")
        train_model(args.csv)

    elif args.action == "live":
        print("Starting live trading bot...")
        print(f"  PCP Hedge : {'DISABLED' if args.no_hedge else 'ENABLED'}")
        if not args.no_hedge:
            print(f"  Hedge ratio     : {args.hedge_ratio}")
            print(f"  Risk-free rate  : {args.risk_free_rate}")
            print(f"  Option days     : {args.option_days}")
            print(f"  Option expiry   : {args.option_expiry or 'auto'}")

        run_live(
            symbol=args.symbol,
            lots=args.lots,
            sl_pips=args.sl,
            tp_pips=args.tp,
            poll_interval=args.poll_interval,
            path=args.mt5_path,
            # PCP hedge
            use_pcp_hedge=not args.no_hedge,
            hedge_ratio=args.hedge_ratio,
            risk_free_rate=args.risk_free_rate,
            option_days_to_expiry=args.option_days,
            option_expiry=args.option_expiry,
        )

    elif args.action == "monitor":
        print("Starting trading monitor...")
        run_monitor_with_path(
            symbol=args.symbol,
            refresh_interval=args.interval,
            mt5_path=args.mt5_path,
        )

    elif args.action == "parity-check":
        if args.option_expiry is None:
            from datetime import datetime, timedelta
            expiry = (datetime.now() + timedelta(days=args.option_days)).strftime("%Y%m%d")
        else:
            expiry = args.option_expiry

        run_parity_check(
            symbol=args.symbol,
            expiry=expiry,
            risk_free_rate=args.risk_free_rate,
            days_to_expiry=args.option_days,
            mt5_path=args.mt5_path,
        )

    else:
        print("Unknown action")


if __name__ == "__main__":
    run()