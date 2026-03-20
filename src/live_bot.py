# src/live_bot.py
import time
import joblib
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta

from src.utils import connect_mt5, disconnect_mt5, get_latest_ticks, place_order, logger
from src.indicators import add_all_indicators
from src.pcp_hedge import (
    hedge_with_arb_check,
    execute_synthetic_hedge,
    scan_arbitrage_opportunities,
    check_parity_for_symbol,
)
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_scalping_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "signal_label_encoder.pkl")
SYMBOL_ENCODER_PATH = os.path.join(MODEL_DIR, "symbol_label_encoder.pkl")


def _get_nearest_expiry(days_ahead: int = 30) -> str:
    """Return the nearest 3rd-Friday expiry as 'YYYYMMDD'."""
    today  = datetime.now()
    target = today + timedelta(days=days_ahead)
    first  = target.replace(day=1)
    fridays = [
        first + timedelta(days=d)
        for d in range(31)
        if (first + timedelta(days=d)).month == target.month
        and (first + timedelta(days=d)).weekday() == 4
    ]
    expiry_date = fridays[2] if len(fridays) >= 3 else target
    return expiry_date.strftime("%Y%m%d")


def run_live(
    symbol,
    lots=0.01,
    sl_pips=8,
    tp_pips=12,
    window=50,
    poll_interval=30,
    login=None,
    password=None,
    server=None,
    path=None,
    # ── PCP parameters ────────────────────────────────────────────────────
    use_pcp_hedge=True,
    use_arb_detection=True,
    auto_execute_arb=False,        # True = fire arb trades automatically
    hedge_ratio=1.0,
    risk_free_rate=0.05,
    option_days_to_expiry=30,
    option_expiry=None,
    min_arb_profit=0.0003,         # min net profit to execute an arb signal
    arb_scan_symbols=None,         # extra symbols to scan for arb (besides trading symbol)
):
    """
    Live trading loop with:
      1. Arbitrage detection  — scans for C+PV(X) ≠ P+S violations
      2. Synthetic hedging    — hedges every spot position via options
      3. Combined mode        — hedge_with_arb_check() runs both atomically

    After every spot BUY  → Synthetic Short (Buy Put + Sell Call)
    After every spot SELL → Synthetic Long  (Buy Call + Sell Put)
    If parity is violated at the time of hedging → arb legs are also fired
    (when auto_execute_arb=True).

    Args:
        use_arb_detection:  Enable standalone arb scan at startup and each poll cycle
        auto_execute_arb:   Automatically execute arb trades when executable signal found
        arb_scan_symbols:   Additional symbols scanned for arb each cycle (e.g. ['GBPUSD'])
        min_arb_profit:     Minimum net profit for an arb signal to be executable
    """
    ok = connect_mt5(login=login, password=password, server=server, path=path)
    if not ok:
        raise SystemExit("MT5 connect failed")

    # ── Load model ────────────────────────────────────────────────────────
    model_data = joblib.load(MODEL_PATH)
    if isinstance(model_data, dict):
        clf           = model_data['model']
        feature_names = model_data.get('feature_names', [])
        logger.info(f"Model loaded — {len(feature_names)} features, "
                    f"trained {model_data.get('training_date','unknown')}")
    else:
        clf, feature_names = model_data, []

    label_enc  = joblib.load(LABEL_ENCODER_PATH)
    symbol_enc = joblib.load(SYMBOL_ENCODER_PATH) if os.path.exists(SYMBOL_ENCODER_PATH) else None

    # ── Resolve expiry ────────────────────────────────────────────────────
    expiry = option_expiry or _get_nearest_expiry(option_days_to_expiry)
    scan_symbols = list({symbol} | set(arb_scan_symbols or []))

    logger.info(
        f"Live loop starting — symbol={symbol} lots={lots} | "
        f"PCP hedge={'ON' if use_pcp_hedge else 'OFF'} | "
        f"Arb detection={'ON' if use_arb_detection else 'OFF'} | "
        f"Auto-execute arb={auto_execute_arb} | "
        f"Expiry={expiry} | Hedge ratio={hedge_ratio}"
    )

    # ── Startup checks ────────────────────────────────────────────────────
    if use_arb_detection:
        logger.info(f"Startup arb scan on: {scan_symbols}")
        scan_arbitrage_opportunities(
            symbols=scan_symbols,
            expiry=expiry,
            risk_free_rate=risk_free_rate,
            days_to_expiry=option_days_to_expiry,
            min_profit_threshold=min_arb_profit,
            auto_execute=False,     # never auto-execute at startup, observe only
            lots=lots,
        )
    elif use_pcp_hedge:
        check_parity_for_symbol(symbol, expiry, risk_free_rate, option_days_to_expiry)

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            # Periodic arb scan (independent of whether a trade fires)
            if use_arb_detection:
                scan_arbitrage_opportunities(
                    symbols=scan_symbols,
                    expiry=expiry,
                    risk_free_rate=risk_free_rate,
                    days_to_expiry=option_days_to_expiry,
                    min_profit_threshold=min_arb_profit,
                    auto_execute=auto_execute_arb,
                    lots=lots,
                )

            # ── Market data + feature engineering ────────────────────────
            df = get_latest_ticks(symbol, n=window)
            if df.empty or len(df) < 10:
                logger.warning("Not enough data, sleeping")
                time.sleep(poll_interval)
                continue

            df['symbol'] = symbol
            df = add_all_indicators(df)
            latest = df.iloc[-1].copy()

            feature_cols = feature_names or (
                ['open','high','low','close','volume','hl_range','oc_change','return',
                 'ema_5','ema_20','sma_5','sma_20','rsi','atr']
                + (['symbol_enc'] if symbol_enc is not None else [])
            )

            if symbol_enc is not None and 'symbol_enc' in feature_cols:
                try:
                    latest['symbol_enc'] = symbol_enc.transform([symbol])[0]
                except Exception as e:
                    logger.warning(f"Symbol encoding failed: {e}")
                    latest['symbol_enc'] = 0

            for col in feature_cols:
                if col not in latest:
                    latest[col] = 0

            X_values = []
            for col in feature_cols:
                try:
                    X_values.append(float(latest[col]))
                except (ValueError, TypeError):
                    X_values.append(0.0)

            X      = np.array(X_values).reshape(1, -1)
            pred   = clf.predict(X)[0]
            signal_label = label_enc.inverse_transform([pred])[0]

            logger.info(f"{symbol} close={latest['close']:.5f} → {signal_label}")

            # ── Skip if position already open ─────────────────────────────
            if mt5.positions_get(symbol=symbol):
                logger.info("Position open — skipping new entry")
                time.sleep(poll_interval)
                continue

            if signal_label == 'hold':
                logger.info("Hold — no trade")
                time.sleep(poll_interval)
                continue

            # ── Fetch tick / symbol info ──────────────────────────────────
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick for {symbol}")
                time.sleep(poll_interval)
                continue

            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"No symbol info for {symbol}")
                time.sleep(poll_interval)
                continue

            digits   = symbol_info.digits
            pip_size = 0.0001 if digits == 5 else 0.001

            # ── Execute spot order ────────────────────────────────────────
            spot_result = None

            if signal_label == 'buy':
                price = tick.ask
                sl, tp = price - sl_pips * pip_size, price + tp_pips * pip_size
                spot_result = place_order(symbol, 'buy', lots=lots, price=price, sl=sl, tp=tp)
                if spot_result and spot_result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"BUY @ {price:.5f} | SL={sl:.5f} TP={tp:.5f}")
                else:
                    logger.warning(f"BUY failed: {spot_result.retcode if spot_result else 'None'}")

            elif signal_label == 'sell':
                price = tick.bid
                sl, tp = price + sl_pips * pip_size, price - tp_pips * pip_size
                spot_result = place_order(symbol, 'sell', lots=lots, price=price, sl=sl, tp=tp)
                if spot_result and spot_result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"SELL @ {price:.5f} | SL={sl:.5f} TP={tp:.5f}")
                else:
                    logger.warning(f"SELL failed: {spot_result.retcode if spot_result else 'None'}")

            # ── PCP: arbitrage check + synthetic hedge ────────────────────
            if (
                use_pcp_hedge
                and spot_result is not None
                and spot_result.retcode == mt5.TRADE_RETCODE_DONE
            ):
                logger.info(f"Running hedge_with_arb_check() for {signal_label.upper()} position...")

                pcp_result = hedge_with_arb_check(
                    trade_direction=signal_label,
                    underlying=symbol,
                    expiry=expiry,
                    trade_lots=lots,
                    risk_free_rate=risk_free_rate,
                    days_to_expiry=option_days_to_expiry,
                    hedge_ratio=hedge_ratio,
                    min_profit_threshold=min_arb_profit,
                    auto_execute_arb=auto_execute_arb,
                )

                arb    = pcp_result["arb_signal"]
                hedge  = pcp_result["hedge"]

                # Log arbitrage outcome
                if arb is not None:
                    if arb.is_executable and auto_execute_arb:
                        logger.info(
                            f"[ARB] Executed — direction={arb.direction} "
                            f"net_profit={arb.net_profit:.5f}"
                        )
                    elif arb.is_executable:
                        logger.warning(
                            f"[ARB] Executable opportunity found but auto_execute_arb=False. "
                            f"net={arb.net_profit:.5f} | Pass --auto-execute-arb to enable."
                        )
                    else:
                        logger.info(
                            f"[ARB] Violation detected but below profit threshold "
                            f"(net={arb.net_profit:.5f})"
                        )

                # Execute hedge
                if hedge is not None:
                    hedge_result = execute_synthetic_hedge(hedge, lots=hedge.hedge_ratio)
                    if hedge_result["success"]:
                        logger.info(
                            f"[HEDGE] {hedge.direction.upper()} complete | "
                            f"net_cost={hedge.net_cost:+.5f}"
                        )
                    else:
                        logger.warning("[HEDGE] Partial failure — review open option legs")
                else:
                    logger.warning(
                        "[HEDGE] Options unavailable — spot running WITHOUT hedge. "
                        "Verify option symbols for your broker."
                    )

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Stopped via KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        disconnect_mt5()