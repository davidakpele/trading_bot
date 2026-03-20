# src/live_bot.py
import time
import joblib
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta

from src.utils import connect_mt5, disconnect_mt5, get_latest_ticks, place_order, logger
from src.indicators import add_all_indicators
from src.pcp_hedge import build_synthetic_hedge, execute_synthetic_hedge, check_parity_for_symbol
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_scalping_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "signal_label_encoder.pkl")
SYMBOL_ENCODER_PATH = os.path.join(MODEL_DIR, "symbol_label_encoder.pkl")


def _get_nearest_expiry(days_ahead: int = 30) -> str:
    """
    Return the nearest standard option expiry (3rd Friday of next month)
    as a string 'YYYYMMDD'. Falls back to days_ahead if no Friday found.
    """
    today = datetime.now()
    target = today + timedelta(days=days_ahead)

    # Walk forward to the 3rd Friday of the target month
    first_of_month = target.replace(day=1)
    fridays = [
        first_of_month + timedelta(days=d)
        for d in range(31)
        if (first_of_month + timedelta(days=d)).month == target.month
        and (first_of_month + timedelta(days=d)).weekday() == 4  # Friday
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
    # ── PCP hedge parameters ──────────────────────────────────────────────
    use_pcp_hedge=True,          # Master switch for hedging
    hedge_ratio=1.0,             # Fraction of position to hedge (0.0 – 1.0)
    risk_free_rate=0.05,         # Annual risk-free rate for PV(X) calculation
    option_days_to_expiry=30,    # Days until option expiry
    option_expiry=None,          # Override expiry string 'YYYYMMDD'; auto if None
):
    """
    Live trading loop with optional Put-Call Parity synthetic hedge.

    After every spot BUY  → opens Synthetic Short (Buy Put + Sell Call)
    After every spot SELL → opens Synthetic Long  (Buy Call + Sell Put)

    The hedge is sized by hedge_ratio * lots and uses the nearest ATM
    options for the given underlying on MT5.

    Args:
        use_pcp_hedge:         Enable/disable the hedge entirely
        hedge_ratio:           1.0 = full hedge, 0.5 = half hedge, etc.
        risk_free_rate:        Used in PV(X) = X * e^(-rT)
        option_days_to_expiry: Calendar days to expiry (used for PV calc)
        option_expiry:         Exact expiry string; auto-computed if None
    """
    # ── Connect ──────────────────────────────────────────────────────────
    ok = connect_mt5(login=login, password=password, server=server, path=path)
    if not ok:
        raise SystemExit("MT5 connect failed")

    # ── Load model ───────────────────────────────────────────────────────
    model_data = joblib.load(MODEL_PATH)
    if isinstance(model_data, dict):
        clf = model_data['model']
        feature_names = model_data.get('feature_names', [])
        logger.info(
            f"Loaded model with {len(feature_names)} features, "
            f"trained on {model_data.get('training_date', 'unknown date')}"
        )
    else:
        clf = model_data
        feature_names = []
        logger.info("Loaded model (old format)")

    label_enc = joblib.load(LABEL_ENCODER_PATH)
    symbol_enc = None
    if os.path.exists(SYMBOL_ENCODER_PATH):
        symbol_enc = joblib.load(SYMBOL_ENCODER_PATH)

    # ── Resolve option expiry ─────────────────────────────────────────────
    expiry = option_expiry or _get_nearest_expiry(option_days_to_expiry)
    logger.info(
        f"Starting live loop for {symbol} (lots={lots}) | "
        f"PCP hedge={'ON' if use_pcp_hedge else 'OFF'} | "
        f"Expiry={expiry} | HedgeRatio={hedge_ratio}"
    )

    if use_pcp_hedge:
        # Run a parity check at startup so we know options are reachable
        logger.info("Running startup parity check...")
        check_parity_for_symbol(symbol, expiry, risk_free_rate, option_days_to_expiry)

    try:
        while True:
            # ── Fetch market data ─────────────────────────────────────────
            df = get_latest_ticks(symbol, n=window)
            if df.empty or len(df) < 10:
                logger.warning("Not enough data, sleeping")
                time.sleep(poll_interval)
                continue

            df['symbol'] = symbol
            df = add_all_indicators(df)
            latest = df.iloc[-1].copy()

            # ── Build feature vector ──────────────────────────────────────
            if feature_names:
                feature_cols = feature_names
            else:
                feature_cols = [
                    'open', 'high', 'low', 'close', 'volume',
                    'hl_range', 'oc_change', 'return',
                    'ema_5', 'ema_20', 'sma_5', 'sma_20', 'rsi', 'atr'
                ]
                if symbol_enc is not None:
                    feature_cols.append('symbol_enc')

            if symbol_enc is not None and 'symbol_enc' in feature_cols:
                try:
                    latest['symbol_enc'] = symbol_enc.transform([symbol])[0]
                except Exception as e:
                    logger.warning(f"Symbol encoding failed: {e}")
                    latest['symbol_enc'] = 0

            for col in feature_cols:
                if col not in latest:
                    logger.warning(f"Missing feature: {col}, setting to 0")
                    latest[col] = 0

            X_values = []
            for col in feature_cols:
                try:
                    X_values.append(float(latest[col]))
                except (ValueError, TypeError):
                    X_values.append(0.0)

            X = np.array(X_values).reshape(1, -1)
            pred = clf.predict(X)[0]
            signal = label_enc.inverse_transform([pred])[0]

            logger.info(f"{symbol} close={latest['close']:.5f} predicted => {signal}")

            # ── Skip if position already open ─────────────────────────────
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                logger.info(f"Already have {len(positions)} open position(s), skipping")
                time.sleep(poll_interval)
                continue

            if signal == 'hold':
                logger.info("Hold signal - no trade executed")
                time.sleep(poll_interval)
                continue

            # ── Get tick / symbol info ────────────────────────────────────
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"Could not get tick for {symbol}")
                time.sleep(poll_interval)
                continue

            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Could not get symbol info for {symbol}")
                time.sleep(poll_interval)
                continue

            digits = symbol_info.digits
            pip_size = 0.0001 if digits == 5 else 0.001

            # ── Execute spot order ────────────────────────────────────────
            spot_result = None

            if signal == 'buy':
                price = tick.ask
                sl = price - sl_pips * pip_size
                tp = price + tp_pips * pip_size
                spot_result = place_order(symbol, 'buy', lots=lots, price=price, sl=sl, tp=tp)

                if spot_result and spot_result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(
                        f"BUY executed @ {price:.5f} | SL={sl:.5f} | TP={tp:.5f}"
                    )
                else:
                    logger.warning(
                        f"BUY failed: {spot_result.retcode if spot_result else 'None'}"
                    )

            elif signal == 'sell':
                price = tick.bid
                sl = price + sl_pips * pip_size
                tp = price - tp_pips * pip_size
                spot_result = place_order(symbol, 'sell', lots=lots, price=price, sl=sl, tp=tp)

                if spot_result and spot_result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(
                        f"SELL executed @ {price:.5f} | SL={sl:.5f} | TP={tp:.5f}"
                    )
                else:
                    logger.warning(
                        f"SELL failed: {spot_result.retcode if spot_result else 'None'}"
                    )

            # ── Build & execute PCP synthetic hedge ───────────────────────
            if (
                use_pcp_hedge
                and spot_result is not None
                and spot_result.retcode == mt5.TRADE_RETCODE_DONE
            ):
                logger.info(
                    f"Building PCP synthetic hedge for {signal.upper()} position..."
                )

                hedge = build_synthetic_hedge(
                    trade_direction=signal,        # 'buy' or 'sell'
                    underlying=symbol,
                    expiry=expiry,
                    trade_lots=lots,
                    risk_free_rate=risk_free_rate,
                    days_to_expiry=option_days_to_expiry,
                    hedge_ratio=hedge_ratio,
                )

                if hedge is not None:
                    hedge_results = execute_synthetic_hedge(hedge, lots=hedge.hedge_ratio)

                    if hedge_results["success"]:
                        logger.info(
                            f"Hedge complete: {hedge.direction.upper()} | "
                            f"Net cost = {hedge.net_cost:+.5f} | "
                            f"C+PV(X) = {hedge.parity.lhs:.5f} | "
                            f"P+S = {hedge.parity.rhs:.5f}"
                        )
                    else:
                        logger.warning(
                            "Hedge execution partially failed — spot position is unhedged. "
                            "Check logs for details."
                        )
                else:
                    logger.warning(
                        "Options unavailable — spot position is running WITHOUT a hedge. "
                        "Verify option symbols on your broker/MT5 terminal."
                    )

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Stopping live loop via KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        disconnect_mt5()