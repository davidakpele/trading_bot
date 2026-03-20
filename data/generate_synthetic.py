# data/generate_synthetic.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_symbol_series(symbol, start_dt, minutes, start_price, tick_vol_mean=150):
    timestamps = [start_dt + timedelta(minutes=i) for i in range(minutes)]
    prices = [start_price]
    for i in range(1, minutes):
        change = np.random.normal(loc=0.0, scale=0.0004)
        if np.random.rand() < 0.001:
            change += np.random.normal(loc=0, scale=0.002)
        prices.append(prices[-1] * (1 + change))
    prices = np.array(prices)
    opens = prices
    closes = prices * (1 + np.random.normal(0, 0.0001, size=minutes))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.0002, size=minutes)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.0002, size=minutes)))
    volumes = np.random.poisson(lam=tick_vol_mean, size=minutes)
    df = pd.DataFrame({
        "symbol": symbol,
        "timestamp": timestamps,
        "open": opens.round(5),
        "high": highs.round(5),
        "low": lows.round(5),
        "close": closes.round(5),
        "volume": volumes
    })
    return df

def label_signals(df):
    df = df.copy()
    df['sma5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['sma20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['signal'] = 'hold'
    df.loc[df['sma5'] > df['sma20'] * 1.00005, 'signal'] = 'buy'
    df.loc[df['sma5'] < df['sma20'] * 0.99995, 'signal'] = 'sell'
    # random noise labels to emulate imperfect labels
    rand_idx = np.random.choice(df.index, size=int(len(df)*0.01), replace=False)
    df.loc[rand_idx, 'signal'] = np.random.choice(['buy','sell','hold'], size=len(rand_idx))
    return df.drop(columns=['sma5','sma20'])

def main(out_path=None, minutes_per_symbol=3000):
    symbols = {
        "EURUSD": 1.1000,
        "GBPUSD": 1.2500,
        "USDJPY": 134.50
    }
    start_dt = datetime(2025, 1, 1, 0, 0)
    frames = []
    for sym, price in symbols.items():
        df_sym = generate_symbol_series(sym, start_dt, minutes_per_symbol, price)
        df_sym = label_signals(df_sym)
        frames.append(df_sym)
    big_df = pd.concat(frames).reset_index(drop=True)
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "scalping_large_dataset.csv")
    big_df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}. Rows: {len(big_df)}")

if __name__ == "__main__":
    main()
