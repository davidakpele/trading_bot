# src/indicators.py
import pandas as pd
import numpy as np

def add_all_indicators(df):
    df = df.copy()
    
    # Ensure data is sorted
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Basic price features
    df['hl_range'] = df['high'] - df['low']
    df['oc_change'] = df['close'] - df['open']
    df['return'] = df.groupby('symbol')['close'].pct_change().fillna(0)
    
    # Moving averages
    df['ema_5'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['sma_5'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['sma_20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    
    # RSI calculation
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    df['rsi'] = df.groupby('symbol')['close'].transform(calculate_rsi)
    
    # ATR calculation - manual approach without groupby complications
    def calculate_atr_single(df_slice):
        high_low = df_slice['high'] - df_slice['low']
        high_close_prev = (df_slice['high'] - df_slice['close'].shift()).abs()
        low_close_prev = (df_slice['low'] - df_slice['close'].shift()).abs()
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = tr.rolling(window=14, min_periods=1).mean()
        return atr.fillna(0)
    
    # Apply ATR calculation per symbol group
    atr_results = []
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        atr = calculate_atr_single(symbol_data)
        atr_results.extend(atr.values)
    
    df['atr'] = atr_results
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    return df