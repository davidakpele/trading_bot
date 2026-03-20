# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_dataset(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

def prepare_features(df):
    df = df.copy()
    df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)
    df['hl_range'] = df['high'] - df['low']
    df['oc_change'] = df['close'] - df['open']
    df['return'] = df.groupby('symbol')['close'].pct_change().fillna(0)
    le = LabelEncoder()
    df['symbol_enc'] = le.fit_transform(df['symbol'])
    return df, le
