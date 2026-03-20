# src/predict.py
import joblib
import pandas as pd
from src.indicators import add_all_indicators

MODEL_DIR = "../models"

def predict_from_row(row_dict):
    clf = joblib.load(f"{MODEL_DIR}/trained_scalping_model.pkl")
    label_enc = joblib.load(f"{MODEL_DIR}/signal_label_encoder.pkl")
    symbol_enc = joblib.load(f"{MODEL_DIR}/symbol_label_encoder.pkl")

    df = pd.DataFrame([row_dict])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['symbol_enc'] = symbol_enc.transform([df.loc[0,'symbol']])[0] if 'symbol' in df.columns else 0
    df = add_all_indicators(df)
    features = ['open','high','low','close','volume','hl_range','oc_change','return',
                'ema_5','ema_20','sma_5','sma_20','rsi','atr','symbol_enc']
    X = df[features].fillna(0)
    pred = clf.predict(X)[0]
    return label_enc.inverse_transform([pred])[0]
