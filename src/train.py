# src/train.py
import joblib
import os
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from src.preprocess import load_dataset, prepare_features
from src.indicators import add_all_indicators

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(csv_path, model_out_path=None):
    df = load_dataset(csv_path)
    df, symbol_le = prepare_features(df)
    df = add_all_indicators(df)

    feature_cols = ['open','high','low','close','volume','hl_range','oc_change','return',
                    'ema_5','ema_20','sma_5','sma_20','rsi','atr','symbol_enc']

    X = df[feature_cols].fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(df['signal'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=-1, 
        random_state=42
    )
    
    clf.fit(X_train, y_train)

    model_data = {
        'model': clf,
        'feature_names': feature_cols,
        'feature_count': len(feature_cols),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'X_train_columns': X_train.columns.tolist()  # Save actual column order
    }

    if model_out_path is None:
        model_out_path = os.path.join(MODEL_DIR, "trained_scalping_model.pkl")
    joblib.dump(model_data, model_out_path)
    joblib.dump(le, os.path.join(MODEL_DIR, "signal_label_encoder.pkl"))
    joblib.dump(symbol_le, os.path.join(MODEL_DIR, "symbol_label_encoder.pkl"))

    print("Training complete.")
    print("Train Acc:", clf.score(X_train, y_train))
    print("Test Acc:", clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nSignal distribution in training data:")
    print(df['signal'].value_counts())
    
    return clf

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/scalping_large_dataset.csv"
    train_model(csv)
