# models/train_xgboost.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.feature_engineering import build_features, FEATURE_COLS, LABEL_COLS

# ── Configuration ────────────────────────────────────────
INPUT_PATH  = "data/dataset.csv"
OUTPUT_DIR  = "models"

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
    # Note: use_label_encoder removed (deprecated in XGBoost >= 1.6)
}
N_SPLITS = 5
# ─────────────────────────────────────────────────────────


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute scale_pos_weight to handle class imbalance.
    = count(negative) / count(positive)
    Tells XGBoost to penalise missing UP predictions more.
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    spw = n_neg / n_pos
    print(f"  Class balance — DOWN: {n_neg}  UP: {n_pos}  "
          f"scale_pos_weight: {spw:.2f}")
    return spw


def evaluate(y_true, y_pred, y_prob, label: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    print(f"\n  [{label}] Test Results")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=["DOWN", "UP"],
                                zero_division=0))
    return {"label": label, "accuracy": acc, "f1": f1, "auc": auc}


def train_for_horizon(df: pd.DataFrame, label_col: str) -> dict:
    horizon = label_col.replace("label_", "T+")
    print(f"\n{'='*55}")
    print(f"  Training XGBoost — {horizon}")
    print(f"{'='*55}")

    X = df[FEATURE_COLS].values
    y = df[label_col].values

    # Chronological 80/20 split — no data leakage
    split_idx        = int(len(X) * 0.8)
    X_train, X_test  = X[:split_idx], X[split_idx:]
    y_train, y_test  = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # Handle class imbalance
    spw = compute_scale_pos_weight(y_train)

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # TimeSeriesSplit cross-validation
    tscv    = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        params = {**XGB_PARAMS, "scale_pos_weight": spw}
        model  = xgb.XGBClassifier(**params)
        model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            verbose=False,
        )
        prob = model.predict_proba(X_train[val_idx])[:, 1]
        try:
            auc = roc_auc_score(y_train[val_idx], prob)
        except ValueError:
            auc = 0.5
        cv_aucs.append(auc)
        print(f"  Fold {fold+1}/{N_SPLITS}  AUC={auc:.4f}")

    print(f"\n  CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

    # Final model on full training set
    final_params = {**XGB_PARAMS, "scale_pos_weight": spw}
    final_model  = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train, verbose=False)

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_prob, horizon)
    metrics["cv_auc_mean"] = np.mean(cv_aucs)
    metrics["cv_auc_std"]  = np.std(cv_aucs)

    # Feature importance
    importance = pd.Series(
        final_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(f"\n  Top 5 features:")
    print(importance.head(5).to_string())

    # Save model and scaler
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(final_model, f"{OUTPUT_DIR}/xgb_{label_col}.pkl")
    joblib.dump(scaler,      f"{OUTPUT_DIR}/scaler_{label_col}.pkl")
    print(f"\n  Saved: {OUTPUT_DIR}/xgb_{label_col}.pkl")

    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("  XGBoost Trainer — T+1 / T+3 / T+5")
    print("=" * 55)

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"\nLoaded {len(df)} rows")

    df = build_features(df)
    all_cols = FEATURE_COLS + LABEL_COLS
    df = df.dropna(subset=all_cols).reset_index(drop=True)
    print(f"After feature engineering: {len(df)} rows\n")

    if len(df) < 100:
        print("WARNING: Dataset too small (<100 rows). "
              "Results may be unreliable.")

    all_metrics = []
    for label_col in LABEL_COLS:
        metrics = train_for_horizon(df, label_col)
        all_metrics.append(metrics)

    # Summary
    print(f"\n{'='*55}")
    print("  Final Results Summary")
    print(f"{'='*55}")
    print(f"{'Horizon':<10} {'Accuracy':<12} {'F1':<10} "
          f"{'AUC-ROC':<12} {'CV AUC'}")
    print("-" * 58)
    for m in all_metrics:
        print(f"{m['label']:<10} {m['accuracy']:<12.4f} {m['f1']:<10.4f} "
              f"{m['auc']:<12.4f} "
              f"{m['cv_auc_mean']:.4f}±{m['cv_auc_std']:.4f}")

    print("\nAll models saved to models/")
    print("Done!")
