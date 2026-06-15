# models/train_lstm.py
"""
LSTM model for stock movement prediction.
Uses sequences of 20 trading days to predict T+1 / T+3 / T+5.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report)
import joblib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.build_historical_dataset import FEATURE_COLS_HIST, LABEL_COLS

# ── Configuration ────────────────────────────────────────
INPUT_PATH  = "data/dataset_historical.csv"
OUTPUT_DIR  = "models"

SEQ_LEN     = 20        # Look back 20 trading days
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
PATIENCE    = 8         # Early stopping patience

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = FEATURE_COLS_HIST + ["ticker_enc"]
# ─────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────
class StockSequenceDataset(Dataset):
    """
    Creates (sequence, label) pairs per ticker.
    Each sequence = SEQ_LEN consecutive days of features.
    """
    def __init__(self, df: pd.DataFrame,
                 feature_cols: list[str],
                 label_col: str,
                 seq_len: int = SEQ_LEN):
        self.sequences = []
        self.labels    = []

        for ticker in df["ticker"].unique():
            t = (df[df["ticker"] == ticker]
                 .sort_values("date")
                 .reset_index(drop=True))

            X = t[feature_cols].values.astype(np.float32)
            y = t[label_col].values.astype(np.float32)

            for i in range(seq_len, len(t)):
                self.sequences.append(X[i - seq_len:i])
                self.labels.append(y[i])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels    = np.array(self.labels,    dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx]),
                torch.tensor(self.labels[idx]))


# ── Model ─────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dim: int  = HIDDEN_DIM,
                 num_layers: int  = NUM_LAYERS,
                 dropout: float   = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            dropout      = dropout if num_layers > 1 else 0,
            batch_first  = True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        last    = out[:, -1, :]   # Take last timestep
        return self.head(last).squeeze(1)


# ── Training ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    all_probs  = []
    all_labels = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        probs   = model(X_batch).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_batch.numpy())
    return np.array(all_probs), np.array(all_labels)


def train_for_horizon(df: pd.DataFrame,
                      label_col: str,
                      scaler: StandardScaler) -> dict:
    horizon = label_col.replace("label_", "T+")
    print(f"\n{'='*55}")
    print(f"  Training LSTM — {horizon}")
    print(f"{'='*55}")
    print(f"  Device: {DEVICE.upper()}")

    # Chronological 80/20 split
    dates      = df["date"].unique()
    dates = sorted(dates)
    split_date = dates[int(len(dates) * 0.8)]
    train_df   = df[df["date"] < split_date]
    test_df    = df[df["date"] >= split_date]

    print(f"  Train dates: {train_df['date'].min()} to {train_df['date'].max()} "
          f"({len(train_df)} rows)")
    print(f"  Test  dates: {test_df['date'].min()} to {test_df['date'].max()} "
          f"({len(test_df)} rows)")

    # Scale features
    train_scaled = train_df.copy()
    test_scaled  = test_df.copy()
    train_scaled[FEATURE_COLS] = scaler.transform(train_df[FEATURE_COLS])
    test_scaled[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

    # Datasets
    train_ds = StockSequenceDataset(train_scaled, FEATURE_COLS, label_col)
    test_ds  = StockSequenceDataset(test_scaled,  FEATURE_COLS, label_col)

    if len(train_ds) == 0 or len(test_ds) == 0:
        print("  ERROR: Not enough data for sequences.")
        return {}

    print(f"  Train sequences: {len(train_ds)}  Test: {len(test_ds)}")

    # Class weight for imbalance
    n_pos = train_ds.labels.sum()
    n_neg = len(train_ds.labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)]).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # Model
    model     = LSTMClassifier(input_dim=len(FEATURE_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.BCELoss()

    # Training loop with early stopping
    best_auc      = 0.0
    best_state    = None
    patience_cnt  = 0

    print(f"\n  {'Epoch':>6} {'Train Loss':>12} {'Val AUC':>10}")
    print(f"  {'-'*30}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_probs, val_labels = evaluate_loader(model, test_loader)

        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.5

        scheduler.step(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6} {train_loss:>12.4f} {val_auc:>10.4f}")

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best AUC={best_auc:.4f})")
                break

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation
    test_probs, test_labels = evaluate_loader(model, test_loader)
    test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(test_labels, test_preds)
    f1  = f1_score(test_labels, test_preds, zero_division=0)
    try:
        auc = roc_auc_score(test_labels, test_probs)
    except ValueError:
        auc = 0.5

    print(f"\n  [{horizon}] Test Results")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(classification_report(test_labels, test_preds,
                                target_names=["DOWN", "UP"],
                                zero_division=0))

    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = f"{OUTPUT_DIR}/lstm_{label_col}.pt"
    torch.save({
        "model_state": best_state,
        "input_dim":   len(FEATURE_COLS),
        "hidden_dim":  HIDDEN_DIM,
        "num_layers":  NUM_LAYERS,
        "dropout":     DROPOUT,
        "seq_len":     SEQ_LEN,
        "label_col":   label_col,
        "feature_cols": FEATURE_COLS,
    }, model_path)
    print(f"  Saved: {model_path}")

    return {"label": horizon, "accuracy": acc, "f1": f1, "auc": auc}


if __name__ == "__main__":
    print("=" * 55)
    print("  LSTM Trainer — T+1 / T+3 / T+5")
    print("=" * 55)

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        print("Please run dataset/build_historical_dataset.py first.")
        exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"\nLoaded {len(df)} rows")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Fit scaler on all data (features only)
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLS])
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler_lstm.pkl")

    all_metrics = []
    for label_col in LABEL_COLS:
        metrics = train_for_horizon(df, label_col, scaler)
        if metrics:
            all_metrics.append(metrics)

    # Summary
    print(f"\n{'='*55}")
    print("  Final Results Summary")
    print(f"{'='*55}")
    print(f"{'Horizon':<10} {'Accuracy':<12} {'F1':<10} {'AUC-ROC'}")
    print("-" * 42)
    for m in all_metrics:
        print(f"{m['label']:<10} {m['accuracy']:<12.4f} "
              f"{m['f1']:<10.4f} {m['auc']:.4f}")

    print(f"\nAll LSTM models saved to {OUTPUT_DIR}/")
    print("Done!")
