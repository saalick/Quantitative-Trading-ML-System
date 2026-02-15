#!/usr/bin/env python3

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.data_loader import load_ohlcv
from src.feature_engineering import compute_all_features, FEATURE_COLUMNS
from src.data_preprocessing import drop_missing_or_fill


def main():
    raw = load_ohlcv(project_root / "data" / "sample_data.csv")
    y = (raw["close"].shift(-1) / raw["close"] - 1 >= 0).astype(int)
    df = compute_all_features(raw)
    df = drop_missing_or_fill(df)
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feat_cols]
    y = y.reindex(X.index).dropna()
    X = X.loc[y.index]

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    imp = np.array(rf.feature_importances_)
    idx = np.argsort(imp)[-15:]
    names = [feat_cols[i] for i in idx]
    values = imp[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), values, color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 feature importance (Random Forest)")
    fig.tight_layout()
    out = project_root / "results" / "figures" / "feature_importance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
