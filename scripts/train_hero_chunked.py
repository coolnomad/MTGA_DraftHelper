"""
Chunked training of hero/state-value model on full bc_dataset with OOF deck bump labels.
Pipeline:
1) Load bc_dataset and deck_effect_oof, inner join on draft_id.
2) Chunked encoding to temporary npy files (X_chunk.npy, y_chunk.npy).
3) Stream chunks into XGBoost (hist) via batch DMatrices.
4) Compute raw + 20-bin metrics (R2, RMSE, slope, intercept); save bins CSV/SVG; record runtime.

Outputs:
- Model: hero_bot/models/state_value.pkl
- Reports: reports/state_value_bins_full.csv, reports/state_value_calibration_full.svg, reports/state_value_metrics_full.json
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Generator, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from hero_bot.train_state_value import _calibration_bins, _save_calibration_svg, REPORTS_DIR, MODEL_DIR
from state_encoding.encoder import encode_state

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "bc_dataset.parquet"
OOF_PATH = REPO_ROOT / "reports" / "deck_effect_oof.parquet"


def load_merged() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    oof = pd.read_parquet(OOF_PATH)[["draft_id", "deck_bump_oof"]]
    df = df.merge(oof, on="draft_id", how="inner")
    return df


def encode_to_chunks(df: pd.DataFrame, chunk_size: int = 50000, out_dir: Path | None = None) -> Generator[Tuple[Path, Path], None, None]:
    """
    Encode rows in chunks to npy files. Yields (X_path, y_path) per chunk.
    """
    if out_dir is None:
        out_dir = REPO_ROOT / "data" / "hero_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(df.itertuples(index=False, name="Row"))
    chunk_idx = 0
    for i in range(0, len(rows), chunk_size):
        xs = []
        ys = []
        for row in rows[i : i + chunk_size]:
            pool = {k: int(v or 0) for k, v in (getattr(row, "pool_counts", {}) or {}).items()}
            pick = getattr(row, "pick", None) or getattr(row, "human_pick", None) or getattr(row, "chosen_card", None)
            if pick is None:
                continue
            pool[pick] = pool.get(pick, 0) + 1
            state_vec = encode_state(
                pool,
                pack_no=getattr(row, "pack_number", 1),
                pick_no=getattr(row, "pick_number", 1),
                skill_bucket=getattr(row, "rank", None) or getattr(row, "skill_bucket", None),
            )
            xs.append(state_vec)
            ys.append(float(getattr(row, "deck_bump_oof")))
        if not xs:
            continue
        X_chunk = np.vstack(xs)
        y_chunk = np.array(ys, dtype=float)
        X_path = out_dir / f"X_{chunk_idx}.npy"
        y_path = out_dir / f"y_{chunk_idx}.npy"
        np.save(X_path, X_chunk)
        np.save(y_path, y_chunk)
        chunk_idx += 1
        yield X_path, y_path


def stream_train(chunks: Generator[Tuple[Path, Path], None, None], params: dict, num_boost_round: int = 400) -> Tuple[xgb.Booster, np.ndarray, np.ndarray]:
    """
    Train by concatenating chunks into a DMatrix (still in memory but bounded by chunk load).
    """
    X_all = []
    y_all = []
    for X_path, y_path in chunks:
        X_all.append(np.load(X_path))
        y_all.append(np.load(y_path))
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)
    return booster, X, y


def main():
    start = time.time()
    df = load_merged()
    print(f"merged rows={len(df)}, drafts={df['draft_id'].nunique()}")
    chunks = list(encode_to_chunks(df, chunk_size=50000))
    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "lambda": 0.5,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "seed": 1337,
    }
    booster, X, y = stream_train(iter(chunks), params, num_boost_round=400)
    y_pred = booster.predict(xgb.DMatrix(X))
    r2 = float(1 - np.sum((y - y_pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-9))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    slope, intercept = np.polyfit(y_pred, y, 1)
    bins = _calibration_bins(y_pred, y)
    r2_bins = float(1 - np.sum((bins["true_mean"] - bins["pred_mean"]) ** 2) / (np.sum((bins["true_mean"] - bins["true_mean"].mean()) ** 2) + 1e-9))
    rmse_bins = float(np.sqrt(np.mean((bins["true_mean"] - bins["pred_mean"]) ** 2)))

    # save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster, MODEL_DIR / "state_value.pkl")
    # save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bins_path = REPORTS_DIR / "state_value_bins_full.csv"
    svg_path = REPORTS_DIR / "state_value_calibration_full.svg"
    bins.to_csv(bins_path, index=False)
    _save_calibration_svg(svg_path, bins, slope, intercept, title="State value calibration (full)")
    metrics = {
        "R2": r2,
        "RMSE": rmse,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "bins_R2": r2_bins,
        "bins_RMSE": rmse_bins,
        "bins_path": str(bins_path),
        "svg_path": str(svg_path),
        "rows": len(X),
        "runtime_sec": time.time() - start,
    }
    (REPORTS_DIR / "state_value_metrics_full.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
