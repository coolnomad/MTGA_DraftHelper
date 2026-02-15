"""
Train XGBoost regressors to predict deck_effect and deck_bump from a full draft pool.

Pipeline:
- Load bc_dataset.parquet (optionally sample)
- Build final pools per draft_id (sum picked card counts)
- For each pool, build a 40-card deck via hero_bot.deck_builder.build_deck
- Targets: evaluate_deck (calibrated) and evaluate_deck_bump on that deck
- Features: pool card counts vectorized by deck_eval/cards_index.json (name_to_idx)
- Train two regressors (effect, bump); save to hero_bot/models/pool_effect.pkl / pool_bump.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from deck_eval.evaluator import evaluate_deck, evaluate_deck_bump, CARD_INDEX_PATH  # noqa: E402
from hero_bot.deck_builder import build_deck  # noqa: E402
from hero_bot.train_state_value import _calibration_bins, _save_calibration_svg  # noqa: E402

BC_DATASET = REPO_ROOT / "data" / "processed" / "bc_dataset.parquet"
MODEL_DIR = REPO_ROOT / "hero_bot" / "models"


def load_card_index() -> Dict[str, int]:
    with open(CARD_INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("name_to_idx", {})


def pools_from_bc(df: pd.DataFrame) -> List[Tuple[str, Dict[str, int]]]:
    """
    Vectorized construction of pools:
    - Use the first non-null among ['pick', 'human_pick', 'chosen_card'] as the chosen card.
    - Count chosen card occurrences per draft_id.
    """
    pick_cols = ["pick", "human_pick", "chosen_card"]

    # Work on a copy to avoid mutating caller's df
    df = df.copy()

    # Select first non-null pick column row-wise
    available_cols = [c for c in pick_cols if c in df.columns]
    if not available_cols:
        raise ValueError("None of the expected pick columns are present in bc_dataset.")

    df["chosen"] = df[available_cols].bfill(axis=1).iloc[:, 0]
    df = df[df["chosen"].notna()]

    pools: List[Tuple[str, Dict[str, int]]] = []
    for draft_id, grp in tqdm(df.groupby("draft_id"), desc="drafts"):
        counts = grp["chosen"].value_counts().to_dict()
        if counts:
            pools.append((str(draft_id), counts))
    return pools


def vectorize_pool(pool: Dict[str, int], name_to_idx: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(name_to_idx), dtype=float)
    for name, cnt in pool.items():
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        vec[idx] += float(cnt)
    return vec


def build_dataset(
    df: pd.DataFrame,
    name_to_idx: Dict[str, int],
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    pools = pools_from_bc(df)
    X_list: List[np.ndarray] = []
    y_eff: List[float] = []
    y_bump: List[float] = []
    draft_ids: List[str] = []

    for draft_id, pool in tqdm(pools, desc="vectorizing"):
        deck = build_deck(pool)
        eff = evaluate_deck(deck)
        bump = evaluate_deck_bump(deck)
        X_list.append(vectorize_pool(pool, name_to_idx))
        y_eff.append(float(eff))
        y_bump.append(float(bump))
        draft_ids.append(draft_id)

    X = np.vstack(X_list)
    y_effect = np.array(y_eff, dtype=float)
    y_bump = np.array(y_bump, dtype=float)
    return draft_ids, X, y_effect, y_bump


def _weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    mean_true = np.average(y_true, weights=weights)
    ss_res = np.average((y_true - y_pred) ** 2, weights=weights)
    ss_tot = np.average((y_true - mean_true) ** 2, weights=weights)
    return float(1 - ss_res / (ss_tot + 1e-9))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 20,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    raw_r2 = float(
        1
        - np.sum((y_true - y_pred) ** 2)
        / (np.sum((y_true - y_true.mean()) ** 2) + 1e-9)
    )
    raw_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    bins = _calibration_bins(y_pred, y_true, n_bins=n_bins)
    if len(bins) == 0:
        bin_r2 = float("nan")
        bin_rmse = float("nan")
    else:
        bin_pred = bins["pred_mean"].to_numpy()
        bin_true = bins["true_mean"].to_numpy()
        weights = bins["count"].to_numpy()
        bin_rmse = float(
            np.sqrt(np.average((bin_true - bin_pred) ** 2, weights=weights))
        )
        bin_r2 = _weighted_r2(bin_true, bin_pred, weights)

    metrics = {
        "raw_r2": raw_r2,
        "raw_rmse": raw_rmse,
        "bin_r2": bin_r2,
        "bin_rmse": bin_rmse,
        "slope": float(slope),
        "intercept": float(intercept),
    }
    return metrics, bins


def cross_val_predict(
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)
    fold_ids = np.zeros(len(y), dtype=int)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        model = XGBRegressor(**params)
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
        fold_ids[test_idx] = fold_idx
    return preds, fold_ids


def save_calibration_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reports_dir: Path,
    prefix: str,
    title: str,
    n_bins: int = 20,
) -> Dict[str, float]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics, bins = compute_metrics(y_true, y_pred, n_bins=n_bins)
    bins_path = reports_dir / f"{prefix}_bins.csv"
    svg_path = reports_dir / f"{prefix}_calibration.svg"
    bins.to_csv(bins_path, index=False)
    _save_calibration_svg(svg_path, bins, metrics["slope"], metrics["intercept"], title=title)
    metrics.update(
        {
            "bins_path": str(bins_path),
            "calibration_svg": str(svg_path),
            "n_bins": n_bins,
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(BC_DATASET))
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="sample rows from bc_dataset before grouping",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="where to write calibration artifacts",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="number of CV folds for metrics / OOF preds",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=600,
        help="XGBoost n_estimators (use smaller for faster dev runs)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=8,
        help="XGBoost max_depth",
    )
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.input))
    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed)

    name_to_idx = load_card_index()
    draft_ids, X, y_eff, y_bump = build_dataset(df, name_to_idx)
    print("dataset", X.shape, "effect mean", y_eff.mean(), "bump mean", y_bump.mean())

    model_params = dict(
        objective="reg:squarederror",
        tree_method="hist",
        max_depth=args.max_depth,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
    )

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 5-fold CV metrics + calibration (these are also the OOF predictions)
    eff_cv_pred, eff_cv_folds = cross_val_predict(
        model_params, X, y_eff, n_splits=args.cv_folds, seed=args.seed
    )
    bump_cv_pred, bump_cv_folds = cross_val_predict(
        model_params, X, y_bump, n_splits=args.cv_folds, seed=args.seed
    )

    eff_cv_metrics = save_calibration_artifacts(
        y_true=y_eff,
        y_pred=eff_cv_pred,
        reports_dir=reports_dir,
        prefix="pool_effect_cv",
        title="Pool -> Deck Effect (CV)",
        n_bins=20,
    )
    bump_cv_metrics = save_calibration_artifacts(
        y_true=y_bump,
        y_pred=bump_cv_pred,
        reports_dir=reports_dir,
        prefix="pool_bump_cv",
        title="Pool -> Deck Bump (CV)",
        n_bins=20,
    )

    # Final models on full data
    model_eff = XGBRegressor(**model_params)
    model_eff.fit(X, y_eff)
    model_bump = XGBRegressor(**model_params)
    model_bump.fit(X, y_bump)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    eff_path = MODEL_DIR / "pool_effect.pkl"
    bump_path = MODEL_DIR / "pool_bump.pkl"
    joblib.dump(model_eff, eff_path)
    joblib.dump(model_bump, bump_path)

    # Train-set calibration (full fit) for reference
    eff_train_pred = model_eff.predict(X)
    bump_train_pred = model_bump.predict(X)
    eff_train_metrics = save_calibration_artifacts(
        y_true=y_eff,
        y_pred=eff_train_pred,
        reports_dir=reports_dir,
        prefix="pool_effect_train",
        title="Pool -> Deck Effect (Train)",
        n_bins=20,
    )
    bump_train_metrics = save_calibration_artifacts(
        y_true=y_bump,
        y_pred=bump_train_pred,
        reports_dir=reports_dir,
        prefix="pool_bump_train",
        title="Pool -> Deck Bump (Train)",
        n_bins=20,
    )

    # OOF predictions from the 5-fold CV (for downstream policy training)
    oof_df = pd.DataFrame(
        {
            "draft_id": draft_ids,
            "fold_eff": eff_cv_folds,
            "fold_bump": bump_cv_folds,
            "effect_true": y_eff,
            "effect_pred": eff_cv_pred,
            "bump_true": y_bump,
            "bump_pred": bump_cv_pred,
        }
    )
    oof_path = reports_dir / "pool_value_oof_5fold.parquet"
    oof_df.to_parquet(oof_path, index=False)
    oof_csv_path = reports_dir / "pool_value_oof_5fold.csv"
    oof_df.to_csv(oof_csv_path, index=False)

    eff_oof_metrics, _ = compute_metrics(y_eff, eff_cv_pred, n_bins=20)
    bump_oof_metrics, _ = compute_metrics(y_bump, bump_cv_pred, n_bins=20)

    meta = {
        "input": str(args.input),
        "max_rows": int(args.max_rows or len(df)),
        "n_samples": int(len(X)),
        "seed": args.seed,
        "feature_dim": int(X.shape[1]),
        "effect_model": str(eff_path),
        "bump_model": str(bump_path),
        "metrics": {
            "cv_folds": args.cv_folds,
            "effect": {
                "cv": eff_cv_metrics,
                "train": eff_train_metrics,
                "oof_5fold": eff_oof_metrics,
            },
            "bump": {
                "cv": bump_cv_metrics,
                "train": bump_train_metrics,
                "oof_5fold": bump_oof_metrics,
            },
        },
        "artifacts": {
            "oof_5fold_parquet": str(oof_path),
            "oof_5fold_csv": str(oof_csv_path),
        },
    }
    meta_path = MODEL_DIR / "pool_models_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
