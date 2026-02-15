"""
Train a state-value model g(s) ≈ E[deck_effect | state] using draft history and the calibrated deck evaluator.

By default uses data/processed/bc_dataset.parquet (pack/pick rows with pool + human pick) to build
state -> deck_effect targets (after hypothetically taking the human pick). You can override with a
DataFrame via df_override for testing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from deck_eval.evaluator import evaluate_deck
from state_encoding.encoder import encode_state

PROCESSED = REPO_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED / "bc_dataset.parquet"
MODEL_DIR = REPO_ROOT / "hero_bot" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "state_value.pkl"
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _train_test_split(n: int, seed: int, test_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * (1 - test_frac))
    if split <= 0:
        split = n
    return idx[:split], idx[split:]


def _train_test_split_by_group(groups: np.ndarray, seed: int, test_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group-aware split: keeps all rows for a given group (e.g., draft_id) on the same side.
    groups: 1D array-like of group keys aligned with rows.
    """
    uniq = pd.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    split = int(len(uniq) * (1 - test_frac))
    if split <= 0:
        split = len(uniq)
    train_groups = set(uniq[:split])
    train_idx = np.where(np.isin(groups, list(train_groups)))[0]
    test_idx = np.where(~np.isin(groups, list(train_groups)))[0]
    if len(test_idx) == 0:
        test_idx = train_idx  # fallback to avoid empty test
    return train_idx, test_idx


def _calibration_bins(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"pred": y_pred, "true": y_true})
    # quantile bins; drop duplicates if not enough unique preds
    df["bin"] = pd.qcut(df["pred"], q=min(n_bins, len(df)), duplicates="drop")
    bins = df.groupby("bin").agg(pred_mean=("pred", "mean"), true_mean=("true", "mean"), count=("true", "count"))
    bins = bins.reset_index(drop=True)
    return bins


def _save_calibration_svg(path: Path, bins: pd.DataFrame, slope: float, intercept: float, title: str):
    # simple svg with points and regression line
    width, height = 420, 300
    margin = 40
    x_min = float(bins["pred_mean"].min())
    x_max = float(bins["pred_mean"].max())
    y_min = float(bins["true_mean"].min())
    y_max = float(bins["true_mean"].max())
    # expand ranges slightly
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    def sx(x):
        return margin + (x - x_min) / (x_max - x_min + 1e-9) * (width - 2 * margin)

    def sy(y):
        return height - margin - (y - y_min) / (y_max - y_min + 1e-9) * (height - 2 * margin)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="none"/>')
    # axes
    parts.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>')
    parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>')
    # regression line
    x0, x1 = x_min, x_max
    y0 = slope * x0 + intercept
    y1 = slope * x1 + intercept
    parts.append(f'<line x1="{sx(x0)}" y1="{sy(y0)}" x2="{sx(x1)}" y2="{sy(y1)}" stroke="red" stroke-width="1.5"/>')
    # points
    for _, row in bins.iterrows():
        cx = sx(float(row["pred_mean"]))
        cy = sy(float(row["true_mean"]))
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="steelblue" opacity="0.8"/>')
    parts.append(f'<text x="{width/2}" y="{margin/2}" text-anchor="middle" font-size="12">{title}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _needed_columns(target_column: str) -> List[str]:
    cols = [
        "pool_counts",
        "pick",
        "human_pick",
        "chosen_card",
        "pack_number",
        "pick_number",
        "rank",
        "skill_bucket",
    ]
    if target_column:
        cols.append(target_column)
    return cols


def _filter_available_columns(all_columns: Sequence[str], requested: Iterable[str]) -> List[str]:
    available = set(all_columns)
    return [c for c in requested if c in available]


def _load_sampled_df(path: Path, max_rows: Optional[int], seed: int, target_column: str) -> pd.DataFrame:
    """
    Load up to max_rows from a parquet file, using row groups and column pruning
    to avoid reading the entire file when max_rows is small.
    """
    requested_cols = _needed_columns(target_column)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        cols = _filter_available_columns(pf.schema.names, requested_cols)

        if max_rows is None or max_rows >= pf.metadata.num_rows:
            return pf.read(columns=cols or None).to_pandas()

        rng = np.random.default_rng(seed)
        rg_indices = list(range(pf.num_row_groups))
        rng.shuffle(rg_indices)

        tables = []
        total = 0
        for rg in rg_indices:
            if total >= max_rows:
                break
            tbl = pf.read_row_group(rg, columns=cols or None)
            tables.append(tbl)
            total += tbl.num_rows

        if not tables:
            return pf.read(columns=cols or None).to_pandas()

        combined = pa.concat_tables(tables)
        df = combined.to_pandas()
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed)
        return df
    except Exception:
        # fallback: normal read + sample (still request a subset of columns if possible)
        try:
            df = pd.read_parquet(path, columns=requested_cols)
        except Exception:
            df = pd.read_parquet(path)
        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed)
        return df


def _build_dataset(
    df: pd.DataFrame,
    target_column: str = "deck_effect",
    n_jobs: int = -1,
    allow_evaluate_fallback: bool = False,
    chunk_size: int = 5000,
    groups: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Encode states and targets from a (possibly already-sampled) dataframe.
    Assumes df has at most max_rows rows; no internal resampling here.
    """

    def process_rows(rows: Sequence[Tuple[int, pd.Series]]) -> Tuple[List[np.ndarray], List[float], List]:
        local_encode_state = encode_state
        out_x: List[np.ndarray] = []
        out_y: List[float] = []
        out_g: List = []
        for idx, row in rows:
            raw_pool = getattr(row, "pool_counts", {}) or {}
            if not raw_pool:
                continue
            # avoid repeated dict allocations where possible
            pool = {k: int(v or 0) for k, v in raw_pool.items()}
            pick = getattr(row, "pick", None) or getattr(row, "human_pick", None) or getattr(row, "chosen_card", None)
            if pick is None or (isinstance(pick, float) and pd.isna(pick)):
                continue
            pool[pick] = pool.get(pick, 0) + 1

            state_vec = local_encode_state(
                pool,
                pack_no=getattr(row, "pack_number", 1),
                pick_no=getattr(row, "pick_number", 1),
                skill_bucket=getattr(row, "rank", None) or getattr(row, "skill_bucket", None),
            )

            target_val = getattr(row, target_column, None)
            if target_val is None or (isinstance(target_val, float) and pd.isna(target_val)):
                if not allow_evaluate_fallback:
                    continue
                target = float(evaluate_deck(pool))
            else:
                target = float(target_val)

            out_x.append(state_vec)
            out_y.append(target)
            if groups is not None:
                out_g.append(groups[idx])
        return out_x, out_y, out_g

    rows = list(enumerate(df.itertuples(index=False, name="Row")))
    if not rows:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float), []

    eff_n_jobs = n_jobs if n_jobs not in (0, None) else -1
    if chunk_size <= 0:
        chunk_size = max(512, len(rows) // max(1, abs(eff_n_jobs) * 4))
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

    backend = "loky" if eff_n_jobs != 1 else "threading"
    try:
        results = Parallel(n_jobs=eff_n_jobs, backend=backend)(
            delayed(process_rows)(chunk) for chunk in chunks
        )
    except PermissionError:
        # Fallback for environments where spawning processes or pipes is restricted
        results = Parallel(n_jobs=eff_n_jobs if eff_n_jobs != -1 else 1, backend="threading")(
            delayed(process_rows)(chunk) for chunk in chunks
        )

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    g_list: List = []
    for xs, ys, gs in results:
        if not xs:
            continue
        X_list.extend(xs)
        y_list.extend(ys)
        g_list.extend(gs)

    if not X_list:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float), []

    return np.vstack(X_list), np.array(y_list, dtype=float), g_list


def train_state_value(
    df_override: Optional[pd.DataFrame] = None,
    seed: int = 1337,
    test_frac: float = 0.2,
    model_dir: Optional[Path] = None,
    max_rows: Optional[int] = 50000,
    target_column: str = "deck_effect",
    n_jobs: int = -1,
    group_column: Optional[str] = "draft_id",
    oof_labels: Optional[Path] = None,
    oof_target_column: Optional[str] = None,
    reports_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Train the state-value model.

    If df_override is provided, it will be used directly (optionally subsampled to max_rows).
    Otherwise, bc_dataset.parquet is loaded via _load_sampled_df with max_rows.
    """
    if df_override is not None:
        df = df_override.copy()
        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed)
    else:
        df = _load_sampled_df(INPUT_PATH, max_rows, seed, target_column)
    # merge OOF labels if provided
    if oof_labels:
        oof_df = pd.read_parquet(oof_labels)
        df = df.merge(oof_df, on="draft_id", how="left")
        if oof_target_column and oof_target_column in df.columns:
            target_column = oof_target_column

    allow_eval = target_column not in df.columns
    groups_used = None
    groups_input = df[group_column].to_numpy() if (group_column and group_column in df.columns) else None
    X, y, used_groups = _build_dataset(
        df,
        target_column=target_column,
        n_jobs=n_jobs,
        allow_evaluate_fallback=allow_eval,
        groups=groups_input,
    )
    if used_groups:
        groups_used = np.array(used_groups)
    if X.size == 0:
        raise ValueError("No training data available for state value model.")

    if groups_used is not None:
        train_idx, test_idx = _train_test_split_by_group(groups_used, seed, test_frac)
    else:
        train_idx, test_idx = _train_test_split(len(X), seed, test_frac)
    X_tr, X_te = X[train_idx], X[test_idx] if len(test_idx) else X
    y_tr, y_te = y[train_idx], y[test_idx] if len(test_idx) else y

    model = XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        n_estimators=400,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=max(1, n_jobs),
        tree_method="hist",
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    r2 = r2_score(y_te, y_pred) if len(y_te) > 1 else 0.0
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    slope, intercept = (0.0, 0.0)
    if len(y_te) > 1:
        slope, intercept = np.polyfit(y_pred, y_te, 1)
    bins = _calibration_bins(y_pred, y_te) if len(y_te) > 1 else pd.DataFrame(columns=["pred_mean", "true_mean", "count"])

    out_dir = model_dir or MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "state_value.pkl")

    rep_dir = reports_dir or REPORTS_DIR
    bins_path = rep_dir / f"state_value_bins_{target_column}.csv"
    svg_path = rep_dir / f"state_value_calibration_{target_column}.svg"
    if not bins.empty:
        bins.to_csv(bins_path, index=False)
        _save_calibration_svg(svg_path, bins, slope, intercept, title=f"State value calibration ({target_column})")

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "model_path": str(out_dir / "state_value.pkl"),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "bins_path": str(bins_path) if not bins.empty else "",
        "svg_path": str(svg_path) if not bins.empty else "",
    }


def load_state_value_model(path: Path = MODEL_PATH):
    return joblib.load(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_PATH),
        help="Path to parquet replay/bc_dataset.",
    )
    parser.add_argument("--max_rows", type=int, default=50000, help="Subsample rows for training")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument(
        "--target_column",
        type=str,
        default="deck_effect",
        help="Column to use as target; fallback to evaluator if missing",
    )
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--group_column",
        type=str,
        default="draft_id",
        help="Optional column for group-wise split (e.g., draft_id). Set empty to disable.",
    )
    parser.add_argument(
        "--oof_labels",
        type=str,
        default=None,
        help="Optional parquet with out-of-fold deck_effect/deck_bump (must include draft_id).",
    )
    parser.add_argument(
        "--oof_target_column",
        type=str,
        default=None,
        help="Optional target column to use from the oof_labels file (e.g., deck_effect_oof).",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default=None,
        help="Directory to save calibration bins/svg (default: reports).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")
    df_in = pd.read_parquet(input_path)

    metrics = train_state_value(
        df_override=df_in,
        seed=args.seed,
        test_frac=args.test_frac,
        max_rows=args.max_rows,
        target_column=args.target_column,
        n_jobs=args.n_jobs,
        group_column=args.group_column or None,
        oof_labels=Path(args.oof_labels) if args.oof_labels else None,
        oof_target_column=args.oof_target_column,
        reports_dir=Path(args.reports_dir) if args.reports_dir else None,
    )
    print(json.dumps(metrics, indent=2))
