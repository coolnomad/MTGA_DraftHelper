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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode states and targets from a (possibly already-sampled) dataframe.
    Assumes df has at most max_rows rows; no internal resampling here.
    """

    def process_rows(rows: Sequence[pd.Series]) -> Tuple[List[np.ndarray], List[float]]:
        local_encode_state = encode_state
        out_x: List[np.ndarray] = []
        out_y: List[float] = []
        for row in rows:
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
        return out_x, out_y

    rows = list(df.itertuples(index=False, name="Row"))
    if not rows:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)

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
    for xs, ys in results:
        if not xs:
            continue
        X_list.extend(xs)
        y_list.extend(ys)

    if not X_list:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)

    return np.vstack(X_list), np.array(y_list, dtype=float)


def train_state_value(
    df_override: Optional[pd.DataFrame] = None,
    seed: int = 1337,
    test_frac: float = 0.2,
    model_dir: Optional[Path] = None,
    max_rows: Optional[int] = 50000,
    target_column: str = "deck_effect",
    n_jobs: int = -1,
    group_column: Optional[str] = "draft_id",
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

    allow_eval = target_column not in df.columns
    X, y = _build_dataset(
        df,
        target_column=target_column,
        n_jobs=n_jobs,
        allow_evaluate_fallback=allow_eval,
    )
    if X.size == 0:
        raise ValueError("No training data available for state value model.")

    if group_column and group_column in df.columns:
        groups = df[group_column].to_numpy()
        train_idx, test_idx = _train_test_split_by_group(groups, seed, test_frac)
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

    out_dir = model_dir or MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "state_value.pkl")

    return {
        "R2": r2,
        "RMSE": rmse,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "model_path": str(out_dir / "state_value.pkl"),
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
    )
    print(json.dumps(metrics, indent=2))
