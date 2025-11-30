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
from typing import Dict, Optional, Tuple

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from deck_eval.evaluator import evaluate_deck
from state_encoding.encoder import encode_state, encode_card, CARDS_DF

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


def _load_sampled_df(path: Path, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    """Load parquet, optionally sampling row-groups to avoid full read."""
    if max_rows is None:
        return pd.read_parquet(path)
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        rng = np.random.default_rng(seed)
        rg_indices = list(range(pf.num_row_groups))
        rng.shuffle(rg_indices)
        tables = []
        total = 0
        for rg in rg_indices:
            if total >= max_rows:
                break
            tbl = pf.read_row_group(rg)
            tables.append(tbl)
            total += tbl.num_rows
        if not tables:
            return pd.read_parquet(path)
        import pyarrow as pa

        combined = pa.concat_tables(tables)
        df = combined.to_pandas()
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed)
        return df
    except Exception:
        df = pd.read_parquet(path)
        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed)
        return df


def _build_dataset(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
    seed: int = 1337,
    target_column: str = "deck_effect",
    n_jobs: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    # optional subsample to keep training tractable
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)

    def process_row(row) -> Optional[Tuple[np.ndarray, float]]:
        raw_pool = getattr(row, "pool_counts", {}) or {}
        pool = {k: int(v or 0) for k, v in raw_pool.items()}
        pick = (
            getattr(row, "pick", None)
            or getattr(row, "human_pick", None)
            or getattr(row, "chosen_card", None)
        )
        if pick is None or (isinstance(pick, float) and pd.isna(pick)):
            return None
        pool[pick] = pool.get(pick, 0) + 1
        state_vec = encode_state(
            pool,
            pack_no=getattr(row, "pack_number", 1),
            pick_no=getattr(row, "pick_number", 1),
            skill_bucket=getattr(row, "rank", None) or getattr(row, "skill_bucket", None),
        )
        if target_column in df.columns:
            target = float(getattr(row, target_column))
        else:
            target = evaluate_deck(pool)
        return state_vec, target

    rows = list(df.itertuples(index=False, name="Row"))
    results = Parallel(n_jobs=max(1, n_jobs), prefer="threads")(delayed(process_row)(r) for r in rows)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for res in results:
        if res is None:
            continue
        x, y = res
        X_list.append(x)
        y_list.append(y)

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
    n_jobs: int = 4,
) -> Dict[str, float]:
    df = df_override.copy() if df_override is not None else _load_sampled_df(INPUT_PATH, max_rows, seed)
    X, y = _build_dataset(df, max_rows=max_rows, seed=seed, target_column=target_column, n_jobs=n_jobs)
    if X.size == 0:
        raise ValueError("No training data available for state value model.")

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
    parser.add_argument("--input", type=str, default=str(INPUT_PATH), help="Path to parquet replay/bc_dataset")
    parser.add_argument("--max_rows", type=int, default=50000, help="Subsample rows for training")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--target_column", type=str, default="deck_effect", help="Column to use as target; fallback to evaluator if missing")
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

    INPUT = Path(args.input)
    df = pd.read_parquet(INPUT)
    metrics = train_state_value(
        df_override=df,
        seed=args.seed,
        test_frac=args.test_frac,
        max_rows=args.max_rows,
        target_column=args.target_column,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(metrics, indent=2))
