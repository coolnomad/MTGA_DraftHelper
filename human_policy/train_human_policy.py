"""
Train per-skill human draft policies on bc_tensors.parquet.

Approach:
- Expand each pick into (state + card) feature rows with label 1 for the chosen card, 0 otherwise.
- Train a binary classifier per skill bucket (and an ALL fallback) and score packs via softmax over candidate scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from state_encoding.encoder import encode_card, encode_state

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED / "bc_tensors.parquet"
MODEL_DIR = REPO_ROOT / "human_policy" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PackMetrics:
    top1_acc: float
    log_loss: float
    n_picks: int
    model_path: str
    n_train: int
    n_test: int


def _build_examples(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Flatten pack rows into per-card examples.
    Returns X, y, pack_lengths for reconstruction.
    """
    rows: List[np.ndarray] = []
    labels: List[int] = []
    pack_lengths: List[int] = []
    for _, row in df.iterrows():
        state_vec = np.array(row["state_vec"], dtype=float)
        card_feats = row["card_features"]
        pack_cards = row["pack_cards"]
        k = len(pack_cards)
        for card_name, feats in zip(pack_cards, card_feats):
            feat_vec = np.array(feats, dtype=float)
            rows.append(np.concatenate([state_vec, feat_vec]))
            labels.append(1 if card_name == row["human_pick"] else 0)
        pack_lengths.append(k)
    X = np.vstack(rows) if rows else np.zeros((0, 0), dtype=float)
    y = np.array(labels, dtype=int)
    return X, y, pack_lengths


def _evaluate_model(model, X: np.ndarray, df: pd.DataFrame, pack_lengths: List[int]) -> Tuple[float, float, int]:
    """Compute top-1 accuracy and avg log loss on a held-out set."""
    if len(df) == 0 or X.size == 0:
        return 0.0, float("inf"), 0

    probs_all = model.predict_proba(X)[:, 1]
    top1 = 0
    total = 0
    logloss = 0.0
    idx = 0
    for (_, row), k in zip(df.iterrows(), pack_lengths):
        pack_probs = probs_all[idx: idx + k]
        # normalize to softmax to avoid unnormalized binary probs
        if pack_probs.sum() > 0:
            pack_probs = pack_probs / pack_probs.sum()
        else:
            pack_probs = np.full(k, 1.0 / k)
        human_idx = row["pack_cards"].index(row["human_pick"]) if row["human_pick"] in row["pack_cards"] else None
        if human_idx is not None:
            if int(np.argmax(pack_probs)) == human_idx:
                top1 += 1
            logloss -= np.log(max(pack_probs[human_idx], 1e-9))
            total += 1
        idx += k
    if total == 0:
        return 0.0, float("inf"), 0
    return top1 / total, logloss / total, total


def _fit_model(train_df: pd.DataFrame) -> Tuple[object, np.ndarray]:
    """Fit logistic model; fallback to constant-prob model if only one class."""
    X_train, y_train, _ = _build_examples(train_df)
    if X_train.size == 0 or len(np.unique(y_train)) < 2:
        # constant model that ignores input shape
        p = float(np.mean(y_train)) if len(y_train) else 0.5

        class ConstantModel:
            def __init__(self, prob: float):
                self.prob = prob

            def predict_proba(self, X):
                n = X.shape[0]
                return np.tile([1 - self.prob, self.prob], (n, 1))

        return ConstantModel(p), X_train
    model = LogisticRegression(
        max_iter=200,
        n_jobs=1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model, X_train


def train_models(df_override: Optional[pd.DataFrame] = None, seed: int = 1337, test_frac: float = 0.1) -> Dict[str, PackMetrics]:
    """Train models per skill bucket plus ALL fallback. Returns metrics dict keyed by bucket."""
    rng = np.random.default_rng(seed)
    df = df_override.copy() if df_override is not None else pd.read_parquet(INPUT_PATH)
    # normalize skill bucket values to strings, fill missing
    df["skill_bucket"] = df["skill_bucket"].fillna("UNKNOWN").astype(str)
    buckets = sorted(df["skill_bucket"].unique().tolist())
    if "ALL" not in buckets:
        buckets.append("ALL")

    results: Dict[str, PackMetrics] = {}

    for bucket in buckets:
        if bucket == "ALL":
            df_bucket = df
        else:
            df_bucket = df[df["skill_bucket"] == bucket]
        if df_bucket.empty:
            continue
        idx = rng.permutation(len(df_bucket))
        split = int(len(idx) * (1 - test_frac))
        if split <= 0:
            split = len(idx)
        train_df = df_bucket.iloc[idx[:split]]
        test_df = df_bucket.iloc[idx[split:]] if split < len(idx) else train_df

        model, X_train = _fit_model(train_df)
        _, y_train, _ = _build_examples(train_df)

        X_test, _, pack_lengths = _build_examples(test_df)
        top1, logloss, n_picks = _evaluate_model(model, X_test, test_df, pack_lengths)

        bucket_slug = bucket.replace(" ", "_")
        model_path = MODEL_DIR / f"human_policy_{bucket_slug}.pkl"
        joblib.dump(model, model_path)

        results[bucket] = PackMetrics(
            top1_acc=top1,
            log_loss=logloss,
            n_picks=n_picks,
            model_path=str(model_path),
            n_train=len(train_df),
            n_test=len(test_df),
        )
    return results


def _load_model(skill_bucket: Optional[str] = None):
    bucket = (skill_bucket or "UNKNOWN")
    bucket_slug = bucket.replace(" ", "_")
    path = MODEL_DIR / f"human_policy_{bucket_slug}.pkl"
    if not path.exists():
        path = MODEL_DIR / "human_policy_ALL.pkl"
    if not path.exists():
        raise FileNotFoundError(f"model not found for {bucket} or ALL in {MODEL_DIR}")
    return joblib.load(path)


def score_pack(
    pack_cards: List[str],
    pool_counts: Dict[str, int],
    pack_number: int,
    pick_number: int,
    skill_bucket: Optional[str] = None,
    state_vec: Optional[np.ndarray] = None,
    card_features: Optional[List[np.ndarray]] = None,
) -> List[Tuple[str, float]]:
    """
    Score a pack using a trained human policy. Returns list of (card, prob) sorted desc.
    Requires models trained and saved to MODEL_DIR.
    """
    if state_vec is None:
        state_vec = encode_state(pool_counts, pack_no=pack_number, pick_no=pick_number, skill_bucket=skill_bucket)
    if card_features is None:
        card_feats = [encode_card(c) for c in pack_cards]
    else:
        card_feats = [np.array(f, dtype=float) for f in card_features]
    model = _load_model(skill_bucket)
    X = np.vstack([np.concatenate([state_vec, f]) for f in card_feats])
    probs = model.predict_proba(X)[:, 1]
    probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(pack_cards), 1 / len(pack_cards))
    ranked = sorted(zip(pack_cards, probs), key=lambda kv: kv[1], reverse=True)
    return ranked


if __name__ == "__main__":
    metrics = train_models()
    print(json.dumps({k: vars(v) for k, v in metrics.items()}, indent=2))
