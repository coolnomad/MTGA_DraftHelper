"""
Distill the hero value function into a stochastic policy:
1) Load bc_dataset (optionally sample max_rows)
2) For each pack row, score every card with the hero value model (state_value.pkl)
3) Convert scores to softmax probabilities with temperature
4) Train a regressor to predict per-card pick probability from (state, card) features
5) Save model to hero_bot/models/hero_policy_distill.pkl

Inference: use hero_policy_distill in hero_bot/hero_policy.py to sample from predicted probs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from hero_bot.train_state_value import MODEL_PATH, load_state_value_model
from state_encoding.encoder import encode_state, encode_card

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "bc_dataset.parquet"
OUT_MODEL = REPO_ROOT / "hero_bot" / "models" / "hero_policy_distill.pkl"
OUT_META = REPO_ROOT / "hero_bot" / "models" / "hero_policy_distill_meta.json"


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    t = max(temperature, 1e-6)
    z = (x / t) - (x / t).max()
    e = np.exp(z)
    return e / (e.sum() + 1e-9)


def build_samples(
    df: pd.DataFrame,
    value_model,
    temperature: float,
    max_cards: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    for row in df.itertuples(index=False, name="Row"):
        pool_raw = getattr(row, "pool_counts", {}) or {}
        pool: Dict[str, int] = {k: int(v or 0) for k, v in dict(pool_raw).items()}
        raw_pack = getattr(row, "pack_card_ids", [])
        if raw_pack is None:
            continue
        if isinstance(raw_pack, np.ndarray):
            pack_cards = [str(x) for x in raw_pack.tolist() if isinstance(x, str) or isinstance(x, (np.str_, np.object_))]
        else:
            pack_cards = list(raw_pack)
        if not pack_cards:
            continue
        total_picks = sum(pool.values())
        pack_no = int(getattr(row, "pack_number", (total_picks // 15) + 1))
        pick_no = int(getattr(row, "pick_number", (total_picks % 15) + 1))
        base_state = encode_state(
            pool,
            pack_no=pack_no,
            pick_no=pick_no,
            skill_bucket=getattr(row, "rank", None) or getattr(row, "skill_bucket", None),
        )
        # score hero Q for each card
        qs = []
        for card in pack_cards:
            new_pool = dict(pool)
            new_pool[card] = new_pool.get(card, 0) + 1
            state_vec = encode_state(
                new_pool,
                pack_no=pack_no,
                pick_no=pick_no,
                skill_bucket=getattr(row, "rank", None) or getattr(row, "skill_bucket", None),
            )
            q = float(value_model.predict(xgb.DMatrix(state_vec.reshape(1, -1)))[0]) if isinstance(value_model, xgb.Booster) else float(value_model.predict(state_vec.reshape(1, -1))[0])
            qs.append(q)
        probs = softmax(np.array(qs, dtype=float), temperature=temperature)
        for card, prob in zip(pack_cards, probs):
            feat = np.concatenate([base_state, encode_card(card)])
            X_list.append(feat)
            y_list.append(float(prob))
            if max_cards and len(X_list) >= max_cards:
                return np.vstack(X_list), np.array(y_list, dtype=float)
    if not X_list:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
    return np.vstack(X_list), np.array(y_list, dtype=float)


def train_policy_distill(
    input_path: Path = INPUT_PATH,
    max_rows: Optional[int] = None,
    temperature: float = 0.25,
    seed: int = 1337,
):
    df = pd.read_parquet(input_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)

    value_model = load_state_value_model(MODEL_PATH)
    if value_model is None:
        raise FileNotFoundError(f"Missing hero value model at {MODEL_PATH}")

    X, y = build_samples(df, value_model, temperature=temperature)
    if X.size == 0:
        raise ValueError("No samples built for policy distillation.")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=400,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_MODEL)
    meta = {"temperature": temperature, "max_rows": max_rows or len(df), "input_path": str(input_path), "model_path": str(OUT_MODEL)}
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(INPUT_PATH), help="bc_dataset parquet path")
    parser.add_argument("--max_rows", type=int, default=200000, help="cap rows for distillation (packs)")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    meta = train_policy_distill(
        input_path=Path(args.input),
        max_rows=args.max_rows,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
