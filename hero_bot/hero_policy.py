from __future__ import annotations

import os
from typing import Dict, List, Optional
import numpy as np
import xgboost as xgb
import joblib

from state_encoding.encoder import encode_state
from state_encoding.encoder import encode_card
from deck_eval.evaluator import evaluate_deck
from hero_bot.train_state_value import MODEL_PATH, load_state_value_model

_STATE_VALUE_MODEL = None
_DISTILL_MODEL = None
_DISTILL_META = None
DISTILL_MODEL_PATH = MODEL_PATH.parent / "hero_policy_distill.pkl"
DISTILL_META_PATH = MODEL_PATH.parent / "hero_policy_distill_meta.json"


def _get_state_value_model():
    global _STATE_VALUE_MODEL
    if _STATE_VALUE_MODEL is None and MODEL_PATH.exists():
        _STATE_VALUE_MODEL = load_state_value_model(MODEL_PATH)
    return _STATE_VALUE_MODEL


def _get_distill_model():
    global _DISTILL_MODEL, _DISTILL_META
    if _DISTILL_MODEL is None and DISTILL_MODEL_PATH.exists():
        _DISTILL_MODEL = joblib.load(DISTILL_MODEL_PATH)
        if DISTILL_META_PATH.exists():
            try:
                import json

                _DISTILL_META = json.loads(DISTILL_META_PATH.read_text(encoding="utf-8"))
            except Exception:
                _DISTILL_META = None
    return _DISTILL_MODEL


def _predict_state_value(model, state_vec: np.ndarray) -> float:
    # Accepts sklearn-like regressor or xgboost Booster.
    if hasattr(model, "predict"):
        try:
            return float(model.predict(state_vec.reshape(1, -1))[0])
        except TypeError:
            pass
    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(state_vec.reshape(1, -1))
        return float(model.predict(dmat)[0])
    raise TypeError(f"Unsupported model type: {type(model)}")


def hero_policy(pack_cards: List[str], pool_counts: Dict[str, int], seat_idx: int, rng: np.random.Generator) -> str:
    """
    Greedy hero policy using learned state-value model when available, otherwise evaluator.
    """
    model = _get_state_value_model()
    best_card = pack_cards[0]
    best_score = -1e9
    total_picks = sum(pool_counts.values())
    pack_number = (total_picks // 15) + 1
    pick_number = (total_picks % 15) + 1
    for card in pack_cards:
        new_pool = dict(pool_counts)
        new_pool[card] = new_pool.get(card, 0) + 1
        if model is not None:
            state_vec = encode_state(new_pool, pack_no=pack_number, pick_no=pick_number)
            try:
                score = _predict_state_value(model, state_vec)
            except Exception:
                score = evaluate_deck(new_pool)
        else:
            score = evaluate_deck(new_pool)
        if score > best_score:
            best_score = score
            best_card = card
    return best_card


def hero_policy_soft(
    pack_cards: List[str],
    pool_counts: Dict[str, int],
    seat_idx: int,
    rng: np.random.Generator,
    temperature: Optional[float] = None,
    epsilon: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    Stochastic hero policy using softmax over hero value estimates.

    - temperature: softmax temperature (default from HERO_TEMP env or 0.25).
    - epsilon: with prob epsilon, pick uniformly from pack (default 0).
    - top_k: restrict sampling to top-k cards after scoring (optional).
    """
    model = _get_state_value_model()
    temp = temperature if temperature is not None else float(os.getenv("HERO_TEMP", "0.25"))
    eps = epsilon if epsilon is not None else float(os.getenv("HERO_EPS", "0.0"))
    total_picks = sum(pool_counts.values())
    pack_number = (total_picks // 15) + 1
    pick_number = (total_picks % 15) + 1

    scores = []
    for card in pack_cards:
        new_pool = dict(pool_counts)
        new_pool[card] = new_pool.get(card, 0) + 1
        if model is not None:
            state_vec = encode_state(new_pool, pack_no=pack_number, pick_no=pick_number)
            try:
                score = _predict_state_value(model, state_vec)
            except Exception:
                score = evaluate_deck(new_pool)
        else:
            score = evaluate_deck(new_pool)
        scores.append(score)

    scores_arr = np.array(scores, dtype=float)
    if eps > 0 and rng.random() < eps:
        return rng.choice(pack_cards)

    # softmax with temperature
    scale = max(temp, 1e-6)
    logits = scores_arr / scale
    logits = logits - logits.max()  # numerical stability
    probs = np.exp(logits)
    probs = probs / probs.sum()

    if top_k is not None and 1 <= top_k < len(pack_cards):
        top_idx = np.argsort(-scores_arr)[:top_k]
        mask = np.zeros_like(probs)
        mask[top_idx] = 1
        probs = probs * mask
        probs = probs / probs.sum()

    choice_idx = rng.choice(len(pack_cards), p=probs)
    return pack_cards[int(choice_idx)]


def hero_policy_distill(
    pack_cards: List[str],
    pool_counts: Dict[str, int],
    seat_idx: int,
    rng: np.random.Generator,
    temperature: Optional[float] = None,
    epsilon: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    Stochastic policy distilled from hero values:
    - Uses pre-trained distill regressor on (state, card) features predicting hero soft targets.
    - Falls back to hero_policy_soft if distill model is missing.
    """
    model = _get_distill_model()
    if model is None:
        return hero_policy_soft(pack_cards, pool_counts, seat_idx, rng, temperature=temperature, epsilon=epsilon, top_k=top_k)

    temp = temperature if temperature is not None else float(os.getenv("HERO_DISTILL_TEMP", "0.25"))
    eps = epsilon if epsilon is not None else float(os.getenv("HERO_EPS", "0.0"))
    total_picks = sum(pool_counts.values())
    pack_number = (total_picks // 15) + 1
    pick_number = (total_picks % 15) + 1
    base_state = encode_state(pool_counts, pack_no=pack_number, pick_no=pick_number)

    feats = []
    expected_len = getattr(model, "n_features_in_", None)
    for card in pack_cards:
        card_vec = np.array(encode_card(card), dtype=float).ravel()
        feat = np.concatenate([base_state, card_vec])
        if expected_len is None:
            expected_len = len(feat)
        # hard align to expected length if the model defines it
        if expected_len is not None:
            if len(feat) < expected_len:
                pad = np.zeros(expected_len - len(feat), dtype=float)
                feat = np.concatenate([feat, pad])
            elif len(feat) > expected_len:
                feat = feat[:expected_len]
        # if still inconsistent (model without n_features_in_), track max length
        else:
            expected_len = len(feat)
        feats.append(feat)
    # final padding to max length in feats to avoid vstack mismatch
    max_len = max(len(f) for f in feats)
    feats_aligned = []
    for f in feats:
        if len(f) < max_len:
            pad = np.zeros(max_len - len(f), dtype=float)
            f = np.concatenate([f, pad])
        feats_aligned.append(f)
    feats_arr = np.vstack(feats_aligned)
    preds = np.array(model.predict(feats_arr), dtype=float)

    if eps > 0 and rng.random() < eps:
        return rng.choice(pack_cards)

    scale = max(temp, 1e-6)
    logits = preds / scale
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    if top_k is not None and 1 <= top_k < len(pack_cards):
        top_idx = np.argsort(-preds)[:top_k]
        mask = np.zeros_like(probs)
        mask[top_idx] = 1
        probs = probs * mask
        probs = probs / probs.sum()
    choice_idx = rng.choice(len(pack_cards), p=probs)
    return pack_cards[int(choice_idx)]


__all__ = ["hero_policy", "hero_policy_soft", "hero_policy_distill"]
