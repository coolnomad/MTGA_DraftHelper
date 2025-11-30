from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from state_encoding.encoder import encode_state
from deck_eval.evaluator import evaluate_deck
from hero_bot.train_state_value import MODEL_PATH, load_state_value_model

_STATE_VALUE_MODEL = None


def _get_state_value_model():
    global _STATE_VALUE_MODEL
    if _STATE_VALUE_MODEL is None and MODEL_PATH.exists():
        _STATE_VALUE_MODEL = load_state_value_model(MODEL_PATH)
    return _STATE_VALUE_MODEL

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
            score = float(model.predict(state_vec.reshape(1, -1))[0])
        else:
            score = evaluate_deck(new_pool)
        if score > best_score:
            best_score = score
            best_card = card
    return best_card


__all__ = ["hero_policy"]
