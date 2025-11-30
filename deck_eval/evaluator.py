"""
Deck evaluator wrapper using the calibrated deck effect model.

Requires:
- models/deck_effect_xgb.json (XGBoost booster)
- models/deck_effect_meta.pkl (theta0/theta1, deck_cols, calibration params)
- deck_eval/cards_index.json (name->index mapping aligned to deck_cols)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
CARD_INDEX_PATH = REPO_ROOT / "deck_eval" / "cards_index.json"


class DeckEvaluator:
    def __init__(self, base_p_cal: float = 0.5):
        self.base_p_cal = base_p_cal
        self.booster = xgb.Booster()
        self.booster.load_model(MODELS_DIR / "deck_effect_xgb.json")
        meta_path = MODELS_DIR / "deck_effect_meta.pkl"
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
        self.deck_cols: List[str] = self.meta.get("deck_cols", [])
        self.theta0 = float(self.meta.get("theta0", 0.0))
        self.theta1 = float(self.meta.get("theta1", 1.0))
        # card index
        with open(CARD_INDEX_PATH, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        self.name_to_idx = idx_data.get("name_to_idx", {})

    def _vectorize(self, deck_counts: Dict[str, int]) -> np.ndarray:
        vec = np.zeros(len(self.deck_cols) + 1, dtype=float)  # base_p_cal + deck counts
        vec[0] = self.base_p_cal
        for name, cnt in deck_counts.items():
            idx = self.name_to_idx.get(name)
            if idx is None:
                continue
            vec[idx + 1] += cnt
        return vec

    def evaluate_bump(self, deck_counts: Dict[str, int]) -> float:
        """Return the raw deck-effect bump s (before logistic calibration)."""
        x = self._vectorize(deck_counts)
        dmat = xgb.DMatrix(x.reshape(1, -1))
        s = float(self.booster.predict(dmat)[0])
        return s

    def evaluate(self, deck_counts: Dict[str, int]) -> float:
        s = self.evaluate_bump(deck_counts)  # deck effect bump
        z = np.log(self.base_p_cal / (1 - self.base_p_cal)) + self.theta0 + self.theta1 * s
        p = 1 / (1 + np.exp(-z))
        return float(np.clip(p, 0.0, 1.0))


_evaluator = None


def evaluate_deck(deck_counts: Dict[str, int]) -> float:
    global _evaluator
    if _evaluator is None:
        _evaluator = DeckEvaluator()
    return _evaluator.evaluate(deck_counts)


def evaluate_deck_bump(deck_counts: Dict[str, int]) -> float:
    global _evaluator
    if _evaluator is None:
        _evaluator = DeckEvaluator()
    return _evaluator.evaluate_bump(deck_counts)


__all__ = ["evaluate_deck", "evaluate_deck_bump", "DeckEvaluator"]
