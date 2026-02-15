from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META = REPO_ROOT / "reports" / "pool_models" / "pool_models_meta.json"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class PoolEvaluator:
    """
    Lightweight predictor for pool -> deck effect/bump using the pool models trained
    by scripts/train_pool_models_from_games.py.
    """

    def __init__(self, base_p: float = 0.5, meta_path: Path | None = None):
        self.base_p = float(base_p)
        meta_path = meta_path or DEFAULT_META
        if not meta_path.exists():
            raise FileNotFoundError(f"Pool model meta not found at {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_cols: List[str] = meta.get("feature_cols", [])
        if not self.feature_cols or self.feature_cols[0] != "base_p":
            raise ValueError("feature_cols missing or does not start with base_p")

        models_dir = Path(meta["models"]["deck_effect"]).parent
        self.model_effect = xgb.Booster()
        self.model_effect.load_model(models_dir / "deck_effect_xgb.json")
        self.model_bump = xgb.Booster()
        self.model_bump.load_model(models_dir / "deck_bump_xgb.json")

        # Use test split calibration if available, otherwise fall back to identity
        eff_res = meta.get("results", {}).get("deck_effect", {}).get("test", {})
        self.theta_eff0 = float(eff_res.get("theta0", 0.0))
        self.theta_eff1 = float(eff_res.get("theta1", 1.0))
        bump_res = meta.get("results", {}).get("deck_bump", {}).get("test", {})
        self.theta_bump0 = float(bump_res.get("theta0", self.theta_eff0))
        self.theta_bump1 = float(bump_res.get("theta1", self.theta_eff1))

        # Map pool_foo -> index
        self.col_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.feature_cols)}

    def _vectorize_pool(self, pool_counts: Dict[str, int]) -> np.ndarray:
        vec = np.zeros(len(self.feature_cols), dtype=float)
        vec[self.col_to_idx["base_p"]] = self.base_p
        for name, cnt in pool_counts.items():
            key = f"pool_{name}"
            idx = self.col_to_idx.get(key)
            if idx is None:
                continue
            vec[idx] += float(cnt)
        return vec

    def _predict_delta(self, booster: xgb.Booster, pool_counts: Dict[str, int]) -> float:
        x = self._vectorize_pool(pool_counts).reshape(1, -1)
        return float(booster.predict(xgb.DMatrix(x))[0])

    def predict_effect(self, pool_counts: Dict[str, int]) -> Tuple[float, float]:
        """
        Returns (p_hat, delta_hat) where delta_hat is raw model output and p_hat is
        base_p calibrated with theta.
        """
        delta = self._predict_delta(self.model_effect, pool_counts)
        z = np.log(self.base_p / (1 - self.base_p)) + self.theta_eff0 + self.theta_eff1 * delta
        p = 1 / (1 + np.exp(-z))
        return _clip01(p), delta

    def predict_bump(self, pool_counts: Dict[str, int]) -> Tuple[float, float]:
        delta = self._predict_delta(self.model_bump, pool_counts)
        z = np.log(self.base_p / (1 - self.base_p)) + self.theta_bump0 + self.theta_bump1 * delta
        p = 1 / (1 + np.exp(-z))
        return _clip01(p), delta


_pool_evaluator: PoolEvaluator | None = None


def evaluate_pool_effect(pool_counts: Dict[str, int], base_p: float = 0.5) -> Tuple[float, float]:
    global _pool_evaluator
    if _pool_evaluator is None or _pool_evaluator.base_p != base_p:
        _pool_evaluator = PoolEvaluator(base_p=base_p)
    return _pool_evaluator.predict_effect(pool_counts)


def evaluate_pool_bump(pool_counts: Dict[str, int], base_p: float = 0.5) -> Tuple[float, float]:
    global _pool_evaluator
    if _pool_evaluator is None or _pool_evaluator.base_p != base_p:
        _pool_evaluator = PoolEvaluator(base_p=base_p)
    return _pool_evaluator.predict_bump(pool_counts)


__all__ = ["evaluate_pool_effect", "evaluate_pool_bump", "PoolEvaluator"]
