from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from data.loaders import load_decks
from .features import build_deck_features


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class ModelMetrics:
    r2: float
    rmse: float


def _train_test_split(X, y, test_frac: float = 0.2, seed: int = 1) -> Tuple:
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    split = int(len(idx) * (1 - test_frac))
    train_idx, test_idx = idx[:split], idx[split:]
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def train_skill_model() -> ModelMetrics:
    decks = load_decks()
    X, y_wr, y_skill = build_deck_features(decks)

    X_tr, X_te, y_tr, y_te = _train_test_split(
        X[["user_skill_proxy"]] if "user_skill_proxy" in X else X.assign(user_skill_proxy=y_skill)[["user_skill_proxy"]],
        y_wr,
    )

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    metrics = ModelMetrics(
        r2=r2_score(y_te, y_pred),
        rmse=np.sqrt(mean_squared_error(y_te, y_pred)),
    )

    joblib.dump(model, MODELS_DIR / "skill_model.pkl")
    return metrics


def train_net_wr_model() -> ModelMetrics:
    decks = load_decks()
    X, y_wr, y_skill = build_deck_features(decks)

    X_tr, X_te, y_tr, y_te = _train_test_split(X, y_wr)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=1,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    metrics = ModelMetrics(
        r2=r2_score(y_te, y_pred),
        rmse=np.sqrt(mean_squared_error(y_te, y_pred)),
    )

    joblib.dump(model, MODELS_DIR / "net_wr_model.pkl")
    return metrics


def train_residual_model() -> ModelMetrics:
    decks = load_decks()
    X, y_wr, y_skill = build_deck_features(decks)

    # simple residual: subtract normalized skill proxy
    skill_centered = (y_skill - y_skill.mean()) / (y_skill.std() + 1e-8)
    y_resid = y_wr - 0.1 * skill_centered  # placeholder

    X_tr, X_te, y_tr, y_te = _train_test_split(X, y_resid)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=2,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    metrics = ModelMetrics(
        r2=r2_score(y_te, y_pred),
        rmse=np.sqrt(mean_squared_error(y_te, y_pred)),
    )

    joblib.dump(model, MODELS_DIR / "residual_model.pkl")
    return metrics


if __name__ == "__main__":
    print("training skill model...")
    print(train_skill_model())

    print("training net wr model...")
    print(train_net_wr_model())

    print("training residual model...")
    print(train_residual_model())
