from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.data.loaders import load_decks
from src.models.features import build_joint_features, build_skill_features, train_test_split_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_joint_model(seed: int = 1337, test_frac: float = 0.2) -> Dict[str, float]:
    """Train M2 joint deck+skill model and report metrics."""
    decks = load_decks()
    X_joint = build_joint_features(decks).fillna(0)
    X_skill = build_skill_features(decks).fillna(0)
    y = decks["run_wr"].astype(float)

    train_idx, test_idx = train_test_split_indices(len(decks), seed=seed, test_frac=test_frac)
    Xj_train, Xj_test = X_joint.iloc[train_idx], X_joint.iloc[test_idx]
    Xs_train, Xs_test = X_skill.iloc[train_idx], X_skill.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        n_estimators=300,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(Xj_train, y_train)

    y_pred = model.predict(Xj_test)
    r2_joint = r2_score(y_test, y_pred)
    rmse_joint = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # skill-only baseline for delta R2
    skill_model = Ridge(alpha=1.0)
    skill_model.fit(Xs_train, y_train)
    y_skill_pred = skill_model.predict(Xs_test)
    r2_skill = r2_score(y_test, y_skill_pred)

    delta_r2_deck = r2_joint - r2_skill

    joblib.dump(model, MODELS_DIR / "joint_model.pkl")

    return {
        "R2_joint": r2_joint,
        "RMSE_joint": rmse_joint,
        "R2_skill_baseline": r2_skill,
        "delta_R2_deck": delta_r2_deck,
    }


if __name__ == "__main__":
    metrics = train_joint_model()
    print(metrics)
