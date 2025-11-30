from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from src.data.loaders import load_decks
from src.models.features import build_skill_features, train_test_split_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_skill_model(seed: int = 1337, test_frac: float = 0.2) -> Dict[str, float]:
    """Train M1 skill-only ridge regression and persist to disk."""
    decks = load_decks()
    X = build_skill_features(decks).fillna(0)
    y = decks["run_wr"].astype(float)

    train_idx, test_idx = train_test_split_indices(len(decks), seed=seed, test_frac=test_frac)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    joblib.dump(model, MODELS_DIR / "skill_model.pkl")

    return {"R2_skill": r2, "RMSE_skill": rmse}


if __name__ == "__main__":
    metrics = train_skill_model()
    print(metrics)
