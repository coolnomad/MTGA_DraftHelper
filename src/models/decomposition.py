from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from src.data.loaders import load_decks
from src.data.paths import DATA_PROCESSED
from src.models.features import build_joint_features, build_skill_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_PATH = DATA_PROCESSED / "decks_with_preds.parquet"


def run_decomposition() -> Dict[str, str]:
    """Compute skill and deck contributions (M3) and persist the augmented table."""
    decks = load_decks()
    if not (MODELS_DIR / "skill_model.pkl").exists():
        raise FileNotFoundError("skill_model.pkl not found in models/. Train M1 first.")
    if not (MODELS_DIR / "joint_model.pkl").exists():
        raise FileNotFoundError("joint_model.pkl not found in models/. Train M2 first.")

    skill_model = joblib.load(MODELS_DIR / "skill_model.pkl")
    joint_model = joblib.load(MODELS_DIR / "joint_model.pkl")

    skill_features = build_skill_features(decks)
    joint_features = build_joint_features(decks)

    decks = decks.copy()
    decks["skill_pred"] = skill_model.predict(skill_features)
    decks["joint_pred"] = joint_model.predict(joint_features)
    decks["deck_boost"] = decks["joint_pred"] - decks["skill_pred"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    decks.to_parquet(OUTPUT_PATH, index=False)
    return {"output_path": str(OUTPUT_PATH)}


if __name__ == "__main__":
    result = run_decomposition()
    print(result)
