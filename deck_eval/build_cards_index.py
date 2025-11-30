"""
Build cards_index.json mapping card names to indices aligned with deck_effect_model deck_cols.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_PATH = REPO_ROOT / "deck_eval" / "cards_index.json"


def main():
    meta_path = MODELS_DIR / "deck_effect_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    deck_cols = meta.get("deck_cols", [])
    names = [c.replace("deck_", "") for c in deck_cols]
    name_to_idx = {name: i for i, name in enumerate(names)}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"name_to_idx": name_to_idx, "deck_cols": deck_cols}, f, indent=2)
    print("wrote", OUTPUT_PATH, "with", len(name_to_idx), "cards")


if __name__ == "__main__":
    main()
