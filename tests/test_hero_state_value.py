import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hero_bot.train_state_value as hsv
import hero_bot.hero_policy as hpolicy


def _fake_bc_tensors(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "draft_id": f"d{i}",
                "pack_number": 1,
                "pick_number": i + 1,
                "skill_bucket": "A",
                "human_pick": "card_a",
                "pool_counts": {"card_a": i},
                "pack_cards": ["card_a", "card_b"],
                "card_features": [[0.0], [1.0]],
            }
        )
    return pd.DataFrame(rows)


def test_train_state_value_and_hero_policy(monkeypatch):
    df = _fake_bc_tensors()
    tmp_dir = Path("tests/tmp_state_value")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # simple evaluator: value = count of card_a
    monkeypatch.setattr(hsv, "evaluate_deck", lambda pool: float(pool.get("card_a", 0)))
    metrics = hsv.train_state_value(df_override=df, seed=0, test_frac=0.2, model_dir=tmp_dir, max_rows=None)
    assert (tmp_dir / "state_value.pkl").exists()
    assert "RMSE" in metrics

    # hero policy should prefer card_a
    monkeypatch.setattr(hpolicy, "MODEL_PATH", tmp_dir / "state_value.pkl")
    # reload model cache
    if hasattr(hpolicy, "_STATE_VALUE_MODEL"):
        hpolicy._STATE_VALUE_MODEL = None
    pick = hpolicy.hero_policy(["card_a", "card_b"], {"card_a": 0}, seat_idx=0, rng=np.random.default_rng(0))
    assert pick == "card_a"

    shutil.rmtree(tmp_dir, ignore_errors=True)
