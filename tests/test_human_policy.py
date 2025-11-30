import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import human_policy.train_human_policy as hp


def _make_fake_bc_tensors(n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        pack_cards = ["card_a", "card_b"]
        card_feats = [[0.0, 0.0], [1.0, 1.0]]
        human_pick = "card_a" if i % 2 == 0 else "card_b"
        rows.append(
            {
                "draft_id": f"d{i}",
                "pack_number": 1,
                "pick_number": i + 1,
                "skill_bucket": "A",
                "human_pick": human_pick,
                "state_vec": [float(i), 0.0, 1.0],
                "pack_cards": pack_cards,
                "card_features": card_feats,
            }
        )
    return pd.DataFrame(rows)


def test_train_models_and_score_pack(monkeypatch):
    df = _make_fake_bc_tensors()
    tmp_dir = Path("tests/tmp_human_models")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hp, "MODEL_DIR", tmp_dir)

    metrics = hp.train_models(df_override=df, seed=0, test_frac=0.5)
    assert "A" in metrics
    assert "ALL" in metrics
    assert (tmp_dir / "human_policy_A.pkl").exists()
    assert (tmp_dir / "human_policy_ALL.pkl").exists()

    ranked = hp.score_pack(
        pack_cards=["card_a", "card_b"],
        pool_counts={},
        pack_number=1,
        pick_number=1,
        skill_bucket="A",
        state_vec=np.array([0.0, 0.0, 1.0]),
        card_features=[[0.0, 0.0], [1.0, 1.0]],
    )
    assert len(ranked) == 2
    probs = [p for _, p in ranked]
    assert np.isclose(sum(probs), 1.0)

    shutil.rmtree(tmp_dir, ignore_errors=True)
