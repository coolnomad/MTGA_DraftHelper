import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.models import (
    build_deck_features,
    build_joint_features,
    build_skill_features,
    train_test_split_indices,
)
from src.models import skill_model as skill_mod
from src.models import joint_model as joint_mod
from src.models import decomposition as decomp_mod


def _make_fake_decks(n: int = 30) -> pd.DataFrame:
    ranks = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Mythic"]
    main_colors = ["W", "U", "B", "R", "G", "WU", "BR", "UG", ""]
    splash_colors = ["", "R", "G", "U", "B"]
    data = {
        "rank": [ranks[i % len(ranks)] for i in range(n)],
        "user_n_games_bucket": np.linspace(0, 100, n),
        "user_game_win_rate_bucket": np.linspace(40, 70, n),
        "run_wr": np.linspace(0.4, 0.7, n),
        "deck_size_avg": np.full(n, 40),
        "main_colors": [main_colors[i % len(main_colors)] for i in range(n)],
        "splash_colors": [splash_colors[i % len(splash_colors)] for i in range(n)],
        "deck_card_a": np.random.randint(0, 4, size=n),
        "deck_card_b": np.random.randint(0, 4, size=n),
    }
    return pd.DataFrame(data)


def test_build_skill_features_encodes_rank_and_user_fields():
    df = _make_fake_decks(5)
    feats = build_skill_features(df)
    assert list(feats.columns) == [
        "rank",
        "user_n_games_bucket",
        "user_game_win_rate_bucket",
    ]
    assert feats["rank"].dtype.kind in ("i", "f")
    assert feats["rank"].iloc[0] == 0  # Bronze -> 0
    assert feats["rank"].max() <= 5


def test_build_deck_features_adds_color_multi_hot():
    df = _make_fake_decks(3)
    df.loc[0, "main_colors"] = ""
    df.loc[0, "splash_colors"] = ""
    feats = build_deck_features(df)
    for col in [
        "deck_card_a",
        "deck_card_b",
        "deck_size_avg",
        "main_W",
        "main_U",
        "main_B",
        "main_R",
        "main_G",
        "splash_W",
        "splash_U",
        "splash_B",
        "splash_R",
        "splash_G",
    ]:
        assert col in feats.columns
    # colorless row should be zeros
    colorless_row = feats.iloc[0]
    assert colorless_row[["main_W", "main_U", "main_B", "main_R", "main_G"]].sum() == 0
    assert colorless_row[["splash_W", "splash_U", "splash_B", "splash_R", "splash_G"]].sum() == 0


def test_train_test_split_indices_is_deterministic():
    first = train_test_split_indices(10, seed=42, test_frac=0.3)
    second = train_test_split_indices(10, seed=42, test_frac=0.3)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_train_skill_model_runs_and_saves(monkeypatch):
    df = _make_fake_decks(25)
    tmp_dir = Path("tests/tmp_skill_models")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(skill_mod, "load_decks", lambda: df)
    monkeypatch.setattr(skill_mod, "MODELS_DIR", tmp_dir)

    metrics = skill_mod.train_skill_model(seed=0, test_frac=0.2)

    assert (tmp_dir / "skill_model.pkl").exists()
    assert "R2_skill" in metrics and "RMSE_skill" in metrics
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_train_joint_model_runs_and_saves(monkeypatch):
    df = _make_fake_decks(30)
    tmp_dir = Path("tests/tmp_joint_models")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(joint_mod, "load_decks", lambda: df)
    monkeypatch.setattr(joint_mod, "MODELS_DIR", tmp_dir)

    metrics = joint_mod.train_joint_model(seed=1, test_frac=0.2)

    assert (tmp_dir / "joint_model.pkl").exists()
    assert {"R2_joint", "RMSE_joint", "delta_R2_deck"} <= metrics.keys()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_decomposition_uses_saved_models(monkeypatch):
    df = _make_fake_decks(20)
    tmp_dir = Path("tests/tmp_decomposition")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # train tiny ridge models as stand-ins
    skill_features = build_skill_features(df)
    joint_features = build_joint_features(df)
    y = df["run_wr"]

    skill_model = Ridge().fit(skill_features, y)
    joint_model = Ridge().fit(joint_features, y)

    # save to temp models dir
    models_dir = tmp_dir / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(skill_model, models_dir / "skill_model.pkl")
    joblib.dump(joint_model, models_dir / "joint_model.pkl")

    monkeypatch.setattr(decomp_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(decomp_mod, "OUTPUT_PATH", tmp_dir / "decks_with_preds.parquet")
    monkeypatch.setattr(decomp_mod, "load_decks", lambda: df)

    result = decomp_mod.run_decomposition()
    assert (tmp_dir / "decks_with_preds.parquet").exists()
    saved = pd.read_parquet(result["output_path"])
    assert {"skill_pred", "joint_pred", "deck_boost"} <= set(saved.columns)
    assert len(saved) == len(df)
    shutil.rmtree(tmp_dir, ignore_errors=True)
