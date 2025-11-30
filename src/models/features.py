from typing import Tuple

import numpy as np
import pandas as pd

COLOR_ORDER = ["W", "U", "B", "R", "G"]
RANK_ORDER = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Mythic"]


def _encode_rank(series: pd.Series) -> pd.Series:
    """Map rank strings to ordered integers; unknowns -> -1."""
    cat = pd.Categorical(series.astype(str), categories=RANK_ORDER, ordered=True)
    # codes assigns -1 to values not in categories
    return pd.Series(cat.codes.astype(float), index=series.index, name="rank")


def _encode_colors(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Multi-hot encode color strings like 'WU' into columns prefix_W...prefix_G."""
    series = series.fillna("").astype(str).str.upper()
    data = {}
    for color in COLOR_ORDER:
        data[f"{prefix}_{color}"] = series.apply(lambda s: float(color in s))
    return pd.DataFrame(data, index=series.index)


def build_skill_features(decks: pd.DataFrame) -> pd.DataFrame:
    """Build skill-only feature frame with rank encoding and user buckets."""
    feats = pd.DataFrame(index=decks.index)
    feats["rank"] = _encode_rank(decks["rank"])
    feats["user_n_games_bucket"] = pd.to_numeric(
        decks["user_n_games_bucket"], errors="coerce"
    )
    feats["user_game_win_rate_bucket"] = pd.to_numeric(
        decks["user_game_win_rate_bucket"], errors="coerce"
    )
    return feats.fillna(0)


def build_deck_features(decks: pd.DataFrame) -> pd.DataFrame:
    """Build deck composition features including card counts, size, and color encodings."""
    deck_cols = [c for c in decks.columns if c.startswith("deck_")]
    deck_df = decks[deck_cols].astype(float).fillna(0)

    deck_df = deck_df.copy()
    deck_df["deck_size_avg"] = pd.to_numeric(
        decks["deck_size_avg"], errors="coerce"
    ).fillna(0)

    # color encodings
    main_encoded = _encode_colors(decks["main_colors"], "main")
    splash_encoded = _encode_colors(decks["splash_colors"], "splash")

    return pd.concat([deck_df, main_encoded, splash_encoded], axis=1)


def build_joint_features(decks: pd.DataFrame) -> pd.DataFrame:
    """Concatenate deck and skill features for joint modeling."""
    deck_feats = build_deck_features(decks)
    skill_feats = build_skill_features(decks)
    return pd.concat([deck_feats, skill_feats], axis=1)


def train_test_split_indices(
    n_rows: int, seed: int = 1337, test_frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic shuffled indices for an 80/20 split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    split = int(n_rows * (1 - test_frac))
    return idx[:split], idx[split:]
