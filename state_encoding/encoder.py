from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS_PATH = REPO_ROOT / "data" / "processed" / "cards.parquet"
CARD_INDEX_PATH = REPO_ROOT / "deck_eval" / "cards_index.json"

# load card metadata
CARDS_DF = pd.read_parquet(CARDS_PATH) if CARDS_PATH.exists() else pd.DataFrame()
with open(CARD_INDEX_PATH, "r", encoding="utf-8") as f:
    CARD_INDEX = json.load(f).get("name_to_idx", {})

# Precompute card lookup tables to avoid repeated DataFrame scans
def _normalize_colors(raw) -> List[str]:
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if raw is None or raw is False:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [c for c in parsed if isinstance(c, str)]
        except Exception:
            return []
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, str)]
    return []


CARD_ROW_MAP: Dict[str, pd.Series] = (
    {row["name"]: row for _, row in CARDS_DF.iterrows()} if not CARDS_DF.empty else {}
)

# color_one_hot, cmc, is_removal, is_fixing
CARD_STATE_FEATURES: Dict[str, Tuple[np.ndarray, float, bool, bool]] = {}
# cached encode_card vectors
CARD_ENCODE_CACHE: Dict[str, np.ndarray] = {}

for name, row in CARD_ROW_MAP.items():
    colors = _normalize_colors(row.get("color_identity"))
    color_vec = np.array(
        [1 if c in colors else 0 for c in ["W", "U", "B", "R", "G"]], dtype=float
    )
    cmc = float(row.get("cmc") or 0)
    type_line = row.get("type_line") or ""
    is_removal = "Instant" in type_line or "Sorcery" in type_line
    is_fixing = "Land" in type_line or "Artifact" in type_line
    CARD_STATE_FEATURES[name] = (color_vec, cmc, is_removal, is_fixing)

    rarity = row.get("rarity") or ""
    rarity_map = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}
    rarity_val = rarity_map.get(str(rarity).lower(), 0)
    is_creature = 1.0 if "Creature" in type_line else 0.0
    idx = float(CARD_INDEX.get(name, -1))
    CARD_ENCODE_CACHE[name] = np.concatenate(
        [color_vec, np.array([cmc, rarity_val, is_creature, idx], dtype=float)]
    )


def _color_one_hot(colors: List[str]) -> List[int]:
    order = ["W", "U", "B", "R", "G"]
    return [1 if c in colors else 0 for c in order]


def _get_card_row(name: str):
    return CARD_ROW_MAP.get(name)


def encode_state(pool_counts: Dict[str, int], pack_no: int, pick_no: int, skill_bucket: str | None = None) -> np.ndarray:
    """Aggregate pool features."""
    total = sum(v or 0 for v in pool_counts.values())
    color_counts = np.zeros(5, dtype=float)
    curve = np.zeros(8, dtype=float)  # cmc bins 0-7+
    removal = 0
    fixing = 0
    for name, cnt in pool_counts.items():
        cnt = cnt or 0
        feats = CARD_STATE_FEATURES.get(name)
        if feats is None:
            continue
        color_vec, cmc, is_removal, is_fixing = feats
        color_counts += cnt * color_vec
        bin_idx = int(min(cmc, 7))
        curve[bin_idx] += cnt
        if is_removal:
            removal += cnt
        if is_fixing:
            fixing += cnt
    skill = 0.0
    if skill_bucket:
        try:
            skill = float(skill_bucket)
        except Exception:
            skill = 0.0
    return np.concatenate(
        [
            np.array([total, pack_no, pick_no, skill], dtype=float),
            color_counts,
            curve,
            np.array([removal, fixing], dtype=float),
        ]
    )


def encode_card(card_name: str) -> np.ndarray:
    """Card-level features from metadata and index."""
    cached = CARD_ENCODE_CACHE.get(card_name)
    if cached is not None:
        return cached
    return np.zeros(10, dtype=float)


__all__ = ["encode_state", "encode_card"]
