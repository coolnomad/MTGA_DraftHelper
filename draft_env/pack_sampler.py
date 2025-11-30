from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS_PARQUET = REPO_ROOT / "data" / "processed" / "cards.parquet"
CARD_INDEX_PATH = REPO_ROOT / "deck_eval" / "cards_index.json"
EMPIRICAL_DRAFTS = REPO_ROOT / "data" / "processed" / "drafts.parquet"
EMPIRICAL_PACKS_PARQUET = REPO_ROOT / "data" / "processed" / "observed_packs.parquet"

# Arena-style pack slots (simplified)
RARE_SLOTS = 1
UNCOMMON_SLOTS = 3
BASIC_LAND_SLOTS = 1
MYTHIC_RATE = 1 / 8  # mythic replaces rare with this probability
RARITIES = ["common", "uncommon", "rare", "mythic"]

# cached rarity pools by set_code (upper) and ALL
_RARITY_POOLS: Dict[str, Dict[str, List[str]]] | None = None
_ALL_CARDS: List[str] | None = None
_CARD_SET: Dict[str, str] = {}
_EMPIRICAL_COUNTS: Dict[str, int] | None = None
_OBSERVED_PACKS: List[List[str]] | None = None


def _load_cards_metadata() -> None:
    """
    Build rarity pools from cards.parquet if available; fallback to cards_index.json
    treating everything as common.
    """
    global _RARITY_POOLS, _ALL_CARDS
    if _RARITY_POOLS is not None and _RARITY_POOLS.get("ALL", {}).get("common"):
        return

    pools: Dict[str, Dict[str, List[str]]] = {}

    if CARDS_PARQUET.exists():
        df = pd.read_parquet(CARDS_PARQUET)
        df["rarity_norm"] = df.get("rarity", "").astype(str).str.lower()
        df["set_code"] = df.get("set", "").astype(str).str.upper().str.strip()
        df["type_line_norm"] = df.get("type_line", "").astype(str)
        _CARD_SET.update({row["name"]: row["set_code"] for _, row in df[["name", "set_code"]].iterrows() if pd.notna(row["name"])})

        def build_for_set(sub: pd.DataFrame) -> Dict[str, List[str]]:
            pools = {
                r: sub.loc[sub["rarity_norm"] == r, "name"].dropna().astype(str).tolist()
                for r in RARITIES
            }
            basics = sub[sub["type_line_norm"].str.contains("Basic", case=False, na=False)]["name"].dropna().astype(str).tolist()
            pools["basic_land"] = basics
            return pools

        # per-set pools
        for set_code, sub in df.groupby("set_code"):
            pools[set_code] = build_for_set(sub)

        # global pool across sets
        pools["ALL"] = build_for_set(df)
    else:
        # fallback to evaluator card index; treat all as commons
        if CARD_INDEX_PATH.exists():
            data = json.loads(CARD_INDEX_PATH.read_text(encoding="utf-8"))
            names = list(data.get("name_to_idx", {}).keys())
        else:
            names = []
        pools["ALL"] = {
            "common": names,
            "uncommon": [],
            "rare": [],
            "mythic": [],
            "basic_land": [],
        }

    _RARITY_POOLS = pools
    # flatten all cards for padding small packs
    _ALL_CARDS = list({name for sets in pools.values() for names in sets.values() for name in names})


def _load_empirical_counts() -> None:
    """
    Build empirical occurrence counts from drafts.parquet pack_card_* columns.
    Uses counts as weights for sampling plausible packs.
    """
    global _EMPIRICAL_COUNTS
    if _EMPIRICAL_COUNTS is not None:
        return
    if not EMPIRICAL_DRAFTS.exists():
        _EMPIRICAL_COUNTS = {}
        return
    # read only pack_card_* columns to save memory
    schema = pq.read_schema(EMPIRICAL_DRAFTS)
    pack_cols = [name for name in schema.names if name.startswith("pack_card_")]
    if not pack_cols:
        _EMPIRICAL_COUNTS = {}
        return
    table = pq.read_table(EMPIRICAL_DRAFTS, columns=pack_cols)
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    counts: Dict[str, int] = {}
    for col in pack_cols:
        name = col.replace("pack_card_", "")
        # values are indicators (0/1)
        counts[name] = int(df[col].fillna(0).sum())
    _EMPIRICAL_COUNTS = counts


def _load_observed_packs() -> None:
    """Load observed packs (as card name lists) from observed_packs.parquet if present."""
    global _OBSERVED_PACKS
    if _OBSERVED_PACKS is not None:
        return
    if not EMPIRICAL_PACKS_PARQUET.exists():
        _OBSERVED_PACKS = []
        return
    df = pd.read_parquet(EMPIRICAL_PACKS_PARQUET)
    packs_col = "pack_cards"
    if packs_col not in df.columns:
        _OBSERVED_PACKS = []
        return
    packs = []
    for vals in df[packs_col]:
        if isinstance(vals, list):
            packs.append([str(v) for v in vals])
    _OBSERVED_PACKS = packs


def load_card_pool(set_code: str | None = None) -> Dict[str, List[str]]:
    """Return rarity pools for a set (upper-cased code) or the global pool."""
    _load_cards_metadata()
    assert _RARITY_POOLS is not None
    key = set_code.upper() if set_code else "ALL"
    pool = _RARITY_POOLS.get(key)
    if not pool or not pool.get("common"):
        pool = _RARITY_POOLS.get("ALL", {})
    return pool


def _sample_without_replacement(rng: np.random.Generator, pool: List[str], k: int) -> List[str]:
    if k <= 0 or not pool:
        return []
    k = min(k, len(pool))
    return rng.choice(pool, size=k, replace=False).tolist()


def _sample_empirical_pack(rng: np.random.Generator, pack_size: int, set_code: str | None) -> List[str]:
    """
    Sample without replacement using empirical occurrence weights from drafts.parquet.
    Ensures at most one basic land (Plains/Island/Swamp/Mountain/Forest).
    """
    _load_empirical_counts()
    if not _EMPIRICAL_COUNTS:
        return []
    # use all occurrences to allow FCA cards to appear alongside main set
    items = list(_EMPIRICAL_COUNTS.items())
    if not items:
        return []

    basics = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
    names = [n for n, _ in items]
    weights = np.array([c for _, c in items], dtype=float)
    weights = weights / weights.sum()

    # force at most one basic, and include one if available
    chosen: List[str] = []
    available = names.copy()
    avail_weights = weights.copy()

    # pick basic first if any
    basic_indices = [i for i, nm in enumerate(available) if nm in basics]
    if basic_indices:
        probs = avail_weights[basic_indices] / avail_weights[basic_indices].sum()
        pick_idx = rng.choice(len(basic_indices), p=probs)
        basic_choice_idx = basic_indices[pick_idx]
        chosen.append(available.pop(basic_choice_idx))
        avail_weights = np.delete(avail_weights, basic_choice_idx)
        # drop other basics
        keep_mask = [nm not in basics for nm in available]
        available = [nm for nm, keep in zip(available, keep_mask) if keep]
        avail_weights = avail_weights[keep_mask]

    # fill remaining without replacement
    for _ in range(min(pack_size - len(chosen), len(available))):
        total = avail_weights.sum()
        if total <= 0:
            break
        probs = avail_weights / total
        pick_idx = rng.choice(len(available), p=probs)
        chosen.append(available.pop(pick_idx))
        avail_weights = np.delete(avail_weights, pick_idx)

    return chosen


def sample_pack(rng: np.random.Generator, pack_size: int = 15, set_code: str | None = None) -> List[str]:
    """
    Sample a realistic pack:
      - prefers empirical pack distribution from drafts.parquet (no replacement, max one basic land)
      - 1 rare (or mythic at MYTHIC_RATE), no duplicate rares/mythics
      - 3 uncommons
      - 1 basic land (falls back to common)
      - remaining commons to reach pack_size
    Falls back to global pool if set-specific pools are missing.
    """
    _load_observed_packs()
    if _OBSERVED_PACKS:
        pack = rng.choice(_OBSERVED_PACKS).copy()
        if len(pack) > pack_size:
            rng.shuffle(pack)
            pack = pack[:pack_size]
        if len(pack) < pack_size:
            _load_cards_metadata()
            extra_pool = [c for c in (_ALL_CARDS or []) if c not in pack]
            extras = _sample_without_replacement(rng, extra_pool, pack_size - len(pack))
            pack.extend(extras)
        return pack[:pack_size]

    # try empirical first
    empirical = _sample_empirical_pack(rng, pack_size, set_code)
    if empirical:
        pools = load_card_pool(set_code)
        basics_names = set(pools.get("basic_land", []))
        # ensure at most one basic land
        basic_in_pack = [c for c in empirical if c in basics_names]
        pack = empirical[:pack_size]
        if basics_names:
            if not basic_in_pack:
                # inject a basic and drop a random non-basic if needed
                basic_choice = rng.choice(list(basics_names))
                pack.append(basic_choice)
                if len(pack) > pack_size:
                    # drop a random non-basic
                    non_basics = [c for c in pack if c not in basics_names]
                    if non_basics:
                        drop = rng.choice(non_basics)
                        pack.remove(drop)
            elif len(basic_in_pack) > 1:
                # trim extras
                to_remove = len(basic_in_pack) - 1
                for _ in range(to_remove):
                    # remove later occurrences
                    for i in range(len(pack) - 1, -1, -1):
                        if pack[i] in basics_names:
                            pack.pop(i)
                            break
        return pack[:pack_size]

    pools = load_card_pool(set_code)
    global_pool = load_card_pool(None)
    assert _ALL_CARDS is not None

    # rare / mythic slot
    if pools["mythic"] and rng.random() < MYTHIC_RATE:
        rares = _sample_without_replacement(rng, pools["mythic"], RARE_SLOTS)
    else:
        rares = _sample_without_replacement(rng, pools["rare"], RARE_SLOTS)
        if len(rares) < RARE_SLOTS:  # fallback to mythic if rare pool empty
            rares += _sample_without_replacement(rng, pools["mythic"], RARE_SLOTS - len(rares))

    # uncommons
    uncommons = _sample_without_replacement(rng, pools["uncommon"], UNCOMMON_SLOTS)

    basics_names = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
    # basic land slot (falls back to commons if none)
    basics_pool = pools.get("basic_land", [])
    basics = _sample_without_replacement(rng, basics_pool, BASIC_LAND_SLOTS)
    if len(basics) < BASIC_LAND_SLOTS:
        # ensure fallback common isn't a second basic by filtering
        common_pool_no_basics = [c for c in pools["common"] if c not in basics_names]
        basics += _sample_without_replacement(rng, common_pool_no_basics, BASIC_LAND_SLOTS - len(basics))

    # commons to fill remaining slots
    remaining = max(pack_size - len(rares) - len(uncommons) - len(basics), 0)
    common_pool_no_basics = [c for c in pools["common"] if c not in basics_names]
    commons = _sample_without_replacement(rng, common_pool_no_basics, remaining)

    pack = rares + uncommons + basics + commons

    # If we still lack cards (e.g., small pools), pad from global pool without duplicates.
    if len(pack) < pack_size:
        need = pack_size - len(pack)
        # avoid duplicates already chosen
        remaining_pool = [c for c in _ALL_CARDS if c not in set(pack)]
        pack.extend(_sample_without_replacement(rng, remaining_pool, need))

    return pack[:pack_size]


__all__ = ["load_card_pool", "sample_pack"]
