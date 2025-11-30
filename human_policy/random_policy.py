from __future__ import annotations

from typing import Dict, List
import numpy as np


def random_policy(pack_cards: List[str], pool_counts: Dict[str, int], seat_idx: int, rng: np.random.Generator) -> str:
    """Pick a random card from the pack."""
    return rng.choice(pack_cards).item()


__all__ = ["random_policy"]
