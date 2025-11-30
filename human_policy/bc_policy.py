from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from human_policy.train_human_policy import score_pack


def make_bc_policy(skill_bucket: Optional[str] = None, pack_size: int = 15, packs_per_player: int = 3) -> Callable:
    """
    Build a draft policy that scores the pack with the trained BC model and picks the argmax.
    pack/pick numbers are inferred from current pool size assuming fixed pack_size/packs_per_player.
    """

    def _policy(pack_cards: List[str], pool_counts: Dict[str, int], seat_idx: int, rng: np.random.Generator):
        total_picks = sum(pool_counts.values())
        pack_number = min(packs_per_player, (total_picks // pack_size) + 1)
        pick_number = (total_picks % pack_size) + 1
        ranked = score_pack(
            pack_cards=pack_cards,
            pool_counts=pool_counts,
            pack_number=pack_number,
            pick_number=pick_number,
            skill_bucket=skill_bucket,
        )
        return ranked[0][0] if ranked else rng.choice(pack_cards)

    return _policy


__all__ = ["make_bc_policy"]
