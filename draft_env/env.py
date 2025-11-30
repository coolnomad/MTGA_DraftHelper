from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional
import numpy as np

from draft_env.pack_sampler import sample_pack

# Policy signature: policy(pack_cards: List[str], pool_counts: Dict[str,int], seat_idx: int, rnd: np.random.Generator) -> str


class DraftEnv:
    def __init__(
        self,
        num_seats: int = 8,
        packs_per_player: int = 3,
        pack_size: int = 15,
        seed: int = 1337,
        set_code: str = "FIN",
    ):
        self.num_seats = num_seats
        self.packs_per_player = packs_per_player
        self.pack_size = pack_size
        self.set_code = set_code
        self.rng = np.random.default_rng(seed)
        self.pools: List[Dict[str, int]] = [defaultdict(int) for _ in range(num_seats)]

    def reset(self):
        self.pools = [defaultdict(int) for _ in range(self.num_seats)]

    def run_draft(self, policies: List[Callable], seed: Optional[int] = None, log_picks: bool = False):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset()
        packs = [
            [sample_pack(self.rng, self.pack_size, set_code=self.set_code) for _ in range(self.num_seats)]
            for _ in range(self.packs_per_player)
        ]

        pick_logs = []
        # pack 1 -> pass left, pack 2 -> pass right, pack 3 -> pass left
        for p_idx in range(self.packs_per_player):
            direction = 1 if p_idx % 2 == 0 else -1
            current_packs = packs[p_idx]
            for pick in range(self.pack_size):
                # each seat picks one card
                new_packs = [None] * self.num_seats
                for seat in range(self.num_seats):
                    pack_cards = current_packs[seat]
                    if not pack_cards:
                        continue
                    choice = policies[seat](pack_cards, self.pools[seat], seat, self.rng)
                    # if invalid choice, fallback to random
                    if choice not in pack_cards:
                        choice = self.rng.choice(pack_cards)
                    if log_picks:
                        pick_logs.append(
                            {
                                "seat": seat,
                                "pack_number": p_idx + 1,
                                "pick_number": pick + 1,
                                "pack_cards": list(pack_cards),
                                "pool_counts": dict(self.pools[seat]),
                                "chosen_card": choice,
                            }
                        )
                    self.pools[seat][choice] += 1
                    pack_cards.remove(choice)
                    new_packs[seat] = pack_cards
                # pass packs
                passed = [None] * self.num_seats
                for seat in range(self.num_seats):
                    target = (seat + direction) % self.num_seats
                    passed[target] = new_packs[seat]
                current_packs = passed
        return self.pools, pick_logs


__all__ = ["DraftEnv"]
