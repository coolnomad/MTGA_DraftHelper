from __future__ import annotations

from typing import Dict, List
from deck_eval.evaluator import evaluate_deck

BASICS = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]


def build_deck(pool_counts: Dict[str, int], target_size: int = 40) -> Dict[str, int]:
    """
    Simple heuristic deck builder:
    - Start with all non-basics sorted by evaluator contribution.
    - Fill remaining slots with basics (cycling colors) to reach target_size.
    """
    pool = {k: int(v) for k, v in pool_counts.items() if v}
    non_basics = {k: v for k, v in pool.items() if k not in BASICS}
    basics = {k: v for k, v in pool.items() if k in BASICS}

    # score non-basics individually
    scores = []
    for card, cnt in non_basics.items():
        # score each copy independently
        for _ in range(cnt):
            scores.append((card, evaluate_deck({card: 1})))
    # take best non-basics first
    scores.sort(key=lambda kv: kv[1], reverse=True)
    deck: Dict[str, int] = {}
    for card, _ in scores:
        if sum(deck.values()) >= target_size:
            break
        deck[card] = deck.get(card, 0) + 1

    # fill with basics, prefer those already in pool
    idx = 0
    while sum(deck.values()) < target_size:
        basic = BASICS[idx % len(BASICS)]
        deck[basic] = deck.get(basic, 0) + 1
        idx += 1

    # clamp to available basics + allow adding more if needed
    for b, cnt in basics.items():
        deck[b] = min(deck.get(b, 0), cnt + 10)  # allow adding extra basics beyond pool

    return deck


__all__ = ["build_deck", "BASICS"]
