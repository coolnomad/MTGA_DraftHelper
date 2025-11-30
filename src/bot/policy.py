from dataclasses import dataclass
from typing import Dict, List

from .env import DraftState


@dataclass
class BotConfig:
    greediness: float = 1.0
    curve_weight: float = 0.0
    color_weight: float = 0.0
    randomness: float = 0.0


class DraftBot:
    """
    draft bot stub that scores cards and picks the best one.

    currently uses a placeholder scoring function; later you'll
    call your residual model + feature builders.
    """

    def __init__(self, config: BotConfig | None = None):
        self.config = config or BotConfig()

    def score_cards(self, state: DraftState) -> Dict[str, float]:
        """
        return a dict of card_id -> score for the cards in the current pack.

        stub: score = current pool count (prefers cards you already have).
        """
        scores: Dict[str, float] = {}
        for c in state.pack_cards:
            scores[c] = state.pool_counts.get(c, 0)
        return scores

    def choose_card(self, state: DraftState) -> str:
        scores = self.score_cards(state)
        if not scores:
            raise ValueError("no cards in pack to choose from")
        # deterministic argmax for now
        return max(scores.items(), key=lambda kv: kv[1])[0]
