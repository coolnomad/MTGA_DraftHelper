from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DraftState:
    pack_cards: List[str]
    pool_counts: Dict[str, int]
    pick_number: int
    pack_number: int
    history: List[str] = field(default_factory=list)


class DraftEnv:
    """
    simple environment stub that replays a single draft's packs.

    later you'll:
      - initialize with real 17lands pack history
      - implement reset() and step(card_id)
    """

    def __init__(self, draft_record: Any):
        self._draft_record = draft_record
        self._state: Optional[DraftState] = None

    def reset(self) -> DraftState:
        # TODO: initialize from draft_record
        self._state = DraftState(
            pack_cards=[],
            pool_counts={},
            pick_number=0,
            pack_number=0,
            history=[],
        )
        return self._state

    def step(self, pick_card_id: str) -> DraftState:
        """
        take a card id as action, update pool + history, and advance to next pack/pick.
        """
        if self._state is None:
            raise RuntimeError("call reset() before step()")

        pool = dict(self._state.pool_counts)
        pool[pick_card_id] = pool.get(pick_card_id, 0) + 1

        self._state.history.append(pick_card_id)
        self._state.pool_counts = pool
        self._state.pick_number += 1

        # TODO: update self._state.pack_cards, pack_number from draft_record

        return self._state
