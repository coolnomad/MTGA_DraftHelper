"""
Interactive pod draft UI with card art, backed by the run_pod_human logic.

Serves a small FastAPI app:
- POST /api/session -> create a draft session (num_pods fixed to 1, 1 human seat).
- GET /api/state?session_id=... -> current pack/pool for the human seat (with art).
- POST /api/pick -> submit a human pick, advances bots and returns next state.
- GET / -> static UI (served from scripts/pod_ui).

Card art is pulled from local MTGA assets:
- Card names/grpIds/artIds from Raw_CardDatabase_*.mtga
- Art bundles from MTGA_Data/Downloads/AssetBundle/<ArtId>_CardArt_*.mtga
- If UnityPy is available, textures are extracted to data URLs; otherwise art_uri is None.

Run:
PYTHONPATH=. .\\.venv\\Scripts\\uvicorn scripts.run_pod_human_ui:app --reload --port 8002
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from draft_env.env import DraftEnv
from hero_bot.hero_policy import hero_policy
from human_policy.random_policy import random_policy
from human_policy.bc_policy import make_bc_policy
from hero_bot.deck_builder import build_deck
from deck_eval.evaluator import evaluate_deck, evaluate_deck_bump
from hero_bot.pool_evaluator import evaluate_pool_effect, evaluate_pool_bump
from draft_env.pack_sampler import sample_pack
from scripts.pod_assets import CardAssetLoader, CardRecord

BASIC_LANDS = {"Island", "Swamp", "Forest", "Mountain", "Plains"}

def human_policy_bot(pack_cards: List[str], pool_counts: Dict[str, int], seat_idx: int, rng: np.random.Generator) -> str:
    bc = make_bc_policy()
    ranked = bc(pack_cards, pool_counts)
    return ranked[0][0] if ranked else pack_cards[0]


POLICY_MAP = {
    "hero": hero_policy,
    "human": human_policy_bot,
    "random": random_policy,
}


@dataclass
class PickLog:
    pod_id: int
    seat_idx: int
    policy_name: str
    pack_number: int
    pick_number: int
    fmt: str
    pack_card_ids: List[str]
    pool_before: Dict[str, int]
    chosen_card: str
    pool_after: Dict[str, int]
    hero_value_before: float | None = None
    hero_value_after: float | None = None
    hero_delta: float | None = None


class PodSession:
    def __init__(self, human_seat: int, bot_policies: List[str], set_code: str, seed: int, asset_loader: CardAssetLoader):
        self.pod_id = 0
        self.human_seat = human_seat
        self.bot_policies = bot_policies
        self.set_code = set_code
        self.rng = np.random.default_rng(seed)
        self.asset_loader = asset_loader

        self.env = DraftEnv(seed=seed, set_code=set_code)
        self.pools: List[Dict[str, int]] = [dict() for _ in range(self.env.num_seats)]
        self.human_pool: Dict[str, int] = {}  # total drafted cards
        self.human_main: Dict[str, int] = {}  # active deck during build screen
        self.human_sideboard: Dict[str, int] = {}  # remaining pool during build screen
        self.pick_logs: List[PickLog] = []
        self.deck_build_initialized = False

        bot_funcs = [POLICY_MAP.get(p, hero_policy) for p in bot_policies]
        self.policies: List[Optional] = []
        b_iter = iter(bot_funcs)
        for seat in range(self.env.num_seats):
            if seat == human_seat:
                self.policies.append(None)
            else:
                self.policies.append(next(b_iter))

        self.pack_number = 0
        self.pick_number = 0
        self.direction = 1
        self.current_packs: List[Optional[List[str]]] = []
        self.finished = False
        self._start_new_pack()
        self._advance_until_human()

    def _start_new_pack(self):
        self.pack_number += 1
        self.pick_number = 0
        self.direction = 1 if (self.pack_number % 2 == 1) else -1
        self.current_packs = [
            sample_pack(self.env.rng, self.env.pack_size, set_code=self.env.set_code) for _ in range(self.env.num_seats)
        ]

    def _pass_packs(self, packs: List[Optional[List[str]]]) -> List[Optional[List[str]]]:
        passed = [None] * self.env.num_seats
        for seat in range(self.env.num_seats):
            target = (seat + self.direction) % self.env.num_seats
            passed[target] = packs[seat]
        return passed

    def _do_pick_round(self, human_choice: Optional[str]) -> str:
        new_packs: List[Optional[List[str]]] = [None] * self.env.num_seats
        for seat in range(self.env.num_seats):
            pack_cards = self.current_packs[seat]
            if not pack_cards:
                continue
            pool_before = dict(self.pools[seat])
            if seat == self.human_seat:
                if human_choice is None:
                    # pause for human input
                    return "await_human"
                chosen = human_choice if human_choice in pack_cards else pack_cards[0]
            else:
                chosen = self.policies[seat](pack_cards, self.pools[seat], seat, self.rng)
                if chosen not in pack_cards:
                    chosen = self.rng.choice(pack_cards)
            if seat == self.human_seat:
                # track total pool plus default main during draft
                self.human_pool[chosen] = self.human_pool.get(chosen, 0) + 1
                self.human_main[chosen] = self.human_main.get(chosen, 0) + 1
                self.pools[seat][chosen] = self.pools[seat].get(chosen, 0) + 1
            else:
                self.pools[seat][chosen] = self.pools[seat].get(chosen, 0) + 1
            self.pick_logs.append(
                PickLog(
                    pod_id=self.pod_id,
                    seat_idx=seat,
                    policy_name="human" if seat == self.human_seat else self.bot_policies[seat if seat < self.human_seat else seat - 1],
                    pack_number=self.pack_number,
                    pick_number=self.pick_number + 1,
                    fmt=self.set_code,
                    pack_card_ids=list(pack_cards),
                    pool_before=pool_before if seat != self.human_seat else dict(self.human_main),
                    chosen_card=chosen,
                    pool_after=dict(self.pools[seat]) if seat != self.human_seat else dict(self.human_main),
                )
            )
            pack_cards.remove(chosen)
            new_packs[seat] = pack_cards

        self.current_packs = self._pass_packs(new_packs)
        self.pick_number += 1
        if self.pick_number >= self.env.pack_size:
            if self.pack_number >= self.env.packs_per_player:
                self.finished = True
            else:
                self._start_new_pack()
        return "continue"

    def _advance_until_human(self):
        if self.finished:
            return
        while True:
            status = self._do_pick_round(human_choice=None)
            if status == "await_human" or self.finished:
                return

    def apply_human_pick(self, card_name: str):
        if self.finished:
            return
        status = self._do_pick_round(human_choice=card_name)
        if status == "continue" and not self.finished:
            self._advance_until_human()

    def build_results(self) -> List[Dict]:
        results = []
        for seat_idx, pool in enumerate(self.pools):
            deck_source = pool if seat_idx != self.human_seat else self.human_main
            deck = build_deck(deck_source)
            deck_effect = evaluate_deck(deck)
            deck_bump = evaluate_deck_bump(deck)
            results.append(
                {
                    "pod_id": self.pod_id,
                    "seat_idx": seat_idx,
                    "policy": "human" if seat_idx == self.human_seat else self.bot_policies[seat_idx if seat_idx < self.human_seat else seat_idx - 1],
                    "pool": deck_source,
                    "deck": deck,
                    "deck_effect": deck_effect,
                    "deck_bump": deck_bump,
                }
            )
        return results

    def current_state(self) -> Dict:
        if self.finished:
            self._init_deck_build()
            return {
                "status": "finished",
                "pack_number": self.pack_number,
                "pick_number": self.pick_number,
                "pool_main": self._serialize_pool_main(),
                "pool_sideboard": self._serialize_pool_sideboard(),
                "basic_lands": self._serialize_basics(),
                "deck_effect": self._deck_scores()[0],
                "deck_bump": self._deck_scores()[1],
                "deck_count": sum(self.human_main.values()),
                "results": self.build_results(),
            }
        pack = self.current_packs[self.human_seat] or []
        base_pool = dict(self.human_main)
        return {
            "status": "awaiting_pick",
            "pack_number": self.pack_number,
            "pick_number": self.pick_number + 1,
            "pack": [self._serialize_card(name, base_pool) for name in pack],
            "pool_main": self._serialize_pool_main(),
            "pool_sideboard": self._serialize_pool_sideboard(),
            "deck_effect": self._deck_scores()[0],
            "deck_bump": self._deck_scores()[1],
            "deck_count": sum(self.human_main.values()),
        }

    def _serialize_card(self, name: str, base_pool: Dict[str, int]) -> Dict:
        card: Optional[CardRecord] = self.asset_loader.find_by_name(name) if self.asset_loader else None
        art_uri = self.asset_loader.art_uri_for_name(name) if card else None
        scry_art = self.asset_loader.scryfall_image_url(name, version="art_crop") if self.asset_loader else None
        scry_card = self.asset_loader.scryfall_image_url(name, version="png") if self.asset_loader else None
        # Projected deck scores if this card is taken
        pool_plus = dict(base_pool)
        pool_plus[name] = pool_plus.get(name, 0) + 1
        try:
            proj_effect, proj_effect_delta = evaluate_pool_effect(pool_plus)
            proj_bump, proj_bump_delta = evaluate_pool_bump(pool_plus)
        except Exception:
            proj_effect = None
            proj_bump = None
            proj_effect_delta = None
            proj_bump_delta = None
        return {
            "id": name,
            "name": name,
            "grp_id": card.grp_id if card else None,
            "art_uri": art_uri,
            "image_url": art_uri or scry_art,
            "card_image_url": scry_card,
            "projected_deck_effect": proj_effect,
            "projected_deck_bump": proj_bump,
            "projected_deck_effect_delta": proj_effect_delta,
            "projected_deck_bump_delta": proj_bump_delta,
        }

    def _serialize_pool(self, seat_idx: int) -> List[Dict]:
        items = []
        for name, cnt in sorted(self.pools[seat_idx].items()):
            card = self.asset_loader.find_by_name(name) if self.asset_loader else None
            art_uri = self.asset_loader.art_uri_for_name(name) if card else None
            scry_art = self.asset_loader.scryfall_image_url(name, version="art_crop") if self.asset_loader else None
            scry_card = self.asset_loader.scryfall_image_url(name, version="png") if self.asset_loader else None
            items.append(
                {
                    "id": name,
                    "name": name,
                    "count": cnt,
                    "grp_id": card.grp_id if card else None,
                    "art_uri": art_uri,
                    "image_url": art_uri or scry_art,
                    "card_image_url": scry_card,
                }
            )
        return items

    def _serialize_pool_main(self) -> List[Dict]:
        items = []
        for name, cnt in sorted(self.human_main.items()):
            card = self.asset_loader.find_by_name(name) if self.asset_loader else None
            art_uri = self.asset_loader.art_uri_for_name(name) if card else None
            scry_art = self.asset_loader.scryfall_image_url(name, version="art_crop") if self.asset_loader else None
            scry_card = self.asset_loader.scryfall_image_url(name, version="png") if self.asset_loader else None
            items.append(
                {
                    "id": name,
                    "name": name,
                    "count": cnt,
                    "grp_id": card.grp_id if card else None,
                    "art_uri": art_uri,
                    "image_url": art_uri or scry_art,
                    "card_image_url": scry_card,
                }
            )
        return items

    def _serialize_pool_sideboard(self) -> List[Dict]:
        items = []
        for name, cnt in sorted(self.human_sideboard.items()):
            card = self.asset_loader.find_by_name(name) if self.asset_loader else None
            art_uri = self.asset_loader.art_uri_for_name(name) if card else None
            scry_art = self.asset_loader.scryfall_image_url(name, version="art_crop") if self.asset_loader else None
            scry_card = self.asset_loader.scryfall_image_url(name, version="png") if self.asset_loader else None
            items.append(
                {
                    "id": name,
                    "name": name,
                    "count": cnt,
                    "grp_id": card.grp_id if card else None,
                    "art_uri": art_uri,
                    "image_url": art_uri or scry_art,
                    "card_image_url": scry_card,
                }
            )
        return items

    def _serialize_basics(self) -> List[Dict]:
        items = []
        for name in sorted(BASIC_LANDS):
            card = self.asset_loader.find_by_name(name) if self.asset_loader else None
            art_uri = self.asset_loader.art_uri_for_name(name) if card else None
            scry_art = self.asset_loader.scryfall_image_url(name, version="art_crop") if self.asset_loader else None
            scry_card = self.asset_loader.scryfall_image_url(name, version="png") if self.asset_loader else None
            items.append(
                {
                    "id": name,
                    "name": name,
                    "count": self.human_main.get(name, 0),
                    "grp_id": card.grp_id if card else None,
                    "art_uri": art_uri,
                    "image_url": art_uri or scry_art,
                    "card_image_url": scry_card,
                }
            )
        return items

    def _init_deck_build(self):
        """After draft ends, move full pool into sideboard and clear main deck."""
        if self.deck_build_initialized:
            return
        self.human_sideboard = dict(self.human_pool)
        self.human_main = {}
        self.deck_build_initialized = True

    def _deck_scores(self) -> Tuple[float, float]:
        try:
            if not self.human_main:
                return 0.0, 0.0
            return float(evaluate_deck(self.human_main)), float(evaluate_deck_bump(self.human_main))
        except Exception:
            return 0.0, 0.0

    def move_card(self, card_name: str, to: str):
        # Allow unlimited basics to be added to main deck
        is_basic = card_name in BASIC_LANDS
        if to == "sideboard":
            if self.human_main.get(card_name, 0) > 0:
                self.human_main[card_name] -= 1
                if self.human_main[card_name] <= 0:
                    self.human_main.pop(card_name, None)
                if not is_basic:
                    self.human_sideboard[card_name] = self.human_sideboard.get(card_name, 0) + 1
        elif to == "main":
            if is_basic:
                self.human_main[card_name] = self.human_main.get(card_name, 0) + 1
            elif self.human_sideboard.get(card_name, 0) > 0:
                self.human_sideboard[card_name] -= 1
                if self.human_sideboard[card_name] <= 0:
                    self.human_sideboard.pop(card_name, None)
                self.human_main[card_name] = self.human_main.get(card_name, 0) + 1


class SessionRequest(BaseModel):
    human_seat: int = 0
    bot_policies: List[str]
    format: str = "FIN"
    seed: int = 1337


class PickRequest(BaseModel):
    session_id: str
    card_id: str


app = FastAPI()
asset_loader = CardAssetLoader()
sessions: Dict[str, PodSession] = {}

STATIC_DIR = REPO_ROOT / "scripts" / "pod_ui"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>UI assets not found</h1>", status_code=500)
    return index_path.read_text(encoding="utf-8")


@app.post("/api/session")
def create_session(req: SessionRequest):
    if len(req.bot_policies) != 7:
        raise HTTPException(status_code=400, detail="bot_policies must have 7 entries")
    if not (0 <= req.human_seat <= 7):
        raise HTTPException(status_code=400, detail="human_seat must be between 0 and 7")
    session_id = str(uuid.uuid4())
    session = PodSession(
        human_seat=req.human_seat,
        bot_policies=req.bot_policies,
        set_code=req.format,
        seed=req.seed,
        asset_loader=asset_loader,
    )
    sessions[session_id] = session
    return {"session_id": session_id, "state": session.current_state()}


@app.get("/api/state")
def get_state(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return session.current_state()


@app.post("/api/pick")
def post_pick(req: PickRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    if session.finished:
        return session.current_state()
    session.apply_human_pick(req.card_id)
    return session.current_state()


@app.post("/api/move")
def move_card(session_id: str, card_id: str, to: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    if to not in ("main", "sideboard"):
        raise HTTPException(status_code=400, detail="to must be 'main' or 'sideboard'")
    session.move_card(card_id, to)
    return session.current_state()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.run_pod_human_ui:app", host="0.0.0.0", port=8002, reload=False)
