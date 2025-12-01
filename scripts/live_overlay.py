"""
Minimal live draft bridge + overlay.

Listens to MTGA Player.log using the brian_staple ArenaScanner and serves a tiny
overlay page that polls for the latest recommendations. This is intended for a
prototype: if the current set is unsupported, cards will show as raw grpIds and
scores fall back to evaluator/neutral ordering.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List
import gzip
import json
import os
import sqlite3

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import numpy as np
import os

REPO_ROOT = Path(__file__).resolve().parents[1]

# brian_staple overlay log scanner
BRIAN_SRC = REPO_ROOT / "docs" / "other_tools" / "brian_staple" / "src"
import sys

sys.path.insert(0, str(REPO_ROOT))  # allow imports from our repo (deck_eval, hero_bot, etc.)
sys.path.insert(0, str(BRIAN_SRC))  # allow src.constants relative imports from brian_staple bundle
sys.path.insert(0, str(BRIAN_SRC.parent))  # so "src" resolves to brian_staple/src, not our project src
try:
    import src.constants as bs_const
    from src.log_scanner import ArenaScanner
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Failed to import ArenaScanner: {e}")

# MTGA logs sometimes emit "BotDraftDraftStatus" (without underscore) for quick drafts; add it to the scanner start strings.
BOT_DRAFT_COMPACT = "[UnityCrossThreadLogger]==> BotDraftDraftStatus "
if hasattr(bs_const, "DRAFT_START_STRINGS") and BOT_DRAFT_COMPACT not in bs_const.DRAFT_START_STRINGS:
    bs_const.DRAFT_START_STRINGS.append(BOT_DRAFT_COMPACT)

# our models
import re

from deck_eval.evaluator import evaluate_deck, evaluate_deck_bump, DeckEvaluator
from hero_bot.hero_policy import _get_state_value_model
from state_encoding.encoder import encode_state


class DummySetList:
    """Minimal stub to satisfy ArenaScanner."""

    def __init__(self):
        self.data: dict = {}


app = FastAPI()


state: Dict[str, object] = {
    "status": "idle",
    "message": "waiting for draft",
    "log_path": None,
    "data_cards_path": None,
    "card_db_path": None,
    "cards_csv_path": None,
    "pack_number": None,
    "pick_number": None,
    "pack_cards": [],
    "pack_cards_named": [],
    "recommendations": [],
    "updated_at": None,
}

GRP_TO_NAME: Dict[str, str] = {}
NAME_TO_SLUG: Dict[str, str] = {}
SLUGS_WITH_INDEX: set = set()
_DECK_EVAL = None


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug


def _find_log_path() -> Path:
    """
    Resolve Player.log by checking typical Arena locations across available drives.
    Returns the first existing path or the default candidate even if missing.
    """
    # Explicit override via env var for machines with non-standard locations.
    env_path = os.environ.get("MTGA_PLAYER_LOG")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
    # Default search.
    default_rel = Path(bs_const.LOG_LOCATION_WINDOWS)
    candidates = []
    # If constant is absolute, try it first.
    if default_rel.is_absolute():
        candidates.append(default_rel)
    else:
        # Prefer current home drive/anchor first.
        home_drive = Path.home().anchor or Path.home().drive
        if home_drive:
            candidates.append(Path(home_drive) / default_rel)
        # Then try all known Windows drives from brian_staple constants.
        for drive in getattr(bs_const, "WINDOWS_DRIVES", []):
            candidates.append(Path(drive) / default_rel)
    # Fallback to common LocalLow path under the current user.
    candidates.append(Path.home() / "AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return last candidate even if none exist so we surface what we tried.
    return candidates[-1]


def _find_data_cards() -> Path | None:
    """Locate MTGA data_cards file on common install paths; newest wins."""
    override = os.environ.get("MTGA_DATA_CARDS")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    drives = getattr(bs_const, "WINDOWS_DRIVES", ["C:/", "D:/", "E:/", "F:/"])
    subpaths = [
        Path("Program Files") / "Wizards of the Coast" / "MTGA" / "MTGA_Data" / "Downloads" / "Data",
        Path("Program Files (x86)") / "Wizards of the Coast" / "MTGA" / "MTGA_Data" / "Downloads" / "Data",
    ]
    for drive in drives:
        for sub in subpaths:
            base = Path(drive) / sub
            if base.exists():
                candidates.extend(base.glob("data_cards*.mtga"))
    # pick newest existing
    existing = [c for c in candidates if c.exists()]
    if not existing:
        return None
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[0]


def _find_card_db() -> Path | None:
    """Locate Raw_CardDatabase_*.mtga (SQLite) on common install paths."""
    override = os.environ.get("MTGA_CARD_DB")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    drives = getattr(bs_const, "WINDOWS_DRIVES", ["C:/", "D:/", "E:/", "F:/"])
    subpaths = [
        Path("Program Files") / "Wizards of the Coast" / "MTGA" / "MTGA_Data" / "Downloads" / "Raw",
        Path("Program Files (x86)") / "Wizards of the Coast" / "MTGA" / "MTGA_Data" / "Downloads" / "Raw",
    ]
    for drive in drives:
        for sub in subpaths:
            base = Path(drive) / sub
            if base.exists():
                candidates.extend(base.glob("Raw_CardDatabase_*.mtga"))
    existing = [c for c in candidates if c.exists()]
    if not existing:
        return None
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[0]


def _load_cards_csv() -> Dict[str, str]:
    """Parse local 17Lands cards.csv (id->name)."""
    csv_path = REPO_ROOT / "data" / "cards.csv"
    if not csv_path.exists():
        return {}
    try:
        import csv

        mapping: Dict[str, str] = {}
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gid = row.get("id")
                name = row.get("name")
                if gid and name:
                    mapping[str(gid)] = name
        if mapping:
            state["cards_csv_path"] = str(csv_path)
        return mapping
    except Exception:
        return {}


def _load_grp_to_name() -> Dict[str, str]:
    """Parse data_cards*.mtga (gzip JSON), Raw_CardDatabase_*.mtga (SQLite), then fallback to cards.csv for grpId->name."""
    path = _find_data_cards()
    mapping: Dict[str, str] = {}
    if path:
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            cards = data.get("cards", [])
            for card in cards:
                grp = card.get("grpid") or card.get("grpId") or card.get("id")
                name = card.get("TitleID") or card.get("titleId") or card.get("name")
                # If titleId is numeric, try English localization when present.
                if isinstance(name, int):
                    locs = card.get("localizedname") or card.get("localizedNames") or card.get("Localization")
                    if isinstance(locs, dict):
                        name = locs.get("en") or locs.get("en_US") or name
                if grp and name:
                    mapping[str(grp)] = str(name)
            if mapping:
                state["data_cards_path"] = str(path)
        except Exception:
            mapping = {}
    # Try Raw_CardDatabase SQLite if mapping is still sparse.
    if not mapping or len(mapping) < 100:
        db_path = _find_card_db()
        if db_path and db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute(
                    "SELECT c.GrpId, l.Loc FROM Cards c JOIN Localizations_enUS l ON c.TitleId = l.LocId"
                )
                for gid, name in cur.fetchall():
                    if gid and name and str(gid) not in mapping:
                        mapping[str(gid)] = str(name)
                conn.close()
                state["card_db_path"] = str(db_path)
            except Exception:
                pass
    # Supplement with 17Lands cards.csv if needed.
    if not mapping or len(mapping) < 100:  # fallback or augment
        csv_map = _load_cards_csv()
        mapping.update({k: v for k, v in csv_map.items() if k not in mapping})
    try:
        return mapping
    except Exception:
        return {}


def _build_slug_maps():
    """Populate NAME_TO_SLUG and SLUGS_WITH_INDEX from current GRP_TO_NAME and deck evaluator."""
    global NAME_TO_SLUG, SLUGS_WITH_INDEX, _DECK_EVAL
    NAME_TO_SLUG = {}
    for gid, name in GRP_TO_NAME.items():
        if not name:
            continue
        slug = _slugify_name(name)
        NAME_TO_SLUG[name] = slug
    try:
        _DECK_EVAL = DeckEvaluator()
        SLUGS_WITH_INDEX = set(_DECK_EVAL.name_to_idx.keys())
    except Exception:
        SLUGS_WITH_INDEX = set()


def _score_pack(pack_cards: List[str], pool_counts: Dict[str, int], pack_no: int, pick_no: int) -> List[Dict[str, object]]:
    """Score pack cards with state-value model when available; fallback to evaluator."""
    model = _get_state_value_model()
    # Build slugged pool for deck bump if we have a deck evaluator index.
    slug_pool: Dict[str, int] = {}
    for gid, cnt in pool_counts.items():
        name = GRP_TO_NAME.get(gid)
        if not name:
            continue
        slug = NAME_TO_SLUG.get(name)
        if slug and slug in SLUGS_WITH_INDEX:
            slug_pool[slug] = slug_pool.get(slug, 0) + cnt

    scored: List[Dict[str, object]] = []
    for card in pack_cards:
        new_pool = dict(pool_counts)
        new_pool[card] = new_pool.get(card, 0) + 1
        # Bump calculation uses slugged names; skip if not known.
        bump_score = None
        name = GRP_TO_NAME.get(card)
        slug = NAME_TO_SLUG.get(name) if name else None
        if slug and slug in SLUGS_WITH_INDEX:
            bump_pool = dict(slug_pool)
            bump_pool[slug] = bump_pool.get(slug, 0) + 1
            try:
                bump_score = float(evaluate_deck_bump(bump_pool))
            except Exception:
                bump_score = None
        if model is not None:
            try:
                vec = encode_state(new_pool, pack_no=pack_no, pick_no=pick_no)
                score = float(model.predict(vec.reshape(1, -1))[0])
            except Exception:
                score = 0.0
        else:
            try:
                score = float(evaluate_deck(new_pool))
            except Exception:
                score = 0.0
        scored.append({"card": card, "score": score, "bump": bump_score})
    # if all scores identical, keep pack order
    scores = [s["score"] for s in scored]
    if len(set(scores)) > 1:
        scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _scanner_loop():
    # Resolve log path; search common drives to reduce "idle" failures when Arena is on a different drive.
    log_path = _find_log_path()
    global state
    state["log_path"] = str(log_path)
    if not log_path.exists():
        state["message"] = f"waiting for draft (log not found: {log_path})"

    # Load grpId->name mapping once.
    global GRP_TO_NAME
    GRP_TO_NAME = _load_grp_to_name()
    _build_slug_maps()
    if not GRP_TO_NAME:
        extra = "card names unavailable (no data_cards/cards.csv mapping)"
        state["message"] = f"{state.get('message') or 'waiting for draft'} | {extra}"

    scanner = ArenaScanner(str(log_path), DummySetList(), sets_location=bs_const.SETS_FOLDER, step_through=False)
    while True:
        try:
            if scanner.draft_start_search():
                state = {
                    "status": "draft_detected",
                    "message": f"Draft detected: {scanner.event_string}",
                    "log_path": str(log_path),
                    "cards_csv_path": state.get("cards_csv_path"),
                    "pack_number": None,
                    "pick_number": None,
                    "pack_cards": [],
                    "recommendations": [],
                    "updated_at": time.time(),
                }
            if scanner.draft_type != bs_const.LIMITED_TYPE_UNKNOWN:
                updated = scanner.draft_data_search()
                if updated:
                    pack_no, pick_no = scanner.retrieve_current_pack_and_pick()
                    # pack_cards are grpIds (strings)
                    idx = max(pick_no - 1, 0) % 8
                    pack_cards = scanner.pack_cards[idx] if idx < len(scanner.pack_cards) else []
                    pack_cards_named = [GRP_TO_NAME.get(c, c) for c in pack_cards]
                    pool_counts = Counter(scanner.taken_cards)
                    recs = _score_pack(pack_cards, dict(pool_counts), pack_no, pick_no)
                    for r in recs:
                        r["name"] = GRP_TO_NAME.get(r["card"], r["card"])
                    state = {
                        "status": "active",
                        "message": "updated",
                        "log_path": str(log_path),
                        "data_cards_path": state.get("data_cards_path"),
                        "cards_csv_path": state.get("cards_csv_path"),
                        "pack_number": pack_no,
                        "pick_number": pick_no,
                        "pack_cards": pack_cards,
                        "pack_cards_named": pack_cards_named,
                        "recommendations": recs,
                        "updated_at": time.time(),
                    }
            time.sleep(0.5)
        except Exception as e:  # pragma: no cover
            state = {
                "status": "error",
                "message": f"{e} (log path: {log_path})",
                "log_path": str(log_path),
                "pack_number": None,
                "pick_number": None,
                "pack_cards": [],
                "recommendations": [],
                "updated_at": time.time(),
            }
            time.sleep(1.0)


@app.on_event("startup")
def _start_scanner():
    t = threading.Thread(target=_scanner_loop, daemon=True)
    t.start()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/overlay")


@app.get("/overlay", response_class=HTMLResponse)
def overlay_page():
    return """
<!doctype html>
<html><head><meta charset="utf-8" />
<title>MTGA Draft Overlay</title>
<style>
  body { font-family: Arial, sans-serif; background: #111; color: #eee; padding: 12px; }
  .card { margin: 4px 0; }
  .score { color: #9cf; }
  .small { font-size: 12px; color: #aaa; }
  .id { color: #666; font-size: 11px; }
</style>
</head>
<body>
  <div id="status">Waiting for draft...</div>
  <div id="pack"></div>
  <div id="recs"></div>
  <div id="path" class="small"></div>
<script>
async function poll() {
  try {
    const res = await fetch('/state');
    const data = await res.json();
    document.getElementById('status').innerText = `${data.status} ${data.message || ''} | P${data.pack_number||'-'}P${data.pick_number||'-'}`;
    const names = data.pack_cards_named && data.pack_cards_named.length ? data.pack_cards_named : data.pack_cards;
    const ids = data.pack_cards || [];
    const packLines = names.map((n, i) => `${n}${ids[i] && ids[i] !== n ? ' ('+ids[i]+')' : ''}`);
    document.getElementById('pack').innerText = 'Pack: ' + packLines.join(', ');
    const recs = data.recommendations || [];
    document.getElementById('recs').innerHTML = recs.map(r => {
      const name = r.name || r.card;
      const id = r.card;
      const idText = id && id !== name ? ` <span class="id">${id}</span>` : '';
      const bump = (r.bump !== null && r.bump !== undefined) ? ` <span class="score">bump ${r.bump.toFixed(3)}</span>` : '';
      return `<div class="card">${name}${idText} <span class="score">${r.score.toFixed(3)}</span>${bump}</div>`;
    }).join('');
    const paths = [];
    if (data.log_path) paths.push(`Log: ${data.log_path}`);
    if (data.data_cards_path) paths.push(`data_cards: ${data.data_cards_path}`);
    if (data.card_db_path) paths.push(`card_db: ${data.card_db_path}`);
    if (data.cards_csv_path) paths.push(`cards.csv: ${data.cards_csv_path}`);
    document.getElementById('path').innerText = paths.join(' | ');
  } catch (e) {
    document.getElementById('status').innerText = 'Error fetching state';
  }
  setTimeout(poll, 1000);
}
poll();
</script>
</body></html>
"""


@app.get("/state")
def get_state():
    return JSONResponse(state)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log", default=None, help="Override path to Player.log")
    args = parser.parse_args()

    if args.log:
        os.environ["MTGA_PLAYER_LOG"] = args.log

    uvicorn.run(app, host=args.host, port=args.port)
