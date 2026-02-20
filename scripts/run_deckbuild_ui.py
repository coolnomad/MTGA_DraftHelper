from __future__ import annotations

import csv
import io
import json
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.pod_assets import CardAssetLoader, CardRecord

BASIC_LANDS = {"Island", "Swamp", "Forest", "Mountain", "Plains"}


class SessionRequest(BaseModel):
    base_p_user: float = 0.55


class LoadPoolRequest(BaseModel):
    session_id: str
    list_text: Optional[str] = None
    csv_text: Optional[str] = None
    locked_list_text: Optional[str] = None
    locked_csv_text: Optional[str] = None
    wobble_list_text: Optional[str] = None
    wobble_csv_text: Optional[str] = None


class MoveRequest(BaseModel):
    session_id: str
    card_id: str
    from_zone: Optional[str] = None
    to_zone: Optional[str] = None
    from_alias: Optional[str] = Field(default=None, alias="from")
    to_alias: Optional[str] = Field(default=None, alias="to")


class BasePRequest(BaseModel):
    session_id: str
    base_p_user: float


class EvaluateRequest(BaseModel):
    session_id: str
    base_p_values: Optional[List[float]] = None


@dataclass
class DeckBuildSession:
    session_id: str
    base_p_user: float
    pool_counts: Dict[str, int]
    locked_counts: Dict[str, int]
    wobble_counts: Dict[str, int]


class DeckBumpScorer:
    def __init__(self) -> None:
        self.model_path = self._resolve_model_path()
        self.feature_cols = self._load_feature_cols(self.model_path)
        if "base_p" not in self.feature_cols:
            raise ValueError("Feature schema must include base_p")
        self.col_to_idx = {c: i for i, c in enumerate(self.feature_cols)}
        self.booster = xgb.Booster()
        self.booster.load_model(self.model_path)

    def _resolve_model_path(self) -> Path:
        candidates = [
            REPO_ROOT / "model" / "deck_bump_model.ubj",
            REPO_ROOT / "models" / "deck_bump_model.ubj",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError("deck_bump_model.ubj not found under model/ or models/")

    def _load_feature_cols(self, model_path: Path) -> List[str]:
        model_dir = model_path.parent
        json_path = model_dir / "deck_cols.json"
        if json_path.exists():
            obj = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                cols = [str(x) for x in obj]
                return cols if "base_p" in cols else ["base_p"] + cols
            if isinstance(obj, dict) and isinstance(obj.get("deck_cols"), list):
                cols = [str(x) for x in obj["deck_cols"]]
                return cols if "base_p" in cols else ["base_p"] + cols

        csv_candidates = [model_dir / "deck_cols.csv", model_dir / "deck_bump_feature_schema.csv"]
        for csv_path in csv_candidates:
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    rows = [r for r in reader if r]
                if not rows:
                    continue
                header = rows[0][0].strip().lower()
                if header in {"deck_cols", "feature", "feature_name", "name"}:
                    cols = [r[0].strip() for r in rows[1:] if r and r[0].strip()]
                    return cols if "base_p" in cols else ["base_p"] + cols
                cols = [r[0].strip() for r in rows if r and r[0].strip()]
                return cols if "base_p" in cols else ["base_p"] + cols

        # last resort: model embedded names
        probe = xgb.Booster()
        probe.load_model(model_path)
        names = probe.feature_names or []
        if names:
            return list(names)
        raise FileNotFoundError("Could not load feature schema (deck_cols.json/csv)")

    def vectorize_deck(self, locked_counts: Dict[str, int], base_p: float) -> np.ndarray:
        x = np.zeros(len(self.feature_cols), dtype=float)
        x[self.col_to_idx["base_p"]] = float(base_p)
        for name, cnt in locked_counts.items():
            if cnt <= 0:
                continue
            key = f"deck_{card_to_feature_token(name)}"
            idx = self.col_to_idx.get(key)
            if idx is not None:
                x[idx] += float(cnt)
        return x

    def predict_many(self, locked_counts: Dict[str, int], base_p_values: List[float]) -> List[Dict[str, float]]:
        if not base_p_values:
            return []
        X = np.vstack([self.vectorize_deck(locked_counts, bp) for bp in base_p_values])
        preds = self.booster.predict(xgb.DMatrix(X, feature_names=self.feature_cols))
        out = []
        for bp, p in zip(base_p_values, preds):
            out.append({"base_p": float(bp), "deck_bump": float(p)})
        return out


def clamp_count(v: int) -> int:
    return max(0, int(v))


def card_to_feature_token(name: str) -> str:
    n = (name or "").strip().lower()
    n = n.replace("\u2019", "'")
    n = n.replace("\u2018", "'")
    n = n.replace("\u2013", "-")
    n = n.replace("\u2014", "-")
    n = re.sub(r"\s+", "_", n)
    return n


def parse_pasted_list(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not text:
        return counts
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s+(.+)$", line)
        if m:
            cnt = int(m.group(1))
            name = m.group(2).strip()
        else:
            cnt = 1
            name = line
        if name:
            counts[name] = counts.get(name, 0) + clamp_count(cnt)
    return counts


def parse_csv_text(csv_text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not csv_text or not csv_text.strip():
        return counts
    buf = io.StringIO(csv_text.lstrip("\ufeff"))
    reader = csv.reader(buf)
    rows = [r for r in reader if any((c or "").strip() for c in r)]
    if not rows:
        return counts

    header = [c.strip().lower() for c in rows[0]]
    has_header = ("name" in header) or ("card_name" in header)

    if has_header:
        name_idx = header.index("name") if "name" in header else header.index("card_name")
        count_idx = header.index("count") if "count" in header else None
        data_rows = rows[1:]
    else:
        name_idx = 0
        count_idx = 1 if len(rows[0]) > 1 and rows[0][1].strip().isdigit() else None
        data_rows = rows

    for row in data_rows:
        if name_idx >= len(row):
            continue
        name = row[name_idx].strip()
        if not name:
            continue
        cnt = 1
        if count_idx is not None and count_idx < len(row):
            raw_cnt = row[count_idx].strip()
            if raw_cnt:
                try:
                    cnt = int(float(raw_cnt))
                except ValueError:
                    cnt = 1
        counts[name] = counts.get(name, 0) + clamp_count(cnt)
    return counts


class DeckBuildService:
    def __init__(self, asset_loader: CardAssetLoader, scorer: DeckBumpScorer):
        self.asset_loader = asset_loader
        self.scorer = scorer
        self.sessions: Dict[str, DeckBuildSession] = {}

    def create_session(self, base_p_user: float) -> DeckBuildSession:
        sid = str(uuid.uuid4())
        ses = DeckBuildSession(
            session_id=sid,
            base_p_user=float(base_p_user),
            pool_counts={},
            locked_counts={},
            wobble_counts={},
        )
        self.sessions[sid] = ses
        return ses

    def get_session(self, session_id: str) -> DeckBuildSession:
        ses = self.sessions.get(session_id)
        if not ses:
            raise HTTPException(status_code=404, detail="session not found")
        return ses

    def _merge_sources(self, list_text: Optional[str], csv_text: Optional[str]) -> Dict[str, int]:
        incoming: Dict[str, int] = {}
        if list_text:
            for k, v in parse_pasted_list(list_text).items():
                incoming[k] = incoming.get(k, 0) + v
        if csv_text:
            for k, v in parse_csv_text(csv_text).items():
                incoming[k] = incoming.get(k, 0) + v
        return incoming

    def load_pool(
        self,
        session_id: str,
        list_text: Optional[str],
        csv_text: Optional[str],
        locked_list_text: Optional[str] = None,
        locked_csv_text: Optional[str] = None,
        wobble_list_text: Optional[str] = None,
        wobble_csv_text: Optional[str] = None,
    ) -> DeckBuildSession:
        ses = self.get_session(session_id)
        pool_in = self._merge_sources(list_text, csv_text)
        locked_in = self._merge_sources(locked_list_text, locked_csv_text)
        wobble_in = self._merge_sources(wobble_list_text, wobble_csv_text)

        has_explicit_locked_or_wobble = bool(locked_in) or bool(wobble_in)
        if has_explicit_locked_or_wobble:
            ses.pool_counts = pool_in
            ses.locked_counts = locked_in
            ses.wobble_counts = wobble_in
        else:
            # Backward-compatible behavior: load all as pool, clear locked/wobble.
            ses.pool_counts = pool_in
            ses.locked_counts = {}
            ses.wobble_counts = {}
        return ses

    def set_base_p(self, session_id: str, base_p_user: float) -> DeckBuildSession:
        ses = self.get_session(session_id)
        ses.base_p_user = float(base_p_user)
        return ses

    def _take(self, zone: Dict[str, int], card: str, n: int = 1) -> bool:
        cur = zone.get(card, 0)
        if cur < n:
            return False
        nxt = cur - n
        if nxt > 0:
            zone[card] = nxt
        else:
            zone.pop(card, None)
        return True

    def _put(self, zone: Dict[str, int], card: str, n: int = 1) -> None:
        zone[card] = zone.get(card, 0) + n

    def move(self, session_id: str, card: str, from_zone: str, to_zone: str) -> DeckBuildSession:
        ses = self.get_session(session_id)
        from_zone = from_zone.lower()
        to_zone = to_zone.lower()
        if from_zone not in {"pool", "locked", "wobble"}:
            raise HTTPException(status_code=400, detail="from_zone must be pool|locked|wobble")
        if to_zone not in {"pool", "locked", "wobble"}:
            raise HTTPException(status_code=400, detail="to_zone must be pool|locked|wobble")
        if from_zone == to_zone:
            return ses

        is_basic = card in BASIC_LANDS
        zones = {"pool": ses.pool_counts, "locked": ses.locked_counts, "wobble": ses.wobble_counts}

        if from_zone == "pool" and to_zone == "locked" and is_basic:
            self._put(ses.locked_counts, card, 1)
            return ses

        src = zones[from_zone]
        dst = zones[to_zone]
        if not self._take(src, card, 1):
            raise HTTPException(status_code=400, detail=f"card not available in {from_zone}")

        if to_zone == "pool" and is_basic:
            return ses

        self._put(dst, card, 1)
        return ses

    def evaluate(self, session_id: str, base_p_values: Optional[List[float]]) -> Dict:
        ses = self.get_session(session_id)
        values = base_p_values or [0.4, 0.5, 0.6, ses.base_p_user]
        vals = [float(v) for v in values]
        preds = self.scorer.predict_many(ses.locked_counts, vals)
        return {
            "deck_count": int(sum(ses.locked_counts.values())),
            "predictions": preds,
        }

    def _serialize_card(self, name: str, count: int) -> Dict:
        card: Optional[CardRecord] = self.asset_loader.find_by_name(name) if self.asset_loader else None
        art_uri = self.asset_loader.art_uri_for_name(name) if card else None
        scry_art = self.asset_loader.scryfall_image_url(name, version="art_crop") if self.asset_loader else None
        scry_card = self.asset_loader.scryfall_image_url(name, version="png") if self.asset_loader else None
        return {
            "id": name,
            "name": name,
            "count": int(count),
            "grp_id": card.grp_id if card else None,
            "art_uri": art_uri,
            "image_url": art_uri or scry_art,
            "card_image_url": scry_card,
        }

    def _serialize_zone(self, counts: Dict[str, int]) -> List[Dict]:
        return [self._serialize_card(name, cnt) for name, cnt in sorted(counts.items())]

    def state(self, session_id: str) -> Dict:
        ses = self.get_session(session_id)
        return {
            "session_id": ses.session_id,
            "base_p_user": ses.base_p_user,
            "pool": self._serialize_zone(ses.pool_counts),
            "locked": self._serialize_zone(ses.locked_counts),
            "wobble": self._serialize_zone(ses.wobble_counts),
            "basics": self._serialize_zone({b: ses.locked_counts.get(b, 0) for b in sorted(BASIC_LANDS)}),
            "deck_count": int(sum(ses.locked_counts.values())),
        }


app = FastAPI()
asset_loader = CardAssetLoader()
scorer = DeckBumpScorer()
service = DeckBuildService(asset_loader=asset_loader, scorer=scorer)

STATIC_DIR = REPO_ROOT / "scripts" / "deckbuild_ui"
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
    ses = service.create_session(req.base_p_user)
    return {"session_id": ses.session_id, "state": service.state(ses.session_id)}


@app.get("/api/state")
def get_state(session_id: str):
    return service.state(session_id)


@app.post("/api/load_pool")
def load_pool(req: LoadPoolRequest):
    ses = service.load_pool(
        req.session_id,
        req.list_text,
        req.csv_text,
        req.locked_list_text,
        req.locked_csv_text,
        req.wobble_list_text,
        req.wobble_csv_text,
    )
    return service.state(ses.session_id)


@app.post("/api/move")
def move(req: MoveRequest):
    from_zone = req.from_zone or req.from_alias
    to_zone = req.to_zone or req.to_alias
    if not from_zone or not to_zone:
        raise HTTPException(status_code=400, detail="from/to (or from_zone/to_zone) are required")
    ses = service.move(req.session_id, req.card_id, from_zone, to_zone)
    return service.state(ses.session_id)


@app.post("/api/set_base_p")
def set_base_p(req: BasePRequest):
    ses = service.set_base_p(req.session_id, req.base_p_user)
    return service.state(ses.session_id)


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    return service.evaluate(req.session_id, req.base_p_values)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.run_deckbuild_ui:app", host="0.0.0.0", port=8003, reload=False)
