from __future__ import annotations

import csv
import io
import json
import re
import sys
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class SuggestSwapsRequest(BaseModel):
    session_id: str
    top_k: int = 20
    rank_mode: str = "user"  # user|mean|min
    base_p_values: Optional[List[float]] = None
    removable_cards: Optional[List[str]] = None
    max_removables: int = 30
    include_basic_adds: bool = False


class AutoIterateRequest(BaseModel):
    session_id: str
    max_steps: int = 10
    rank_mode: str = "user"  # user|mean|min
    base_p_values: Optional[List[float]] = None
    removable_cards: Optional[List[str]] = None
    max_removables: int = 30
    include_basic_adds: bool = False


class OptimizeBeamRequest(BaseModel):
    session_id: str
    steps: int = 8
    beam_width: int = 10
    top_children_per_parent: int = 60
    R: int = 12
    rank_mode: str = "user"
    mode_removable: str = "auto"  # auto|manual
    removable_cards: Optional[List[str]] = None
    include_basic_adds: bool = False
    include_basic_tweaks: bool = False
    base_p_values: Optional[List[float]] = None
    dedupe: bool = True
    stop_if_no_improvement: bool = True


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

    def predict_batch(self, decks: List[Dict[str, int]], base_p: float) -> np.ndarray:
        if not decks:
            return np.array([], dtype=float)
        X = np.vstack([self.vectorize_deck(deck, base_p) for deck in decks])
        return self.booster.predict(xgb.DMatrix(X, feature_names=self.feature_cols))


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
        self._score_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._score_cache_max = 50000

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
        deck_eval, warnings = self._deck_for_eval(ses.locked_counts)
        preds = self.scorer.predict_many(deck_eval, vals)
        return {
            "deck_count": int(sum(ses.locked_counts.values())),
            "deck_count_eval": int(sum(deck_eval.values())),
            "predictions": preds,
            "warnings": warnings,
        }

    def _deck_for_eval(self, locked_counts: Dict[str, int]) -> tuple[Dict[str, int], List[str]]:
        deck = {k: int(v) for k, v in locked_counts.items() if int(v) > 0}
        warnings: List[str] = []
        total = sum(deck.values())
        if total < 40:
            deck["Plains"] = deck.get("Plains", 0) + (40 - total)
            warnings.append(f"Deck had {total} cards; auto-filled to 40 with Plains.")
        elif total > 40:
            over = total - 40
            for b in sorted(BASIC_LANDS):
                if over <= 0:
                    break
                take = min(deck.get(b, 0), over)
                if take > 0:
                    deck[b] -= take
                    if deck[b] <= 0:
                        deck.pop(b, None)
                    over -= take
            if over > 0:
                for name in sorted([n for n in deck.keys() if n not in BASIC_LANDS]):
                    if over <= 0:
                        break
                    take = min(deck.get(name, 0), over)
                    if take > 0:
                        deck[name] -= take
                        if deck[name] <= 0:
                            deck.pop(name, None)
                        over -= take
            warnings.append(f"Deck had {total} cards; trimmed to 40 for evaluation.")
        basic_lands = sum(deck.get(b, 0) for b in BASIC_LANDS)
        if basic_lands < 16 or basic_lands > 18:
            warnings.append(f"Basic land count is {basic_lands}; target is 16-18.")
        return deck, warnings

    def _rank_score(self, predictions: List[Dict[str, float]], rank_mode: str, base_p_user: float) -> float:
        mode = (rank_mode or "user").lower()
        if not predictions:
            return 0.0
        values = [float(p["deck_bump"]) for p in predictions]
        if mode == "mean":
            return float(np.mean(values))
        if mode == "min":
            return float(np.min(values))
        best = min(predictions, key=lambda p: abs(float(p["base_p"]) - float(base_p_user)))
        return float(best["deck_bump"])

    def _deck_key(self, deck_counts: Dict[str, int]) -> tuple:
        return tuple(sorted((k, int(v)) for k, v in deck_counts.items() if int(v) > 0))

    def _cache_get(self, key: tuple) -> Optional[dict]:
        val = self._score_cache.get(key)
        if val is None:
            return None
        self._score_cache.move_to_end(key)
        return val

    def _cache_put(self, key: tuple, value: dict) -> None:
        self._score_cache[key] = value
        self._score_cache.move_to_end(key)
        while len(self._score_cache) > self._score_cache_max:
            self._score_cache.popitem(last=False)

    def _score_decks(
        self,
        decks: List[Dict[str, int]],
        base_p_values: List[float],
        rank_mode: str,
        base_p_user: float,
    ) -> List[dict]:
        if not decks:
            return []
        base_ps = tuple(float(x) for x in base_p_values)
        mode = (rank_mode or "user").lower()
        out: List[Optional[dict]] = [None] * len(decks)
        miss_idx: List[int] = []
        miss_decks: List[Dict[str, int]] = []
        miss_keys: List[tuple] = []

        for i, d in enumerate(decks):
            d_key = self._deck_key(d)
            c_key = (d_key, base_ps, mode, float(base_p_user))
            cached = self._cache_get(c_key)
            if cached is not None:
                out[i] = cached
            else:
                miss_idx.append(i)
                miss_decks.append(d)
                miss_keys.append(c_key)

        if miss_decks:
            n = len(miss_decks)
            t = len(base_ps)
            if hasattr(self.scorer, "booster") and hasattr(self.scorer, "vectorize_deck"):
                vecs = []
                base_idx = self.scorer.col_to_idx["base_p"]
                for d in miss_decks:
                    v = self.scorer.vectorize_deck(d, 0.5)
                    vecs.append(v)
                X_base = np.vstack(vecs)
                X = np.repeat(X_base, t, axis=0)
                for j, bp in enumerate(base_ps):
                    X[j::t, base_idx] = float(bp)
                y = self.scorer.booster.predict(xgb.DMatrix(X, feature_names=self.scorer.feature_cols))
                y2 = y.reshape((n, t))
            else:
                y2 = np.zeros((n, t), dtype=float)
                for j, bp in enumerate(base_ps):
                    preds = self.scorer.predict_batch(miss_decks, float(bp))
                    for i, pv in enumerate(preds):
                        y2[i, j] = float(pv)
            for row in range(n):
                preds = [{"base_p": float(base_ps[j]), "deck_bump": float(y2[row, j])} for j in range(t)]
                rank = self._rank_score(preds, mode, base_p_user)
                val = {"predictions": preds, "rank_score": float(rank)}
                self._cache_put(miss_keys[row], val)
                out[miss_idx[row]] = val

        return [x for x in out if x is not None]

    def _predict_decks(self, decks: List[Dict[str, int]], base_p_values: List[float], rank_mode: str, base_p_user: float) -> List[List[Dict[str, float]]]:
        scored = self._score_decks(decks, base_p_values, rank_mode, base_p_user)
        return [s["predictions"] for s in scored]

    def suggest_swaps(
        self,
        session_id: str,
        top_k: int = 20,
        rank_mode: str = "user",
        base_p_values: Optional[List[float]] = None,
        removable_cards: Optional[List[str]] = None,
        max_removables: int = 30,
        include_basic_adds: bool = False,
    ) -> Dict[str, Any]:
        ses = self.get_session(session_id)
        base_vals = [float(v) for v in (base_p_values or [0.4, 0.5, 0.6, ses.base_p_user])]
        current_deck, warnings = self._deck_for_eval(ses.locked_counts)
        current_preds = self.scorer.predict_many(current_deck, base_vals)
        current_rank = self._rank_score(current_preds, rank_mode, ses.base_p_user)

        if removable_cards:
            removable_raw = [c for c in removable_cards if current_deck.get(c, 0) > 0]
        else:
            removable_raw = [c for c, n in current_deck.items() if n > 0 and c not in BASIC_LANDS]

        removable_nom: List[Dict[str, float]] = []
        removable_pool: List[str] = []
        if removable_raw:
            rem_decks: List[Dict[str, int]] = []
            rem_cards: List[str] = []
            for r in removable_raw:
                nd = dict(current_deck)
                nd[r] -= 1
                if nd[r] <= 0:
                    nd.pop(r, None)
                if sum(nd.values()) < 40:
                    nd["Plains"] = nd.get("Plains", 0) + (40 - sum(nd.values()))
                rem_decks.append(nd)
                rem_cards.append(r)
            rem_preds = self._predict_decks(rem_decks, base_vals, rank_mode, ses.base_p_user)
            for card, preds in zip(rem_cards, rem_preds):
                rank = self._rank_score(preds, rank_mode, ses.base_p_user)
                delta = rank - current_rank
                removable_nom.append({"card": card, "delta_remove": float(delta)})
            removable_nom.sort(key=lambda x: x["delta_remove"], reverse=True)
            if removable_cards:
                removable_pool = [r["card"] for r in removable_nom][: max(1, int(max_removables))]
            else:
                removable_pool = [r["card"] for r in removable_nom if r["delta_remove"] > 0][: max(1, int(max_removables))]

        addable = [c for c, n in ses.wobble_counts.items() if n > 0 and c not in BASIC_LANDS]
        if include_basic_adds:
            addable.extend(sorted(BASIC_LANDS))

        candidate_decks: List[Dict[str, int]] = []
        candidate_meta: List[Dict[str, str]] = []
        for r in removable_pool:
            for a in addable:
                if a == r:
                    continue
                nd = dict(current_deck)
                if nd.get(r, 0) <= 0:
                    continue
                nd[r] -= 1
                if nd[r] <= 0:
                    nd.pop(r, None)
                nd[a] = nd.get(a, 0) + 1
                candidate_decks.append(nd)
                candidate_meta.append({"remove": r, "add": a})

        suggestions: List[Dict[str, Any]] = []
        if candidate_decks:
            cand_preds = self._predict_decks(candidate_decks, base_vals, rank_mode, ses.base_p_user)
            for meta, preds in zip(candidate_meta, cand_preds):
                rank = self._rank_score(preds, rank_mode, ses.base_p_user)
                delta = rank - current_rank
                if delta <= 0:
                    continue
                suggestions.append(
                    {
                        "remove": meta["remove"],
                        "add": meta["add"],
                        "delta": float(delta),
                        "new_rank_score": float(rank),
                        "new_predictions": preds,
                    }
                )
            suggestions.sort(key=lambda x: x["delta"], reverse=True)
            suggestions = suggestions[: max(1, int(top_k))]

        return {
            "current": {
                "deck_count": int(sum(ses.locked_counts.values())),
                "deck_count_eval": int(sum(current_deck.values())),
                "predictions": current_preds,
                "rank_score": float(current_rank),
                "warnings": warnings,
            },
            "removable_nominated": removable_nom[: max(1, int(max_removables))],
            "suggestions": suggestions,
        }

    def auto_iterate(
        self,
        session_id: str,
        max_steps: int = 10,
        rank_mode: str = "user",
        base_p_values: Optional[List[float]] = None,
        removable_cards: Optional[List[str]] = None,
        max_removables: int = 30,
        include_basic_adds: bool = False,
    ) -> Dict[str, Any]:
        ses = self.get_session(session_id)
        applied: List[Dict[str, Any]] = []
        for step in range(max(0, int(max_steps))):
            out = self.suggest_swaps(
                session_id=session_id,
                top_k=1,
                rank_mode=rank_mode,
                base_p_values=base_p_values,
                removable_cards=removable_cards,
                max_removables=max_removables,
                include_basic_adds=include_basic_adds,
            )
            if not out["suggestions"]:
                break
            best = out["suggestions"][0]
            remove_card = best["remove"]
            add_card = best["add"]
            # Apply swap into session zones.
            if ses.locked_counts.get(remove_card, 0) <= 0:
                break
            ses.locked_counts[remove_card] -= 1
            if ses.locked_counts[remove_card] <= 0:
                ses.locked_counts.pop(remove_card, None)
            ses.locked_counts[add_card] = ses.locked_counts.get(add_card, 0) + 1
            if add_card in BASIC_LANDS:
                pass
            else:
                ses.wobble_counts[add_card] = max(0, ses.wobble_counts.get(add_card, 0) - 1)
                if ses.wobble_counts.get(add_card, 0) <= 0:
                    ses.wobble_counts.pop(add_card, None)
            ses.wobble_counts[remove_card] = ses.wobble_counts.get(remove_card, 0) + 1
            applied.append(
                {
                    "step": step + 1,
                    "remove": remove_card,
                    "add": add_card,
                    "delta": float(best["delta"]),
                    "new_rank_score": float(best["new_rank_score"]),
                }
            )

        final_eval = self.evaluate(session_id, base_p_values)
        final_deck, _warnings = self._deck_for_eval(ses.locked_counts)
        final_rank = self._rank_score(final_eval["predictions"], rank_mode, ses.base_p_user)
        return {
            "applied_swaps": applied,
            "final": {
                "deck_count": int(sum(ses.locked_counts.values())),
                "deck_count_eval": int(sum(final_deck.values())),
                "predictions": final_eval["predictions"],
                "rank_score": float(final_rank),
                "warnings": final_eval.get("warnings", []),
            },
            "state": self.state(session_id),
        }

    def _pool_limit(self, ses: DeckBuildSession, card: str) -> int:
        return int(
            max(
                ses.pool_counts.get(card, 0),
                ses.locked_counts.get(card, 0) + ses.wobble_counts.get(card, 0),
            )
        )

    def optimize_beam(
        self,
        session_id: str,
        steps: int = 8,
        beam_width: int = 10,
        top_children_per_parent: int = 300,
        R: int = 12,
        rank_mode: str = "user",
        mode_removable: str = "auto",
        removable_cards: Optional[List[str]] = None,
        include_basic_adds: bool = False,
        include_basic_tweaks: bool = False,
        base_p_values: Optional[List[float]] = None,
        dedupe: bool = True,
        stop_if_no_improvement: bool = True,
    ) -> Dict[str, Any]:
        ses = self.get_session(session_id)
        base_vals = [float(v) for v in (base_p_values or [0.4, 0.5, 0.6, ses.base_p_user])]
        start_deck, warnings = self._deck_for_eval(ses.locked_counts)
        start_score = self._score_decks([start_deck], base_vals, rank_mode, ses.base_p_user)[0]
        start_node = {"deck_counts": start_deck, "score": start_score, "path": []}
        beam = [start_node]
        best = start_node
        trajectory = [{"step": 0, "rank_score": float(start_score["rank_score"])}]

        addable = [c for c, n in ses.wobble_counts.items() if n > 0 and c not in BASIC_LANDS]
        if include_basic_adds:
            addable.extend(sorted(BASIC_LANDS))
        addable = sorted(set(addable))

        mode_rem = (mode_removable or "auto").lower()
        manual_rem = set(removable_cards or [])

        for step_i in range(1, max(1, int(steps)) + 1):
            all_children: List[Dict[str, Any]] = []
            for parent in beam:
                p_deck = parent["deck_counts"]
                p_score = float(parent["score"]["rank_score"])

                if mode_rem == "manual" and manual_rem:
                    rem_cards = [c for c in sorted(manual_rem) if p_deck.get(c, 0) > 0]
                    rem_nom = [{"card": c, "delta_remove": 0.0} for c in rem_cards]
                else:
                    cand = sorted([c for c, n in p_deck.items() if n > 0], key=lambda x: (-p_deck[x], x))[:40]
                    rem_decks: List[Dict[str, int]] = []
                    for c in cand:
                        nd = dict(p_deck)
                        nd[c] -= 1
                        if nd[c] <= 0:
                            nd.pop(c, None)
                        if sum(nd.values()) < 40:
                            nd["Plains"] = nd.get("Plains", 0) + (40 - sum(nd.values()))
                        rem_decks.append(nd)
                    rem_scores = self._score_decks(rem_decks, base_vals, rank_mode, ses.base_p_user)
                    rem_nom = []
                    for c, s in zip(cand, rem_scores):
                        rem_nom.append({"card": c, "delta_remove": float(s["rank_score"] - p_score)})
                    rem_nom.sort(key=lambda x: x["delta_remove"], reverse=True)
                    rem_cards = [x["card"] for x in rem_nom[: max(1, int(R))]]

                cand_decks: List[Dict[str, int]] = []
                cand_moves: List[Dict[str, str]] = []
                for r in rem_cards:
                    if p_deck.get(r, 0) <= 0:
                        continue
                    for a in addable:
                        if a == r:
                            continue
                        nd = dict(p_deck)
                        nd[r] -= 1
                        if nd[r] <= 0:
                            nd.pop(r, None)
                        nd[a] = nd.get(a, 0) + 1
                        if a not in BASIC_LANDS and nd.get(a, 0) > self._pool_limit(ses, a):
                            continue
                        cand_decks.append(nd)
                        cand_moves.append({"remove": r, "add": a})

                if include_basic_tweaks:
                    basics_present = [b for b in sorted(BASIC_LANDS) if p_deck.get(b, 0) > 0]
                    for bx in basics_present:
                        for by in sorted(BASIC_LANDS):
                            if bx == by:
                                continue
                            nd = dict(p_deck)
                            nd[bx] -= 1
                            if nd[bx] <= 0:
                                nd.pop(bx, None)
                            nd[by] = nd.get(by, 0) + 1
                            cand_decks.append(nd)
                            cand_moves.append({"remove": bx, "add": by})

                if not cand_decks:
                    continue
                c_scores = self._score_decks(cand_decks, base_vals, rank_mode, ses.base_p_user)
                rows = list(zip(cand_decks, cand_moves, c_scores))
                rows.sort(key=lambda x: float(x[2]["rank_score"]), reverse=True)
                rows = rows[: max(1, int(top_children_per_parent))]
                for d, m, s in rows:
                    all_children.append(
                        {
                            "deck_counts": d,
                            "score": s,
                            "path": parent["path"]
                            + [
                                {
                                    "remove": m["remove"],
                                    "add": m["add"],
                                    "delta": float(s["rank_score"] - p_score),
                                }
                            ],
                        }
                    )

            if not all_children:
                break

            if dedupe:
                best_by_key: Dict[tuple, Dict[str, Any]] = {}
                for ch in all_children:
                    k = self._deck_key(ch["deck_counts"])
                    prev = best_by_key.get(k)
                    if prev is None or float(ch["score"]["rank_score"]) > float(prev["score"]["rank_score"]):
                        best_by_key[k] = ch
                all_children = list(best_by_key.values())

            all_children.sort(key=lambda x: float(x["score"]["rank_score"]), reverse=True)
            beam = all_children[: max(1, int(beam_width))]
            step_best = beam[0]
            improved = float(step_best["score"]["rank_score"]) > float(best["score"]["rank_score"])
            if improved:
                best = step_best
            trajectory.append({"step": step_i, "rank_score": float(best["score"]["rank_score"])})
            if stop_if_no_improvement and not improved:
                #break
                continue

        return {
            "start": {
                "rank_score": float(start_score["rank_score"]),
                "predictions": start_score["predictions"],
            },
            "best": {
                "rank_score": float(best["score"]["rank_score"]),
                "predictions": best["score"]["predictions"],
                "deck_counts": best["deck_counts"],
            },
            "path": best["path"],
            "trajectory": trajectory,
            "warnings": warnings,
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


@app.post("/api/suggest_swaps")
def suggest_swaps(req: SuggestSwapsRequest):
    return service.suggest_swaps(
        session_id=req.session_id,
        top_k=req.top_k,
        rank_mode=req.rank_mode,
        base_p_values=req.base_p_values,
        removable_cards=req.removable_cards,
        max_removables=req.max_removables,
        include_basic_adds=req.include_basic_adds,
    )


@app.post("/api/auto_iterate")
def auto_iterate(req: AutoIterateRequest):
    return service.auto_iterate(
        session_id=req.session_id,
        max_steps=req.max_steps,
        rank_mode=req.rank_mode,
        base_p_values=req.base_p_values,
        removable_cards=req.removable_cards,
        max_removables=req.max_removables,
        include_basic_adds=req.include_basic_adds,
    )


@app.post("/api/optimize_beam")
def optimize_beam(req: OptimizeBeamRequest):
    return service.optimize_beam(
        session_id=req.session_id,
        steps=req.steps,
        beam_width=req.beam_width,
        top_children_per_parent=req.top_children_per_parent,
        R=req.R,
        rank_mode=req.rank_mode,
        mode_removable=req.mode_removable,
        removable_cards=req.removable_cards,
        include_basic_adds=req.include_basic_adds,
        include_basic_tweaks=req.include_basic_tweaks,
        base_p_values=req.base_p_values,
        dedupe=req.dedupe,
        stop_if_no_improvement=req.stop_if_no_improvement,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.run_deckbuild_ui:app", host="0.0.0.0", port=8003, reload=False)
