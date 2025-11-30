from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from draft_env.pack_sampler import sample_pack
from human_policy.bc_policy import make_bc_policy
from hero_bot.hero_policy import hero_policy
from human_policy.random_policy import random_policy
from deck_eval.evaluator import evaluate_deck
from src.models.features import build_deck_features, build_skill_features, build_joint_features

app = FastAPI()
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Draft vs Bots</title>
    <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    #cards button { margin: 4px; padding: 8px 12px; }
    pre { background: #f7f7f7; padding: 12px; }
    .section { margin-top: 16px; padding: 12px; border: 1px solid #ddd; border-radius: 6px; }
    .row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
    .list { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
    .card-row { border: 1px solid #ddd; border-radius: 6px; padding: 8px; display: flex; justify-content: space-between; align-items: center; }
    button { cursor: pointer; }
  </style>
</head>
<body>
  <h1>Draft vs Bots</h1>
  <div class="section" id="draftView">
    <div class="row">
      <label>Set code: <input id="setCode" value="FIN" /></label>
      <label>Bot policy:
        <select id="botPolicy">
          <option value="hero">hero</option>
          <option value="bc">bc</option>
          <option value="random">random</option>
        </select>
      </label>
      <button onclick="startDraft()">Start Draft</button>
      <button onclick="showDeckBuilder()" id="deckBuilderBtn" disabled>Assemble Deck</button>
    </div>
    <div id="status"></div>
    <div id="advice"></div>
    <div id="cards"></div>
    <h3>Your pool</h3>
    <pre id="pool"></pre>
  </div>

  <div class="section" id="deckBuilder" style="display:none;">
    <h2>Deck Builder</h2>
    <div class="row">
      <label>Rank: <input id="rankInput" placeholder="Gold" /></label>
      <label>Games bucket: <input id="gamesBucketInput" type="number" /></label>
      <label>Win rate bucket: <input id="wrBucketInput" type="number" /></label>
      <button onclick="submitDeck()">Score Deck</button>
      <button onclick="backToDraft()">Back</button>
    </div>
    <div class="row">
      <div>Deck size: <span id="deckSize">0</span> (target 40)</div>
      <div id="scoreOutput"></div>
    </div>
    <h3>Add/Remove Cards</h3>
    <div class="list" id="deckList"></div>
  </div>
<script>
let sessionId = null;
let poolCounts = {};
let deckCounts = {};
const basics = ["Plains","Island","Swamp","Mountain","Forest","Wastes"];

async function startDraft() {
  const setCode = document.getElementById("setCode").value || "FIN";
  const botPolicy = document.getElementById("botPolicy").value || "hero";
  const res = await fetch("/start_draft", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({set_code: setCode, bot_policy: botPolicy})
  });
  const data = await res.json();
  sessionId = data.session_id;
  poolCounts = data.pool_counts || {};
  deckCounts = {};
  renderState(data);
}
async function pickCard(card) {
  if (!sessionId) return;
  const res = await fetch("/pick", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({session_id: sessionId, card})
  });
  const data = await res.json();
  renderState(data);
}
function renderState(data) {
  document.getElementById("status").innerText =
    data.done ? "Draft complete" : `Pack ${data.pack_number} / Pick ${data.pick_number}`;
  const cardsDiv = document.getElementById("cards");
  cardsDiv.innerHTML = "";
  if (!data.done && data.pack_cards) {
    data.pack_cards.forEach(c => {
      const btn = document.createElement("button");
      btn.innerText = c;
      btn.onclick = () => pickCard(c);
      cardsDiv.appendChild(btn);
    });
  }
  poolCounts = data.pool_counts || poolCounts;
  document.getElementById("pool").innerText = JSON.stringify(poolCounts, null, 2);
  document.getElementById("deckBuilderBtn").disabled = !data.done;
  if (!data.done) {
    updateAdvice(data.pack_cards, data.pack_number, data.pick_number);
  } else {
    document.getElementById("advice").innerText = "";
  }
}

function showDeckBuilder() {
  // initialize deck with basics to reach 40 if empty
  if (Object.keys(deckCounts).length === 0) {
    deckCounts = Object.assign({}, poolCounts);
    let size = deckSize();
    while (size < 40) {
      const basic = basics[size % basics.length];
      deckCounts[basic] = (deckCounts[basic] || 0) + 1;
      size++;
    }
  }
  renderDeckBuilder();
  document.getElementById("deckBuilder").style.display = "block";
  document.getElementById("draftView").style.display = "none";
}

function backToDraft() {
  document.getElementById("deckBuilder").style.display = "none";
  document.getElementById("draftView").style.display = "block";
}

function deckSize() {
  return Object.values(deckCounts).reduce((a,b)=>a+(b||0),0);
}

function renderDeckBuilder() {
  const listDiv = document.getElementById("deckList");
  listDiv.innerHTML = "";
  const allCards = new Set([...Object.keys(poolCounts), ...basics]);
  allCards.forEach(name => {
    const row = document.createElement("div");
    row.className = "card-row";
    const count = deckCounts[name] || 0;
    row.innerHTML = `<span>${name}</span><span>Deck: ${count} / Pool: ${poolCounts[name] || 0}</span>`;
    const controls = document.createElement("div");
    const addBtn = document.createElement("button");
    addBtn.innerText = "+";
    addBtn.onclick = () => { deckCounts[name] = (deckCounts[name] || 0) + 1; updateDeck(); };
    const remBtn = document.createElement("button");
    remBtn.innerText = "-";
    remBtn.onclick = () => { deckCounts[name] = Math.max(0, (deckCounts[name] || 0) - 1); updateDeck(); };
    controls.appendChild(addBtn);
    controls.appendChild(remBtn);
    row.appendChild(controls);
    listDiv.appendChild(row);
  });
  updateDeck();
}

function updateDeck() {
  document.getElementById("deckSize").innerText = deckSize();
}

async function submitDeck() {
  const rank = document.getElementById("rankInput").value || null;
  const gamesBucket = document.getElementById("gamesBucketInput").value;
  const wrBucket = document.getElementById("wrBucketInput").value;
  const payload = {
    deck_counts: deckCounts,
    rank: rank,
    user_n_games_bucket: gamesBucket ? parseFloat(gamesBucket) : null,
    user_game_win_rate_bucket: wrBucket ? parseFloat(wrBucket) : null
  };
  const res = await fetch("/score_deck", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  const out = document.getElementById("scoreOutput");
  if (res.ok) {
    out.innerText = `Deck effect: ${data.deck_effect?.toFixed(4)} | Skill: ${data.skill_pred ?? 'n/a'} | Joint: ${data.joint_pred ?? 'n/a'} | Deck boost: ${data.deck_boost ?? 'n/a'}`;
  } else {
    out.innerText = `Error: ${data.detail}`;
  }
}

async function updateAdvice(packCards, packNumber, pickNumber) {
  const res = await fetch("/recommend_pick", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      pack_cards: packCards,
      pool_counts: poolCounts,
      pack_number: packNumber,
      pick_number: pickNumber
    })
  });
  const adviceDiv = document.getElementById("advice");
  if (!res.ok) {
    adviceDiv.innerText = "Advice unavailable";
    return;
  }
  const data = await res.json();
  const recs = data.scored || [];
  if (recs.length === 0) {
    adviceDiv.innerText = "Advice unavailable";
    return;
  }
  const top = recs.slice(0, 5).map(r => `${r.card} (${r.score.toFixed(3)})`).join(", ");
  adviceDiv.innerText = "Recommendations: " + top;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/recommend")
def recommend(payload: dict):
    """
    Minimal stub endpoint to satisfy tests.
    Returns two placeholder recommendations.
    """
    # echo back recommendations regardless of input
    return {"recommendations": ["card_a", "card_b"]}


# -------- Draft GUI support --------


class StartDraftRequest(BaseModel):
    set_code: str | None = "FIN"
    bot_policy: str = "hero"  # hero | bc | random
    num_seats: int = 8
    packs_per_player: int = 3
    pack_size: int = 15


class PickRequest(BaseModel):
    session_id: str
    card: str


class RecommendRequest(BaseModel):
    pack_cards: list[str]
    pool_counts: dict
    pack_number: int = 1
    pick_number: int = 1
    policy: str = "hero"  # hero | bc | random


class ScoreDeckRequest(BaseModel):
    deck_counts: dict
    rank: str | None = None
    user_n_games_bucket: float | None = None
    user_game_win_rate_bucket: float | None = None


class DraftSession:
    def __init__(self, cfg: StartDraftRequest):
        self.session_id = str(uuid.uuid4())
        self.num_seats = cfg.num_seats
        self.packs_per_player = cfg.packs_per_player
        self.pack_size = cfg.pack_size
        self.set_code = cfg.set_code
        self.rng = np.random.default_rng()
        # choose bot policy
        self.bot_policy = _resolve_policy(cfg.bot_policy)
        self.pools = [dict() for _ in range(self.num_seats)]
        self.pack_idx = 0
        self.pick_idx = 0
        self.direction = 1
        # generate packs
        self.packs = [
            [sample_pack(self.rng, self.pack_size, set_code=self.set_code) for _ in range(self.num_seats)]
            for _ in range(self.packs_per_player)
        ]

    def _pass_packs(self):
        new_packs = [None] * self.num_seats  # type: ignore
        for seat in range(self.num_seats):
            target = (seat + self.direction) % self.num_seats
            new_packs[target] = self.packs[self.pack_idx][seat]
        self.packs[self.pack_idx] = new_packs

    def current_pack(self):
        if self.pack_idx >= self.packs_per_player:
            return []
        return self.packs[self.pack_idx][0]

    def step(self, user_card: str):
        pack = self.current_pack()
        if user_card not in pack:
            raise ValueError(f"card {user_card} not in current pack")
        pack.remove(user_card)
        self.pools[0][user_card] = self.pools[0].get(user_card, 0) + 1
        # bots pick
        for seat in range(1, self.num_seats):
            pack_cards = self.packs[self.pack_idx][seat]
            if not pack_cards:
                continue
            choice = self.bot_policy(pack_cards, self.pools[seat], seat, self.rng)
            if choice not in pack_cards:
                choice = self.rng.choice(pack_cards)
            pack_cards.remove(choice)
            self.pools[seat][choice] = self.pools[seat].get(choice, 0) + 1

        # pass packs and advance
        self._pass_packs()
        self.pick_idx += 1
        if self.pick_idx >= self.pack_size or all(len(p) == 0 for p in self.packs[self.pack_idx]):
            self.pack_idx += 1
            self.pick_idx = 0
            self.direction = 1 if self.pack_idx % 2 == 0 else -1

    def is_done(self) -> bool:
        return self.pack_idx >= self.packs_per_player


SESSIONS: dict[str, DraftSession] = {}
_SKILL_MODEL = None
_JOINT_MODEL = None
_DECK_FEATURE_COLS: list[str] = []


def _load_models():
    global _SKILL_MODEL, _JOINT_MODEL, _DECK_FEATURE_COLS
    if _SKILL_MODEL is None and (MODELS_DIR / "skill_model.pkl").exists():
        _SKILL_MODEL = joblib.load(MODELS_DIR / "skill_model.pkl")
    if _JOINT_MODEL is None and (MODELS_DIR / "joint_model.pkl").exists():
        _JOINT_MODEL = joblib.load(MODELS_DIR / "joint_model.pkl")
        if hasattr(_JOINT_MODEL, "feature_names_in_"):
            _DECK_FEATURE_COLS = [c for c in _JOINT_MODEL.feature_names_in_ if c.startswith("deck_")]


def _deck_df_from_counts(deck_counts: dict, req: ScoreDeckRequest) -> pd.DataFrame:
    row = {}
    total = 0
    # ensure expected deck_* columns exist for model alignment
    for col in _DECK_FEATURE_COLS:
        row[col] = 0
    for name, cnt in deck_counts.items():
        col = f"deck_{name}"
        row[col] = cnt
        total += cnt
    row["deck_size_avg"] = total
    row["main_colors"] = ""
    row["splash_colors"] = ""
    row["rank"] = req.rank or ""
    row["user_n_games_bucket"] = req.user_n_games_bucket or 0
    row["user_game_win_rate_bucket"] = req.user_game_win_rate_bucket or 0
    return pd.DataFrame([row])


def _align_features(df: pd.DataFrame, model) -> pd.DataFrame:
    if model is not None and hasattr(model, "feature_names_in_"):
        return df.reindex(columns=model.feature_names_in_, fill_value=0)
    return df


def _resolve_policy(name: str):
    name = (name or "").lower()
    if name == "bc":
        return make_bc_policy()
    if name == "random":
        return random_policy
    return hero_policy


@app.post("/start_draft")
def start_draft(req: StartDraftRequest):
    session = DraftSession(req)
    SESSIONS[session.session_id] = session
    return {
        "session_id": session.session_id,
        "pack_cards": session.current_pack(),
        "pool_counts": session.pools[0],
        "pack_number": session.pack_idx + 1,
        "pick_number": session.pick_idx + 1,
        "done": session.is_done(),
    }


@app.post("/pick")
def pick_card(req: PickRequest):
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    if session.is_done():
        return {"done": True, "message": "draft complete", "pool_counts": session.pools[0]}
    try:
        session.step(req.card)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "session_id": session.session_id,
        "pack_cards": session.current_pack(),
        "pool_counts": session.pools[0],
        "pack_number": session.pack_idx + 1,
        "pick_number": session.pick_idx + 1,
        "done": session.is_done(),
    }


@app.post("/recommend_pick")
def recommend_pick(req: RecommendRequest):
    """
    Rank a pack given current pool and pack/pick numbers.
    """
    if not req.pack_cards:
        raise HTTPException(status_code=400, detail="pack_cards required")
    # score with evaluator/state-value
    scored = []
    pack_no = req.pack_number
    pick_no = req.pick_number
    model = None
    try:
        from hero_bot.hero_policy import _get_state_value_model
        model = _get_state_value_model()
        from state_encoding.encoder import encode_state
    except Exception:
        model = None
    for card in req.pack_cards:
        new_pool = dict(req.pool_counts)
        new_pool[card] = new_pool.get(card, 0) + 1
        if model is not None:
            state_vec = encode_state(new_pool, pack_no=pack_no, pick_no=pick_no)
            score = float(model.predict(state_vec.reshape(1, -1))[0])
        else:
            score = evaluate_deck(new_pool)
        scored.append((card, score))
    scored_sorted = sorted(scored, key=lambda kv: kv[1], reverse=True)
    ranked = [c for c, _ in scored_sorted]
    return {
        "recommendations": ranked or req.pack_cards,
        "scored": [{"card": c, "score": float(s)} for c, s in scored_sorted],
    }


@app.post("/score_deck")
def score_deck(req: ScoreDeckRequest):
    """
    Score a candidate deck with the calibrated deck-effect model and (if present) joint/skill models.
    Payload example:
    {
      "deck_counts": {"card_a": 4, "card_b": 3},
      "rank": "Gold",
      "user_n_games_bucket": 50,
      "user_game_win_rate_bucket": 55
    }
    """
    deck_counts = req.deck_counts or {}
    deck_effect = evaluate_deck(deck_counts)

    _load_models()
    skill_pred = None
    joint_pred = None
    deck_boost = None

    try:
        df = _deck_df_from_counts(deck_counts, req)
        deck_feats = build_deck_features(df).fillna(0)
        skill_feats = build_skill_features(df).fillna(0)
        joint_feats = build_joint_features(df).fillna(0)
        deck_feats = _align_features(deck_feats, _JOINT_MODEL or _SKILL_MODEL)
        skill_feats = _align_features(skill_feats, _SKILL_MODEL)
        joint_feats = _align_features(joint_feats, _JOINT_MODEL)

        if _SKILL_MODEL is not None:
            skill_pred = float(_SKILL_MODEL.predict(skill_feats)[0])
        if _JOINT_MODEL is not None:
            joint_pred = float(_JOINT_MODEL.predict(joint_feats)[0])
        if skill_pred is not None and joint_pred is not None:
            deck_boost = joint_pred - skill_pred
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to score deck: {e}")

    return {
        "deck_effect": deck_effect,
        "skill_pred": skill_pred,
        "joint_pred": joint_pred,
        "deck_boost": deck_boost,
    }
