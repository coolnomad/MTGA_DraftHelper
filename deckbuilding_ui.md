# deckbuilding-only ui (python + fastapi + static frontend)

goal: create a new script/app that launches a deckbuilding ui only (no draft simulation). it should reuse the existing card art + serialization patterns from `scripts/run_pod_human_ui.py`. :contentReference[oaicite:0]{index=0}

## source reference to reuse (do not change)
- reuse the `CardAssetLoader` + `CardRecord` usage and the “serialize card with art_uri + scryfall fallback” logic found in `scripts/run_pod_human_ui.py`. :contentReference[oaicite:1]{index=1}
- reuse static ui serving pattern:
  - `FastAPI()` backend
  - mount `/static` from an adjacent directory
  - `GET /` returns `index.html`

## new deliverable
create a new python script (do not modify existing) that runs a fastapi app for deckbuilding-only.

suggested path/name:
- `scripts/run_deckbuild_ui.py`
and a new static frontend directory:
- `scripts/deckbuild_ui/` (with `index.html`, `app.js` or similar)

## model artifact
- xgboost model file: `model/deck_bump_model.ubj`
- feature schema file: REQUIRED. create/assume one of:
  - `model/deck_cols.json` (preferred) OR `model/deck_cols.csv`
  - it must list feature columns in exact order used by the model.
- inference inputs:
  - `base_p` feature value
  - deck composition feature vector aligned to `deck_cols`

## ui requirements (elements)
the page must contain these panels/elements:

1) deck entry
- textarea: user can paste a list of card names (one per line), with optional leading counts like:
  - `2 Lightning Bolt`
  - `Lightning Bolt`
- upload control: user can upload a csv containing card names
  - accept these csv formats:
    - single column named `name` OR `card_name` OR no header (assume first column is name)
    - optional `count` column; if absent count=1 per row
- optional: a numeric input for user-provided `base_p_user` (default 0.55)

2) pool window
- displays each card in the uploaded pool, with:
  - card art (art_uri or scryfall fallback)
  - name
  - count in pool
- each card row has a button “add to locked” and “add to wobble”

3) deck window (locked-in maindeck)
- list/grid of cards the user has locked in
- each item shows count and art
- each item has buttons:
  - `-` / `+` count (bounded by availability unless basic lands)
  - “move to wobble” (one copy)
  - “remove” (back to pool)

4) wobble window (swap candidates)
- list/grid of cards marked as wobble (consider swapping)
- each item shows count and art
- each item has buttons:
  - “move to locked” (one copy)
  - “remove” (back to pool)

5) evaluate button
- when clicked, backend returns deck bump predictions for:
  - base_p = 0.4
  - base_p = 0.5
  - base_p = 0.6
  - base_p = base_p_user (user-provided)
- show results in a small table and include:
  - deck card count (sum of locked counts)
  - optionally show lands breakdown if basic land columns exist in `deck_cols`

## allowed basics
- allow unlimited basic lands to be added to locked deck even if not in pool
- basics set: `{"Island","Swamp","Forest","Mountain","Plains"}` (match existing constant usage) :contentReference[oaicite:2]{index=2}
- provide a “basics” section in pool window or separate bar with `+` buttons for each basic land.

---

## backend: data model + endpoints

### in-memory session
- on first load, create a session_id (uuid) similar to the draft ui pattern. :contentReference[oaicite:3]{index=3}
- store per session:
  - `pool_counts: Dict[str,int]` (uploaded pool)
  - `locked_counts: Dict[str,int]`
  - `wobble_counts: Dict[str,int]`
  - `base_p_user: float`
- do not persist to disk for v1.

### endpoints

#### `POST /api/session`
create a new session
request:
```json
{ "base_p_user": 0.55 }