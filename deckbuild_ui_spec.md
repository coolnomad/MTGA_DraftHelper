# Deckbuilding-Only UI (Python + FastAPI + Static Frontend)

## Goal

Create a new script/app that launches a deckbuilding-only UI (no draft
simulation).\
It should reuse the existing card art + serialization patterns from:

`scripts/run_pod_human_ui.py`

------------------------------------------------------------------------

## Source Reference to Reuse (Do Not Modify)

Reuse from `scripts/run_pod_human_ui.py`:

-   `CardAssetLoader`
-   `CardRecord`
-   Card serialization logic (art_uri + scryfall fallback)
-   FastAPI + static serving pattern
    -   `FastAPI()` backend
    -   mount `/static`
    -   `GET /` returns `index.html`

------------------------------------------------------------------------

## New Deliverable

Create:

-   `scripts/run_deckbuild_ui.py`
-   `scripts/deckbuild_ui/`
    -   `index.html`
    -   `app.js`
    -   optional CSS

Do NOT modify the draft UI.

------------------------------------------------------------------------

## Model Artifact

Model file: - `model/deck_bump_model.ubj`

Feature schema (required): - `model/deck_cols.json` (preferred) OR -
`model/deck_cols.csv`

Inference features: - `base_p` - deck composition aligned exactly to
`deck_cols`

------------------------------------------------------------------------

## UI Requirements

### 1) Deck Entry Panel

-   Textarea for pasted list (one per line)
    -   `2 Lightning Bolt`
    -   `Lightning Bolt`
-   CSV upload
    -   Accept formats:
        -   `name` column
        -   `card_name` column
        -   or no header (assume first column is name)
    -   Optional `count` column
-   Numeric input for `base_p_user` (default 0.55)

------------------------------------------------------------------------

### 2) Pool Window

Display each card in pool with: - Art - Name - Count - Buttons: - "Add
to Locked" - "Add to Wobble"

------------------------------------------------------------------------

### 3) Locked Deck Window

-   Cards considered maindeck
-   Show count + art
-   Buttons:
    -   `+` / `-`
    -   "Move to Wobble"
    -   "Remove"

------------------------------------------------------------------------

### 4) Wobble Window

-   Swap candidates
-   Show count + art
-   Buttons:
    -   "Move to Locked"
    -   "Remove"

------------------------------------------------------------------------

### 5) Evaluate Button

Returns predictions for: - base_p = 0.4 - base_p = 0.5 - base_p = 0.6 -
base_p = user value

Response example:

``` json
{
  "deck_count": 40,
  "predictions": [
    {"base_p":0.4,"deck_bump":0.0123},
    {"base_p":0.5,"deck_bump":0.0101},
    {"base_p":0.6,"deck_bump":0.0087},
    {"base_p":0.57,"deck_bump":0.0091}
  ]
}
```

------------------------------------------------------------------------

## Basics

Unlimited basics allowed in locked deck:

    Island
    Swamp
    Forest
    Mountain
    Plains

------------------------------------------------------------------------

## Backend API

### POST /api/session

Create session

Request:

``` json
{ "base_p_user": 0.55 }
```

------------------------------------------------------------------------

### POST /api/load_pool

Load from text or CSV

------------------------------------------------------------------------

### POST /api/move

``` json
{
  "session_id":"...",
  "card_id":"Card Name",
  "from":"pool|locked|wobble",
  "to":"pool|locked|wobble"
}
```

------------------------------------------------------------------------

### POST /api/set_base_p

``` json
{
  "session_id":"...",
  "base_p_user": 0.57
}
```

------------------------------------------------------------------------

### POST /api/evaluate

``` json
{
  "session_id":"...",
  "base_p_values":[0.4,0.5,0.6,0.57]
}
```

------------------------------------------------------------------------

## Model Scoring (Python)

Load once on startup:

-   Load booster from `model/deck_bump_model.ubj`
-   Load `deck_cols`
-   Build column index mapping
-   Ensure `base_p` exists in schema

Feature encoding:

-   Create zero vector length = len(deck_cols)
-   Map locked deck counts to corresponding `deck_<card_name>` features
-   Set `base_p` column

Prediction:

``` python
dmat = xgb.DMatrix(X)
pred = bst.predict(dmat)
```

Batch across base_p values for efficiency.

------------------------------------------------------------------------

## Tests (Pytest)

1.  Parsing pasted list
2.  Move logic
3.  Feature encoding correctness
4.  Model smoke test

------------------------------------------------------------------------

## Run

``` bash
PYTHONPATH=. .\.venv\Scripts\uvicorn scripts.run_deckbuild_ui:app --reload --port 8003
```

------------------------------------------------------------------------

## Acceptance Criteria

-   UI launches
-   Pool renders with art
-   Cards move between buckets
-   Evaluate returns 4 predictions
-   No draft simulation logic included
