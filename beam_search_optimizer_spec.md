# Beam Search Optimizer Spec: "Improve My Deck" (Multi‑Swap)

## Goal

Extend the existing deckbuilding backend to support a **true multi-step
optimizer** that considers **sequences of swaps** (not just 1-for-1
ranked deltas). The optimizer should operate in "improve my deck" mode:

-   **Addable set** = wobble cards (plus optional basics when enabled)
-   **Removable set** = user-specified OR auto-nominated per deck state
-   **Moves** = primarily 1-for-1 swaps
-   **Search** = **beam search** over swap sequences

Return: - best final deck found - the **swap path** to reach it - score
trajectory and per-tier predictions

Model: - `model/deck_bump_model.ubj` - schema: `deck_cols.json|csv`
(must include `base_p`)

------------------------------------------------------------------------

## Definitions

### Deck state

-   Dict `counts: Dict[str,int]` representing a 40-card evaluated deck

### Canonical key (for dedupe/cache)

-   `key = tuple(sorted(counts.items()))`

### Score

-   Compute bump predictions at base_p values:
    -   `[0.4, 0.5, 0.6, base_p_user]` (default)
-   `rank_score` = scalar used for search ranking; modes:
    -   `user` (default): bump at base_p\_user
    -   `mean`: mean bump across tiers
    -   `min`: min bump across tiers

------------------------------------------------------------------------

## Move Set

### Primary move: 1-for-1 swap

Given removable card `r` and addable card `a`: - `D' = D - 1×r + 1×a`
Constraints: - `r` must exist in deck with count \>= 1 - `a` must be
addable: - if nonbasic: availability \<= pool_count\[a\] - if basic:
unlimited (optional feature; defaults off for addable)

### Optional move: basic-to-basic tweak

If enabled, allow: - `D' = D - 1×basic_x + 1×basic_y` Only basics. Keeps
deck size constant.

------------------------------------------------------------------------

## Inputs to Optimizer

Optimizer operates on **current session**: - `locked_counts` (starting
point) - `wobble_counts` (addable set) - `pool_counts` (availability
check) - `base_p_user`

Deck size handling: - If locked != 40, build `D0` via existing "complete
to 40 with basics" policy. - Return warning if auto-completion occurred.

------------------------------------------------------------------------

## Beam Search Algorithm

Parameters (defaults): - `steps` = 8 - `beam_width` = 10 -
`top_children_per_parent` = 60 - `R` = 12 removable candidates per
parent - `rank_mode` = `user` - `include_basic_adds` = false -
`include_basic_tweaks` = false - `dedupe` = true -
`stop_if_no_improvement` = true

### Data tracked per node

Each node in the beam: - `deck_counts` - `score_current` (rank_score +
tier vector) - `path`: list of swaps applied from start - `parent_key`
(optional)

### Step 0: initialization

-   `D0` = starting evaluated deck (40 cards)
-   `score(D0)` computed
-   beam = `[Node(D0, score, path=[])]`
-   best = D0

### For each step t = 1..steps

For each node `N` in current beam:

1)  Determine removable set `R_N`:
    -   if user provided removables: filter to those present in N.deck
    -   else auto-nominate per node using LOO removal (batched):
        -   for each candidate card `c` in N.deck (sample/cap to 25):
            -   create `D_minus = N.deck - 1×c + 1×basic_fill`
            -   compute
                `delta_remove(c) = score(D_minus) - score(N.deck)`
        -   take top `R` by `delta_remove` (descending)
2)  Determine addable set `A`:
    -   `A_nonbasic` = wobble cards with count \> 0 and not basic
    -   if `include_basic_adds`: also allow the 5 basics
    -   cap optionally to `max_addable` (if needed)
3)  Generate candidate children decks:
    -   for each `r in R_N` and `a in A`:
        -   create `D_child = N.deck - r + a`
        -   validate availability and non-negativity
    -   optionally apply `basic_tweaks` generation
    -   if too many children, keep only a random subset or heuristically
        pre-filter, but the preferred control is
        `top_children_per_parent`:
        -   score all children (batched), then keep top
            `top_children_per_parent` by rank_score

After processing all beam parents: - collect all surviving children -
dedupe by canonical key (keep highest score version) - sort by
rank_score desc - new_beam = top `beam_width` - update global `best` if
any child beats it

Termination: - if `stop_if_no_improvement` and best doesn't improve in
an entire step, break early.

Return best node found (deck + path).

------------------------------------------------------------------------

## Scoring Implementation (Batching)

Implement a batch scorer:
`score_decks(decks: List[Dict[str,int]], base_ps: List[float], rank_mode: str) -> List[ScoreResult]`

Approach: 1) Build feature vectors for each deck once (base_p unset). 2)
Replicate each deck vector for each base_p tier and set base_p column.
3) Single `xgb.predict(DMatrix)` call. 4) Reshape preds to (n_decks,
n_tiers). 5) Compute `rank_score` based on rank_mode.

Required outputs per deck: - `predictions`: list of (base_p, bump) -
`rank_score`: float

Cache: - Add LRU cache keyed by `(deck_key, tuple(base_ps), rank_mode)`
to avoid re-scoring duplicates in beam search.

------------------------------------------------------------------------

## API

Add endpoint: \### POST `/api/optimize_beam` Request:

``` json
{
  "session_id": "...",
  "steps": 8,
  "beam_width": 10,
  "top_children_per_parent": 60,
  "R": 12,
  "rank_mode": "user",
  "mode_removable": "auto",
  "removable_cards": ["..."],
  "include_basic_adds": false,
  "include_basic_tweaks": false,
  "base_p_values": [0.4,0.5,0.6,0.57]
}
```

Response:

``` json
{
  "start": { "rank_score": 0.0123, "predictions": [...] },
  "best":  {
    "rank_score": 0.0159,
    "predictions": [...],
    "deck_counts": { "Card A":2, "Island":9, ... }
  },
  "path": [
    { "remove":"Card R1", "add":"Card A1", "delta":0.0012 },
    { "remove":"Card R2", "add":"Card A2", "delta":0.0008 }
  ],
  "trajectory": [
    { "step":0, "rank_score":0.0123 },
    { "step":1, "rank_score":0.0135 },
    { "step":2, "rank_score":0.0159 }
  ],
  "warnings": ["auto-filled 2 basics to reach 40"]
}
```

Frontend behavior: - show the path as an ordered list with "apply all"
button - applying the path uses existing `/api/move` calls (two moves
per swap)

------------------------------------------------------------------------

## Tests (Pytest)

1)  **Beam search correctness**

-   With a mocked scorer that assigns known scores, ensure beam returns
    the optimal multi-step path.

2)  **Dedupe**

-   Same deck reached via different paths keeps best scoring node and
    does not duplicate.

3)  **Constraints**

-   Children never exceed pool availability for nonbasics.
-   Deck size remains 40.

4)  **Caching**

-   Repeated scoring requests hit cache (can assert call counts on
    mocked predict).

------------------------------------------------------------------------

## Acceptance Criteria

-   Endpoint returns a multi-swap path that improves score versus start
    deck
-   Results are stable and fast (batch scoring + caching)
-   Wobble cards define addable candidates (plus optional basics if
    enabled)
-   Output includes per-tier predictions and a scalar rank_score
