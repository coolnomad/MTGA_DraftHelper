
# Spec: Min-Count Locks (Not-Swappable Cards) for Stage 2 Optimization

## Goal

Add a **min-count lock** feature to the deckbuilding + optimization UI so the user can prevent certain cards from being removed (fully or partially) during optimization.

This is a **constraint** on removable choices, not a scoring heuristic.

---

## Core Concept

Maintain a per-card minimum count constraint:

- `min_locked_counts: Dict[str, int]`

Meaning:
- During optimization, the system must never reduce a card `c` below `min_locked_counts[c]` copies in the maindeck (`locked_counts`).

Examples:
- If locked has `2 Ice Magic` and user locks min=2, optimizer cannot remove Ice Magic at all.
- If user sets min=1, optimizer may remove at most 1 copy (cannot go below 1).

Default:
- If a card is not in `min_locked_counts`, its min is 0 (fully swappable).

---

## Backend: Session State Changes

Extend the session object/state to include:

- `min_locked_counts: Dict[str, int]`

Initialize:
- empty dict `{}`

Include this field in `/api/state` responses so the frontend can render lock indicators.

---

## Backend: API Endpoints

### 1) Set/Update a Min Lock

`POST /api/set_min_lock`

Request:
```json
{
  "session_id": "...",
  "card": "Ice Magic",
  "min_count": 2
}
```

Rules:
- `min_count` must be an integer >= 0.
- If `min_count == 0`, remove the key from the dict (unlock).
- If card is not currently in `locked_counts`, allow storing the constraint anyway (future-proof), but UI typically only sets locks for locked cards.
- Optional safety: clamp `min_count` to current `locked_counts[card]` when toggling “lock all copies”.

Response:
- updated session state

### 2) Convenience Toggle: Lock All Current Copies (Optional)

`POST /api/toggle_lock_all`

Request:
```json
{
  "session_id": "...",
  "card": "Ice Magic",
  "lock": true
}
```

Behavior:
- if `lock=true`: set `min_count = locked_counts[card]`
- if `lock=false`: set `min_count = 0`

Response:
- updated session state

---

## Backend: Enforcing Constraints

### 1) Removable Candidate Filtering

Everywhere removable candidates are generated (manual or auto):

A card `r` is removable iff:
- `locked_counts[r] >= 1`
- `locked_counts[r] - 1 >= min_locked_counts.get(r, 0)`

Equivalent:
```python
allowed = locked_counts[r] > min_locked_counts.get(r, 0)
```

Apply this filter in:
- `/api/suggest_swaps`
- `optimize_beam()` removable nomination
- beam expansion when generating children

### 2) Auto-Nominated Removables (LOO Removal)

When nominating `R` cards via remove-one deltas:
- skip any candidate card where `locked_counts[c] <= min_locked_counts.get(c,0)`
- only score legal remove-one candidates

If all candidates are locked and no removables exist:
- return an empty suggestions list
- include warning: `"No removable cards available due to min locks"`

### 3) Beam Expansion

When generating children `D_child = D - r + a`:
- enforce same check: `D[r] > min_locked_counts.get(r,0)` before creating the child
- this prevents violating locks anywhere in the search

---

## Frontend UI Requirements

### Locked Deck Window Enhancements

For each card row in the locked deck list:

1) Display current count and a lock control:
- A “lock” toggle (pin icon or checkbox): **lock all current copies**
- A numeric input: **min count** (0..current count)

Recommended UI behavior:
- Toggle ON sets min to current count
- Toggle OFF sets min to 0
- Numeric input sets exact min; if user sets min > current count, clamp to current count

2) Visual indicator:
- show a lock badge if `min_count > 0`
- show min value: e.g., “min: 2”

### Wobble/Pool Windows
No lock controls needed.

---

## Interaction With Existing Optimization

### Suggest Swaps
- All swap suggestions must respect min locks.
- If user manually selects removables, backend must still filter them by min locks.

### Beam Search Optimization
- Must never return a path that violates min locks at any step.

---

## State Serialization

Extend `/api/state` response with:

```json
{
  "min_locked_counts": {
    "Ice Magic": 2,
    "Zack Fair": 1
  }
}
```

Frontend uses this to render lock UI state.

---

## Tests (Pytest)

1) **Lock prevents removal**
- locked: {"Ice Magic":2}
- set min=2
- ensure no suggestion removes Ice Magic

2) **Partial lock**
- locked: {"Ice Magic":2}
- min=1
- ensure suggestions may remove at most 1 copy (never below 1)

3) **Beam respects locks**
- create scenario where best path would remove locked card
- ensure optimizer does not produce illegal path; returns next-best or none

4) **Manual removable filtering**
- user provides removable list containing locked-at-min cards
- backend filters them out

---

## Acceptance Criteria

- User can set per-card min count locks from the locked deck UI.
- Swap suggestions and beam optimizer never violate min locks.
- Locks persist in session state and are reflected in UI.
- When all cards are locked, optimizer returns no suggestions with a clear warning.
