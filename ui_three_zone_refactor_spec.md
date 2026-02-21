
# Spec: UI Refactor – Three-Zone Deckbuilder + Land-Only Swap Flag

## Goal

Refactor the Stage 2 deckbuilding UI into three clean zones, enforce correct pool semantics for beam search, and improve UX layout.

---

# High-Level Changes

## New Zone Structure (Left → Right)

1. Main Deck  
2. Swappable Sideboard  
3. Ignorable Sideboard  

Window order must be:

[ Main Deck ] → [ Swappable Sideboard ] → [ Ignorable Sideboard ]

---

# Zone Definitions

## 1) Main Deck

Represents:
- Cards currently in the maindeck
- Used for evaluation and optimization

Backend field:
locked_counts

Supports:
- Min-count locks (min_locked_counts)
- Count adjustments

---

## 2) Swappable Sideboard

Represents:
- Cards eligible to be swapped into the maindeck
- These are the optimizer candidate set

Backend field:
wobble_counts

Rules:
- Must contribute to pool availability
- When swapped into main, decrement from wobble

---

## 3) Ignorable Sideboard

Represents:
- Cards excluded from optimization
- Informational only

Backend field:
ignored_counts

Rules:
- Not included in addable set
- Not included in pool limits

---

# Critical Backend Invariant

Pool availability must equal:

pool_counts = locked_counts + wobble_counts

Ignorable cards must not affect pool limits.

---

# New Feature: "Land-Only Swap" Flag

## Motivation

Certain cards (e.g., nonbasic lands) should only swap with lands.

## Backend State

Add:

land_swap_only: Dict[str, bool]

Meaning:
If True for card C:
- C may only be swapped with BASIC_LANDS

---

# Enforcement Logic

When generating swap (remove r, add a):

If land_swap_only[a] == True and r not in BASIC_LANDS:
    skip

If land_swap_only[r] == True and a not in BASIC_LANDS:
    skip

---

# UI Layout Changes

Move to LEFT panel:
- Base P User slider
- Evaluate button
- Evaluation output

Layout:

[ Base P User ]
[ Evaluate Button ]

Evaluation Output:
- base_p = 0.4 → bump
- base_p = 0.5 → bump
- base_p = 0.6 → bump
- base_p = user → bump

---

# Movement Rules

Allowed:

Main ↔ Swappable  
Swappable ↔ Ignorable  
Main → Ignorable  

Not allowed:

Ignorable → Main (must go through Swappable)

---

# API Updates

## /api/state must include:

{
  "locked": [...],
  "wobble": [...],
  "ignored": [...],
  "min_locked_counts": {...},
  "land_swap_only": {...}
}

---

## New Endpoint

POST /api/set_land_swap_only

Request:
{
  "session_id": "...",
  "card": "Card Name",
  "flag": true
}

Behavior:
- flag true → set land_swap_only[card] = True
- flag false → remove key

---

# Acceptance Criteria

- Beam search respects 3-zone structure
- Ignorable cards never used in optimization
- Land-only flags enforced
- Evaluation UI moved to left panel
- Window order is Main → Swappable → Ignorable
