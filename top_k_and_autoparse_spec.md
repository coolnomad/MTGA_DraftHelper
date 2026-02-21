
# Spec: Top-K Beam Results + Auto Deck/Sideboard Parsing

## Goal

1. Allow the user to browse the Top K optimized decks from beam search.
2. Automatically parse pasted lists containing `Deck` and `Sideboard` headers.
3. Maintain compatibility with the 3-zone system:
   - Main Deck
   - Swappable Sideboard
   - Ignorable Sideboard

---

# Part 1 — Return & Browse Top K Optimized Decks

## Backend Changes

### Extend `/api/optimize_beam`

Add request parameter:

return_top_k: int (default = 5)

After beam search completes:

1. Sort final beam by rank_score descending.
2. Take top return_top_k decks.
3. Return them in response.

### Response Format

{
  "start": {...},
  "best": {...},
  "top_decks": [
    {
      "rank": 1,
      "rank_score": float,
      "predictions": [...],
      "deck_counts": {...},
      "path": [...]
    },
    {
      "rank": 2,
      ...
    }
  ],
  "trajectory": [...],
  "warnings": [...]
}

Notes:
- Deduplicate by canonical deck key before returning.
- best should equal top_decks[0].
- Do not modify locked deck automatically.

---

## Frontend Changes

### Add "Optimized Results" Panel

Elements:

- Numeric input: "Top K" (default 5)
- Dropdown/List: Deck #1 ... Deck #K
- Deck preview window
- Score preview
- Buttons:
  - "Apply This Deck"
  - "Apply Swap Path"

### Apply This Deck

Overwrites Main Deck (locked_counts) with selected deck.

### Apply Swap Path (Optional)

Sequentially apply swaps from path using existing move logic.

---

# Part 2 — Auto-Parse Deck / Sideboard Headers

## Supported Input Format

Deck
1 Card A
2 Card B

Sideboard
1 Card C
1 Card D

---

## Parsing Logic (State Machine)

Initialize:

section = "deck"

For each line:

- If matches ^deck\b (case-insensitive) → section = "deck"
- If matches ^sideboard\b → section = "sideboard"
- Ignore blank lines
- Otherwise parse card line:
  Pattern: ^(\d+)?\s*(.+)$
  Default count = 1 if missing

---

## Zone Routing

If section == "deck":
    → Main Deck (locked_counts)

If section == "sideboard":
    → Swappable Sideboard (wobble_counts)

Ignorable Sideboard remains user-managed via UI.

---

## Pool Invariant

After parsing:

pool_counts = locked_counts + wobble_counts

Ignorable cards must NOT affect pool availability.

---

## Error Handling

Return warnings:

{
  "unparsed_lines": [...],
  "unknown_cards": [...]
}

Do not fail entire request on minor parse errors.

---

# Acceptance Criteria

- User can browse Top K optimized decks.
- Selecting a deck previews it.
- Applying deck overwrites Main Deck.
- Deck/Sideboard headers automatically populate correct zones.
- Pool availability remains correct.
- No regression in beam search.

---

# Performance Notes

Beam width ≤ 10 and wobble ≤ 14 means Top K extraction is trivial.
No meaningful runtime impact expected.
