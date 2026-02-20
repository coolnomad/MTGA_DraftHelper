# Stage 2 Optimizer Specification: "Improve My Deck"

## Objective

Given:

-   `locked_counts`: current maindeck (user locked cards)
-   `wobble_counts`: candidate swap pool
-   `pool_counts`: full pool (for availability checks)
-   Unlimited basics (Island, Swamp, Forest, Mountain, Plains)
-   XGBoost model: `model/deck_bump_model.ubj`
-   Feature schema: `deck_cols.json` or `deck_cols.csv`

Produce:

-   Ranked 1-for-1 swap suggestions that improve predicted deck bump
-   Optional iterative improvement mode
-   Predictions at base_p = 0.4, 0.5, 0.6, base_p\_user

Scope: Improve-my-deck mode only. Wobble set defines addable candidates.

------------------------------------------------------------------------

## Definitions

### Zones

-   `locked`: maindeck cards
-   `wobble`: candidate swap-in cards
-   `pool`: entire pool

### Basics

Unlimited add/remove allowed: Island, Swamp, Forest, Mountain, Plains

------------------------------------------------------------------------

## Deck Size Policy

Deck must evaluate as 40 cards.

If locked deck != 40: - Warn user - Auto-complete to 40 using basics
(default behavior)

Land plausibility target: 16--18 total lands.

------------------------------------------------------------------------

## Scoring

Use existing scorer.

Return: - Tier predictions - rank_score (default = bump at
base_p\_user) - delta vs current

Rank modes: - user (default) - mean - min

------------------------------------------------------------------------

## Swap Generation

### Addable Set A

-   Non-basic cards in wobble
-   Optional basics if land optimization enabled

### Removable Set R

Manual OR auto-nominated.

------------------------------------------------------------------------

## Auto Removable Nomination (LOO Approximation)

For candidate removable cards: 1. Remove 1 copy 2. Add 1 basic to
maintain 40 3. Score delta 4. Select top R by positive delta

------------------------------------------------------------------------

## Suggest Swaps Algorithm

1.  Compute baseline deck score
2.  Determine R
3.  Determine A
4.  Generate swaps (R × A)
5.  Batch score all candidates
6.  Rank by delta
7.  Return top K

------------------------------------------------------------------------

## Output Schema

``` json
{
  "current": {
    "deck_count": 40,
    "predictions": [...],
    "rank_score": 0.0123
  },
  "removable_nominated": [
    {"card":"X","delta_remove":0.0012}
  ],
  "suggestions": [
    {
      "remove":"R",
      "add":"A",
      "delta":0.0021,
      "new_rank_score":0.0144,
      "new_predictions":[...]
    }
  ]
}
```

------------------------------------------------------------------------

## Auto Iterate Mode

Repeat best improving swap up to N steps or until no improvement.

Return: - Final deck - Applied swaps - Final predictions

------------------------------------------------------------------------

## Performance Requirements

-   Batch scoring only
-   Support \~2000 candidate swaps per request
-   Interactive latency

------------------------------------------------------------------------

## API Endpoints

POST `/api/suggest_swaps`

POST `/api/auto_iterate`

------------------------------------------------------------------------

## Acceptance Criteria

-   User clicks "Suggest Improvements"
-   Receives ranked swap suggestions
-   Wobble-only addable cards
-   Predictions returned for 4 base_p values
-   Applying swaps updates state and improves score
