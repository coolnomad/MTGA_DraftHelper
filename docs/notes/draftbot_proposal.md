---
## Iterative Learning Loop Plan

Goal: make the bots improve over time via train–evaluate–update cycles using self-play data.

1) Self-play data collection
- Run batches of multi-seat drafts with current policies (hero/BC).
- Log per-pick tuples: (state encoding, pack_cards, chosen card, pack_number, pick_number, policy name) and final outcomes per seat (deck_effect probability, deck_bump).
- Store as a replay buffer (parquet) for training.

2) Train/update models
- State-value model g(s): retrain on (state, deck_effect/deck_bump) pairs from self-play (not just historical human data). Use the latest replay buffer or a rolling window.
- Optional pick policy: train a policy to maximize expected g(s_after_pick) (e.g., advantage = g(s_after_pick) – baseline). Or distill hero greedy choices into a pick model for speed.

3) Evaluation & selection
- After each training round, run un_tournament.py (or a faster eval harness) with the updated models to get mean deck_effect/deck_bump across tables.
- Keep/checkpoint the best-performing model; track metrics over iterations.

4) Loop driver
- Add a script to alternate: self-play → train state-value (and policy, if enabled) → evaluate → checkpoint → repeat for N rounds.
- Configurable batch sizes (tables per round), replay sampling, and stopping criteria.

5) Deck builder refinement
- Improve deck construction heuristics (colors, mana base) so evaluation reflects true potential; poor builders can bottleneck scores even if picks improve.
