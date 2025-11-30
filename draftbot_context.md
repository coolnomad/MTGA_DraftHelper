DraftBot current status (session log)

- Deck evaluator integration
  - Calibrated deck-effect model available: `models/deck_effect_xgb.json`, `deck_effect_meta.pkl`.
  - Deck evaluator wrapper `deck_eval/evaluator.py` loads the booster + calibration and uses `deck_eval/cards_index.json` for name→index mapping. `evaluate_deck(deck_counts)` returns calibrated deck-effect.
  - Card index built via `deck_eval/build_cards_index.py` (364 cards from model deck_cols).

- Data prep
  - `data/processed/drafts.parquet` (FIN PremierDraft) used to build BC dataset.
  - `scripts/build_bc_dataset.py` produces `data/processed/bc_dataset.parquet` (4,095,462 rows) with pack_card_ids, pool_counts (pre-pick), human_pick, user buckets.
  - Card metadata noted: `data/raw/FIN.json` (691 cards, 313 names) and `data/raw/FCA.json` (66 cards, 65 names) for pack sampling/indexing.

- Deck-effect reporting
  - `scripts/deck_effect_model.py`: R-style deck effect pipeline (OOF base_p, grouped calibration, XGBoost on bump, 2-parameter logistic overlay). Outputs reports in `reports/` and artifacts in `models/`.
  - `scripts/train_calibrated_models.py` + `scripts/evaluate_models.py`: skill/bump models, calibration plots, noise ceilings.

- DraftBot scaffolding
  - Pack sampler (`draft_env/pack_sampler.py`) uses cards_index to sample packs (uniform, no rarity logic yet).
  - Draft env (`draft_env/env.py`) with 8-seat passing; hero policy (`hero_bot/hero_policy.py`) uses calibrated evaluator; random policy stub.
  - State encoder placeholder (`state_encoding/encoder.py`).

- Next targets
  - Build real state/card encoders using card metadata (colors, cmc, rarity, tags).
  - Implement BC dataset builder for tensors and train human policies per skill bucket; add pick API.
  - Add personality wrapper, refine pack sampler with rarity/duplication, strengthen draft env, and train hero state-value model.
