Run order and script notes

Data prep
- `scripts/build_data.py`: top-level fetch/decompress of raw assets (MTGA data, card DBs). Run once to populate `data/raw`.
- `scripts/build_cards_parquet.py`: extracts card metadata into parquet for downstream lookups.
- `scripts/build_bc_dataset.py` / `scripts/build_bc_dataset_from_raw.py`: build the bot-competition draft dataset (`data/processed/bc_dataset.parquet`) used by pool-value training.

Deck-level models
- `scripts/train_calibrated_models.py`: trains calibrated deck effect models from historical games; outputs `models/deck_effect_xgb.json` and `models/deck_effect_meta.pkl` used by `deck_eval.evaluator`.
- `scripts/deck_effect_model.py`: helper/analysis for deck models (feature importance, calibration).

Pool models (desired for draft UI live value)
- `scripts/train_pool_models_from_games.py`: trains pool→deck effect/bump models from `data/processed/games.parquet` using real game outcomes; writes models to `hero_bot/models/deck_effect_xgb.json` and `hero_bot/models/deck_bump_xgb.json` plus metadata/plots in `reports/pool_models/`.
- `scripts/train_pool_value.py`: older surrogate pool→auto-built-deck evaluator (uses `evaluate_deck/evaluate_deck_bump`); superseded by `train_pool_models_from_games.py` for real outcomes.

Hero/agent training
- `scripts/train_hero_full.py` / `train_hero_chunked.py`: train the drafting agent.
- `scripts/train_hero_policy_distill.py`: distills hero policy for faster inference.

Simulation / UI
- `scripts/run_pod_human.py`: CLI pod draft runner; uses pool models for live projections, logs picks/pools, writes results/picks parquet.
- `scripts/run_pod_human_ui.py`: FastAPI + static UI for interactive drafting; surfaces projected deck effect/bump on pack cards.
- `scripts/run_tournament.py`: runs multi-pod tournaments between policies.
- `scripts/live_overlay.py`: overlay for in-game display (deck predictions).

Artifacts to expect after training
- Deck model: `models/deck_effect_xgb.json`, `models/deck_effect_meta.pkl`.
- Pool models: `hero_bot/models/deck_effect_xgb.json`, `hero_bot/models/deck_bump_xgb.json`, metadata/plots in `reports/pool_models/`.
