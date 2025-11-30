Updated evaluator bump access:

- deck_eval/evaluator.py now exposes evaluate_bump on DeckEvaluator and a helper evaluate_deck_bump(deck_counts) to return the raw deck-effect bump (pre-calibration). evaluate_deck still returns the calibrated win-rate probability.
- Tests remain green: pytest tests/test_api_stub.py tests/test_hero_state_value.py -q.

New additions:
- Observed-pack sampling: sampler uses data/processed/observed_packs.parquet (pack_cards) when present; falls back to empirical/rarity sampler otherwise.
- Live draft advisor: UI calls /recommend_pick after each state to show ranked recommendations; deck builder still present.
- State-value training: hero_bot/train_state_value.py trains from bc_dataset.parquet (subsample default 50k), targets deck_effect, saves hero_bot/models/state_value.pkl.
- Deck builder + tournament runner: hero_bot/deck_builder.py builds 40-card lists; scripts/run_tournament.py runs multi-seat drafts, builds decks, scores deck_effect/deck_bump, and with --log_picks can log a replay stub.

Runtime estimate for hero_bot/train_state_value.py (no execution):
- bc_dataset.parquet is ~4.1M rows x 370 cols (4 row groups). Default max_rows=50k still forces a full parquet read (~10–30s) then subsampling.
- Encoding loop dominates: df.iterrows with a per-card DataFrame scan in _get_card_row across ~22k cards, repeated for every card in each pool (~15–25 cards per row). Expect several minutes to tens of minutes (roughly 10–25 min end-to-end) on a midrange CPU; using all rows would push into hours.
- XGBoost fit on 50k x ~19 features, 400 estimators, n_jobs=1 is comparatively light (~1–3 min), so not the choke point.

Bottleneck clarification:
- The choke point is dataset encoding (_build_dataset), not the XGBoost fit; the nested DataFrame scans per card per row dominate wall time. XGBoost training is modest by comparison.

Speed-up suggestions (28-core machine):
- Cache card lookups: prebuild a dict {name: row or precomputed feature vector} to avoid scanning the cards DataFrame per card in the pool.
- Vectorize encoding: replace df.iterrows with array ops; explode pools and aggregate with pandas/numpy instead of per-row Python loops.
- Parallelize encoding: chunk the sampled rows and process in parallel (joblib/dask/multiprocessing), sharing the card-feature dict read-only across workers.
- Reduce IO: stop after sampling 50k rows directly from Parquet row groups (pyarrow) instead of fully reading 4.1M rows; or persist a pre-encoded dataset.
- Faster XGBoost fit: set n_jobs=28 and consider tree_method='hist'; DMatrix can trim overhead. Model fit isn’t the choke point, but can still shrink.

Updates:
- hero_bot/train_state_value.py now has CLI args (input/max_rows/target_column/seed/test_frac/n_jobs), uses cached card features + chunked encoding, and hist-based XGBoost with n_jobs for faster training.
- Supports replay targets (pick logs with deck_effect/deck_bump), still subsamples before encoding.
- Tests: pytest tests/test_hero_state_value.py -q
