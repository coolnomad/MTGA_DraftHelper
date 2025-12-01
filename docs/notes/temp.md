Updated evaluator bump access:

- deck_eval/evaluator.py now exposes evaluate_bump on DeckEvaluator and a helper evaluate_deck_bump(deck_counts) to return the raw deck-effect bump (pre-calibration). evaluate_deck still returns the calibrated win-rate probability.
- Tests remain green: pytest tests/test_api_stub.py tests/test_hero_state_value.py -q.

New additions:
- Observed-pack sampling: sampler uses data/processed/observed_packs.parquet (pack_cards) when present; falls back to empirical/rarity sampler otherwise.
- Live draft advisor: UI calls /recommend_pick after each state to show ranked recommendations; deck builder still present.
- State-value training: hero_bot/train_state_value.py trains from bc_dataset.parquet (subsample default 50k), targets deck_effect, saves hero_bot/models/state_value.pkl.
- Deck builder + tournament runner: hero_bot/deck_builder.py builds 40-card lists; scripts/run_tournament.py runs multi-seat drafts, builds decks, scores deck_effect/deck_bump, and with --log_picks can log a replay stub.

Runtime estimate for hero_bot/train_state_value.py (no execution):
- bc_dataset.parquet is ~4.1M rows x 370 cols (4 row groups). Default max_rows=50k now samples row groups and prunes columns when pyarrow is present, avoiding a full-file read.
- Encoding loop remains the dominant cost: per-row pool aggregation in Python, though now chunked and parallelized via joblib (process backend). Expect several minutes on a midrange CPU; using all rows would still be much longer.
- XGBoost fit on 50k x ~19 features, 400 estimators, n_jobs configurable (default -1) is comparatively light, so it is not the choke point.

Bottleneck clarification:
- The choke point is dataset encoding (_build_dataset), not the XGBoost fit; the nested DataFrame scans per card per row dominate wall time. XGBoost training is modest by comparison.

Speed-up suggestions (28-core machine):
- Cache card lookups: prebuild a dict {name: row or precomputed feature vector} to avoid scanning the cards DataFrame per card in the pool.
- Vectorize encoding: replace df.iterrows with array ops; explode pools and aggregate with pandas/numpy instead of per-row Python loops.
- Parallelize encoding: chunk the sampled rows and process in parallel (joblib/dask/multiprocessing), sharing the card-feature dict read-only across workers.
- Reduce IO: stop after sampling 50k rows directly from Parquet row groups (pyarrow) instead of fully reading 4.1M rows; or persist a pre-encoded dataset.
- Faster XGBoost fit: set n_jobs=28 and consider tree_method='hist'; DMatrix can trim overhead. Model fit isn't the choke point, but can still shrink.

Updates:
- hero_bot/train_state_value.py now has CLI args (input/max_rows/target_column/seed/test_frac/n_jobs), uses pyarrow row-group sampling with column pruning when available, and hist-based XGBoost with configurable n_jobs; encoding is still per-row but chunked.
- Supports replay targets (pick logs with deck_effect/deck_bump), still subsamples before encoding.
- Tests: pytest tests/test_hero_state_value.py -q

Wall-clock improvements applied in hero_bot/train_state_value.py:
- IO: pyarrow path now prunes to only needed columns (pool_counts, pick/human_pick/chosen_card, pack_number, pick_number, rank/skill_bucket, target_column) and samples row groups up to max_rows; fallback uses column-pruned pandas read when possible.
- Avoid double sampling: _build_dataset assumes the input is already capped; no second subsample.
- Encoding: rows are chunked (default 5k) and processed in parallel with joblib's process backend and n_jobs=-1 by default (falling back to threads if processes are disallowed), reducing per-row Python overhead and GIL contention.
- Targets: evaluate_deck fallback is skipped unless the target column is absent; rows without targets are dropped instead of triggering expensive deck scoring.

Remaining ideas to squeeze more time out:
- Further vectorize encoding (pool expansion + numpy aggregation) to reduce Python loops inside encode_state.
- Consider persisting a pre-encoded dataset (.npz/parquet) for repeated training runs.
- Add early stopping to XGBoost (eval_set + early_stopping_rounds) so it halts before n_estimators if the metric plateaus.

Improvements observed versus the previous version:
- IO is now column-pruned and row-group sampled, not a full read.
- Encoding is chunked and process-parallel with n_jobs=-1 by default, improving CPU utilization over the prior per-row threading.
- evaluate_deck fallback is avoided when targets exist, removing expensive deck scoring in the normal path.
- Defaults (max_rows=50k, hist tree_method, modest n_estimators) still cap dataset and model size to keep training bounded.

Test run: pytest tests/test_api_stub.py tests/test_hero_state_value.py -q (passes; 1 skipped). Pytest cache warnings remain due to existing files.

Baseline run on 5k-draft sample:
- Sample: data/processed/bc_dataset_sample_5k.parquet (210k rows across 5k draft_ids).
- Added deck_effect targets: aggregated human picks per draft_id, normalized to slug ids matching deck_eval/cards_index.json, and scored with deck_eval.evaluate_deck; wrote data/processed/bc_dataset_sample_5k_with_effect.parquet. deck_effect spread: mean 0.412, std 0.050 (min 0.220, max 0.602).
- Command (with targets): python hero_bot/train_state_value.py --input data/processed/bc_dataset_sample_5k_with_effect.parquet --max_rows 50000 --n_jobs -1
- Runtime: ~1m40s wall on this machine.
- Metrics: R2≈0.234, RMSE≈0.0438, n_train=40k, n_test=10k. Model saved to hero_bot/models/state_value.pkl.

Recommend pick endpoint smoke:
- recommend_pick now returns scored cards; UI shows top 5 with scores.
- Direct function call smoke: RecommendRequest(pack_cards=['Forest','Island','Mountain'], pool_counts={'Plains':2}) → Forest (0.4302), Mountain (0.4154), Island (0.4130). (fastapi.testclient skipped: httpx missing.)

Is this enough for a live-draft overlay?
- What’s ready: working `/recommend_pick` with state-value scoring; web UI demonstrates recommendations during a simulated draft.
- Missing for live overlay: ingesting live MTGA draft state (pack/pool/pack#/pick#) from logs or memory; an overlay surface that sits on top of the game (vs. the browser UI). You’d need a log watcher that emits the current pack/pool and calls `/recommend_pick`, plus a lightweight overlay (Electron/Overwolf/DirectX) to render the top picks without alt-tabbing.

Notes from brian_staple overlay (docs/other_tools/brian_staple/src):
- log_scanner.ArenaScanner tails Player.log (path from constants.LOG_LOCATION_WINDOWS/OSX) and detects drafts via DRAFT_START_STRINGS. After draft_start_search(), draft_data_search() parses pack/pick events for Premier v1/v2, Quick, Trad, Sealed. It tracks current_pack/pick, pack_cards (grpIds), initial_pack, picked_cards, taken_cards. Helpers: retrieve_current_pack_and_pick(), retrieve_current_pack_cards(), retrieve_current_missing_cards(), retrieve_taken_cards(). Can write Logs/DraftLog_<set>_<event>_<id>.log when enabled.
- constants.py: log locations, draft start strings, limited-type enums (Premier/Quick/Trad/Sealed) and mappings.
- Card lookup: card_logic + file_extractor + set data files; retrieve_card_data(set_data, grpId) maps grpId to card metadata; retrieve_data_sources/retrieve_set_data locate/load set files.
- To feed our recommender: poll ArenaScanner.draft_start_search(), then loop draft_data_search(); on update, collect pack_cards (grpIds) and taken_cards, map grpIds to names, build pool_counts and pack/pick numbers, call /recommend_pick and render scores in an overlay window.

Prototype status and next steps:
- Status: have a working `/recommend_pick` with ML scoring and a simulated draft UI; trained a baseline state-value model on a 5k-draft sample with deck_effect targets. We now know how to read live MTGA logs via brian_staple’s ArenaScanner (grpId-based pack/pick tracking).
- Recommendation: reuse the log scanner, but build our own lightweight overlay that calls `/recommend_pick` and renders top-N picks with scores (Electron/Overwolf/local web overlay), instead of adapting his UI.
- Next steps to reach a prototype overlay:
  1) Write a small bridge script: instantiate ArenaScanner pointing at Player.log; on updates, map grpIds→card names, build `pack_cards`/`pool_counts`/pack#/pick# payload, call `/recommend_pick`, and publish the scored list (e.g., via websocket or local HTTP).
  2) Add a minimal overlay client that subscribes to that feed and draws the top recommendations on top of the game.
  3) Optional: wire a draft_id-level split in training when we retrain; add a smoke test for `/recommend_pick` (direct call).

Prototype bridge/overlay added:
- scripts/live_overlay.py: tails Player.log via ArenaScanner (grpIds), scores with state-value model when available (fallback to evaluator/zero) and serves a tiny overlay at http://localhost:8001/overlay. Cards show as grpIds if names are unknown; intended to validate live tracking/flow even on unsupported sets.

Path to a polished draft assistant from current state:
- Validate & baseline: run training on the 5k-draft sample (`hero_bot/train_state_value.py --input data/processed/bc_dataset_sample_5k.parquet --max_rows 1000000 --n_jobs -1`) to establish runtime/metrics; add early stopping/eval_set and log R²/RMSE/time; benchmark `_build_dataset`/`encode_state`.
- Data & targets: ensure `bc_dataset.parquet` always carries the target column (`deck_effect`/`deck_bump`) to avoid evaluator fallbacks; add a data check; pick a training slice (full vs draft-sampled vs cached pre-encoded states) and persist pre-encoded features if retraining often.
- Model quality: do a small hyperparameter sweep (depth/eta/estimators/colsample) on a representative sample; train/val split by draft_id to avoid leakage; record metrics.
- Performance & reliability: finish encoding optimizations (vectorized pool aggregation, reused card encodings), keep process/thread fallback and sensible chunk sizing, add a profiling script to catch regressions; automate a smoke test for `/recommend_pick` (API + UI).
- Product integration: wire the trained model into the live advisor so it loads the latest `state_value.pkl` and returns ranked picks; verify deck builder and tournament runner use updated evaluator/state-value outputs consistently.
- Testing & tooling: expand tests (data integrity check, model load/predict, API smoke, minimal end-to-end replay), add CI for lint/tests and a “train-on-sample” sanity job to ensure training works.
- UX & delivery: polish UI (recommendation display, explanations, performance indicator); document runbooks for retrain/deploy/local advisor; package artifacts and version model weights with data snapshot notes.
- Scaling up: if needed, run full-data training overnight (4M rows) or distributed encoding; consider a pre-encoded feature store or incremental updates as new drafts arrive.

Recent work (log scan check):
- Reviewed scripts/live_overlay.py and brian_staple src/log_scanner.py to confirm Player.log resolution uses constants.LOG_LOCATION_WINDOWS (home-based) and falls back to AppData/LocalLow path.
- Verified Player.log at C:\Users\dimuc\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log exists (~3.29 MB, current mtime) and contains BotDraftDraftStatus lines (no Draft.Notify/Event_PlayerDraftMakePick yet).
- Ran a quick Python probe to summarize tail + marker counts:
  python -c "from pathlib import Path; import itertools, time; p=Path(r'C:\Users\dimuc\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log'); print('exists:', p.exists()); print('size MB:', round(p.stat().st_size/1024**2,2)); print('mtime:', time.ctime(p.stat().st_mtime)); lines=p.read_text(encoding='utf-8', errors='replace').splitlines(); print('\nLast 5 lines:'); [print(l) for l in lines[-5:]]; markers=['Draft.Notify','BotDraftDraftStatus','Event_PlayerDraftMakePick','DraftDeck']; print('\nMarker hits (first up to 3 lines):');\nfor m in markers:\n    ms=[ln for ln in lines if m in ln]; print(m, 'count', len(ms)); [print('  ', ln[:200]) for ln in ms[:3]]"

Overlay pack names note:
Those numbers are MTGA grpIds from DraftPack; we pass DummySetList to ArenaScanner, so set_data is None and we echo raw IDs. Brian resolves names via set files in Sets/ (built from local MTGA card DB + 17Lands) and passes a real SetList, so retrieve_card_data can map grpId to card metadata. To get names here, either generate the brian set files and use SetList instead of DummySetList, or add a lightweight grpId→name map in live_overlay.py by reading your local MTGA card database (Raw_CardDatabase_*.mtga/data_cards JSON) and translating before display.

Overlay naming update:
- live_overlay.py now searches for MTGA data_cards*.mtga (gzip JSON) on common install paths or an MTGA_DATA_CARDS override, loads it once, and builds a grpId→name map.
- /state now includes data_cards_path, and the overlay footer shows both log path and cards path.
- Pack cards display as Name (grpId) when names are known; recommendations include names as well. If no data_cards is found, it falls back to grpIds and notes that in the status.
- Run with --log to point at Player.log and optional MTGA_DATA_CARDS to point at a custom data_cards file.

How brian_staple maps grpIds to names:
- It builds set files under Sets/ using src/file_extractor.py (fed by MTGA’s data_cards*.mtga and 17Lands ratings). The set JSON is keyed by grpId and contains card metadata (name, images, ratings).
- limited_sets.LimitedSets loads those set files; ArenaScanner is constructed with that SetList (not a DummySetList).
- On draft updates, ArenaScanner retrieves set_data for the active set (retrieve_data_sources → retrieve_set_data) and then retrieve_card_data uses set_data["card_ratings"][grpId] to map each grpId to a card object (name, images, etc.).
- The packaged overlay (src/overlay.py) assumes Sets/ exists and LimitedSets was passed in; that’s why names appear there. With DummySetList (our overlay), set_data is None so grpIds are echoed.

Next step for names:
- Our lightweight data_cards loader didn't find any data_cards*.mtga on common paths, so names still show as grpIds. Brian's mapping requires either (a) a data_cards*.mtga path (override with MTGA_DATA_CARDS) or (b) prebuilt set files under Sets/. Without one of those sources locally (network is restricted), we can't construct grpId→name. We should point the overlay at the actual data_cards*.mtga file from your MTGA install, or copy in a set file for the current set.

How brian_staple builds the prebuilt set files:
- The builder (docs/other_tools/brian_staple/MTGA_Draft_17Lands-main/src/file_extractor.py + overlay UI) reads MTGA's local card data (Raw_CardDatabase_* and data_cards*.mtga; paths from constants/configuration like LOCAL_DATA_FILE_PREFIX_CARDS/Raw_cards_* and LOCAL_CARDS_KEY_GROUP_ID='grpid').
- It merges that with 17Lands ratings for the set to produce <SET>_<DraftType>_Data.json under Sets/ (SET_FILE_SUFFIX='Data.json', SETS_FOLDER defaults to ./Sets). Each JSON is keyed by grpId with card_ratings (name, images, ratings, etc.).
- limited_sets.LimitedSets loads those set files; ArenaScanner retrieves the matching set_data via retrieve_data_sources → retrieve_set_data so retrieve_card_data can map grpId→card info. Without those set files locally, names cannot be resolved.

17Lands cards.csv fallback added:
- Downloaded https://17lands-public.s3.amazonaws.com/analysis_data/cards/cards.csv to data/cards.csv (contains columns id, expansion, name, etc.).
- live_overlay.py now supplements grpId→name mapping with cards.csv when data_cards*.mtga is missing; /state exposes cards_csv_path and the overlay footer shows it.

Raw_CardDatabase (SQLite) lookup added:
- Found lookup_card.py uses MTGA Raw_CardDatabase_*.mtga (SQLite) at Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw; query Cards.GrpId joined to Localizations_enUS for English names.
- live_overlay.py now searches for Raw_CardDatabase_*.mtga (or MTGA_CARD_DB override), loads grpId→name from that DB if data_cards is missing/sparse, then falls back to cards.csv. /state shows card_db_path when used, and the overlay footer lists it.

Dynamic deck building plan:
- Existing: hero_bot/deck_builder.py builds 40-card lists from a pool; tournament runner uses it offline.
- Goal: keep a live pool from ArenaScanner.taken_cards, and after each pick call deck_builder (or a lighter heuristic) to suggest a best deck and curve summary.
- Integration ideas: add a /deck endpoint that returns current best deck (main/side, colors, curve, stats); overlay can show a compact summary (colors, mana curve, key cards). Compute asynchronously to avoid UI lag.
- Data needed: card names (already mapped), card colors/types (from set file or card DB), pool counts (taken_cards), and possibly ratings for tie-breakers.
- Caution: ensure performance by caching card features and running deck building every N picks or on demand (button/hotkey) rather than every poll.

Electron overlay shell added:
- Added overlay-shell/ with package.json, main.js, README.md. It opens the existing overlay URL in a transparent, always-on-top Electron window, click-through by default, toggled via F8 (configurable with OVERLAY_TOGGLE_KEY). Run server (live_overlay.py) then npm install && npm start from overlay-shell.
- Added index.html with a draggable top bar; main.js now loads this local HTML (iframe to OVERLAY_URL). Default click-through is off; toggle with F8. Opacity via OVERLAY_OPACITY. Drag the top bar to move the window.
- Added OVERLAY_FRONT_KEY (default Shift+F8) hotkey to bring the window to front, focus, and disable click-through for repositioning.

Deck bump surfaced in overlay:
- live_overlay.py now slugifies card names, maps grpIds to names, and when possible computes deck bump via deck_eval.evaluate_deck_bump using slugged pool counts. Recommendations include a bump field (shown alongside the score).
