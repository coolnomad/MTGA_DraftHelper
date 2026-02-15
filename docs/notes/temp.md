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

2025-12-01 offline focus:
- Model quality: retrain/validate state-value and deck-effect/bump models on a representative sample; add a small hyperparam sweep, draft_id-based split, and log metrics; ensure target columns are present to avoid evaluator fallback.
- Deck builder integration: add a live /deck endpoint (using hero_bot/deck_builder.py or a fast heuristic) to suggest best deck + curve from current pool; add a smoke test.
- Data hygiene: add checks for model artifacts (deck_effect_xgb.json/meta, cards_index.json) and card mappings; add a reload for name/slug maps.
- Set data: optionally build/import set files or MTGA card DB locally to enrich metadata (colors/rarity/types).
- Tests/tooling: add quick tests for /state (mock pack/pool), deck bump calculation, and model load/predict; consider a minimal CI/script to run them.
- Performance: profile/optimize encode_state and deck scoring (cache card features, chunk/parallelize) to reduce per-row Python overhead.

2025-12-01 notes (recent Q&A):
- Human policy metrics were degenerate because the sample is tiny/imbalanced; it doesn’t use deck bump/effect at all—just pick probabilities from human choices.
- Human policy trains per skill bucket (plus ALL); you choose which bucket model to use at inference. Without specifying, you’d use ALL (average). No automatic shift unless you pick a bucket (e.g., high-skill => use the matching model).
- Hero/state-value model trained on the 5k sample: R2≈0.0763, RMSE≈0.0476 in ~68s; human policy training now saves models after fixing a pickling bug (ConstantModel moved to module scope).

Deck bump overfit concern (clarification needed):
- The `deck_effect`/`deck_bump` in bc_dataset_sample_5k_with_effect.parquet come from the fixed deck_eval model. If the concern is overfitting that model, we’d need to retrain it; otherwise, to avoid in-sample labels for hero training, we can generate out-of-fold (OOF) `deck_effect_oof`/`deck_bump_oof` via a 2-fold split: train deck-effect on half, predict the other half, swap, write OOF columns, then train state-value on those. Need confirmation whether to (1) retrain deck_eval in folds, or (2) produce OOF labels from this parquet (which would differ from shipped deck_eval).

2025-12-01 OOF deck_effect/bump:
- Added two-fold OOF generation in scripts/deck_effect_model.py; fold models saved under models/fold_models/deck_effect_xgb_foldA/B + meta.
- Generated reports/deck_effect_oof.parquet (n=77,867) with columns draft_id, deck_effect_oof, deck_bump_oof. Main deck_eval artifacts not overwritten.
- hero_bot/train_state_value.py now accepts --oof_labels and --oof_target_column to merge OOF labels (by draft_id) and train on them. Use: `python hero_bot/train_state_value.py --input data/processed/bc_dataset_sample_5k_with_effect.parquet --oof_labels reports/deck_effect_oof.parquet --oof_target_column deck_effect_oof`.

2025-12-01 hero train (OOF target):
- Command: `python hero_bot/train_state_value.py --input data/processed/bc_dataset_sample_5k_with_effect.parquet --oof_labels reports/deck_effect_oof.parquet --oof_target_column deck_effect_oof --max_rows 50000 --n_jobs -1`
- Metrics: R2=0.0121, RMSE=0.0738, n_train=31,918; n_test=8,031; model saved to hero_bot/models/state_value.pkl.
- Interpretation: training on OOF deck_effect labels avoids in-sample leakage but yields modest fit (very low R2, higher RMSE than prior in-sample run). Could try deck_bump_oof as target, larger sample, or feature/model tweaks to improve.

2025-12-01 hero train (OOF bump target):
- Command: `python hero_bot/train_state_value.py --input data/processed/bc_dataset_sample_5k_with_effect.parquet --oof_labels reports/deck_effect_oof.parquet --oof_target_column deck_bump_oof --max_rows 50000 --n_jobs -1`
- Metrics: R2=0.0606, RMSE=0.0363, n_train=31,918; n_test=8,031; model saved to hero_bot/models/state_value.pkl (overwrites prior).
- Interpretation: OOF bump target fits better than OOF deck_effect (higher R2, lower RMSE). Still modest; consider more data/hyperparam tuning if higher fidelity needed.

2025-12-01 hero train (OOF net deck_effect target):
- Command: `python hero_bot/train_state_value.py --input data/processed/bc_dataset_sample_5k_with_effect.parquet --oof_labels reports/deck_effect_oof.parquet --oof_target_column deck_effect_oof --max_rows 50000 --n_jobs -1`
- Metrics: R2=0.0121, RMSE=0.0738, n_train=31,918; n_test=8,031; calibration slope≈0.561, intercept≈0.236; bins at reports/state_value_bins_deck_effect_oof.csv; SVG at reports/state_value_calibration_deck_effect_oof.svg. Model saved to hero_bot/models/state_value.pkl (overwrites prior).
- Interpretation: Net OOF target remains weak fit (low R2, higher RMSE) and under-calibrated (slope<1). Bump target still preferable. Binned calibration R2≈0.446 (true_mean vs pred_mean).

Calibration bin R2:
- deck_bump_oof bins: R2≈0.847
- deck_effect_oof bins: R2≈0.446

2025-12-01 human policy train on 5k tensors:
- Rebuilt tensors from bc_dataset_sample_5k_with_effect.parquet: `python human_policy/bc_dataset_builder.py --input data/processed/bc_dataset_sample_5k_with_effect.parquet --output data/processed/bc_tensors_5k.parquet` (210k rows; human_pick/pack_cards slugged).
- Trained human policy on bc_tensors_5k: `PYTHONPATH=. python human_policy/train_human_policy.py --input data/processed/bc_tensors_5k.parquet`
- Metrics (top1_acc ~0.22–0.24, log_loss ~1.79–1.81):
  - ALL: top1=0.229, log_loss=1.804, n_picks=21,000 (n_train=189k/n_test=21k)
  - platinum: top1=0.219, log_loss=1.791, n_picks=7,644
  - diamond: top1=0.224, log_loss=1.783, n_picks=3,667
  - gold: top1=0.240, log_loss=1.799, n_picks=2,924
  - silver: top1=0.240, log_loss=1.785, n_picks=2,004
  - bronze: top1=0.223, log_loss=1.811, n_picks=1,059
  - mythic: top1=0.216, log_loss=1.812, n_picks=1,974
  - UNKNOWN: top1=0.229, log_loss=1.797, n_picks=1,731

2025-12-01 human policy train:
- Command: `PYTHONPATH=. python human_policy/train_human_policy.py --input data/processed/bc_tensors.parquet`
- Metrics remain degenerate on the tiny tensor sample: top1_acc=0, log_loss=Inf, n_picks=0 across buckets (UNKNOWN/bronze/silver/gold/platinum/diamond/mythic/ALL). Models saved under human_policy/models/*. This indicates insufficient/imbalanced data; need larger/cleaner tensors for meaningful human pick models.

2025-12-02 plan (hero full, chunked):
- Goal: train hero/state-value on full bc_dataset.parquet (~4.1M rows, ~97k drafts) using OOF labels (77,867 drafts in deck_effect_oof); inner-join on draft_id to drop missing. Target: deck_bump_oof.
- Method: add chunked encoder (encode_state) writing features/targets to disk in batches; train XGBoost by streaming chunks (bounded memory). Save raw+20-bin metrics and calibration plots.
- Metrics to record (raw and 20-bin): R2, RMSE, slope, intercept; save bins CSV and SVG under reports/state_value_*_full. Also record wall-clock runtime.
- Steps:
  1) Implement chunked encoding of merged df (bc_dataset ∩ deck_effect_oof), write temp chunk files.
  2) Train XGBoost on streamed chunks.
  3) Compute metrics + 20-bin calibration; save bins CSV/SVG; log runtime.

2025-12-02 hero full attempt:
- Implemented scripts/train_hero_chunked.py to merge bc_dataset with deck_effect_oof (inner join on draft_id), encode in chunks to npy, stream into XGBoost, and save metrics/bins/SVG/runtime.
- Merge size: ~3,270,414 rows, 77,867 drafts (only drafts with OOF labels).
- Run failed due to PermissionError writing first chunk to system temp (C:\Users\dimuc\AppData\Local\Temp\hero_chunks_...).
- Next step: redirect chunk output to a writable path (e.g., Temp/hero_chunks under the repo or set TMPDIR/TEMP env) and rerun the chunked training; then record metrics (raw + 20-bin R2/RMSE/slope/intercept) and runtime to reports + temp.md.

2025-12-02 hero full train (chunked, deck_bump_oof target):
- Chunk path redirected to data/hero_chunks; full merge rows=3,270,414 across 77,867 drafts; target deck_bump_oof.
- Command: `PYTHONPATH=. python scripts/train_hero_chunked.py` (batch encoder + XGBoost hist).
- Metrics (raw): R2=0.1260, RMSE=0.0345, slope=1.1010, intercept=0.00079.
- 20-bin calibration: bins_R2=0.9923, bins_RMSE=0.00111; bins at reports/state_value_bins_full.csv; SVG at reports/state_value_calibration_full.svg.
- Rows trained: 3,270,414; runtime ≈ 2,789 seconds (~46.5 minutes). Model saved to hero_bot/models/state_value.pkl.
- Tournament smoke (hero policy with new model): `PYTHONPATH=. python scripts/run_tournament.py --games 1 --policy hero` → deck_effect mean ~0.379 across 8 seats (reports/tournament_hero.parquet).

2025-12-02 roadmap status:
- Hero bot: trained on full data with OOF deck_bump_oof; strong calibration, modest R2. Model deployed in overlay and sim drafts.
- Human bot: trained on 5k tensors (0.22–0.24 top1, log_loss ~1.8); better than tiny sample but still weak; needs richer tensors/encoding for higher fidelity.
- Overlay: live pack/pool ingest, name mapping (data_cards/raw DB/17Lands), bump scores displayed; Electron on-top wrapper with drag/opacity/hotkeys.
- OOF deck effect/bump: 2-fold labels generated (reports/deck_effect_oof.parquet), fold models saved separately.
- Next/high-value items: add /deck endpoint with live deck builder; improve human policy with richer tensors; further hero tuning (hyperparams, features) if needed; integrate deck summaries into overlay; add tests/CI for /state, model load, and recommend_pick.

2025-12-02 human policy train on full bc_dataset:
- Rebuilt tensors from full bc_dataset.parquet: `PYTHONPATH=. python human_policy/bc_dataset_builder.py --input data/processed/bc_dataset.parquet --output data/processed/bc_tensors_full.parquet` (4,095,462 rows, 97,511 drafts).
- Trained human policy on bc_tensors_full: `PYTHONPATH=. python human_policy/train_human_policy.py --input data/processed/bc_tensors_full.parquet` (~23 min observed).
- Metrics (top1_acc ~0.223–0.229, log_loss ~1.797–1.800):
  - ALL: top1=0.2230, log_loss=1.7992, n_picks=409,547 (n_train=3,685,915/n_test=409,547); topk (k=1..14): [0.255, 0.358, 0.414, 0.439, 0.443, 0.433, 0.413, 0.386, 0.349, 0.306, 0.257, 0.201, 0.139, 0.072]
  - platinum: top1=0.2233, log_loss=1.7983, n_picks=147,689; topk≈[0.255,0.358,0.413,0.437,0.442,0.433,0.412,0.384,0.350,0.307,0.257,0.202,0.138,0.071]
  - diamond: top1=0.2227, log_loss=1.7985, n_picks=70,842; topk≈[0.255,0.359,0.416,0.439,0.443,0.432,0.413,0.385,0.350,0.308,0.258,0.202,0.140,0.072]
  - gold: top1=0.2250, log_loss=1.7984, n_picks=57,948; topk≈[0.253,0.356,0.413,0.437,0.443,0.434,0.414,0.386,0.349,0.307,0.258,0.199,0.138,0.070]
  - silver: top1=0.2259, log_loss=1.7999, n_picks=40,026; topk≈[0.252,0.355,0.413,0.438,0.440,0.433,0.410,0.382,0.348,0.303,0.254,0.202,0.139,0.071]
  - bronze: top1=0.2293, log_loss=1.7973, n_picks=19,770; topk≈[0.255,0.356,0.411,0.436,0.440,0.432,0.414,0.386,0.348,0.308,0.260,0.205,0.140,0.072]
  - mythic: top1=0.2224, log_loss=1.7979, n_picks=39,661; topk≈[0.256,0.361,0.416,0.439,0.441,0.431,0.409,0.382,0.349,0.305,0.259,0.202,0.139,0.070]
  - UNKNOWN: top1=0.2235, log_loss=1.8002, n_picks=33,613; topk≈[0.252,0.352,0.411,0.441,0.442,0.432,0.412,0.385,0.352,0.307,0.261,0.205,0.139,0.074]

2025-12-02 ingestion helper:
- Added scripts/build_bc_dataset_from_raw.py to build bc_dataset_full.parquet directly from draft_data_public*.csv.gz (wide one-hot pack_/pool_ columns). Slugifies card names, builds pack_card_ids/pool_counts/human_pick, keeps metadata (pack_number/pick_number/rank/buckets).

2025-12-02 run_pod_human spec -> work items:
- Need a new scripts/run_pod_human.py that runs an 8-seat draft with 1 human seat + 7 bot policies, packs generated locally (L/R/L passing), logs per-pick and summary to parquet, and optional replay.
- Define a DraftPolicy interface (pick(pack_card_ids, pool_counts, pack_number, pick_number, seat_idx, meta)) and adapters for hero, human_policy, random.
- Use DraftEnv if compatible; otherwise implement pack generation and pass/pick loop per spec. Ensure pack_card_ids are slugged.
- Add CLI per spec: --num_pods, --human_seat, --bot_policies (7 entries), --format, --output, --seed, optional verbosity/overlay and --auto_human for testing.
- Implement human CLI prompt (pack display, pool size, optional hero delta), input validation, 'q' to quit.
- Logging: per-pick schema (pod_id, seat_idx, policy_name, pack_number, pick_number, format, pack_card_ids, pool_before/after, chosen_card, optional hero_value_before/after/delta); summary parquet with deck_effect; optional replay parquet.
- Deck scoring: build deck from pool (or via deck_builder) and score with evaluate_deck/evaluate_deck_bump.
- Validation/tests: check bot policies count, human_seat bounds; add bots-only smoke and auto-human mode; ensure reproducibility with seed.

You can run end-to-end by:

  1. PYTHONPATH=. python scripts/build_bc_dataset_from_raw.py --input data/raw/draft_data_public.FIN.PremierDraft.csv.gz --output data/processed/ 
     bc_dataset_full.parquet
  2. PYTHONPATH=. python scripts/train_hero_chunked.py (uses full bc_dataset with OOF labels)
  3. PYTHONPATH=. python human_policy/train_human_policy.py --input data/processed/bc_tensors_full.parquet (after rebuilding tensors with
     bc_dataset_builder.py if needed).

How to run run_pod_human.py:
- Bots-only/auto-human smoke:
  ```
  PYTHONPATH=. .\.venv\Scripts\python.exe scripts\run_pod_human.py ^
    --num_pods 1 ^
    --human_seat 0 ^
    --bot_policies hero,hero,hero,hero,hero,hero,hero ^
    --format FIN ^
    --output data\processed\human_pods.parquet ^
    --seed 42 ^
    --auto_human
  ```
- Omit --auto_human for an interactive human prompt.
- Outputs: summary parquet at --output; picks log at <output_stem>_picks.parquet.

Plan: UI wrapper for run_pod_human with card art (Raw_ArtCropDatabase)
- Goal: point-and-click pod draft UI against bots, showing card art for pack/pool/history.
- Data: read grpId/name/ArtId from Raw_CardDatabase_*.mtga; fetch art blobs from Raw_ArtCropDatabase_*.mtga (same directory). If blobs are compressed, wrap with zlib.decompress; otherwise base64-encode directly for data URLs.
- API wrapper: expose run_pod_human core via FastAPI/Starlette.
  - POST /api/session {num_pods,human_seat,bot_policies,format,seed} -> session_id.
  - GET /api/state?session_id=... -> current pack (cards: name, grpId, art_uri, optional score), pool, pack#/pick#, pod meta.
  - POST /api/pick {session_id, card_id} -> apply human pick, run bots to next human turn.
  - Optional WS /api/stream for push updates.
- Frontend: static page (React/Vite or vanilla) served by StaticFiles.
  - Pack grid with art tiles (click to pick), pool sidebar with art thumbnails sorted by color/type, picks history, pack/pick counter.
  - Use art_uri data URLs to avoid separate file hosting.
- Asset helper: module to load/cache grpId -> {name, color, art_uri}.
  - Query Raw_CardDatabase: Cards join Localizations_enUS for names and ArtId (or equivalent) for art lookup.
  - Query Raw_ArtCropDatabase: list tables, PRAGMA columns; likely ArtCrops(ArtId, Image). Cache per grpId, preload active-set IDs to avoid per-request DB hits; optionally persist to Temp/arts_<set>.json for reuse.
- Flow: UI calls /api/state -> renders pack with art -> user clicks -> POST /api/pick -> server advances bots -> UI refreshes state.
- Validation: support --auto_human for smoke tests; ensure grpId mapping matches pack generator; include log paths in /api/state for debugging.

2025-12-02 pod UI with art:
- Added scripts/run_pod_human_ui.py (FastAPI): POST /api/session, GET /api/state, POST /api/pick; serves static UI from scripts/pod_ui.
- Added scripts/pod_assets.py: loads card names/grpIds/artIds from Raw_CardDatabase_*.mtga and extracts art from AssetBundle/<ArtId>_CardArt_*.mtga via UnityPy (optional; falls back to placeholders).
- Added static UI (scripts/pod_ui/index.html, app.js, styles.css, placeholder.svg): pack grid with art, pool sidebar, simple controls for format/seed/human seat/bots.
- Run: PYTHONPATH=. .\.venv\Scripts\uvicorn scripts.run_pod_human_ui:app --port 8002 --reload; open http://localhost:8002. Install UnityPy (added to requirements.txt) to see art; otherwise placeholders render.

Strategy: upgrade hero-bot policy
- Targets: use OOF bump labels (deck_bump_oof) to avoid leakage and improve calibration vs. deck_effect.
- Training path (full, chunked): PYTHONPATH=. .\.venv\Scripts\python.exe scripts/train_hero_chunked.py (merges bc_dataset.parquet with reports/deck_effect_oof.parquet, trains XGB on deck_bump_oof, saves hero_bot/models/state_value.pkl).
- Alternatives: scripts/train_hero_full.py (full in-memory merge, also deck_bump_oof) if RAM allows; for samples/hparam sweeps, adjust hero_bot/train_state_value.py (max_depth/eta/estimators/early stopping) and run on a capped input.
- Validation: smoke with PYTHONPATH=. .\.venv\Scripts\python.exe scripts/run_tournament.py --games 1 --policy hero to ensure model loads/scores; check reports outputs (bins/svg) if rerun with chunked trainer.
- Rollout: restart services (live_overlay, run_pod_human_ui) to pick up new state_value.pkl; keep OOF labels current if data refreshes.

Hero policy flexibility:
- Added hero_policy_soft (hero_bot/hero_policy.py) to sample picks via softmax over hero values (env HERO_TEMP for temperature, HERO_EPS for epsilon; optional top_k). Added alias to run_pod_human POLICY_MAP as hero_soft. Default greedy hero remains hero.

Hero policy change (soft/stochastic option):
- Added hero_policy_soft in hero_bot/hero_policy.py: samples picks via softmax over hero values (temperature from HERO_TEMP, epsilon from HERO_EPS, optional top_k). Falls back to evaluator if model is missing.
- run_pod_human policy map includes hero_soft; use --bot_policies hero_soft,... to enable.
- Tunables: HERO_TEMP (default 0.25), HERO_EPS (default 0), top_k param if desired.
- If full soft value distillation is needed (train a separate policy to match hero’s soft targets), add a pipeline to generate soft labels from hero Q-values and train a policy head; current change is inference-time stochasticity without retrain.

Hero policy distillation (per hero_policy_upgrade.md):
- Added scripts/train_hero_policy_distill.py: builds soft targets from hero value model (state_value.pkl), converts pack hero Q to softmax (temperature), trains XGB regressor on (state + card) -> prob, saves hero_bot/models/hero_policy_distill.pkl (+ meta).
- Added hero_policy_distill in hero_bot/hero_policy.py; uses distilled model to sample picks (softmax with HERO_DISTILL_TEMP/HERO_EPS/top_k). Wired into run_pod_human as hero_distill.
- Current run: PYTHONPATH=. .\.venv\Scripts\python.exe scripts/train_hero_policy_distill.py --input data/processed/bc_dataset_sample_5k.parquet --max_rows 2000 --temperature 0.25; outputs hero_bot/models/hero_policy_distill.pkl (meta: temperature 0.25, max_rows 2000).
- Notes: trained on a small 2k-row sample for speed; for better fidelity, rerun on larger bc_dataset slice (e.g., max_rows 50k-200k) after ensuring hero value model is fixed.

Hero picks logging (where to see hero draft paths):
- bc_dataset.parquet is human history only.
- Hero picks are logged when running sims:
  - scripts/run_pod_human.py writes <output>.parquet (summary) and <output>_picks.parquet (per-pick with pack_card_ids, pool_before/after, chosen_card, pack/pick numbers, policy_name). Example: run with --bot_policies hero,... and --auto_human to watch hero draft.
  - scripts/run_tournament.py with --log_picks writes a picks parquet alongside the output.
- To inspect hero’s draft decisions, load the *_picks.parquet emitted by these runs; that contains the pick path (pack/pool before/after, chosen_card).

Inspecting pick replays in Python (example: reports/replay_hero.parquet):
```
import pandas as pd
df = pd.read_parquet("reports/replay_hero.parquet")
print("rows:", len(df), "cols:", df.columns.tolist())

# seat 0 picks in order
seat0 = df[df["seat"] == 0].sort_values(["pack_number","pick_number"])
print(seat0[["pack_number","pick_number","chosen_card","pack_cards"]].head())

# inspect pool/cards for a single pick
row = seat0.iloc[0]
print("pack:", row.pack_cards)
print("pool before:", row.pool_counts)
print("chosen:", row.chosen_card, "deck_effect:", row.deck_effect, "deck_bump:", row.deck_bump)

# replay text
for _, pick in seat0.iterrows():
    print(f"P{pick.pack_number} Pick {pick.pick_number}: chose {pick.chosen_card}")
```
Schema: seat, pack_number, pick_number, pack_cards (list), pool_counts (dict before pick), chosen_card, deck_effect, deck_bump (after pick).
