MTGA Draft Helper — proposed execution plan

1) Data prep
- Load `data/processed/drafts.parquet`, `games.parquet`, and `decks.parquet`.
- Filter to a single expansion and `event_type == "PremierDraft"`.
- Build deck features: mean of all `deck_*` per run, compute `deck_size_avg`, multi-hot encode main/splash colors, map rank to ordered int, keep user buckets; drop incomplete runs.

2) Train/test split
- Create fixed 80/20 split with seed 1337; reuse the same indices for all models.

3) Model M1 (skill-only)
- Features: rank encoded, user_n_games_bucket, user_game_win_rate_bucket.
- Model: ridge or gradient boosting (per spec) with MSE; save to `models/skill_model.pkl`.
- Metrics: R2_skill, RMSE_skill on the shared test split.

4) Model M2 (joint deck + skill)
- Features: all deck_* means, deck_size_avg, color encodings, skill features (no n_games unless chosen as optional numeric).
- Model: gradient boosting/XGBoost with depth 6, lr 0.05, subsample 0.8; MSE loss.
- Metrics: R2_joint; compute deck incremental `delta_R2_deck = R2_joint - R2_skill`; save to `models/joint_model.pkl`.

5) Decomposition (M3)
- For each run in the test set (and optionally full data), compute `skill_pred`, `joint_pred`, `deck_boost = joint_pred - skill_pred`.
- Save table to `data/processed/decks_with_preds.parquet`.

6) Calibration (optional)
- If needed, fit isotonic or Platt scaling on validation data for M2; report calibrated vs uncalibrated metrics.

7) Validation & reports
- Summarize metrics, deck_boost distribution, and key feature importance (if supported by model) in a short report/notebook.
- Verify artifacts written to the specified paths and document reproducible commands (including seed).
