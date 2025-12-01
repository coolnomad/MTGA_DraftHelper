# MTGA Draft Helper – Implementation Tasks (Models M1–M3)

Context:
- See design.md for causal spec, data schema, and modeling assumptions.

## 1. Feature builder module: src/models/features.py

Implement `src/models/features.py` with:

- `build_skill_features(decks: pd.DataFrame) -> pd.DataFrame`
  - Inputs: decks from data/processed/decks.parquet
  - Columns used:
    - rank
    - user_n_games_bucket
    - user_game_win_rate_bucket
  - Behavior:
    - Encode rank as ordered integer (Bronze < Silver < Gold < Platinum < Diamond < Mythic).
    - Coerce user_* columns to numeric.
    - Return a DataFrame with same index as decks.

- `build_deck_features(decks: pd.DataFrame) -> pd.DataFrame`
  - Inputs: decks
  - Columns used:
    - all columns starting with "deck_"
    - deck_size_avg
    - main_colors
    - splash_colors
  - Behavior:
    - Use all deck_* columns as float features.
    - Include deck_size_avg as numeric.
    - Encode main_colors and splash_colors as multi-hot for W/U/B/R/G:
      - 5 columns main_W, main_U, main_B, main_R, main_G
      - 5 columns splash_W, splash_U, splash_B, splash_R, splash_G
    - Colorless decks have all zeros in these 10 color columns.
    - Return DataFrame with same index as decks.

- `build_joint_features(decks: pd.DataFrame) -> pd.DataFrame`
  - Behavior:
    - Concatenate build_deck_features and build_skill_features along columns.

- `train_test_split_indices(n_rows: int, seed: int = 1337, test_frac: float = 0.2)`
  - Behavior:
    - Deterministic 80/20 split using given seed.
    - Return (train_idx, test_idx) as numpy arrays of row indices.

## 2. Skill-only model: src/models/skill_model.py (M1)

Implement `train_skill_model()`:

- Load decks via existing loader.
- Build X_skill via build_skill_features.
- Target y = decks["run_wr"].astype(float).
- Use train_test_split_indices with seed=1337 and test_frac=0.2.
- Model: Ridge regression (sklearn.linear_model.Ridge, alpha=1.0).
- Train on train set, evaluate on test set:
  - Report R2_skill and RMSE_skill.
- Save model to models/skill_model.pkl with joblib.

Allow running as a script:

```python
if __name__ == "__main__":
    metrics = train_skill_model()
    print(metrics)

3. Joint deck+skill model: src/models/joint_model.py (M2)

Implement train_joint_model():

Load decks.

Build X_joint via build_joint_features.

Target y = decks["run_wr"].astype(float).

Use same train/test indices from train_test_split_indices (seed=1337).

Model: gradient boosting or XGBoost with:

max_depth = 6

learning_rate = 0.05

subsample = 0.8

Train and evaluate:

Report R2_joint and RMSE_joint.

To compute ΔR2_deck:

Either:

Recompute R2_skill on the same test set using build_skill_features and Ridge, or

Load skill_model.pkl and evaluate it on the same test set.

ΔR2_deck = R2_joint − R2_skill.

Save model to models/joint_model.pkl.

Make it runnable via:
if __name__ == "__main__":
    metrics = train_joint_model()
    print(metrics)

4. Decomposition layer: src/models/decomposition.py (M3)

Implement annotate_decks_with_preds():

Load decks from data/processed/decks.parquet.

Load models/skill_model.pkl and models/joint_model.pkl.

Build X_skill and X_joint using the feature builders.

Compute:

skill_pred = skill_model.predict(X_skill)

joint_pred = joint_model.predict(X_joint)

deck_boost = joint_pred − skill_pred

Attach these as new columns on a copy of decks:

"skill_pred", "joint_pred", "deck_boost"

Save to data/processed/decks_with_preds.parquet.

Also allow running as a script:
if __name__ == "__main__":
    annotate_decks_with_preds()

5. Tests

Add pytest tests under tests/:

tests/test_features.py:

Load decks.

Check that build_skill_features, build_deck_features, build_joint_features:

return DataFrames with same number of rows as decks

have at least 1 column.

tests/test_models.py:

Call train_skill_model() and train_joint_model().

Assert that returned R2 values are between 0 and 1 (or at least > −1 and < 1).
