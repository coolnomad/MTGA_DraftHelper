Project: MTGA Draft Helper — Causal Modeling Spec

This document defines the data schema, causal assumptions, targets, and modeling plan for estimating the effect of deck composition on run win rate in MTG Arena Premier Draft (17Lands).

It serves as authoritative context for ML pipelines in this repository.

1. Data Sources / Canonical Tables

All data comes from processed parquet tables in data/processed/:

1.1 drafts.parquet — one row per pick

Columns include:

expansion (e.g., "FIN")

event_type (e.g., "PremierDraft")

draft_id

draft_time

event_match_wins, event_match_losses

pack_number, pick_number

rank

user stats:

user_n_games_bucket

user_game_win_rate_bucket

card visibility & picks:

pack_card_*

pick_*

Purpose: run-level outcomes + draft behavior.

1.2 games.parquet — one row per game

Columns include:

identifiers:

draft_id, draft_time, game_time

build_index

match_number, game_number

player features:

rank, main_colors, splash_colors

user_n_games_bucket, user_game_win_rate_bucket

opponent features:

opp_rank, opp_colors

game outcome & metadata:

won

on_play, num_mulligans, opp_num_mulligans, num_turns

deck features:

deck_*

sideboard_*

opening_hand_*, drawn_*, tutored_*

Purpose: deck compositions, per-game state, skill proxies.

1.3 decks.parquet — one row per completed run

Built by aggregating games + drafts:

draft_id

averaged deck composition: mean of all deck_* columns over all games in the run

deck_size_avg (sum of mean deck_*; filtered to >= 40)

player features:

rank

user stats: all columns containing "user"

main_colors, splash_colors

run outcome:

event_match_wins, event_match_losses

run_wr = wins / (wins + losses)

only completed runs retained (7 wins or 3 losses)

n_games (games played in the run)

Purpose: unit of analysis for causal modeling.

2. Target Definition
Outcome 
𝑊
W

run_wr (run win rate) at the run level:

𝑊
=
event_match_wins
event_match_wins
+
event_match_losses
W=
event_match_wins+event_match_losses
event_match_wins
	​

Unit of analysis: a single draft run.

We remove all incomplete runs.

2a. Causal estimand (plain-text math, single expansion)

For now we restrict analysis to a single expansion at a time (for example, only FIN).
That means the environment is effectively fixed and does not need to appear as a separate variable in the causal model.

We focus on three conceptual variables:

S = player skill (unobserved true skill, approximated by rank and user buckets)

D = deck composition (card counts and deck-level characteristics)

W = run win rate for a completed run (wins divided by total matches for that draft)

The main causal question is:

How much does changing the deck composition D change the expected run win rate W for a player of a given skill S?

To express this in plain text:

Let f(D, S) be the expected run win rate predicted by the joint model when the model is given:

a specific deck representation D

and a specific skill feature vector S

Consider two deck compositions, D1 and D2, that a player with skill features S might play.

The “deck effect” for this player when switching from D1 to D2 is:

deck_effect(D1 → D2 | S) = f(D2, S) minus f(D1, S)

In words:

“For a player with skill S, the effect of changing their deck from D1 to D2 is the difference between the predicted win rate with D2 and the predicted win rate with D1, holding skill fixed.”

Two types of quantities matter in practice:

Individual-level deck effect

For a given player (with known skill features S_i) and a specific change to their deck (from D_i to D_i_new), we evaluate:

f(D_i_new, S_i) minus f(D_i, S_i)

This is what will be used for recommendations and “what if I add/remove this card?” queries.

How much deck composition matters overall

Model M1 (skill-only) predicts W using only S, giving R2_skill.

Model M2 (joint) predicts W using both D and S, giving R2_joint.

The incremental contribution of deck composition is:

delta_R2_deck = R2_joint minus R2_skill

This is interpreted as:

“the share of variance in run win rate explained by deck composition in addition to what is already explained by player skill.”

2b. Interpretation of the three models in plain text

Model M1 (skill-only):

Input: skill features S (rank and user_* buckets).

Output: predicted run win rate that we attribute to skill alone.

This provides a baseline expected win rate for a player given their skill profile.

Model M2 (joint deck + skill):

Input: full deck representation D (deck_* card counts, deck_size_avg, encoded colors) plus skill features S.

Output: predicted run win rate that accounts for both skill and deck.

This model is used to evaluate counterfactual changes to the deck while holding skill fixed.

Model M3 (decomposition, no new training):

For each run i:

skill_pred_i = prediction from M1 using S_i only

joint_pred_i = prediction from M2 using D_i and S_i

deck_boost_i = joint_pred_i minus skill_pred_i

Interpretation:

skill_pred_i is the baseline win rate expected from the player’s skill alone.

deck_boost_i is the additional win rate (positive or negative) that this specific deck appears to provide on top of their skill baseline, according to the joint model.

No additional causal adjustment happens in M3.
All causal adjustment is handled by M1 and M2 (through conditioning on skill).
M3 only decomposes the predictions into a “skill part” and a “deck part” for interpretability and UX.

3. Exposure / Treatment
Deck Composition 
𝐷
D

Representation includes:

All card count columns: deck_*

Aggregated deck-level metrics:

deck_size_avg

main_colors, splash_colors (encoded)

n_games (for stability, optional)

These represent the “treatment” whose causal effect on win rate we seek.

4. Confounders

From DAG assumptions:

Player Skill 
𝑆
S

Observed via:

rank

user_n_games_bucket

user_game_win_rate_bucket

Skill influences:

deck construction (S → D)

win rate (S → W)

Environment 
𝐸
E

For current project:

expansion

event_type

We generally restrict to a single expansion (e.g., FIN), making 
𝐸
E effectively constant.

5. DAG

Causal structure:
     S  ------------>  W
      \               ^
       \             /
        --> D ------

And environment:
E → D
E → W

Unobserved randomness:
L → W
O → W

Where:

S: player skill

D: deck composition

W: run win rate

E: environment/format

O: opponent pool strength

L: in-game randomness

Target estimand:

𝐸
[
𝑊
∣
𝑑
𝑜
(
𝐷
=
𝑑
)
,
𝑆
=
𝑠
,
𝐸
=
𝑒
]
−
𝐸
[
𝑊
∣
𝑑
𝑜
(
𝐷
=
𝑑
′
)
,
𝑆
=
𝑠
,
𝐸
=
𝑒
]
E[W∣do(D=d),S=s,E=e]−E[W∣do(D=d
′
),S=s,E=e]

6. Modeling Strategy (Three Models)

We train three complementary models to examine skill effects, deck effects, and decompositions.

6.1 Model M1 — Skill-Only Model

Predict:

𝑊
=
𝑔
(
𝑆
)
+
𝜀
W=g(S)+ε

Features:

rank (encoded)

all user* columns

Outputs:

skill_model.pkl

metrics: R2_skill, RMSE_skill

Purpose: quantify how much win rate is explained by skill alone.

6.2 Model M2 — Joint Model (Deck + Skill)

Predict:

𝑊
=
ℎ
(
𝐷
,
𝑆
)
W=h(D,S)

Features:

all deck_* columns

deck meta (deck_size_avg, encoded main_colors, etc.)

all skill proxies

Outputs:

joint_model.pkl

metrics: R2_joint

Compute incremental deck contribution:

Δ
𝑅
deck
2
=
𝑅
joint
2
−
𝑅
skill
2
ΔR
deck
2
	​

=R
joint
2
	​

−R
skill
2
	​


Purpose: estimate the effect of deck composition on win rate after adjusting for skill.

6.3 Model M3 — Decomposition Layer

For each run 
𝑖
i:

skill_pred_i = g(S_i)

joint_pred_i = h(D_i, S_i)

deck_boost_i = joint_pred_i - skill_pred_i

This gives:

“baseline WR due to skill”

“boost due to deck given skill”

Outputs a table:

decks_with_preds.parquet

Used for:

interpretability

ranking decks/cards conditional on skill

7. Optional: Orthogonalized / DML-style Deck Model

If needed later:

cross-fit skill model 
𝑔
(
𝑆
)
g(S) → get residuals

cross-fit deck-on-skill models → residualize deck features

fit deck-only model on residualized features

Provides more formal causal identification but is not needed for v1.

8. Evaluation

Track:

R2_skill

R2_joint

ΔR2_deck

distribution of deck_boost

calibration of joint model

Cross-validation and/or fixed train/test splits (common RNG seed) should be used for comparability.

9. Future Extensions

card-level marginal effects:
counterfactuals with 
ℎ
(
𝐷
,
𝑆
)
h(D,S) while holding 
𝑆
S fixed

archetype-stratified models (e.g., by main color pair)

sequence models (using games.parquet)

contextual bandit for draft decision-time evaluation

## 10. Implementation Clarifications
10. Implementation Clarifications
10.1 Card Feature Representation

All deck_* columns are integer card counts representing the number of copies present in the player’s deck for each game.

In decks.parquet, these are averaged over all games in the run.

This is intentional: players sometimes switch builds between games.

Exposure 
𝐷
D for causal modeling is the mean composition of the 40–45 cards they actually played across the run.

10.2 Colors Encoding

main_colors and splash_colors are string fields defined by 17Lands.

Encode them as:

multi-hot binary indicators for each color: {W, U, B, R, G}

For splashes, include separate indicators: {is_splash_W, ..., is_splash_G}.

Treat “colorless” as all zeros.

10.3 Required Covariates

n_games is optional and should not be used as a confounder.

Models must use:

skill: rank (encoded), user_n_games_bucket, user_game_win_rate_bucket

deck: all deck_*, deck_size_avg, encoded colors

Everything else is ignored unless later added to the spec.

10.4 Handling Event Type / Build Index

Only retain event_type == "PremierDraft".

build_index is ignored for M1/M2; it is only used to compute averaged deck features when building decks.parquet.

10.5 Train/Test Split

Always use a fixed split:

80/20, random shuffle, seed = 1337

Same split reused for M1 and M2 to ensure:

comparable R2_skill and R2_joint

meaningful ΔR2_deck

10.6 Loss / Objective Functions

run_wr is a continuous target in 
[
0
,
1
]
[0,1].

Use MSE loss for training and report:

R2

RMSE

For calibration: post-hoc isotonic regression or Platt scaling is allowed, but optional.

10.7 Model Families

M1 (skill-only): ridge regression or gradient boosting

M2 (joint): gradient boosting (LightGBM or XGBoost) with:

max_depth: 6

learning_rate: 0.05

subsample: 0.8

regularized to avoid overfit due to high-dim card features

M3: purely derived from M1 and M2; no training occurs here.

10.8 Output Artifacts

Save models to:

models/skill_model.pkl

models/joint_model.pkl

Save decomposition table to:

data/processed/decks_with_preds.parquet

11. Further clarifications
Encoding and cleanup (colors, rank, buckets, n_games)

main_colors / splash_colors: keep them as strings and later encode them as multi-hot for W/U/B/R/G main colors plus binary splash flags.

rank: encode as an ordered categorical (e.g. Bronze < Silver < Gold < Platinum < Diamond < Mythic) mapped to integers.

user_n_games_bucket and user_game_win_rate_bucket: treat as numeric features.

n_games: optional; can be included as a numeric feature but is not required as a confounder.

Deck aggregation

Confirmed: using the mean of deck_* across all games in the run is the intended exposure.

build_index only affects how games are grouped during the canonical build; after decks.parquet is created, we do not use build_index in M1–M3.

Filters

Expansion: work on one expansion at a time (e.g., FIN only). The models are fit per expansion, so expansion does not appear as a feature in these models.

Event type: keep only event_type == "PremierDraft" for now.

Rank filters: no rank filters in v1; all ranks included.

Model families and loss

M1: ridge regression or gradient boosting on skill features with mean squared error as the loss.

M2: gradient boosting (or random forest / XGBoost) on deck + skill features with mean squared error as the loss.

Evaluation: report R2 and RMSE on an 80/20 train/test split with fixed seed 1337. Calibration is a nice-to-have but optional in v1.

Outputs

Models directory: models/skill_model.pkl and models/joint_model.pkl.

Decomposition table: data/processed/decks_with_preds.parquet.