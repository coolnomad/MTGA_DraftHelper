
# DraftBot Project: Schema & Requirements

## 0. Goal

Build a draft simulator + agents for MTG Arena Limited where:

- We have a **deck evaluator** `f(deck) -> deck_effect` (Δp).
- We train **human-like drafting policies** from ~100k 17Lands logs.
- We define **bot personalities** (weak, shark, rare-drafter, color-hater, etc.).
- We run **multi-agent simulations** (8 seats) where each seat uses a policy.
- We train a **hero bot** that maximizes `deck_effect` when other seats follow a realistic, mixed-strength meta.

This spec is for the CLI robot to scaffold the repo and core modules.

---

## 1. High-Level Architecture

Core modules:

1. `data/`
   - Load and normalize draft logs, card data, and deck evaluator outputs.

2. `deck_eval/`
   - Python wrapper around existing deck-effect model.
   - API: `evaluate_deck(deck_counts) -> float` (Δp or similar).

3. `state_encoding/`
   - Encode partial draft state into feature vectors for models.

4. `human_policy/`
   - Behavior-cloned policies from human logs (stratified by skill).

5. `personality_policy/`
   - Wrapper that modifies human policies into personalities via utility shaping.

6. `draft_env/`
   - Multi-agent (8-seat) draft environment with pack generation and passing.

7. `hero_bot/`
   - State-value model and/or RL policy for the hero seat.

8. `experiments/`
   - Scripts to train/evaluate:
     - human policies,
     - personality bots,
     - hero bot,
     - offline evaluation + simulated tables.

---

## 2. Data Sources & Schemas

### 2.1 Draft Logs

Assume 17Lands-style CSV/Parquet:

**File:** `draft_data_public.FIN.PremierDraft.csv`

Required per-pick fields:

- `draft_id`
- `user_id`
- `event_date`
- `rank`
- `user_win_rate_overall`
- `pack_number`
- `pick_number`
- `pack_cards` (list of card_ids)
- `pick` (chosen card)
- `cards_in_pool` or reconstructable pool state

### 2.2 Card Metadata

**File:** `cards.parquet`

Fields:

- `card_id`
- `name`
- `color_identity`
- `cmc`
- `rarity`
- `card_type`
- optional tags (`is_removal`, `is_fixing`, `curve_bin`)

### 2.3 Deck Evaluator Output

**File:** `deck_effects.parquet`

Fields:

- `draft_id`
- `deck_counts` (vector or columns)
- `deck_effect` (Δp)
- optional: `base_p`, `p_post_draft`

---

## 3. State Representation

### 3.1 Deck/Pool

- `pool_counts` vector (size = num cards)
- `picks_so_far`
- `picks_left`
- `pack_number`, `pick_number`

### 3.2 Derived Features

- color counts
- curve histogram
- removal/fixing counts
- creatures vs spells
- skill features from user metadata

### 3.3 Per-Card Features (for action scoring)

- color identity
- cmc
- rarity
- card_type
- curve_bin
- synergy flags (optional)
- color-match score
- curve-smoothing effect

---

## 4. Human Drafting Policies (Behavior Cloning)

### 4.1 Skill Buckets

Ex:
- `low` = bottom 40%
- `mid` = 40–80%
- `high` = top 20%

### 4.2 Behavior Cloning Dataset

**File:** `bc_dataset.parquet`

Fields:

- `draft_id`
- `skill_bucket`
- `pack_number`
- `pick_number`
- `pool_counts`
- `pack_card_ids`
- `chosen_card_id`
- derived features

### 4.3 Model

Per-card score model:

- logits = `score(s, c)`
- `π(a=c|s,P) = softmax(logits over cards in pack)`
- loss = cross-entropy on human pick

---

## 5. Personality Policies

Utility:

\[
U(a, s) = w_{deck} \cdot f(pool^{+a}) + w_{rare} \cdot V_{gem}(a) + w_{bias}\cdot b(a)
\]

Combined with human model:

`logits_personal = log π_human + τ * U(a,s)`

Personalities (examples):

- weak drafter (`w_deck < 0`)
- rare drafter (`w_rare >> 0`)
- red-hater (`w_bias < 0` for red cards)
- shark (`w_deck >> 0`)

Configs stored in `config/personalities.yaml`.

---

## 6. Draft Environment

`DraftEnv`:

- `reset()`
- `step_one_pick(policies)`
- `is_done()`
- `final_decks()`
- `final_values()`

Uses:

- pack sampler (`pack_sampler.py`)
- deck evaluator (`evaluate_deck(deck)`)

---

## 7. Hero Bot

### 7.1 State-Value Model g(s)

Dataset **`state_value_dataset.parquet`**:

- one row per pick:
  - state features
  - target = `deck_effect` for final deck

Train regression model:

`g(s) ≈ E[deck_effect | s]`

### 7.2 Greedy Hero Policy

At pick t:

- for each card c in pack:
  - form state s’ = updated with c
  - compute score(c) = g(s’)
- pick `argmax score(c)`

### 7.3 Mixed-Meta Environment

Define meta distribution over seat types:

- 40% low-skill human-like
- 40% mid-skill
- 15% high-skill
- 5% weird personalities

Simulate many drafts:

- hero seat uses greedy (or learned) policy
- other seats sampled from meta
- evaluate expected `deck_effect`

---

## 8. Evaluation & Metrics

### 8.1 Offline vs Humans

Compare:

- hero pick vs human pick  
- using `g(s_after_pick)` for counterfactual

Metrics:

- fraction where hero > human
- avg(g(hero) − g(human))

### 8.2 Simulated Tables

Metrics:

- hero mean deck_effect
- distribution over simulations
- archetype/color/curve distributions

---

## 9. Repository Layout

```
draftbot/
  data/
    raw/
    processed/
  deck_eval/
    evaluator.py
  state_encoding/
    encoder.py
  human_policy/
    bc_dataset_builder.py
    train_human_policy.py
  personality_policy/
    personalities.py
  draft_env/
    env.py
    pack_sampler.py
  hero_bot/
    train_state_value.py
    hero_policy.py
  experiments/
    simulate_meta.py
    eval_vs_humans.py
  config/
    personalities.yaml
    meta_distribution.yaml
  README.md
  requirements.txt
```

---

## 10. Requirements

- Python 3.10+
- pandas, numpy
- pyarrow
- scikit-learn
- xgboost
- pyyaml
- tqdm

Deck evaluator must be callable as:
`evaluate_deck(deck_counts) -> float`.

