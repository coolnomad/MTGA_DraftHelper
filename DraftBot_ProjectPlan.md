
# DraftBot_ProjectPlan.md  
*End-to-end system for drafting MTG decks that maximize calibrated deck-effect (Δp) using 17Lands data + your calibrated evaluator.*

---

## 1. Overview

The goal is to build a full draft-simulation ecosystem that produces a **draft bot** capable of making picks that lead to decks with maximal *deck-effect uplift* (Δp).

This system uses:
- Deck evaluator (calibrated Δp model)
- Human behavior cloning
- Draft environment (8 seats)
- Hero bot maximizing expected deck-effect
- Meta simulation

---

## 2. Directory Structure

draftbot/
  deck_eval/
    evaluator.py
    cards_index.json
  data/
    raw/
      FIN.json
      draft_logs_raw.parquet
    processed/
      cards.parquet
      bc_dataset.parquet
      state_value_dataset.parquet
      deck_effects.parquet
  state_encoding/
    encoder.py
  human_policy/
    bc_dataset_builder.py
    train_human_policy.py
    models/
  personality_policy/
    personalities.py
    config/
      personalities.yaml
  draft_env/
    env.py
    pack_sampler.py
  hero_bot/
    train_state_value.py
    hero_policy.py
    models/
  experiments/
    simulate_meta.py
    eval_vs_humans.py
  configs/
    meta_distribution.yaml
    paths.yaml
  utils/
    io.py
    seed.py

---

## 3. Deck Evaluator Integration

evaluate_deck(deck_counts) → float  
Returns calibrated deck-effect Δp.  
Requires fixed card-index mapping stored in deck_eval/cards_index.json.

---

## 4. Data Preparation

Inputs:
- 17Lands draft logs
- FIN.json set metadata

Outputs:
- cards.parquet
- draft_logs_raw.parquet
- bc_dataset.parquet
- deck_effects.parquet

bc_dataset schema (one row per pick):
- draft_id
- pick_number
- pack_number
- skill_bucket
- pool_counts
- pack_card_ids
- human_pick_card_id

deck_effects schema:
- draft_id
- final_deck_counts
- deck_effect (Δp)

---

## 5. State Encoding

encoder.py must expose:
- encode_state(pool_counts, pack_no, pick_no, skill_bucket)
- encode_card(state_vec, card_id)

Features:
- pool stats: color, curve, removal, fixing
- context: pack/pick, skill
- card features: color id, cmc, rarity, type, tags

---

## 6. Human Behavior Cloning

bc_dataset_builder.py → builds tensors of (state, card) pairs.  
train_human_policy.py → trains per-skill models.

Metrics:
- cross-entropy
- top-1 / top-k
- KL divergence

---

## 7. Personality Policies

logits_personal = log_p_human + τ * utility  
utilities include deck_gain, rarity_bonus, color_avoid.  
Configs in personalities.yaml.

---

## 8. Draft Environment

env.py:
- 8 seats
- pack sampler
- step(action) returns (state, reward, done, info)
- logs full drafts

---

## 9. Hero Bot

Goal: learn g(s) = E[Δp | state].  
train_state_value.py produces model.  
hero_policy.py chooses pick maximizing g(s_after_pick).

---

## 10. Experiments

simulate_meta.py → run tables with mixed bots.  
eval_vs_humans.py → counterfactual comparison hero vs human picks.

---

## 11. Milestones

M1 – evaluator + data  
M2 – encoder + dataset  
M3 – human policies  
M4 – personalities + env  
M5 – hero g(s) + hero policy  
M6 – meta sims + reporting

---

## 12. Dependencies

- python 3.11
- numpy, pandas, pyarrow
- sklearn or xgboost or lightgbm
- pytorch (optional)
- pydantic
