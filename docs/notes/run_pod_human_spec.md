# run_pod_human.py – Specification for Offline Human-vs-Bot Drafts

## 1. Purpose

`run_pod_human.py` runs **one or more 8-player Limited pods** where:

- One seat is controlled by a **human player** (via CLI interaction).
- The other seats are controlled by **configurable bot policies** (hero / human-like / blends).
- Packs are generated locally.
- Drafts are simulated fully offline.
- Results (full draft logs + final decks + deck_effects) are saved to Parquet for analysis.

This script is the main entrypoint for an **offline draft-vs-bots experience**.

---

## 2. Assumptions & Dependencies

### 2.1. Repo structure (expected)

- `REPO_ROOT/`
  - `hero_bot/`
    - `models/state_value.pkl` (XGBoost critic)
    - `policy.py` (hero pick interface – to be implemented/used)
  - `human_policy/`
    - `models/human_policy.pkl` (behavior model)
    - `policy.py` (human-like pick interface – to be implemented/used)
  - `deck_eval/`
    - `evaluator.py` → `evaluate_deck(deck_counts) -> float`
  - `state_encoding/`
    - `encoder.py` → `encode_state(pool, pack_no, pick_no, skill_bucket) -> np.ndarray`
  - `data/`
    - `raw/cards.json` or similar card metadata
    - `processed/` (output logs)
  - `scripts/`
    - `run_pod_human.py` (this script)
    - `run_tournament.py` (already exists; can be reused for shared logic)

### 2.2. Policy abstraction

All bot policies must conform to:

```
class DraftPolicy:
    def pick(
        self,
        pack_card_ids: list[str],
        pool_counts: dict[str, int],
        pack_number: int,
        pick_number: int,
        seat_idx: int,
        meta: dict | None = None,
    ) -> str:
        """Return chosen card ID (slug) from pack_card_ids."""
```

---

## 3. CLI Interface

Example:

```
PYTHONPATH=. python scripts/run_pod_human.py   --num_pods 1   --human_seat 0   --bot_policies hero,hero,hero,bot_avg,bot_weak,bot_human,bot_human   --format FIN   --output data/processed/human_pods.parquet   --seed 1337
```

### Required arguments

- `--num_pods` (int, default=1)
- `--human_seat` (int, default=0)
- `--bot_policies` (str, required) — 7 comma-separated policies for non-human seats.
- `--format` (str, default="FIN")
- `--output` (str, default="data/processed/human_pods.parquet")
- `--seed` (int, default=1337)

Optional:

- `--verbose`
- `--overlay hero|human_policy|blend`

---

## 4. Draft Simulation Behavior

### 4.1. Pod structure
- 8 seats.
- 3 packs × 15 cards.
- Passing: L/R/L for packs 1–3.

### 4.2. Pack generation

Function:

```
generate_pod_packs(format, rng) -> packs[pack_no][seat_idx] = list[str]
```

### 4.3. Draft loop

For each podcast:
1. Initialize `pool_counts[seat_idx] = {}`.
2. For `pack_number in [1,2,3]`:
3. For `pick_number in [1..15]`:
   - For seat in 0..7:
     - Get pack for seat.
     - If human: prompt interactive pick.
     - Else: call policy.pick.
     - Remove chosen card.
     - Update pool.

4. After all packs:
   - Build final deck = pool.
   - Compute deck_effect with `evaluate_deck`.

---

## 5. Human Interaction (CLI)

### Display:

```
Pod 0 | Seat 0 | Pack 2 | Pick 5
Pool size: 24

Pack:
  [1] play_with_fire      hero_delta=+0.012
  [2] consider            hero_delta=+0.008
  ...
Enter pick index:
```

### Input:
- Accept int.
- Reject invalid.
- Optional: 'q' to quit gracefully.

---

## 6. Logging Schema

Each pick row has:

- `pod_id`
- `seat_idx`
- `policy_name`
- `pack_number`
- `pick_number`
- `format`
- `pack_card_ids`
- `pool_counts_before`
- `chosen_card`
- `pool_counts_after`
- `hero_value_before` (optional)
- `hero_value_after` (optional)
- `hero_delta` (optional)

### Summary file

`human_pods_summary.parquet`:

- `pod_id`
- `seat_idx`
- `policy_name`
- `pool_counts`
- `deck_effect`

---

## 7. Error Handling

- Validate `--bot_policies` has 7 entries.
- Validate `human_seat` ∈ [0,7].
- Exit gracefully on `q`.

---

## 8. Tests

1. **Bots-only smoke test**  
   - 360 picks per pod  
   - 45 cards per pool

2. **Auto-human test**  
   - `--auto_human` makes human seat auto-pick first card.

3. **Reproducibility**  
   - Same seed → identical draft.

