# pool-only deckbuilder ui spec (new script)

## goal
create a **new** python script (do not modify existing draft tournament + deckbuild script) that runs a local web ui where:

1) the app selects a **real draft pool** at random from `final_pools.parquet`
2) the user builds a **40-card main deck** from that pool plus **unlimited basic lands**
3) the user submits the deck for scoring by the existing deck model (same scoring logic as current project)
4) the app displays:
   - current main deck list + counts
   - sideboard/pool list + counts
   - basic lands picker + counts
   - model score output (bump, p_hat, EV, whatever the model currently returns)

## non-goals
- no drafting, no bots, no packs, no pick-by-pick logic
- no attempting to generate synthetic pools
- no new modeling/training work; this is purely ui + wiring to existing scorer

---

## inputs / data
### pool source
- file: `final_pools.parquet`
- each row represents one **final pool**
- The column names are the **card names** in the pool
- the non-NA entries in a row are counts of the corresponding cards in the pool
- duplicates across columns represent multiple copies (each cell = one copy)
- rows may contain non-card columns (ids, metrics). treat only **string** cells as card names.

### basic lands
- unlimited: `Plains`, `Island`, `Swamp`, `Mountain`, `Forest`
- user can add any number (integer >= 0) to main deck

### model scorer
- reuse the project’s existing scoring machinery (same functions/classes used by the current human ui)
- scoring input should be “deck counts by card name”, including basics
- missing cards (not in model vocab) should be ignored and reported (count of ignored cards + examples)
Reference scripts/run_pod_human_ui.py for how to display card art and mouse enabled selection.
---

## new script deliverable
create a new script file, for example:

- `scripts/run_pool_human_ui.py`

it should be runnable like:

```bash
python scripts/run_pool_human_ui.py --set FIN --port 8001 --pools final_pools.parquet --seed 1337
```
