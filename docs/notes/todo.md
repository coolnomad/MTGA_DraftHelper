## TODO (next steps)

- In-game overlay: render recommendations on top of Arena (e.g., Overwolf/Electron overlay, transparent always-on-top window, or RivaTuner-style), driven by the existing `/state` feed.
- Live deck view: keep a running pool from `ArenaScanner.taken_cards` and expose a `/deck` endpoint that returns best deck + curve summary (use `hero_bot/deck_builder.py` or a fast heuristic); render a compact summary in the overlay.
- Name/metadata completeness: load set files (or MTGA card DB) to show colors/rarity/type; add a cache-reload endpoint so mapping updates without restart.
- Scoring sanity: add a smoke test/endpoint to verify the state-value model is loaded; fallback to 17Lands ratings ordering when the model returns flat scores.
- Overlay polish: highlight top pick, show P#/P# progress, add hotkey to hide/show, and a manual “refresh names” control.
- Logs & resilience: surface last log update time and recent parse errors in `/state`; make log path/mapping sources configurable via UI fields.
- Tests: add a small test that injects a fake pack/pool and checks `/state` returns names and sorted recommendations.
