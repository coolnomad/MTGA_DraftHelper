# MTGA Draft Overlay (Electron shell)

Minimal Electron wrapper that displays the running FastAPI overlay (`scripts/live_overlay.py`) as an always-on-top, transparent window over Arena.

## Usage

1) Ensure the overlay server is running:
   ```
   .\.venv\Scripts\python.exe scripts\live_overlay.py --log "C:/Users/<you>/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"
   ```
2) Install deps and start Electron:
   ```
   cd overlay-shell
   npm install
   npm start
   ```
3) Controls:
   - Overlay URL: `OVERLAY_URL` env var (default `http://127.0.0.1:8001/overlay`).
   - Toggle click-through: `OVERLAY_TOGGLE_KEY` env var (default `F8`). When click-through is off, you can drag/resize the window; when on, clicks go to Arena.
   - Bring to front: `OVERLAY_FRONT_KEY` env var (default `Shift+F8`), forces topmost, focuses, and disables click-through to let you reposition.
   - Opacity: `OVERLAY_OPACITY` env var (default `0.9`), 0=transparent to 1=opaque.
   - Drag handle: use the top bar of the Electron window to move it.

This shell does not bundle the overlay UI; it just loads the live page from the Python server.
