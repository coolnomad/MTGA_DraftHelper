# Deckbuild UI on GitHub Pages

## What this setup does

- Publishes `scripts/deckbuild_ui` to GitHub Pages via:
  - `.github/workflows/deploy-deckbuild-ui-pages.yml`
- Rewrites asset paths for Pages hosting:
  - `/static/styles.css` -> `styles.css`
  - `/static/app.js` -> `app.js`

## Before using the site

GitHub Pages can host only the static frontend.  
You still need a hosted backend API for endpoints like `/api/session`, `/api/load_pool`, `/api/optimize_beam`, etc.

## Configure the backend URL in the UI

In the top bar:
- Set `API Base` to your backend URL, for example:
  - `https://your-backend.example.com`
- Click `Save`.

You can also pass it via query string:
- `?api=https://your-backend.example.com`

## CORS

`scripts/run_deckbuild_ui.py` enables CORS via `DECKBUILD_UI_CORS_ORIGINS`.

- Default: `*`
- Recommended for production:
  - `DECKBUILD_UI_CORS_ORIGINS=https://<your-user>.github.io`
  - or for project pages:
  - `DECKBUILD_UI_CORS_ORIGINS=https://<your-user>.github.io,https://<your-user>.github.io/<repo>`
