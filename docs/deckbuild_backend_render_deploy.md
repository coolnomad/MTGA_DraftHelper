# Deckbuild Backend Deploy Workflow (Render)

This repo includes a GitHub Actions workflow that triggers Render deploys:

- `.github/workflows/deploy-deckbuild-backend-render.yml`

## 1. Create Render web service

Use this repo as the Render source and configure:

- Build Command:
  - `pip install -r requirements.txt`
- Start Command:
  - `uvicorn scripts.run_deckbuild_ui:app --host 0.0.0.0 --port $PORT`

Set environment variable on Render:

- `DECKBUILD_UI_CORS_ORIGINS`
  - Example:
  - `https://<your-user>.github.io`
  - or if you use project pages:
  - `https://<your-user>.github.io,https://<your-user>.github.io/<repo>`

## 2. Create Render deploy hook

In Render service settings, create a Deploy Hook and copy the URL.

## 3. Add GitHub repo secrets

In GitHub repository settings -> Secrets and variables -> Actions:

- Required:
  - `RENDER_DEPLOY_HOOK_URL` = `<your render deploy hook url>`
- Optional:
  - `RENDER_HEALTHCHECK_URL` = `<your backend health URL>`
  - Example:
  - `https://your-backend.onrender.com/api/state?session_id=invalid`
  - (any endpoint that returns quickly is fine)

## 4. Trigger behavior

The workflow triggers on push to `main` when these paths change:

- `scripts/run_deckbuild_ui.py`
- `scripts/pod_assets.py`
- `requirements.txt`
- `model/**`
- `models/**`

It can also be run manually via `workflow_dispatch`.
