# Deckbuild Backend Deploy Workflow (Koyeb)

This repo includes a GitHub Actions workflow to deploy the backend to Koyeb:

- `.github/workflows/deploy-deckbuild-backend-koyeb.yml`

## 1. Create Koyeb API token

In Koyeb dashboard:

- Account -> API
- Create token

## 2. Add GitHub secret and variables

In GitHub repo settings -> Secrets and variables -> Actions:

- Secret (required):
  - `KOYEB_API_TOKEN`
- Variables (required):
  - `KOYEB_APP_NAME` (example: `mtga-drafthelper`)
  - `KOYEB_SERVICE_NAME` (example: `deckbuild-api`)
- Variable (optional):
  - `KOYEB_REGION` (default in workflow: `was`)

## 3. Run the workflow

- Push backend changes to `main`, or
- Actions -> `Deploy Deckbuild Backend (Koyeb)` -> `Run workflow`

The workflow creates/updates a web service with:

- Build: buildpack
- Run command:
  - `uvicorn scripts.run_deckbuild_ui:app --host 0.0.0.0 --port 8000`
- Port: `8000`

## 4. Set CORS in Koyeb service

In Koyeb service environment variables, set:

- `DECKBUILD_UI_CORS_ORIGINS`

Recommended value for your project pages:

- `https://coolnomad.github.io/MTGA_DraftHelper`

If needed, include multiple origins comma-separated.

## 5. Connect GitHub Pages UI to Koyeb backend

Use Pages URL with `api` query param:

- `https://coolnomad.github.io/MTGA_DraftHelper/?api=https://<your-koyeb-domain>`
