# mtga_draft_helper

draft and deck helper for magic: the gathering arena limited formats, using 17lands data.

main components:

- **data**: load and preprocess 17lands draft/game data into canonical parquet tables.
- **models**: train composite models to predict run win rate and isolate card effects.
- **bot**: simulate drafts and implement tunable draft policies.
- **overlay**: desktop overlay to recommend picks in real time.
- **api**: local inference server to connect overlay to models.
- **tests**: unit tests for core functionality.

## quickstart

1. create and activate a virtualenv:

```bash
python -m venv .venv
# windows powershell:
.venv\Scripts\Activate.ps1
# or cmd:
.venv\Scripts\activate.bat
