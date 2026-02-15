import joblib, json
from pathlib import Path
model = joblib.load(Path('hero_bot/models/hero_policy_distill.pkl'))
print(type(model))
print(model.get_params())
meta = json.loads(Path('hero_bot/models/hero_policy_distill_meta.json').read_text())
print('meta', meta)
