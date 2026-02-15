import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from hero_bot.train_state_value import MODEL_PATH, load_state_value_model
from state_encoding.encoder import encode_state, encode_card

INPUT = Path('data/processed/bc_dataset.parquet')
OUT_MODEL = Path('hero_bot/models/hero_policy_distill.pkl')
OUT_META = Path('hero_bot/models/hero_policy_distill_meta.json')
max_rows = 50000
temperature = 0.25
seed=1337

df = pd.read_parquet(INPUT)
if len(df) > max_rows:
    df = df.sample(n=max_rows, random_state=seed)
val_model = load_state_value_model(MODEL_PATH)
if val_model is None:
    raise SystemExit('missing state value model')

X_list=[]; y_list=[]
start=time.time()
for row in df.itertuples(index=False, name='Row'):
    pool = dict(getattr(row, 'pool_counts', {}) or {})
    pack = list(getattr(row, 'pack_card_ids', []) or [])
    if not pack:
        continue
    total = sum(pool.values())
    pack_no = int(getattr(row,'pack_number', (total//15)+1))
    pick_no = int(getattr(row,'pick_number', (total%15)+1))
    base_state = encode_state(pool, pack_no=pack_no, pick_no=pick_no, skill_bucket=getattr(row,'rank', None) or getattr(row,'skill_bucket', None))
    qs=[]
    for c in pack:
        new_pool = dict(pool)
        new_pool[c] = new_pool.get(c,0)+1
        svec = encode_state(new_pool, pack_no=pack_no, pick_no=pick_no, skill_bucket=getattr(row,'rank', None) or getattr(row,'skill_bucket', None))
        q = float(val_model.predict(xgb.DMatrix(svec.reshape(1,-1)))[0]) if isinstance(val_model, xgb.Booster) else float(val_model.predict(svec.reshape(1,-1))[0])
        qs.append(q)
    probs = np.exp((np.array(qs)/max(temperature,1e-6) - np.max(np.array(qs)/max(temperature,1e-6))))
    probs = probs / probs.sum()
    for c,p in zip(pack, probs):
        X_list.append(np.concatenate([base_state, encode_card(c)]))
        y_list.append(float(p))

X = np.vstack(X_list)
y = np.array(y_list, dtype=float)
print('samples', X.shape, 'build_time', time.time()-start)
model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, n_estimators=300, random_state=seed, n_jobs=-1)
model.fit(X,y)
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, OUT_MODEL)
meta={'temperature':temperature,'max_rows':max_rows,'input':str(INPUT)}
OUT_META.write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('done', meta)
