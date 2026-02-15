import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from hero_bot.train_state_value import _calibration_bins, _save_calibration_svg, REPORTS_DIR, MODEL_DIR
import time
import joblib

dir = Path('data/hero_chunks')
X_files = sorted(dir.glob('X_*.npy'), key=lambda p: int(p.stem.split('_')[1]))
y_files = sorted(dir.glob('y_*.npy'), key=lambda p: int(p.stem.split('_')[1]))
assert len(X_files)==len(y_files)
print('chunks', len(X_files))
X_list=[]; y_list=[]
for xf,yf in zip(X_files,y_files):
    X_list.append(np.load(xf))
    y_list.append(np.load(yf))
X = np.vstack(X_list)
y = np.concatenate(y_list)
print('loaded', X.shape, y.shape, 'mean', y.mean())
params = {
    'objective':'reg:squarederror',
    'eta':0.05,
    'max_depth':6,
    'min_child_weight':2,
    'subsample':0.8,
    'colsample_bytree':1.0,
    'lambda':0.5,
    'tree_method':'hist',
    'eval_metric':'rmse',
    'seed':1337,
}
start=time.time()
dtrain = xgb.DMatrix(X, label=y)
booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=400)
elapsed=time.time()-start
pred = booster.predict(xgb.DMatrix(X))
r2 = float(1 - ((y - pred)**2).sum() / (((y - y.mean())**2).sum()+1e-9))
rmse = float(np.sqrt(((y - pred)**2).mean()))
slope, intercept = np.polyfit(pred, y, 1)
bins = _calibration_bins(pred, y)
r2_bins = float(1 - ((bins['true_mean'] - bins['pred_mean'])**2).sum() / (((bins['true_mean'] - bins['true_mean'].mean())**2).sum()+1e-9))
rmse_bins = float(np.sqrt(((bins['true_mean'] - bins['pred_mean'])**2).mean()))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(booster, MODEL_DIR / 'state_value.pkl')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
bins_path = REPORTS_DIR / 'state_value_bins_full.csv'
svg_path = REPORTS_DIR / 'state_value_calibration_full.svg'
bins.to_csv(bins_path, index=False)
_save_calibration_svg(svg_path, bins, slope, intercept, title='State value calibration (full)')
metrics = {
    'R2': r2,
    'RMSE': rmse,
    'calibration_slope': float(slope),
    'calibration_intercept': float(intercept),
    'bins_R2': r2_bins,
    'bins_RMSE': rmse_bins,
    'bins_path': str(bins_path),
    'svg_path': str(svg_path),
    'rows': int(len(y)),
    'runtime_sec': elapsed,
}
(REPORTS_DIR / 'state_value_metrics_full.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(json.dumps(metrics, indent=2))
