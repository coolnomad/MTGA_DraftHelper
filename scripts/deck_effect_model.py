"""
Deck effect pipeline aligned with the R implementation.

Steps:
1) Load game-level data (Premier Draft) and compute draft-level stop-rule stats (w_stop, l_stop, A, B, p_mle, p_post).
2) Compute out-of-fold hierarchical baseline base_p over (user_game_win_rate_bucket, user_n_games_bucket).
3) Fit grouped logistic calibration of base_p vs p_mle to obtain base_p_cal.
4) Merge deck features (deck_* means) and build features [base_p_cal + deck_*].
5) Split stratified by base_p_cal into train/val/test; train XGBoost on target bump = p_post - base_p_cal with weights = A+B.
6) Fit 2-parameter logistic calibration of deck effect on the validation set.
7) Evaluate on test: net_pred = calibrated base + deck effect; bump_pred = net_pred - base_p_cal; report metrics, calibration plots, and save artifacts.

Outputs:
- models/deck_effect_xgb.json
- models/deck_effect_meta.pkl (theta0/theta1, calibration params)
- reports/deck_effect_metrics.json, deck_effect_bins_net.csv, deck_effect_bins_boost.csv
- reports/deck_effect_calibration_net.svg, deck_effect_calibration_boost.svg
- reports/deck_effect_report.pdf
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.data.loaders import load_games, load_decks  # noqa: E402

# OOF output path
OOF_PATH = REPO_ROOT / "reports" / "deck_effect_oof.parquet"


W_STOP = 7
L_STOP = 3
ALPHA = 0.5
BETA = 0.5
K_FOLDS = 5
DEFAULT_K = 40.0
SEED = 1337

MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def clip01(p, eps=1e-9):
    return np.minimum(np.maximum(p, eps), 1 - eps)


def coerce_won01(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    if np.issubdtype(series.dtype, np.number):
        return (series.to_numpy() != 0).astype(int)
    s = series.astype(str).str.lower()
    return s.isin(["true", "t", "1", "yes"]).astype(int).to_numpy()


def run_nll(p: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    p = clip01(p)
    return float(-np.mean(A * np.log(p) + B * np.log(1.0 - p)))


def posterior_mean_by_draft(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["draft_id", "won", "match_number"]].copy()
    df["won01"] = coerce_won01(df["won"])
    out = []
    for draft_id, g in df.sort_values("match_number").groupby("draft_id"):
        won01 = g["won01"].to_numpy()
        cw = np.cumsum(won01)
        cl = np.cumsum(1 - won01)
        idx_candidates = np.where((cw >= W_STOP) | (cl >= L_STOP))[0]
        if len(idx_candidates) > 0:
            idx = idx_candidates[0]
            w_stop = int(cw[idx])
            l_stop = int(cl[idx])
            if l_stop >= L_STOP:
                a_post = ALPHA + w_stop
                b_post = BETA + L_STOP
            else:
                a_post = ALPHA + W_STOP
                b_post = BETA + l_stop
        else:
            w_stop = int(cw[-1])
            l_stop = int(cl[-1])
            a_post = ALPHA + w_stop
            b_post = BETA + l_stop
        p_post = a_post / (a_post + b_post)
        out.append({"draft_id": draft_id, "w_stop": w_stop, "l_stop": l_stop, "p_post": p_post})
    return pd.DataFrame(out)


def derive_k_by_gp(gp_series: pd.Series, k_hi: float = 80.0, k_lo: float = 15.0) -> Dict[str, float]:
    lev = sorted(set(gp_series.astype(str)))
    try:
        lev_num = np.array(lev, dtype=float)
        lev = [lev[i] for i in np.argsort(lev_num)]
    except ValueError:
        pass
    t = np.linspace(0.0, 1.0, num=len(lev))
    k_vals = (1.0 - t) * k_hi + t * k_lo
    return dict(zip(lev, k_vals))


def oof_base_p_two_buckets(
    df: pd.DataFrame,
    draft_col: str = "draft_id",
    won_col: str = "won",
    wr_col: str = "user_game_win_rate_bucket",
    gp_col: str = "user_n_games_bucket",
    k_by_gp: Dict[str, float] | None = None,
    default_k: float = DEFAULT_K,
    seed: int = SEED,
    k_folds: int = K_FOLDS,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draft = df[draft_col].astype(str).to_numpy()
    won01 = coerce_won01(df[won_col])
    wr = df[wr_col].astype(str).to_numpy()
    gp = df[gp_col].astype(str).to_numpy()
    key = np.char.add(np.char.add(wr, "||"), gp)

    if k_by_gp is None:
        k_by_gp = derive_k_by_gp(df[gp_col])

    unique_drafts = np.unique(draft)
    fold_ids = np.arange(k_folds)
    rng.shuffle(unique_drafts)
    draft_to_fold = {d: fold_ids[i % k_folds] for i, d in enumerate(unique_drafts)}
    fold = np.array([draft_to_fold[d] for d in draft])

    base_hat = np.zeros(len(df))
    for k in fold_ids:
        tr = fold != k
        va = fold == k
        wr_tr = wr[tr]
        key_tr = key[tr]
        won_tr = won01[tr]

        wr_levels, wr_idx = np.unique(wr_tr, return_inverse=True)
        W_wr = np.bincount(wr_idx, weights=won_tr, minlength=len(wr_levels))
        N_wr = np.bincount(wr_idx, minlength=len(wr_levels))
        base_wr = (W_wr + ALPHA) / (N_wr + ALPHA + BETA)
        base_wr_map = dict(zip(wr_levels, base_wr))

        joint_levels, joint_idx = np.unique(key_tr, return_inverse=True)
        W_joint = np.bincount(joint_idx, weights=won_tr, minlength=len(joint_levels))
        N_joint = np.bincount(joint_idx, minlength=len(joint_levels))
        wr_of_key = np.array([jk.split("||")[0] for jk in joint_levels])
        gp_of_key = np.array([jk.split("||")[1] for jk in joint_levels])
        m0_wr = np.array([base_wr_map[w] for w in wr_of_key])
        k_gp_vec = np.array([k_by_gp.get(g, default_k) for g in gp_of_key])
        base_joint = (W_joint + k_gp_vec * m0_wr + ALPHA) / (N_joint + k_gp_vec + ALPHA + BETA)
        base_joint_map = dict(zip(joint_levels, base_joint))
        base_global = (W_joint.sum() + ALPHA) / (N_joint.sum() + ALPHA + BETA)

        wr_va = wr[va]
        key_va = key[va]
        bh = np.empty(len(key_va))
        for i, (kv, wv) in enumerate(zip(key_va, wr_va)):
            if kv in base_joint_map:
                bh[i] = base_joint_map[kv]
            elif wv in base_wr_map:
                bh[i] = base_wr_map[wv]
            else:
                bh[i] = base_global
        base_hat[va] = bh
    return clip01(base_hat)


@dataclass
class GroupedCalibration:
    theta0: float
    theta1: float
    gamma: Dict[str, float]
    phi: Dict[str, float]
    gp_levels: list


def fit_basep_logit_cal_grouped(d_draft: pd.DataFrame) -> GroupedCalibration:
    s = logit(clip01(d_draft["base_p"].to_numpy()))
    gp = d_draft["gp_bucket"].astype(str)
    gp_levels = sorted(gp.unique())
    gp_cat = pd.Categorical(gp, categories=gp_levels)
    Xgp = pd.get_dummies(gp_cat, drop_first=False)
    G = Xgp.shape[1]
    A = d_draft["A"].to_numpy().astype(float)
    B = d_draft["B"].to_numpy().astype(float)

    def nll_from_par(par: np.ndarray) -> float:
        theta0 = par[0]
        theta1 = par[1]
        gamma_free = par[2 : 2 + (G - 1)]
        phi_free = par[2 + (G - 1) : 2 + 2 * (G - 1)]
        gamma = np.concatenate(([0.0], gamma_free))
        phi = np.concatenate(([0.0], phi_free))
        gamma_vec = Xgp.to_numpy() @ gamma
        phi_vec = Xgp.to_numpy() @ phi
        z = s + theta0 + theta1 * s + gamma_vec + phi_vec * s
        p = expit(z)
        return run_nll(p, A, B)

    par0 = np.zeros(2 + 2 * (G - 1))
    opt = minimize(nll_from_par, par0, method="BFGS")
    par = opt.x
    theta0 = float(par[0])
    theta1 = float(par[1])
    gamma_free = par[2 : 2 + (G - 1)]
    phi_free = par[2 + (G - 1) : 2 + 2 * (G - 1)]
    gamma = np.concatenate(([0.0], gamma_free))
    phi = np.concatenate(([0.0], phi_free))
    gamma_map = dict(zip(Xgp.columns.tolist(), gamma))
    phi_map = dict(zip(Xgp.columns.tolist(), phi))
    return GroupedCalibration(theta0=theta0, theta1=theta1, gamma=gamma_map, phi=phi_map, gp_levels=gp_levels)


def apply_basep_logit_cal_grouped(base_p: np.ndarray, gp_bucket: pd.Series, cal: GroupedCalibration) -> np.ndarray:
    s = logit(clip01(base_p))
    gp = gp_bucket.astype(str)
    gp_cat = pd.Categorical(gp, categories=cal.gp_levels)
    X = pd.get_dummies(gp_cat, drop_first=False)
    gamma = np.array([cal.gamma.get(col, 0.0) for col in X.columns])
    phi = np.array([cal.phi.get(col, 0.0) for col in X.columns])
    gamma_vec = X.to_numpy() @ gamma
    phi_vec = X.to_numpy() @ phi
    z = s + cal.theta0 + cal.theta1 * s + gamma_vec + phi_vec * s
    return clip01(expit(z))


def stratified_split(base_p: np.ndarray, seed: int = SEED, val_frac: float = 0.15, test_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    quantiles = np.quantile(base_p, np.linspace(0, 1, 11))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    strata = np.digitize(base_p, quantiles, right=True)
    idx = np.arange(len(base_p))
    train_idx, val_idx, test_idx = [], [], []
    for s in np.unique(strata):
        mask = strata == s
        ids = idx[mask]
        if len(ids) == 0:
            continue
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(np.ceil(val_frac * n)))
        n_test = max(1, int(np.ceil(test_frac * (n - n_val))))
        val_idx.extend(ids[:n_val])
        test_idx.extend(ids[n_val : n_val + n_test])
        train_idx.extend(ids[n_val + n_test :])
    return np.array(train_idx, int), np.array(val_idx, int), np.array(test_idx, int)


def calibration_bins(pred: np.ndarray, obs: np.ndarray, weights: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({"pred": pred, "obs": obs, "w": weights})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    agg = (
        df.groupby("bin")
        .apply(
            lambda g: pd.Series(
                {
                    "pred_mean": np.average(g["pred"], weights=g["w"]),
                    "obs_mean": np.average(g["obs"], weights=g["w"]),
                    "count": g.shape[0],
                    "weight_sum": g["w"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return agg


def weighted_regression(pred: np.ndarray, obs: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    w = weights / (weights.sum() + 1e-9)
    x_bar = np.sum(w * pred)
    y_bar = np.sum(w * obs)
    cov = np.sum(w * (pred - x_bar) * (obs - y_bar))
    var = np.sum(w * (pred - x_bar) ** 2)
    slope = cov / (var + 1e-9)
    intercept = y_bar - slope * x_bar
    y_hat = intercept + slope * pred
    r2 = 1 - np.sum(w * (obs - y_hat) ** 2) / (np.sum(w * (obs - y_bar) ** 2) + 1e-9)
    rmse = float(np.sqrt(np.sum(w * (obs - y_hat) ** 2)))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(r2),
        "RMSE": rmse,
    }


def save_calibration_svg(bins: pd.DataFrame, path: Path, title: str, xlabel: str, ylabel: str, reg: Dict[str, float]):
    width, height = 700, 500
    margin = 60
    xmin = float(min(bins["pred_mean"].min(), bins["obs_mean"].min()))
    xmax = float(max(bins["pred_mean"].max(), bins["obs_mean"].max()))
    padding = 0.02
    x0, x1 = xmin - padding, xmax + padding
    y0, y1 = x0, x1

    def sx(x):
        return margin + (x - x0) / (x1 - x0 + 1e-9) * (width - 2 * margin)

    def sy(y):
        return height - margin - (y - y0) / (y1 - y0 + 1e-9) * (height - 2 * margin)

    circles = [
        f'<circle cx="{sx(r.pred_mean):.2f}" cy="{sy(r.obs_mean):.2f}" r="4" fill="#1f77b4" />'
        for r in bins.itertuples()
    ]
    line_ideal = f'<line x1="{sx(x0):.2f}" y1="{sy(y0):.2f}" x2="{sx(x1):.2f}" y2="{sy(y1):.2f}" stroke="#888" stroke-dasharray="4 2" />'
    y_start = reg["intercept"] + reg["slope"] * x0
    y_end = reg["intercept"] + reg["slope"] * x1
    line_reg = f'<line x1="{sx(x0):.2f}" y1="{sy(y_start):.2f}" x2="{sx(x1):.2f}" y2="{sy(y_end):.2f}" stroke="#d62728" stroke-width="1.5" />'
    reg_text = f'slope={reg["slope"]:.3f}, intercept={reg["intercept"]:.3f}, R2={reg["R2"]:.3f}, RMSE={reg["RMSE"]:.3f}'
    text = [
        f'<text x="{width/2:.1f}" y="20" text-anchor="middle" font-family="Helvetica" font-size="16">{title}</text>',
        f'<text x="{width/2:.1f}" y="{height-15}" text-anchor="middle" font-family="Helvetica" font-size="12">{xlabel}</text>',
        f'<text x="15" y="{height/2:.1f}" transform="rotate(-90 15,{height/2:.1f})" text-anchor="middle" font-family="Helvetica" font-size="12">{ylabel}</text>',
        f'<text x="{width/2:.1f}" y="{height-30}" text-anchor="middle" font-family="Helvetica" font-size="11" fill="#d62728">{reg_text}</text>',
    ]
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            line_ideal,
            line_reg,
            *circles,
            *text,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


def write_simple_pdf(path: Path, lines: List[str]):
    safe = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    content_lines = []
    y = 770
    for line in safe:
        content_lines.append(f"BT /F1 12 Tf 50 {y} Td ({line}) Tj ET")
        y -= 16
    stream = "\n".join(content_lines)
    objects = []
    objects.append("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    objects.append("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    objects.append(
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj"
    )
    objects.append("4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    objects.append(f"5 0 obj << /Length {len(stream.encode('utf-8'))} >> stream\n{stream}\nendstream endobj")
    xref_positions = []
    parts = ["%PDF-1.4"]
    for obj in objects:
        xref_positions.append(sum(len(p.encode("utf-8")) + 1 for p in parts))
        parts.append(obj)
    xref_start = sum(len(p.encode("utf-8")) + 1 for p in parts)
    parts.append("xref")
    parts.append(f"0 {len(objects)+1}")
    parts.append("0000000000 65535 f ")
    for pos in xref_positions:
        parts.append(f"{pos:010d} 00000 n ")
    parts.append("trailer")
    parts.append(f"<< /Size {len(objects)+1} /Root 1 0 R >>")
    parts.append("startxref")
    parts.append(str(xref_start))
    parts.append("%%EOF")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for part in parts:
            f.write(part.encode("utf-8"))
            f.write(b"\n")


def main():
    games = load_games()
    decks = load_decks()
    _, _, metrics = train_eval_save(games, decks)
    return metrics


def train_eval_save(
    games: pd.DataFrame,
    decks: pd.DataFrame,
    booster_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
):

    # stop-rule stats
    post_tbl = posterior_mean_by_draft(games)

    # OOF baseline
    base_p = oof_base_p_two_buckets(games)
    games = games.assign(base_p=base_p)

    # draft-level baseline table
    draft_base = (
        games.groupby("draft_id")
        .agg(base_p=("base_p", "mean"), gp_bucket=("user_n_games_bucket", "first"))
        .reset_index()
        .merge(post_tbl, on="draft_id", how="left")
    )
    draft_base["A"] = np.minimum(draft_base["w_stop"].to_numpy(), W_STOP)
    draft_base["B"] = np.minimum(draft_base["l_stop"].to_numpy(), L_STOP)
    draft_base["p_mle"] = draft_base["A"] / (draft_base["A"] + draft_base["B"])

    # grouped calibration of baseline
    cal_base = fit_basep_logit_cal_grouped(draft_base)
    draft_base["base_p_cal"] = apply_basep_logit_cal_grouped(
        draft_base["base_p"].to_numpy(), draft_base["gp_bucket"], cal_base
    )

    # merge deck features (use deck_* from decks.parquet)
    deck_cols = [c for c in decks.columns if c.startswith("deck_")]
    decks_sub = decks[["draft_id"] + deck_cols].copy()
    decks_sub["draft_id"] = decks_sub["draft_id"].astype(str)

    D = draft_base.merge(decks_sub, on="draft_id", how="inner")
    base_p_vec = D["base_p_cal"].to_numpy()
    target = (D["p_post"] - D["base_p_cal"]).to_numpy()
    weights = (D["A"] + D["B"]).to_numpy()
    feature_cols = ["base_p_cal"] + deck_cols
    X = D[feature_cols].to_numpy(dtype=float)

    train_idx, val_idx, test_idx = stratified_split(base_p_vec, seed=SEED, val_frac=0.15, test_frac=0.15)
    dtrain = xgb.DMatrix(X[train_idx], label=target[train_idx], weight=weights[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=target[val_idx], weight=weights[val_idx])
    dtest = xgb.DMatrix(X[test_idx])

    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "lambda": 0.5,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "seed": SEED,
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )

    # calibration of deck effect (theta0/theta1) on validation
    s_val = booster.predict(dval)
    base_p_val = base_p_vec[val_idx]
    A_val = D["A"].to_numpy()[val_idx]
    B_val = D["B"].to_numpy()[val_idx]

    def nll_theta(theta: np.ndarray) -> float:
        t0, t1 = theta
        z = logit(clip01(base_p_val)) + t0 + t1 * s_val
        p = expit(z)
        return run_nll(p, A_val, B_val)

    opt_theta = minimize(nll_theta, x0=np.array([0.0, 1.0]), method="BFGS")
    theta0, theta1 = opt_theta.x

    # evaluate on test
    s_test = booster.predict(dtest)
    base_p_test = base_p_vec[test_idx]
    A_test = D["A"].to_numpy()[test_idx]
    B_test = D["B"].to_numpy()[test_idx]
    obs_test = D["p_mle"].to_numpy()[test_idx]

    net_pred = clip01(expit(logit(clip01(base_p_test)) + theta0 + theta1 * s_test))
    bump_pred = net_pred - base_p_test
    bump_obs = obs_test - base_p_test

    def metr(y_true, y_pred):
        return {
            "R2": float(1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-9)),
            "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "MAE": float(np.mean(np.abs(y_true - y_pred))),
        }

    m_net = metr(obs_test, net_pred)
    m_bump = metr(bump_obs, bump_pred)

    # calibration bins and regression
    bins_net = calibration_bins(net_pred, obs_test, weights=A_test + B_test)
    bins_boost = calibration_bins(bump_pred, bump_obs, weights=A_test + B_test)
    reg_net = weighted_regression(net_pred, obs_test, A_test + B_test)
    reg_boost = weighted_regression(bump_pred, bump_obs, A_test + B_test)

    bins_net.to_csv(REPORTS_DIR / "deck_effect_bins_net.csv", index=False)
    bins_boost.to_csv(REPORTS_DIR / "deck_effect_bins_boost.csv", index=False)
    save_calibration_svg(
        bins_net,
        REPORTS_DIR / "deck_effect_calibration_net.svg",
        title="Deck effect net calibration",
        xlabel="Predicted run p (net)",
        ylabel="Observed p_mle",
        reg=reg_net,
    )
    save_calibration_svg(
        bins_boost,
        REPORTS_DIR / "deck_effect_calibration_boost.svg",
        title="Deck effect bump calibration",
        xlabel="Predicted bump",
        ylabel="Observed bump",
        reg=reg_boost,
    )

    metrics_out = {
        "net": m_net,
        "bump": m_bump,
        "calibration_net": reg_net,
        "calibration_bump": reg_boost,
        "theta0": float(theta0),
        "theta1": float(theta1),
        "early_stopping_rounds": booster.best_iteration + 1,
    }
    (REPORTS_DIR / "deck_effect_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    lines = [
        "Deck Effect Model (R-style) Evaluation",
        f"Net: R2={m_net['R2']:.3f}, RMSE={m_net['RMSE']:.3f}, MAE={m_net['MAE']:.3f}",
        f"Bump: R2={m_bump['R2']:.3f}, RMSE={m_bump['RMSE']:.3f}, MAE={m_bump['MAE']:.3f}",
        f"Calibration net slope={reg_net['slope']:.3f}, intercept={reg_net['intercept']:.3f}, R2={reg_net['R2']:.3f}",
        f"Calibration bump slope={reg_boost['slope']:.3f}, intercept={reg_boost['intercept']:.3f}, R2={reg_boost['R2']:.3f}",
        f"Theta0={theta0:.3f}, Theta1={theta1:.3f}, best_round={booster.best_iteration+1}",
        "Artifacts: deck_effect_xgb.json, deck_effect_meta.pkl",
    ]
    write_simple_pdf(REPORTS_DIR / "deck_effect_report.pdf", lines)

    # save artifacts
    booster_path = booster_path or (MODELS_DIR / "deck_effect_xgb.json")
    booster.save_model(booster_path)
    meta = {
        "theta0": float(theta0),
        "theta1": float(theta1),
        "deck_cols": deck_cols,
        "gp_levels": cal_base.gp_levels,
        "gamma": cal_base.gamma,
        "phi": cal_base.phi,
        "k_by_gp": derive_k_by_gp(games["user_n_games_bucket"]),
        "W": W_STOP,
        "L": L_STOP,
        "alpha": ALPHA,
        "beta": BETA,
    }
    meta_path = meta_path or (MODELS_DIR / "deck_effect_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    return booster, meta, metrics_out


def oof_predictions_two_fold():
    """Two-fold out-of-fold deck effect/bump predictions saved to reports/deck_effect_oof.parquet."""
    games = load_games()
    decks = load_decks()
    # shuffle drafts, split in half
    draft_ids = games["draft_id"].unique().tolist()
    rng = np.random.default_rng(SEED)
    rng.shuffle(draft_ids)
    mid = len(draft_ids) // 2
    drafts_a, drafts_b = set(draft_ids[:mid]), set(draft_ids[mid:])

    def fit_on(draft_subset: set, tag: str):
        g = games[games["draft_id"].isin(draft_subset)].copy()
        d = decks[decks["draft_id"].isin(draft_subset)].copy()
        fold_dir = MODELS_DIR / "fold_models"
        fold_dir.mkdir(parents=True, exist_ok=True)
        booster_path = fold_dir / f"deck_effect_xgb_{tag}.json"
        meta_path = fold_dir / f"deck_effect_meta_{tag}.pkl"
        booster, meta, _ = train_eval_save(g, d, booster_path=booster_path, meta_path=meta_path)
        return booster, meta

    # recompute features on full data (same as train_eval_save)
    post_tbl = posterior_mean_by_draft(games)
    base_p = oof_base_p_two_buckets(games)
    games = games.assign(base_p=base_p)
    draft_base = (
        games.groupby("draft_id")
        .agg(base_p=("base_p", "mean"), gp_bucket=("user_n_games_bucket", "first"))
        .reset_index()
        .merge(post_tbl, on="draft_id", how="left")
    )
    draft_base["A"] = np.minimum(draft_base["w_stop"].to_numpy(), W_STOP)
    draft_base["B"] = np.minimum(draft_base["l_stop"].to_numpy(), L_STOP)
    draft_base["p_mle"] = draft_base["A"] / (draft_base["A"] + draft_base["B"])
    cal_base = fit_basep_logit_cal_grouped(draft_base)
    draft_base["base_p_cal"] = apply_basep_logit_cal_grouped(
        draft_base["base_p"].to_numpy(), draft_base["gp_bucket"], cal_base
    )
    deck_cols = [c for c in decks.columns if c.startswith("deck_")]
    decks_sub = decks[["draft_id"] + deck_cols].copy()
    decks_sub["draft_id"] = decks_sub["draft_id"].astype(str)
    D = draft_base.merge(decks_sub, on="draft_id", how="inner")
    base_p_vec = D["base_p_cal"].to_numpy()
    feature_cols = ["base_p_cal"] + deck_cols
    X = D[feature_cols].to_numpy(dtype=float)

    # train on A, predict B
    booster_a, meta_a = fit_on(drafts_a, "foldA")
    theta0_a, theta1_a = meta_a["theta0"], meta_a["theta1"]
    mask_b = D["draft_id"].isin(drafts_b)
    dmat_b = xgb.DMatrix(X[mask_b])
    s_b = booster_a.predict(dmat_b)
    base_b = base_p_vec[mask_b]
    net_b = clip01(expit(logit(clip01(base_b)) + theta0_a + theta1_a * s_b))
    bump_b = net_b - base_b

    # train on B, predict A
    booster_b, meta_b = fit_on(drafts_b, "foldB")
    theta0_b, theta1_b = meta_b["theta0"], meta_b["theta1"]
    mask_a = D["draft_id"].isin(drafts_a)
    dmat_a = xgb.DMatrix(X[mask_a])
    s_a = booster_b.predict(dmat_a)
    base_a = base_p_vec[mask_a]
    net_a = clip01(expit(logit(clip01(base_a)) + theta0_b + theta1_b * s_a))
    bump_a = net_a - base_a

    deck_effect_oof = np.zeros(len(D))
    deck_bump_oof = np.zeros(len(D))
    deck_effect_oof[mask_b] = net_b
    deck_bump_oof[mask_b] = bump_b
    deck_effect_oof[mask_a] = net_a
    deck_bump_oof[mask_a] = bump_a

    out = pd.DataFrame(
        {
            "draft_id": D["draft_id"],
            "deck_effect_oof": deck_effect_oof,
            "deck_bump_oof": deck_bump_oof,
        }
    )
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OOF_PATH, index=False)
    return {"path": str(OOF_PATH), "n": len(D)}


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
