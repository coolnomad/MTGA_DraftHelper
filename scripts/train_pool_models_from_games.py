"""
Train pool-level models (net deck effect and deck bump) from real game outcomes.

Pipeline mirrors docs/notes/R_pool_model.md:
- Load games.parquet with per-game outcomes and deck/sideboard columns.
- Build pool composition per draft_id (deck + sideboard counts; drop basic lands).
- Compute out-of-fold base win rates using user buckets (winrate bucket + games-played).
- Compute draft-level stop stats under 7W/3L with Jeffreys prior to get p_mle/p_post.
- Targets:
    * net deck effect: p_post_draft - base_p (posterior-smoothed uplift vs base skill)
    * deck bump     : p_mle - base_p      (raw uplift vs base skill)
- Train XGBoost regressors with base_p as a feature + pool counts.
- Calibrate predicted probabilities via logit(base_p) + theta0 + theta1 * s_hat.
- Emit metrics, OOF preds, and calibration plots (binned reliability + raw scatter with fit).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - dependency is expected in runtime env
    raise SystemExit(
        "xgboost is required to run this script. Install with `pip install xgboost`."
    ) from exc

try:
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover - dependency is expected in runtime env
    raise SystemExit(
        "scipy is required for calibration. Install with `pip install scipy`."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency is expected in runtime env
    raise SystemExit(
        "matplotlib is required for plots. Install with `pip install matplotlib`."
    ) from exc


# ------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------
def clip01(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.clip(arr, eps, 1.0 - eps)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip01(p)
    return np.log(p / (1.0 - p))


def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------
# Data prep: pools and base win rates
# ------------------------------------------------------------
def coerce_won01(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    if np.issubdtype(series.dtype, np.number):
        return (series.to_numpy() != 0).astype(int)
    lower = series.astype(str).str.lower()
    return lower.isin({"true", "t", "1", "yes"}).astype(int).to_numpy()


def derive_k_by_gp(gp_vec: Iterable[str], k_hi: float = 80.0, k_lo: float = 15.0) -> Dict[str, float]:
    levels = sorted({str(x) for x in gp_vec})
    # numeric levels sorted numerically if possible, otherwise lexicographically
    try:
        levels = sorted(levels, key=lambda s: float(s))
    except ValueError:
        levels = sorted(levels)
    t = np.linspace(0.0, 1.0, num=len(levels))
    return {lev: (1 - ti) * k_hi + ti * k_lo for lev, ti in zip(levels, t)}


def oof_base_p_two_buckets_hier(
    df: pd.DataFrame,
    draft_col: str,
    won_col: str,
    wr_col: str,
    gp_col: str,
    n_folds: int,
    alpha: float,
    beta: float,
    k_by_gp: Dict[str, float] | None,
    default_k: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draft_ids = df[draft_col].astype(str).unique()
    rng.shuffle(draft_ids)
    fold_ids = np.tile(np.arange(n_folds, dtype=int), int(np.ceil(len(draft_ids) / n_folds)))[: len(draft_ids)]
    draft_to_fold = dict(zip(draft_ids, fold_ids, strict=False))

    draft = df[draft_col].astype(str).to_numpy()
    won01 = coerce_won01(df[won_col])
    wr = df[wr_col].astype(str).to_numpy()
    gp = df[gp_col].astype(str).to_numpy()
    key = np.char.add(np.char.add(wr, "||"), gp)
    fold = np.array([draft_to_fold[d] for d in draft], dtype=int)

    if k_by_gp is None:
        k_by_gp = derive_k_by_gp(gp)

    def get_k(g: str) -> float:
        return float(k_by_gp.get(g, default_k))

    base_hat = np.zeros(len(df), dtype=float)
    for k in range(n_folds):
        tr = fold != k
        va = fold == k

        # wr-only marginal
        W_wr = pd.Series(won01[tr]).groupby(wr[tr]).sum()
        N_wr = pd.Series(1, index=np.arange(tr.sum())).groupby(wr[tr]).sum()
        base_wr = (W_wr + alpha) / (N_wr + alpha + beta)

        # joint (wr, gp)
        W_joint = pd.Series(won01[tr]).groupby(key[tr]).sum()
        N_joint = pd.Series(1, index=np.arange(tr.sum())).groupby(key[tr]).sum()

        # align shrinkage term
        joint_keys = np.array(W_joint.index, dtype=str)
        wr_of_key = np.array([k.split("||")[0] for k in joint_keys])
        gp_of_key = np.array([k.split("||")[1] for k in joint_keys])
        m0_wr = base_wr.reindex(wr_of_key).to_numpy()
        k_gp = np.array([get_k(g) for g in gp_of_key])

        base_joint = (W_joint.to_numpy() + k_gp * m0_wr + alpha) / (N_joint.to_numpy() + k_gp + alpha + beta)
        base_joint_map = dict(zip(joint_keys, base_joint, strict=False))
        base_wr_map = base_wr.to_dict()
        base_global = float((won01[tr].sum() + alpha) / (tr.sum() + alpha + beta))

        k_va = key[va]
        wr_va = wr[va]
        out = np.array([base_joint_map.get(kv, np.nan) for kv in k_va])
        missing = np.isnan(out)
        if missing.any():
            out[missing] = [base_wr_map.get(wrv, np.nan) for wrv in wr_va[missing]]
            missing = np.isnan(out)
        if missing.any():
            out[missing] = base_global
        base_hat[va] = out

    return clip01(base_hat)


def posterior_mean_by_draft(
    df: pd.DataFrame,
    W: int,
    L: int,
    alpha: float,
    beta: float,
    draft_col: str,
    won_col: str,
    match_col: str,
) -> pd.DataFrame:
    won01 = coerce_won01(df[won_col])
    df_local = df.copy()
    df_local["_won01"] = won01
    df_local = df_local.sort_values([draft_col, match_col])

    rows = []
    for draft_id, grp in df_local.groupby(draft_col):
        cw = grp["_won01"].cumsum().to_numpy()
        cl = (1 - grp["_won01"]).cumsum().to_numpy()
        stop_idx_arr = np.where((cw >= W) | (cl >= L))[0]
        if len(stop_idx_arr) > 0:
            idx = int(stop_idx_arr[0])
            w_stop = int(cw[idx])
            l_stop = int(cl[idx])
            if l_stop >= L:
                A = alpha + w_stop
                B = beta + L
            else:
                A = alpha + W
                B = beta + l_stop
        else:
            w_stop = int(cw[-1])
            l_stop = int(cl[-1])
            A = alpha + w_stop
            B = beta + l_stop
        p_post = float(A / (A + B))
        rows.append(
            {
                draft_col: draft_id,
                "w_stop": w_stop,
                "l_stop": l_stop,
                "A": A,
                "B": B,
                "p_post_draft": p_post,
            }
        )
    return pd.DataFrame(rows)


def build_pool_features(df: pd.DataFrame) -> pd.DataFrame:
    deck_cols = [c for c in df.columns if c.startswith("deck_")]
    sideboard_cols = [c for c in df.columns if c.startswith("sideboard_")]
    side_from_deck = [c.replace("deck_", "sideboard_", 1) for c in deck_cols]
    missing_side = [c for c in side_from_deck if c not in sideboard_cols]
    if missing_side:
        raise ValueError(f"Missing sideboard columns for deck_* counterparts: {missing_side[:5]}")

    pool_cols = [c.replace("deck_", "pool_", 1) for c in deck_cols]
    df_local = df.copy()
    for d_col, s_col, p_col in zip(deck_cols, side_from_deck, pool_cols):
        df_local[p_col] = df_local[d_col].astype(float).fillna(0.0) + df_local[s_col].astype(float).fillna(0.0)

    pool_only = [c for c in pool_cols if c not in {"pool_island", "pool_swamp", "pool_forest", "pool_mountain", "pool_plains"}]
    agg = df_local[[ "draft_id", *pool_only ]].groupby("draft_id").agg(max).reset_index()
    return agg


# ------------------------------------------------------------
# Calibration helpers
# ------------------------------------------------------------
@dataclass
class GroupedLogitCal:
    theta0: float
    theta1: float
    gamma: Dict[str, float]
    phi: Dict[str, float]
    gp_levels: List[str]


def fit_basep_logit_cal_grouped(base_p: np.ndarray, gp_bucket: Iterable[str], A: np.ndarray, B: np.ndarray) -> GroupedLogitCal:
    s = logit(base_p)
    gp_cat = pd.Categorical(gp_bucket)
    Xgp = pd.get_dummies(gp_cat, drop_first=False)
    gp_levels = list(Xgp.columns)
    Xgp_mat = Xgp.to_numpy()
    G = Xgp_mat.shape[1]

    def nll(params: np.ndarray) -> float:
        theta0 = params[0]
        theta1 = params[1]
        gamma = np.concatenate(([0.0], params[2 : 2 + G - 1]))
        phi = np.concatenate(([0.0], params[2 + G - 1 :]))
        z = s + theta0 + theta1 * s + Xgp_mat @ gamma + (Xgp_mat @ phi) * s
        p = inv_logit(z)
        p = clip01(p)
        return float(-np.mean(A * np.log(p) + B * np.log(1.0 - p)))

    par0 = np.zeros(2 + 2 * (G - 1), dtype=float)
    res = minimize(nll, par0, method="BFGS")
    params = res.x
    gamma = np.concatenate(([0.0], params[2 : 2 + G - 1]))
    phi = np.concatenate(([0.0], params[2 + G - 1 :]))
    return GroupedLogitCal(
        theta0=float(params[0]),
        theta1=float(params[1]),
        gamma=dict(zip(gp_levels, gamma, strict=False)),
        phi=dict(zip(gp_levels, phi, strict=False)),
        gp_levels=gp_levels,
    )


def apply_basep_logit_cal_grouped(base_p: np.ndarray, gp_bucket: Iterable[str], cal: GroupedLogitCal) -> np.ndarray:
    s = logit(base_p)
    gp_cat = pd.Categorical(gp_bucket, categories=cal.gp_levels)
    Xgp = pd.get_dummies(gp_cat, drop_first=False).to_numpy()
    gamma = np.array([cal.gamma.get(c, 0.0) for c in cal.gp_levels])
    phi = np.array([cal.phi.get(c, 0.0) for c in cal.gp_levels])
    z = s + cal.theta0 + cal.theta1 * s + Xgp @ gamma + (Xgp @ phi) * s
    return clip01(inv_logit(z))


# ------------------------------------------------------------
# Metrics and plotting
# ------------------------------------------------------------
def nll_from_probs(p: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    p = clip01(p)
    return float(-np.mean(A * np.log(p) + B * np.log(1.0 - p)))


def reliability_table(p_pred: np.ndarray, A: np.ndarray, B: np.ndarray, nbins: int) -> pd.DataFrame:
    p_pred = p_pred.astype(float)
    bins = np.unique(np.quantile(p_pred, np.linspace(0.0, 1.0, nbins + 1)))
    if len(bins) < 3:
        bins = np.linspace(p_pred.min(), p_pred.max(), nbins + 1)
    bin_idx = np.digitize(p_pred, bins[1:-1], right=True)
    df = pd.DataFrame({"bin": bin_idx, "p_pred": p_pred, "A": A, "B": B})
    grouped = df.groupby("bin").agg(
        p_mean=("p_pred", "mean"),
        A=("A", "sum"),
        B=("B", "sum"),
    )
    grouped["p_true"] = grouped["A"] / (grouped["A"] + grouped["B"] + 1e-12)
    grouped["weight"] = grouped["A"] + grouped["B"]
    return grouped.reset_index(drop=True)


def plot_reliability(table: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(table["p_mean"], table["p_true"], s=np.clip(table["weight"], 5, None), alpha=0.9)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("Predicted p (bin mean)")
    plt.ylabel("Observed p (bin mean)")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_raw_scatter(x: np.ndarray, y: np.ndarray, w: np.ndarray, title: str, path: Path) -> Tuple[float, float]:
    # weighted fit line to show trend
    coeffs = np.polynomial.polynomial.polyfit(x, y, deg=1, w=w)
    intercept, slope = float(coeffs[0]), float(coeffs[1])
    x_line = np.array([x.min(), x.max()])
    y_line = intercept + slope * x_line

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=np.clip(w, 5, None), alpha=0.6)
    plt.plot(x_line, y_line, color="red", label=f"fit slope={slope:.3f}")
    plt.xlabel("Predicted delta")
    plt.ylabel("Observed delta")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    return slope, intercept


# ------------------------------------------------------------
# Model training
# ------------------------------------------------------------
def fit_theta(base_p: np.ndarray, s_hat: np.ndarray, A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    def obj(params: np.ndarray) -> float:
        theta0, theta1 = params
        p = inv_logit(logit(base_p) + theta0 + theta1 * s_hat)
        return nll_from_probs(p, A, B)

    res = minimize(obj, np.array([0.0, 1.0], dtype=float), method="BFGS")
    return float(res.x[0]), float(res.x[1])


def stratified_split(base: np.ndarray, seed: int, val_frac: float = 0.15, test_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0.0, 1.0, 11)
    bins = np.unique(np.quantile(base, quantiles))
    if len(bins) < 3:
        bins = np.linspace(base.min(), base.max(), num=11)
    strata = np.digitize(base, bins[1:-1], right=True)
    idx = np.arange(len(base))
    val_idx: List[int] = []
    test_idx: List[int] = []
    for s in np.unique(strata):
        mask = strata == s
        stratum_idx = idx[mask]
        rng.shuffle(stratum_idx)
        n = len(stratum_idx)
        n_val = int(np.ceil(n * val_frac))
        n_test = int(np.ceil(n * test_frac))
        val_idx.extend(stratum_idx[:n_val])
        test_idx.extend(stratum_idx[n_val : n_val + n_test])
    val_idx = np.array(sorted(set(val_idx)), dtype=int)
    test_idx = np.array(sorted(set(test_idx)), dtype=int)
    train_idx = np.array(sorted(set(idx) - set(val_idx) - set(test_idx)), dtype=int)
    return train_idx, val_idx, test_idx


def train_xgb_regressor(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    params: dict,
    nrounds: int = 5000,
    early_stopping_rounds: int = 150,
) -> Tuple[xgb.Booster, np.ndarray, np.ndarray, int]:
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], weight=w[train_idx])
    dvalid = xgb.DMatrix(X[val_idx], label=y[val_idx], weight=w[val_idx])
    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=nrounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )
    best_ntree = booster.best_iteration or booster.num_boosted_rounds()
    return booster, booster.predict(xgb.DMatrix(X[val_idx])), booster.predict(xgb.DMatrix(X[train_idx])), best_ntree


def evaluate_split(
    name: str,
    base_p: np.ndarray,
    s_hat: np.ndarray,
    y_true: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    split: str,
    out_dir: Path,
) -> Dict[str, float]:
    theta0, theta1 = fit_theta(base_p, s_hat, A, B)
    p_hat = clip01(inv_logit(logit(base_p) + theta0 + theta1 * s_hat))
    nll_model = nll_from_probs(p_hat, A, B)
    nll_base = nll_from_probs(base_p, A, B)
    rel_gain = 1.0 - nll_model / nll_base

    # calibration plots
    rel_table = reliability_table(p_hat, A, B, nbins=20)
    plot_reliability(rel_table, f"{name} reliability ({split})", out_dir / f"{name}_{split}_reliability.png")

    delta_obs = y_true
    delta_pred = s_hat
    slope, intercept = plot_raw_scatter(
        delta_pred,
        delta_obs,
        w=A + B,
        title=f"{name} raw ({split})",
        path=out_dir / f"{name}_{split}_raw.png",
    )

    rmse = float(np.sqrt(np.average((delta_obs - delta_pred) ** 2, weights=A + B)))
    return {
        "theta0": theta0,
        "theta1": theta1,
        "nll_model": nll_model,
        "nll_base": nll_base,
        "relative_gain": rel_gain,
        "rmse": rmse,
        "slope": slope,
        "intercept": intercept,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games_path", type=str, default="data/processed/games.parquet")
    parser.add_argument("--reports_dir", type=str, default="reports/pool_models")
    parser.add_argument("--models_dir", type=str, default="hero_bot/models")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--W", type=int, default=7, help="win stop for posterior")
    parser.add_argument("--L", type=int, default=3, help="loss stop for posterior")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--default_k",
        type=float,
        default=40.0,
        help="fallback shrink strength for base_p when gp bucket not in map",
    )
    parser.add_argument("--eta", type=float, default=0.05, help="XGBoost learning rate")
    parser.add_argument("--max_depth", type=int, default=6, help="XGBoost max_depth")
    parser.add_argument("--n_estimators", type=int, default=3000, help="XGBoost rounds")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    models_dir = Path(args.models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load games
    df_games = pd.read_parquet(Path(args.games_path))
    needed = ["draft_id", "won", "user_n_games_bucket", "user_game_win_rate_bucket", "match_number"]
    missing = [c for c in needed if c not in df_games.columns]
    if missing:
        raise ValueError(f"Missing required columns in games parquet: {missing}")

    # 2) Base table for base_p
    x2 = df_games[needed].copy()

    # 3) Posterior per draft under 7W/3L
    post_tbl = posterior_mean_by_draft(
        x2,
        W=args.W,
        L=args.L,
        alpha=args.alpha,
        beta=args.beta,
        draft_col="draft_id",
        won_col="won",
        match_col="match_number",
    )

    # 4) OOF base_p with hierarchical shrink
    k_map = derive_k_by_gp(x2["user_n_games_bucket"].astype(str))
    x2["base_p"] = oof_base_p_two_buckets_hier(
        x2,
        draft_col="draft_id",
        won_col="won",
        wr_col="user_game_win_rate_bucket",
        gp_col="user_n_games_bucket",
        n_folds=args.cv_folds,
        alpha=args.alpha,
        beta=args.beta,
        k_by_gp=k_map,
        default_k=args.default_k,
        seed=args.seed,
    )

    # 5) Draft-level table for calibration to MLE
    draft_base = (
        x2.groupby("draft_id")["base_p"].mean().rename("base_p").reset_index()
    ).merge(
        post_tbl[["draft_id", "w_stop", "l_stop", "A", "B", "p_post_draft"]],
        on="draft_id",
        how="inner",
    )
    draft_base["gp_bucket"] = (
        x2.groupby("draft_id")["user_n_games_bucket"].first().reindex(draft_base["draft_id"]).values
    )
    draft_base["p_mle"] = draft_base["A"] / (draft_base["A"] + draft_base["B"])

    # 6) Fit grouped calibration for base_p only (optional QA)
    base_cal = fit_basep_logit_cal_grouped(
        draft_base["base_p"].to_numpy(),
        draft_base["gp_bucket"],
        draft_base["A"].to_numpy(),
        draft_base["B"].to_numpy(),
    )
    draft_base["base_p_cal"] = apply_basep_logit_cal_grouped(
        draft_base["base_p"].to_numpy(), draft_base["gp_bucket"], base_cal
    )

    # 7) Pool feature matrix (pool_* counts per draft)
    pool_df = build_pool_features(df_games)
    pool_df = pool_df.rename(columns={"draft_id": "draft_id"})
    pool_cols = [c for c in pool_df.columns if c != "draft_id"]

    # 8) Merge draft data with pool features
    data = draft_base.merge(pool_df, on="draft_id", how="inner")
    feature_cols = ["base_p"] + pool_cols
    X = data[feature_cols].to_numpy(dtype=float)
    ids = data["draft_id"].astype(str).to_numpy()
    base_p_vec = data["base_p"].to_numpy(dtype=float)
    A = data["A"].to_numpy(dtype=float)
    B = data["B"].to_numpy(dtype=float)
    w = A + B

    # Targets
    y_effect = data["p_post_draft"].to_numpy(dtype=float) - base_p_vec
    y_bump = data["p_mle"].to_numpy(dtype=float) - base_p_vec

    # 9) Split data
    train_idx, val_idx, test_idx = stratified_split(base_p_vec, seed=args.seed)

    # 10) Train models
    params = {
        "objective": "reg:squarederror",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 0.5,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "seed": args.seed,
    }

    results: Dict[str, dict] = {}
    models: Dict[str, Path] = {}
    split_preds: Dict[str, Dict[str, np.ndarray]] = {
        "deck_effect": {},
        "deck_bump": {},
    }

    for name, target in [("deck_effect", y_effect), ("deck_bump", y_bump)]:
        booster, val_pred, _, best_rounds = train_xgb_regressor(
            X,
            target,
            w,
            train_idx=train_idx,
            val_idx=val_idx,
            params=params,
            nrounds=args.n_estimators,
        )
        # evaluate on val + test
        test_pred = booster.predict(xgb.DMatrix(X[test_idx]))
        split_preds[name]["val"] = val_pred
        split_preds[name]["test"] = test_pred
        val_metrics = evaluate_split(
            name=name,
            base_p=base_p_vec[val_idx],
            s_hat=val_pred,
            y_true=target[val_idx],
            A=A[val_idx],
            B=B[val_idx],
            split="val",
            out_dir=reports_dir,
        )
        test_metrics = evaluate_split(
            name=name,
            base_p=base_p_vec[test_idx],
            s_hat=test_pred,
            y_true=target[test_idx],
            A=A[test_idx],
            B=B[test_idx],
            split="test",
            out_dir=reports_dir,
        )

        # final fit on all data
        d_all = xgb.DMatrix(X, label=target, weight=w, feature_names=feature_cols)
        final_model = xgb.train(
            params={k: v for k, v in params.items() if k != "eval_metric"},
            dtrain=d_all,
            num_boost_round=best_rounds or args.n_estimators,
            verbose_eval=False,
        )
        model_path = models_dir / f"{name}_xgb.json"
        final_model.save_model(model_path)
        models[name] = model_path

        # in-sample fit stats
        all_pred = final_model.predict(d_all)
        slope_all, intercept_all = plot_raw_scatter(
            all_pred,
            target,
            w=w,
            title=f"{name} raw (all)",
            path=reports_dir / f"{name}_all_raw.png",
        )
        rmse_all = float(np.sqrt(np.average((target - all_pred) ** 2, weights=w)))

        results[name] = {
            "val": val_metrics,
            "test": test_metrics,
            "best_rounds": int(best_rounds),
            "rmse_all": rmse_all,
            "slope_all": slope_all,
            "intercept_all": intercept_all,
            "feature_dim": X.shape[1],
            "n_samples": int(X.shape[0]),
            "target_mean": float(target.mean()),
        }

    # 11) Save OOF predictions (val/test combined) and metadata
    oof_df = pd.DataFrame(
        {
            "draft_id": ids[val_idx].tolist() + ids[test_idx].tolist(),
            "split": ["val"] * len(val_idx) + ["test"] * len(test_idx),
            "base_p": np.concatenate([base_p_vec[val_idx], base_p_vec[test_idx]]),
            "A": np.concatenate([A[val_idx], A[test_idx]]),
            "B": np.concatenate([B[val_idx], B[test_idx]]),
            "y_effect": np.concatenate([y_effect[val_idx], y_effect[test_idx]]),
            "y_bump": np.concatenate([y_bump[val_idx], y_bump[test_idx]]),
            "pred_effect": np.concatenate(
                [split_preds["deck_effect"]["val"], split_preds["deck_effect"]["test"]]
            ),
            "pred_bump": np.concatenate(
                [split_preds["deck_bump"]["val"], split_preds["deck_bump"]["test"]]
            ),
        }
    )
    oof_path = reports_dir / "pool_models_oof.parquet"
    oof_df.to_parquet(oof_path, index=False)

    meta = {
        "games_path": str(args.games_path),
        "reports_dir": str(reports_dir),
        "models_dir": str(models_dir),
        "cv_folds_base_p": args.cv_folds,
        "W": args.W,
        "L": args.L,
        "alpha": args.alpha,
        "beta": args.beta,
        "default_k": args.default_k,
        "xgb_params": params,
        "results": results,
        "artifacts": {
            "oof_predictions": str(oof_path),
            "reliability_plots": {
                "deck_effect_val": str(reports_dir / "deck_effect_val_reliability.png"),
                "deck_effect_test": str(reports_dir / "deck_effect_test_reliability.png"),
                "deck_bump_val": str(reports_dir / "deck_bump_val_reliability.png"),
                "deck_bump_test": str(reports_dir / "deck_bump_test_reliability.png"),
            },
            "raw_scatter_plots": {
                "deck_effect_val": str(reports_dir / "deck_effect_val_raw.png"),
                "deck_effect_test": str(reports_dir / "deck_effect_test_raw.png"),
                "deck_bump_val": str(reports_dir / "deck_bump_val_raw.png"),
                "deck_bump_test": str(reports_dir / "deck_bump_test_raw.png"),
                "deck_effect_all": str(reports_dir / "deck_effect_all_raw.png"),
                "deck_bump_all": str(reports_dir / "deck_bump_all_raw.png"),
            },
        },
        "models": {name: str(path) for name, path in models.items()},
        "pool_feature_cols": pool_cols,
        "feature_cols": feature_cols,
    }
    meta_path = reports_dir / "pool_models_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
