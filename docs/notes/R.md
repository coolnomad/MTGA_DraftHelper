"""
deck_effect_model.py

Python version of the R pipeline:

1) Read Arena game-level data and build draft-level deck features.
2) Compute posterior mean per draft under 7–3 stop rule (Jeffreys prior).
3) Compute out-of-fold baseline skill probabilities base_p via hierarchical
   shrinkage across (winrate_bucket, games_bucket).
4) Fit grouped logistic calibration of base_p vs MLE p under the stop rule.
5) Build draft-level dataset with:
   - base_p_cal (calibrated baseline)
   - p_post_draft (posterior mean)
   - p_mle (MLE under stop rule)
   - deck features
6) Train an xgboost model on Δp = p_post_draft - base_p_cal, with sample weights A+B.
7) Fit a 2-parameter logistic calibration for the deck effect:
   logit(p_hat) = logit(base_p_cal) + theta0 + theta1 * s_hat
   by minimizing NLL of (A,B).
8) Report calibration diagnostics and save artifacts.

This is structurally equivalent to your R code.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from scipy.special import expit, logit
from scipy.optimize import minimize
import xgboost as xgb


# -------------------------------------------------------------------
# config
# -------------------------------------------------------------------

DATA_PATH = r"C:/Users/dimuc/OneDrive/Desktop/Magic Arena Analysis/game_data_public.FIN.PremierDraft.csv.gz"

W_STOP = 7
L_STOP = 3
ALPHA = 0.5  # Jeffreys prior
BETA = 0.5

K_FOLDS = 5
DEFAULT_K = 40.0  # shrink strength fallback

RANDOM_SEED = 123


# -------------------------------------------------------------------
# low-level helpers
# -------------------------------------------------------------------

def clip01(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.minimum(np.maximum(p, eps), 1.0 - eps)


def coerce_won01(series: pd.Series) -> np.ndarray:
    """Convert won column to 0/1 as in R coerce_won01()."""
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    if np.issubdtype(series.dtype, np.number):
        return (series.to_numpy() != 0).astype(int)
    s = series.astype(str).str.lower()
    return s.isin(["true", "t", "1", "yes"]).astype(int).to_numpy()


def run_nll(p: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """Negative log-likelihood for Binomial under A wins, B losses."""
    p = clip01(p)
    return float(-np.mean(A * np.log(p) + B * np.log(1.0 - p)))


# -------------------------------------------------------------------
# posterior mean per draft under stop rule (Jeffreys)
# -------------------------------------------------------------------

def posterior_mean_by_draft(
    df: pd.DataFrame,
    W: int = W_STOP,
    L: int = L_STOP,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> pd.DataFrame:
    """
    For each draft_id, respect the W/L stop rule to compute:
    - w_stop, l_stop
    - p_post_draft: posterior mean of p under Beta(alpha, beta)
      with special handling if run stopped by W or L.
    This matches the R/posterior_mean_by_draft logic.
    """
    df = df[["draft_id", "won", "match_number"]].copy()
    df["won01"] = coerce_won01(df["won"])

    results = []

    for draft_id, g in df.sort_values("match_number").groupby("draft_id"):
        won01 = g["won01"].to_numpy()

        cw = np.cumsum(won01)
        cl = np.cumsum(1 - won01)

        idx_candidates = np.where((cw >= W) | (cl >= L))[0]
        if len(idx_candidates) > 0:
            idx = idx_candidates[0]
            w_stop = int(cw[idx])
            l_stop = int(cl[idx])
            # match R: if stopped via losses, treat B as beta + L; else treat A as alpha + W
            if l_stop >= L:
                a_post = alpha + w_stop
                b_post = beta + L
            else:
                a_post = alpha + W
                b_post = beta + l_stop
        else:
            w_stop = int(cw[-1])
            l_stop = int(cl[-1])
            a_post = alpha + w_stop
            b_post = beta + l_stop

        p_post = a_post / (a_post + b_post)
        results.append({"draft_id": draft_id, "w_stop": w_stop, "l_stop": l_stop, "p_post_draft": p_post})

    return pd.DataFrame(results)


# -------------------------------------------------------------------
# hierarchical OOF baseline: oof_base_p_two_buckets_hier
# -------------------------------------------------------------------

def derive_k_by_gp(gp_series: pd.Series, k_hi: float = 80.0, k_lo: float = 15.0) -> Dict[str, float]:
    """
    Approximate R derive_k_by_gp(): map gp levels to shrink strengths.
    Levels are sorted; k decreases linearly from k_hi to k_lo.
    """
    lev = sorted(set(gp_series.astype(str)))
    # attempt numeric sort if possible
    try:
        lev_num = np.array(lev, dtype=float)
        order = np.argsort(lev_num)
        lev = [lev[i] for i in order]
    except ValueError:
        # fall back to lexicographic
        pass
    t = np.linspace(0.0, 1.0, num=len(lev))
    k_vals = (1.0 - t) * k_hi + t * k_lo
    return dict(zip(lev, k_vals))


def oof_base_p_two_buckets_hier(
    df: pd.DataFrame,
    draft_col: str = "draft_id",
    won_col: str = "won",
    wr_col: str = "user_game_win_rate_bucket",
    gp_col: str = "user_n_games_bucket",
    K: int = K_FOLDS,
    alpha: float = ALPHA,
    beta: float = BETA,
    k_by_gp: Dict[str, float] | None = None,
    default_k: float = DEFAULT_K,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Python version of R oof_base_p_two_buckets_hier().

    Returns an array base_hat of length n_rows with out-of-fold baseline
    probability estimates using hierarchical shrinkage over (wr,gp).
    """
    rng = np.random.default_rng(seed)

    draft = df[draft_col].astype(str).to_numpy()
    won01 = coerce_won01(df[won_col])
    wr = df[wr_col].astype(str).to_numpy()
    gp = df[gp_col].astype(str).to_numpy()
    key = np.char.add(np.char.add(wr, "||"), gp)

    if k_by_gp is None:
        k_by_gp = derive_k_by_gp(df[gp_col])

    def get_k(g: str) -> float:
        return float(k_by_gp.get(g, default_k))

    # assign folds at draft level
    unique_drafts = np.unique(draft)
    fold_ids = np.arange(K) + 1
    assigned = rng.choice(fold_ids, size=len(unique_drafts), replace=True)
    draft_to_fold = dict(zip(unique_drafts, assigned))
    fold = np.array([draft_to_fold[d] for d in draft])

    base_hat = np.zeros(len(df), dtype=float)

    for k in fold_ids:
        tr = fold != k
        va = fold == k

        if not np.any(va):
            continue

        wr_tr = wr[tr]
        key_tr = key[tr]
        won_tr = won01[tr]

        # WR-only marginal
        wr_levels, wr_idx = np.unique(wr_tr, return_inverse=True)
        W_wr = np.bincount(wr_idx, weights=won_tr, minlength=len(wr_levels))
        N_wr = np.bincount(wr_idx, minlength=len(wr_levels))
        base_wr = (W_wr + alpha) / (N_wr + alpha + beta)
        base_wr_map = dict(zip(wr_levels, base_wr))

        # joint (wr,gp) counts
        joint_levels, joint_idx = np.unique(key_tr, return_inverse=True)
        W_joint = np.bincount(joint_idx, weights=won_tr, minlength=len(joint_levels))
        N_joint = np.bincount(joint_idx, minlength=len(joint_levels))

        # decode wr and gp from joint_keys
        wr_of_key = np.array([jk.split("||")[0] for jk in joint_levels])
        gp_of_key = np.array([jk.split("||")[1] for jk in joint_levels])

        m0_wr = np.array([base_wr_map[w] for w in wr_of_key])
        k_gp_vec = np.array([get_k(g) for g in gp_of_key])

        base_joint = (W_joint + k_gp_vec * m0_wr + alpha) / (N_joint + k_gp_vec + alpha + beta)
        base_joint_map = dict(zip(joint_levels, base_joint))

        # global prior
        base_global = (W_joint.sum() + alpha) / (N_joint.sum() + alpha + beta)

        # fill validation rows
        wr_va = wr[va]
        key_va = key[va]
        bh = np.empty(len(key_va), dtype=float)
        for i, (kv, wv) in enumerate(zip(key_va, wr_va)):
            if kv in base_joint_map:
                bh[i] = base_joint_map[kv]
            elif wv in base_wr_map:
                bh[i] = base_wr_map[wv]
            else:
                bh[i] = base_global
        base_hat[va] = bh

    return clip01(base_hat)


# -------------------------------------------------------------------
# grouped logistic calibration of baseline vs MLE
# -------------------------------------------------------------------

@dataclass
class GroupedCalibration:
    theta0: float
    theta1: float
    gamma: Dict[str, float]
    phi: Dict[str, float]
    gp_levels: list


def fit_basep_logit_cal_grouped(
    d_draft: pd.DataFrame,
    W: int = W_STOP,
    L: int = L_STOP,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> GroupedCalibration:
    """
    Python version of fit_basep_logit_cal_grouped().

    d_draft must contain columns:
      - base_p
      - gp_bucket
      - A, B (wins/losses after clipping)
    """
    s = logit(clip01(d_draft["base_p"].to_numpy()))
    gp = d_draft["gp_bucket"].astype(str)
    gp_levels = sorted(gp.unique())
    gp_cat = pd.Categorical(gp, categories=gp_levels)
    Xgp = pd.get_dummies(gp_cat, drop_first=False)  # one-hot

    G = Xgp.shape[1]
    A = d_draft["A"].to_numpy().astype(float)
    B = d_draft["B"].to_numpy().astype(float)

    def nll_from_par(par: np.ndarray) -> float:
        theta0 = par[0]
        theta1 = par[1]
        # gamma, phi with first level fixed at 0
        gamma_free = par[2 : 2 + (G - 1)]
        phi_free = par[2 + (G - 1) : 2 + 2 * (G - 1)]
        gamma = np.concatenate(([0.0], gamma_free))
        phi = np.concatenate(([0.0], phi_free))

        gamma_vec = Xgp.to_numpy() @ gamma
        phi_vec = Xgp.to_numpy() @ phi

        z = s + theta0 + theta1 * s + gamma_vec + phi_vec * s
        p = expit(z)
        return run_nll(p, A, B)

    # starting point
    par0 = np.zeros(2 + 2 * (G - 1), dtype=float)
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

    return GroupedCalibration(
        theta0=theta0,
        theta1=theta1,
        gamma=gamma_map,
        phi=phi_map,
        gp_levels=gp_levels,
    )


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


# -------------------------------------------------------------------
# deck effect model + calibration
# -------------------------------------------------------------------

@dataclass
class DeckEffectArtifacts:
    booster_model: xgb.Booster
    theta0: float
    theta1: float
    deck_cols: list
    # baseline calibration artifacts
    basep_calibration: GroupedCalibration
    k_by_gp: Dict[str, float]


def train_deck_effect_model(
    df_games: pd.DataFrame,
    X_deck: pd.DataFrame,
    W: int = W_STOP,
    L: int = L_STOP,
    alpha: float = ALPHA,
    beta: float = BETA,
    random_state: int = RANDOM_SEED,
) -> Tuple[DeckEffectArtifacts, pd.DataFrame]:
    """
    Main training pipeline: given game-level df_games and draft-level X_deck (row index=draft_id),
    fit the deck-effect model and return artifacts + draft-level frame.
    """

    # ensure draft_id is str
    df_games = df_games.copy()
    df_games["draft_id"] = df_games["draft_id"].astype(str)

    # 1) posterior per draft
    post_tbl = posterior_mean_by_draft(df_games, W=W, L=L, alpha=alpha, beta=beta)

    # 2) OOF base_p via hierarchical shrink
    df_games["base_p_raw"] = oof_base_p_two_buckets_hier(
        df_games,
        draft_col="draft_id",
        won_col="won",
        wr_col="user_game_win_rate_bucket",
        gp_col="user_n_games_bucket",
        K=K_FOLDS,
        alpha=alpha,
        beta=beta,
        k_by_gp=None,
        default_k=DEFAULT_K,
        seed=random_state,
    )

    # 3) build draft-level table for baseline calibration
    # aggregate base_p by draft, merge w stop stats & gp_bucket
    draft_base = (
        df_games.groupby("draft_id")
        .agg(
            base_p=("base_p_raw", "mean"),
            gp_bucket=("user_n_games_bucket", "first"),
        )
        .reset_index()
        .merge(post_tbl, on="draft_id", how="left")
    )

    # clip A,B as in R
    draft_base["A"] = np.minimum(draft_base["w_stop"].to_numpy(), W)
    draft_base["B"] = np.minimum(draft_base["l_stop"].to_numpy(), L)
    draft_base["p_mle"] = draft_base["A"] / (draft_base["A"] + draft_base["B"])

    # 4) grouped calibration of baseline vs MLE
    cal_base = fit_basep_logit_cal_grouped(draft_base, W=W, L=L, alpha=alpha, beta=beta)
    draft_base["base_p_cal"] = apply_basep_logit_cal_grouped(
        draft_base["base_p"].to_numpy(),
        draft_base["gp_bucket"],
        cal_base,
    )

    # 5) merge deck features
    # X_deck: rows = draft_id index or column; make sure
    if "draft_id" in X_deck.columns:
        X_df = X_deck.copy()
        X_df["draft_id"] = X_df["draft_id"].astype(str)
        X_df.set_index("draft_id", inplace=True)
    else:
        X_df = X_deck.copy()
        X_df.index = X_df.index.astype(str)

    D = draft_base.set_index("draft_id").join(X_df, how="inner")
    D = D.reset_index()  # draft_id back as column

    # 6) features & target for deck effect model
    # deck columns = all deck_* columns (same as your R code)
    deck_cols = [c for c in D.columns if c.startswith("deck_")]
    feature_cols = ["base_p_cal"] + deck_cols

    X = D[feature_cols].to_numpy(dtype=float)
    y = (D["p_post_draft"] - D["base_p_cal"]).to_numpy(dtype=float)  # surrogate Δp
    w = (D["A"] + D["B"]).to_numpy(dtype=float)
    base_p_vec = D["base_p_cal"].to_numpy(dtype=float)

    # 7) stratified train/val/test split by base_p deciles (same idea as R)
    rng = np.random.default_rng(random_state)
    quantiles = np.quantile(base_p_vec, np.linspace(0, 1, 11))
    # guard against edge duplicates
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    strata = np.digitize(base_p_vec, quantiles, right=True)

    indices = np.arange(len(D))
    train_idx = []
    val_idx = []
    test_idx = []

    for s in np.unique(strata):
        mask = strata == s
        idx_s = indices[mask]
        if len(idx_s) == 0:
            continue
        rng.shuffle(idx_s)
        n = len(idx_s)
        n_val = max(1, int(np.ceil(0.15 * n)))
        n_test = max(1, int(np.ceil(0.15 * (n - n_val))))
        val_idx.extend(idx_s[:n_val])
        test_idx.extend(idx_s[n_val : n_val + n_test])
        train_idx.extend(idx_s[n_val + n_test :])

    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], weight=w[train_idx])
    dvalid = xgb.DMatrix(X[val_idx], label=y[val_idx], weight=w[val_idx])

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
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=150,
        verbose_eval=False,
    )

    # 8) 2-parameter logistic calibration for deck effect (VALID set)
    s_val = booster.predict(dvalid)  # Δp surrogate
    A_val = D["A"].to_numpy()[val_idx]
    B_val = D["B"].to_numpy()[val_idx]
    base_p_val = base_p_vec[val_idx]

    def nll_theta(theta: np.ndarray) -> float:
        t0, t1 = theta
        z = logit(clip01(base_p_val)) + t0 + t1 * s_val
        p = expit(z)
        return run_nll(p, A_val, B_val)

    opt_theta = minimize(nll_theta, x0=np.array([0.0, 1.0]), method="BFGS")
    theta0, theta1 = opt_theta.x

    # attach calibrated predictions on TEST for sanity
    dtest = xgb.DMatrix(X[test_idx])
    s_test = booster.predict(dtest)
    base_p_test = base_p_vec[test_idx]
    A_test = D["A"].to_numpy()[test_idx]
    B_test = D["B"].to_numpy()[test_idx]
    p_hat_test = clip01(expit(logit(clip01(base_p_test)) + theta0 + theta1 * s_test))
    p_base_test = clip01(base_p_test)

    nll_test_model = run_nll(p_hat_test, A_test, B_test)
    nll_test_base = run_nll(p_base_test, A_test, B_test)

    print("TEST NLL (model vs base):", nll_test_model, nll_test_base)
    print("relative gain:", 1.0 - nll_test_model / nll_test_base)

    artifacts = DeckEffectArtifacts(
        booster_model=booster,
        theta0=float(theta0),
        theta1=float(theta1),
        deck_cols=deck_cols,
        basep_calibration=cal_base,
        k_by_gp=derive_k_by_gp(df_games["user_n_games_bucket"]),
    )

    return artifacts, D


# -------------------------------------------------------------------
# example main
# -------------------------------------------------------------------

def main():
    # read raw game-level data
    df = pd.read_csv(DATA_PATH)
    # keep just columns we need for this pipeline
    cols_needed = [
        "draft_id",
        "won",
        "match_number",
        "user_n_games_bucket",
        "user_game_win_rate_bucket",
    ]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # build deck features per draft (same idea as your R x3 / deck_by_draft)
    deck_cols = [c for c in df.columns if c.startswith("deck_")]
    if not deck_cols:
        raise ValueError("No deck_* columns found in the raw data.")

    deck_df = df[["draft_id"] + deck_cols].copy()
    deck_df["draft_id"] = deck_df["draft_id"].astype(str)

    # average deck_* over games in the same draft (fractions)
    X_deck = (
        deck_df.groupby("draft_id")[deck_cols]
        .mean()
        .reset_index()
    )

    # train deck effect model
    artifacts, D = train_deck_effect_model(df[cols_needed], X_deck)

    # example: save artifacts
    os.makedirs("models", exist_ok=True)
    artifacts.booster_model.save_model("models/deck_effect_xgb.json")
    # save calibration + metadata
    meta = {
        "theta0": artifacts.theta0,
        "theta1": artifacts.theta1,
        "deck_cols": artifacts.deck_cols,
        "basep_gp_levels": artifacts.basep_calibration.gp_levels,
        "gamma": artifacts.basep_calibration.gamma,
        "phi": artifacts.basep_calibration.phi,
        "k_by_gp": artifacts.k_by_gp,
        "W": W_STOP,
        "L": L_STOP,
        "alpha": ALPHA,
        "beta": BETA,
    }
    import pickle

    with open("models/deck_effect_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # optionally, write the draft-level frame for diagnostics
    D.to_parquet("models/draft_level_deck_effect.parquet", index=False)


if __name__ == "__main__":
    main()
