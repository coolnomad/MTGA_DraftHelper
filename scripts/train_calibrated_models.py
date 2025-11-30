"""
Train calibrated skill and deck-effect models following the R pipeline:
- Out-of-fold hierarchical base_p by skill/experience buckets from games
- Stop-rule aggregation to p_mle per draft
- Calibration of base_p to p_mle (linear weighted)
- Skill model targets base_p_cal
- Deck model targets bump = p_mle - base_p_cal

Outputs:
- models/skill_model.pkl
- models/joint_model.pkl (predicts deck effect / bump)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.data.loaders import load_decks, load_games  # noqa: E402
from src.models.features import build_joint_features, build_skill_features, train_test_split_indices  # noqa: E402
import joblib  # noqa: E402


MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main(seed: int = 1337, test_frac: float = 0.2):
    games = load_games()
    decks = load_decks()

    base_p = _oof_base_p_two_buckets(games, seed=seed)
    games = games.assign(base_p=base_p)
    draft_stats = _draft_level_stop_stats(games)

    # merge draft-level stats
    decks = decks.merge(
        draft_stats[["draft_id", "base_p_cal", "p_mle"]],
        on="draft_id",
        how="inner",
    )

    skill_feats = build_skill_features(decks).fillna(0)
    joint_feats = build_joint_features(decks).fillna(0)

    y_skill = decks["base_p_cal"].to_numpy()
    y_bump = (decks["p_mle"] - decks["base_p_cal"]).to_numpy()

    train_idx, test_idx = train_test_split_indices(len(decks), seed=seed, test_frac=test_frac)

    # skill model tuning (ridge alpha)
    alphas = [0.1, 1.0, 10.0]
    best_alpha, _ = _tune_ridge(skill_feats, y_skill, alphas, seed=seed)
    skill_model = Ridge(alpha=best_alpha)
    skill_model.fit(skill_feats.iloc[train_idx], y_skill[train_idx])
    joblib.dump(skill_model, MODELS_DIR / "skill_model.pkl")

    # deck effect model (predict bump) hyperparam sweep (shallow grid, subsampled for speed)
    xgb_params_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 400},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 500},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 400},
    ]
    best_params, _ = _tune_xgb(joint_feats, y_bump, xgb_params_grid, seed=seed)
    joint_model = XGBRegressor(
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        **best_params,
    )
    joint_model.fit(joint_feats.iloc[train_idx], y_bump[train_idx])
    joblib.dump(joint_model, MODELS_DIR / "joint_model.pkl")

    return {"train_size": len(train_idx), "test_size": len(test_idx)}


def _tune_ridge(X: pd.DataFrame, y: np.ndarray, alphas, seed: int = 1337):
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    best_alpha = None
    best_r2 = -np.inf
    for alpha in alphas:
        r2s = []
        model = Ridge(alpha=alpha)
        for tr, va in kf.split(X):
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
            r2s.append(_r2(y[va], pred))
        mean_r2 = float(np.mean(r2s))
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_alpha = alpha
    return best_alpha, best_r2


def _tune_xgb(X: pd.DataFrame, y: np.ndarray, grid, seed: int = 1337):
    # subsample for speed if very large
    max_rows = 25000
    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X_sub = X.iloc[idx]
        y_sub = y[idx]
    else:
        X_sub, y_sub = X, y

    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    best_params = None
    best_r2 = -np.inf
    for params in grid:
        r2s = []
        for tr, va in kf.split(X_sub):
            model = XGBRegressor(
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=-1,
                **params,
            )
            model.fit(X_sub.iloc[tr], y_sub[tr])
            pred = model.predict(X_sub.iloc[va])
            r2s.append(_r2(y_sub[va], pred))
        mean_r2 = float(np.mean(r2s))
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_params = params
    return best_params, best_r2


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-9)


def _oof_base_p_two_buckets(
    games: pd.DataFrame,
    draft_col: str = "draft_id",
    won_col: str = "won",
    wr_col: str = "user_game_win_rate_bucket",
    gp_col: str = "user_n_games_bucket",
    k_map: Dict[str, float] | None = None,
    default_k: float = 40.0,
    k_folds: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
    seed: int = 1337,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    drafts = games[draft_col].astype(str).unique()
    rng.shuffle(drafts)
    fold_ids = {d: i % k_folds for i, d in enumerate(drafts)}
    fold = games[draft_col].astype(str).map(fold_ids).to_numpy()

    won01 = games[won_col].astype(int).to_numpy()
    wr = games[wr_col].astype(str).to_numpy()
    gp = games[gp_col].astype(str).to_numpy()
    key = np.char.add(np.char.add(wr, "||"), gp)

    if k_map is None:
        k_map = {"1": 15, "5": 30, "10": 55, "50": 75, "100": 95, "500": 115, "1000": 125}

    def get_k(g):
        return k_map.get(g, default_k)

    base_hat = np.zeros(len(games))
    for k in range(k_folds):
        tr = fold != k
        va = fold == k

        df_tr = pd.DataFrame({"wr": wr[tr], "won": won01[tr]})
        grouped_wr = df_tr.groupby("wr")["won"].agg(["sum", "count"])
        base_wr = (grouped_wr["sum"] + alpha) / (grouped_wr["count"] + alpha + beta)

        df_joint = pd.DataFrame({"key": key[tr], "wr": wr[tr], "gp": gp[tr], "won": won01[tr]})
        grouped_joint = df_joint.groupby("key")["won"].agg(["sum", "count"])
        wr_of_key = grouped_joint.index.to_series().str.split("||").str[0]
        gp_of_key = grouped_joint.index.to_series().str.split("||").str[1]
        m0_wr = base_wr.reindex(wr_of_key).fillna(base_wr.mean())
        k_gp = gp_of_key.map(get_k).astype(float)
        base_joint = (grouped_joint["sum"] + k_gp.to_numpy() * m0_wr.to_numpy() + alpha) / (
            grouped_joint["count"] + k_gp.to_numpy() + alpha + beta
        )
        base_joint.index = grouped_joint.index

        base_global = (grouped_joint["sum"].sum() + alpha) / (grouped_joint["count"].sum() + alpha + beta)

        keys_va = key[va]
        wr_va = wr[va]
        bh = pd.Series(base_joint).reindex(keys_va).to_numpy()
        miss = np.isnan(bh)
        if miss.any():
            bh[miss] = base_wr.reindex(wr_va[miss]).fillna(base_global).to_numpy()
        base_hat[va] = bh
    return np.clip(base_hat, 1e-6, 1 - 1e-6)


def _draft_level_stop_stats(games: pd.DataFrame) -> pd.DataFrame:
    def _stop_counts(df):
        df = df.sort_values("match_number")
        wins = 0
        losses = 0
        for _, row in df.iterrows():
            wins += int(row["won"])
            losses += int(not row["won"])
            if wins >= 7 or losses >= 3:
                break
        return pd.Series({"w_stop": wins, "l_stop": losses})

    stop_df = games.groupby("draft_id", observed=False).apply(_stop_counts, include_groups=False).reset_index()
    base_means = games.groupby("draft_id")["base_p"].mean().reset_index(name="base_p")
    out = stop_df.merge(base_means, on="draft_id", how="left")
    out["A"] = np.where(out["w_stop"] >= 7, 7, out["w_stop"])
    out["B"] = np.where(out["l_stop"] >= 3, 3, out["l_stop"])
    out["p_mle"] = out["A"] / (out["A"] + out["B"])

    weights = out["A"] + out["B"]
    X = out["base_p"].to_numpy()
    y = out["p_mle"].to_numpy()
    w = weights.to_numpy()
    w_norm = w / (w.sum() + 1e-9)
    x_bar = np.sum(w_norm * X)
    y_bar = np.sum(w_norm * y)
    slope = np.sum(w_norm * (X - x_bar) * (y - y_bar)) / (np.sum(w_norm * (X - x_bar) ** 2) + 1e-9)
    intercept = y_bar - slope * x_bar
    out["base_p_cal"] = np.clip(intercept + slope * out["base_p"], 1e-6, 1 - 1e-6)
    return out


if __name__ == "__main__":
    print(main())
