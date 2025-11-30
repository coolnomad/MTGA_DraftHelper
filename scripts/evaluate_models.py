"""
Offline evaluation script for M1/M2/M3 with reporting artifacts.

Outputs:
- reports/model_metrics.json : key metrics, CV stats, distribution summaries
- reports/model_bins_net.csv : 20-bin calibration for joint_pred vs run_wr
- reports/model_bins_boost.csv : 20-bin calibration for deck_boost vs observed bump
- reports/model_report.pdf : minimal PDF summary of results
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.data.loaders import load_decks, load_games
from src.models.features import (
    build_joint_features,
    build_skill_features,
    train_test_split_indices,
)

MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


@dataclass
class SplitMetrics:
    r2: float
    rmse: float
    mae: float

    def to_dict(self, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_R2": self.r2,
            f"{prefix}_RMSE": self.rmse,
            f"{prefix}_MAE": self.mae,
        }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> SplitMetrics:
    return SplitMetrics(
        r2=r2_score(y_true, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
    )


def _load_models():
    skill_model = joblib.load(MODELS_DIR / "skill_model.pkl")
    joint_model = joblib.load(MODELS_DIR / "joint_model.pkl")
    return skill_model, joint_model


def _calibration_bins(
    pred: np.ndarray, obs: np.ndarray, weights: np.ndarray | None = None, n_bins: int = 20
) -> pd.DataFrame:
    if weights is None:
        weights = np.ones_like(pred)
    df = pd.DataFrame({"pred": pred, "obs": obs, "w": weights})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    agg = df.groupby("bin", observed=False).apply(
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
    agg = agg.reset_index()
    return agg


def _weighted_regression(pred: np.ndarray, obs: np.ndarray, weights: np.ndarray | None = None):
    if weights is None:
        weights = np.ones_like(pred)
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


def _simulate_stop_rule_r2(deck_boost_pred: np.ndarray, skill_pred: np.ndarray, runs: int = 200):
    """
    Estimate an upper-bound R2/RMSE ceiling due to stop-rule noise by treating
    joint_pred as true p, simulating many 7/3 runs, and comparing observed bumps
    to the presumed true deck boost (deck_boost_pred).
    """
    p_true = np.clip(skill_pred + deck_boost_pred, 1e-6, 1 - 1e-6)

    def simulate_once(p):
        wins = 0
        losses = 0
        while wins < 7 and losses < 3:
            if np.random.rand() < p:
                wins += 1
            else:
                losses += 1
        return wins / (wins + losses)

    all_bumps = []
    for _ in range(runs):
        wr_sim = np.array([simulate_once(p) for p in p_true])
        bump_sim = wr_sim - skill_pred
        all_bumps.append(bump_sim)
    sim_bumps = np.vstack(all_bumps)
    noise = sim_bumps - deck_boost_pred  # deviation from presumed true boost
    noise_var = float(np.var(noise))
    signal_var = float(np.var(deck_boost_pred))
    r2_upper = signal_var / (signal_var + noise_var + 1e-9)
    rmse_floor = float(np.sqrt(noise_var))
    return {"R2_upper": r2_upper, "RMSE_floor": rmse_floor}


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
    """Out-of-fold hierarchical base_p akin to R pipeline."""
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
        # rough mapping inspired by R K_MAP
        k_map = {"1": 15, "5": 30, "10": 55, "50": 75, "100": 95, "500": 115, "1000": 125}
    def get_k(g):
        return k_map.get(g, default_k)

    base_hat = np.zeros(len(games))
    for k in range(k_folds):
        tr = fold != k
        va = fold == k

        # WR marginal
        df_tr = pd.DataFrame({"wr": wr[tr], "won": won01[tr]})
        grouped_wr = df_tr.groupby("wr")["won"].agg(["sum", "count"])
        base_wr = (grouped_wr["sum"] + alpha) / (grouped_wr["count"] + alpha + beta)

        # joint
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
    """Compute stop-rule stats and calibrate base_p vs p_mle."""
    # compute w_stop, l_stop per draft
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

    stop_df = games.groupby("draft_id").apply(_stop_counts).reset_index()
    # aggregate base_p mean per draft
    base_means = games.groupby("draft_id")["base_p"].mean().reset_index(name="base_p")
    out = stop_df.merge(base_means, on="draft_id", how="left")
    out["A"] = np.where(out["w_stop"] >= 7, 7, out["w_stop"])
    out["B"] = np.where(out["l_stop"] >= 3, 3, out["l_stop"])
    out["p_mle"] = out["A"] / (out["A"] + out["B"])

    # weighted linear fit p_mle ~ base_p to calibrate
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


def run_evaluation():
    decks = load_decks()
    games = load_games()

    # build OOF hierarchical base_p from games (skill + experience buckets)
    base_p_game = _oof_base_p_two_buckets(games)
    games = games.assign(base_p=base_p_game)
    draft_stats = _draft_level_stop_stats(games)
    # merge calibrated base_p into decks
    decks = decks.merge(
        draft_stats[["draft_id", "base_p", "base_p_cal", "p_mle"]],
        on="draft_id",
        how="inner",
    )

    y = decks["p_mle"].astype(float).to_numpy()
    skill_features = build_skill_features(decks).fillna(0)
    joint_features = build_joint_features(decks).fillna(0)

    # load trained models
    skill_model, joint_model = _load_models()
    skill_pred_model = skill_model.predict(skill_features)
    joint_pred = joint_model.predict(joint_features)

    # joint model predicts deck effect (bump); skill predicts calibrated baseline
    deck_effect_pred = joint_model.predict(joint_features)
    skill_pred = skill_model.predict(skill_features)
    skill_pred_cal = decks["base_p_cal"].to_numpy()
    # align skill_pred to calibrated baseline if model differs
    skill_pred = skill_pred_cal
    joint_pred_cal = skill_pred + deck_effect_pred
    deck_boost_pred = deck_effect_pred
    observed_boost = y - skill_pred  # observed delta relative to calibrated skill baseline

    # deterministic holdout metrics using same split
    train_idx, test_idx = train_test_split_indices(len(decks), seed=1337, test_frac=0.2)
    y_test = y[test_idx]
    skill_test = skill_pred[test_idx]
    joint_test = joint_pred_cal[test_idx]
    boost_test = deck_boost_pred[test_idx]
    obs_boost_test = observed_boost[test_idx]

    m_skill = _metrics(y_test, skill_test)
    m_joint = _metrics(y_test, joint_test)
    m_boost = _metrics(obs_boost_test, boost_test)

    delta_r2_deck = m_joint.r2 - m_skill.r2

    # CV metrics
    cv_skill: List[float] = []
    cv_joint: List[float] = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1337)
    for train, test in kf.split(joint_features):
        Xs_tr, Xs_te = skill_features.iloc[train], skill_features.iloc[test]
        Xj_tr, Xj_te = joint_features.iloc[train], joint_features.iloc[test]
        y_tr, y_te = y[train], y[test]

        sm = Ridge(alpha=1.0)
        sm.fit(Xs_tr, y_tr)
        cv_skill.append(r2_score(y_te, sm.predict(Xs_te)))

        jm = XGBRegressor(
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=200,
            objective="reg:squarederror",
            random_state=1337,
            n_jobs=-1,
        )
        jm.fit(Xj_tr, y_tr)
        cv_joint.append(r2_score(y_te, jm.predict(Xj_te)))

    # distributions
    deck_boost_stats = {
        "mean": float(np.mean(deck_boost_pred)),
        "var": float(np.var(deck_boost_pred)),
        "pct_positive": float(np.mean(deck_boost_pred > 0)),
        "p50": float(np.percentile(deck_boost_pred, 50)),
        "p90": float(np.percentile(deck_boost_pred, 90)),
        "p95": float(np.percentile(deck_boost_pred, 95)),
    }

    # calibration bins
    weights = decks["n_games"].to_numpy() if "n_games" in decks.columns else None
    bins_net = _calibration_bins(joint_pred_cal, y, weights=weights)
    bins_boost = _calibration_bins(deck_boost_pred, observed_boost, weights=weights)
    bins_net.to_csv(REPORTS_DIR / "model_bins_net.csv", index=False)
    bins_boost.to_csv(REPORTS_DIR / "model_bins_boost.csv", index=False)
    reg_net = _weighted_regression(joint_pred_cal, y, weights=weights)
    reg_boost = _weighted_regression(deck_boost_pred, observed_boost, weights=weights)
    _save_calibration_svg(
        bins_net,
        REPORTS_DIR / "model_calibration_net.svg",
        title="Calibration: joint_pred vs observed run_wr",
        xlabel="Predicted run_wr",
        ylabel="Observed run_wr",
        reg=reg_net,
    )
    _save_calibration_svg(
        bins_boost,
        REPORTS_DIR / "model_calibration_boost.svg",
        title="Calibration: deck_boost vs observed bump",
        xlabel="Predicted deck_boost",
        ylabel="Observed deck_bump (run_wr - skill_pred)",
        reg=reg_boost,
    )

    # feature importance (gain) from trained joint model if available
    feature_importance = {}
    if hasattr(joint_model, "get_booster"):
        booster = joint_model.get_booster()
        imp = booster.get_score(importance_type="gain")
        feature_importance = dict(
            sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:30]
        )

    noise_bounds = _simulate_stop_rule_r2(deck_boost_pred, skill_pred)

    metrics_out = {
        **m_skill.to_dict("skill_test"),
        **m_joint.to_dict("joint_test"),
        **m_boost.to_dict("deck_boost_test"),
        "delta_R2_deck": delta_r2_deck,
        "cv_R2_skill_mean": float(np.mean(cv_skill)),
        "cv_R2_skill_std": float(np.std(cv_skill)),
        "cv_R2_joint_mean": float(np.mean(cv_joint)),
        "cv_R2_joint_std": float(np.std(cv_joint)),
        "deck_boost_stats": deck_boost_stats,
        "feature_importance_top30": feature_importance,
        "calibration_reg_net": reg_net,
        "calibration_reg_boost": reg_boost,
        "noise_bounds": noise_bounds,
        "skill_baseline": "base_p_cal",
    }

    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    # Build a minimal PDF summary with key metrics
    lines = [
        "MTGA Draft Helper Model Evaluation",
        f"Skill model: R2={m_skill.r2:.3f}, RMSE={m_skill.rmse:.3f}, MAE={m_skill.mae:.3f}",
        f"Joint model: R2={m_joint.r2:.3f}, RMSE={m_joint.rmse:.3f}, MAE={m_joint.mae:.3f}",
        f"Delta R2 (deck contribution): {delta_r2_deck:.3f}",
        f"Deck boost vs observed bump: R2={m_boost.r2:.3f}, RMSE={m_boost.rmse:.3f}, MAE={m_boost.mae:.3f}",
        f"CV skill R2 mean±std: {np.mean(cv_skill):.3f} ± {np.std(cv_skill):.3f}",
        f"CV joint R2 mean±std: {np.mean(cv_joint):.3f} ± {np.std(cv_joint):.3f}",
        f"Deck boost mean={deck_boost_stats['mean']:.3f}, var={deck_boost_stats['var']:.4f}, pct>0={deck_boost_stats['pct_positive']:.3f}",
        f"Calibration net slope={reg_net['slope']:.3f}, intercept={reg_net['intercept']:.3f}, R2={reg_net['R2']:.3f}",
        f"Calibration boost slope={reg_boost['slope']:.3f}, intercept={reg_boost['intercept']:.3f}, R2={reg_boost['R2']:.3f}",
        f"Noise ceiling (stop rule): R2_upper={noise_bounds['R2_upper']:.3f}, RMSE_floor={noise_bounds['RMSE_floor']:.3f}",
        "Calibration bins saved: model_bins_net.csv, model_bins_boost.csv",
    ]
    _write_simple_pdf(REPORTS_DIR / "model_report.pdf", lines)

    return metrics_out


def _write_simple_pdf(path: Path, lines: List[str]):
    """
    Write a single-page PDF with basic text using only the standard library.
    This avoids external dependencies (matplotlib/reportlab).
    """
    # escape parentheses
    safe_lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    content_lines = []
    y = 770
    for line in safe_lines:
        content_lines.append(f"BT /F1 12 Tf 50 {y} Td ({line}) Tj ET")
        y -= 16
    content = "\n".join(content_lines)
    stream = f"{content}"
    objects = []

    # 1: catalog
    objects.append("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    # 2: pages
    objects.append("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    # 3: page
    objects.append(
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj"
    )
    # 4: font
    objects.append("4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    # 5: content stream
    objects.append(f"5 0 obj << /Length {len(stream.encode('utf-8'))} >> stream\n{stream}\nendstream endobj")

    # build xref
    xref_positions = []
    pdf_parts = ["%PDF-1.4"]
    for obj in objects:
        xref_positions.append(sum(len(p.encode("utf-8")) + 1 for p in pdf_parts))  # +1 for newline
        pdf_parts.append(obj)
    xref_start = sum(len(p.encode("utf-8")) + 1 for p in pdf_parts)
    pdf_parts.append("xref")
    pdf_parts.append(f"0 {len(objects)+1}")
    pdf_parts.append("0000000000 65535 f ")
    for pos in xref_positions:
        pdf_parts.append(f"{pos:010d} 00000 n ")
    pdf_parts.append("trailer")
    pdf_parts.append(f"<< /Size {len(objects)+1} /Root 1 0 R >>")
    pdf_parts.append("startxref")
    pdf_parts.append(str(xref_start))
    pdf_parts.append("%%EOF")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for part in pdf_parts:
            f.write(part.encode("utf-8"))
            f.write(b"\n")


def _save_calibration_svg(bins: pd.DataFrame, path: Path, title: str, xlabel: str, ylabel: str, reg: Dict[str, float] | None = None):
    """
    Write a simple SVG scatter with 1:1 reference using only stdlib.
    Expects bins with pred_mean and obs_mean columns.
    """
    width, height = 700, 500
    margin = 60
    xmin = float(min(bins["pred_mean"].min(), bins["obs_mean"].min()))
    xmax = float(max(bins["pred_mean"].max(), bins["obs_mean"].max()))
    padding = 0.02
    x0, x1 = xmin - padding, xmax + padding
    y0, y1 = x0, x1  # same scale for 1:1

    def scale_x(x):
        return margin + (x - x0) / (x1 - x0 + 1e-9) * (width - 2 * margin)

    def scale_y(y):
        return height - margin - (y - y0) / (y1 - y0 + 1e-9) * (height - 2 * margin)

    circles = []
    for _, row in bins.iterrows():
        cx = scale_x(row["pred_mean"])
        cy = scale_y(row["obs_mean"])
        r = 4
        circles.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="#1f77b4" stroke="none" />')

    line = f'<line x1="{scale_x(x0):.2f}" y1="{scale_y(y0):.2f}" x2="{scale_x(x1):.2f}" y2="{scale_y(y1):.2f}" stroke="#888" stroke-dasharray="4 2" />'
    reg_line = ""
    reg_text = ""
    if reg is not None:
        y_start = reg["intercept"] + reg["slope"] * x0
        y_end = reg["intercept"] + reg["slope"] * x1
        reg_line = f'<line x1="{scale_x(x0):.2f}" y1="{scale_y(y_start):.2f}" x2="{scale_x(x1):.2f}" y2="{scale_y(y_end):.2f}" stroke="#d62728" stroke-width="1.5" />'
        reg_text = f'slope={reg["slope"]:.3f}, intercept={reg["intercept"]:.3f}, R2={reg["R2"]:.3f}, RMSE={reg["RMSE"]:.3f}'

    text = [
        f'<text x="{width/2:.1f}" y="20" text-anchor="middle" font-family="Helvetica" font-size="16">{title}</text>',
        f'<text x="{width/2:.1f}" y="{height-15}" text-anchor="middle" font-family="Helvetica" font-size="12">{xlabel}</text>',
        f'<text x="15" y="{height/2:.1f}" transform="rotate(-90 15,{height/2:.1f})" text-anchor="middle" font-family="Helvetica" font-size="12">{ylabel}</text>',
        f'<text x="{width/2:.1f}" y="{height-30}" text-anchor="middle" font-family="Helvetica" font-size="11" fill="#d62728">{reg_text}</text>' if reg else "",
    ]

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            line,
            reg_line,
            *circles,
            *text,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    results = run_evaluation()
    print(json.dumps(results, indent=2))
