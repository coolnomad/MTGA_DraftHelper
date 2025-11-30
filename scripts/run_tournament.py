"""
Run multi-seat drafts with bots, build decks, and report evaluator scores.

Usage:
  python scripts/run_tournament.py --games 10 --policy hero
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from draft_env.env import DraftEnv
from hero_bot.hero_policy import hero_policy
from human_policy.bc_policy import make_bc_policy
from human_policy.random_policy import random_policy
from hero_bot.deck_builder import build_deck
from deck_eval.evaluator import evaluate_deck, evaluate_deck_bump
from scripts.self_play_logger import log_picks


POLICY_MAP = {
    "hero": hero_policy,
    "bc": make_bc_policy(),
    "random": random_policy,
}


def run_table(policy_name: str, set_code: str = "FIN", seed: int = 0, log_picks: bool = False) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    policies = [POLICY_MAP.get(policy_name, hero_policy) for _ in range(8)]
    env = DraftEnv(seed=seed, set_code=set_code)
    pools, picks = env.run_draft(policies=policies, seed=seed, log_picks=log_picks)
    results = []
    for seat, pool in enumerate(pools):
        deck = build_deck(pool)
        score = evaluate_deck(deck)  # calibrated win rate
        bump = evaluate_deck_bump(deck)  # raw deck-effect score (s)
        results.append(
            {
                "seat": seat,
                "policy": policy_name,
                "deck_effect": score,
                "deck_bump": bump,
                "deck_size": sum(deck.values()), 
                "n_pool_cards": sum(pool.values()),
            }
        )
    # backfill outcomes onto picks for this table
    if log_picks and picks:
        picks_with_outcome = []
        for log in picks:
            seat = log.get("seat")
            # find matching result
            seat_res = next((r for r in results if r["seat"] == seat), None)
            if seat_res:
                log = dict(log)
                log["deck_effect"] = seat_res["deck_effect"]
                log["deck_bump"] = seat_res["deck_bump"]
            picks_with_outcome.append(log)
        picks = picks_with_outcome
    return results, picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5, help="number of draft tables to run")
    parser.add_argument("--policy", type=str, default="hero", choices=list(POLICY_MAP.keys()))
    parser.add_argument("--set", dest="set_code", type=str, default="FIN")
    parser.add_argument("--log_picks", action="store_true", help="if set, log picks to replay parquet")
    args = parser.parse_args()

    all_results = []
    all_picks = []
    for i in range(args.games):
        res, picks = run_table(args.policy, args.set_code, seed=i, log_picks=args.log_picks)
        all_results.extend(res)
        all_picks.extend(picks)

    df = pd.DataFrame(all_results)
    if df.empty:
        print("no results")
        return
    print(df.groupby("policy")["deck_effect"].describe())
    out_path = Path("reports") / f"tournament_{args.policy}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved results to {out_path}")
    if args.log_picks and all_picks:
        log_picks(all_picks, Path("reports") / f"replay_{args.policy}.parquet")
        print("Saved replay to", Path("reports") / f"replay_{args.policy}.parquet")


if __name__ == "__main__":
    main()
