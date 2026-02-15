"""
Run one or more 8-seat pods with 1 human seat and 7 bot seats.

Usage example:
PYTHONPATH=. python scripts/run_pod_human.py --num_pods 1 --human_seat 0 --bot_policies hero,hero,hero,human,random,hero,human --format FIN --output data/processed/human_pods.parquet --seed 1337
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from draft_env.env import DraftEnv
from hero_bot.hero_policy import hero_policy
from hero_bot.hero_policy import hero_policy_distill
from human_policy.random_policy import random_policy
from human_policy.bc_policy import make_bc_policy
from hero_bot.deck_builder import build_deck
from deck_eval.evaluator import evaluate_deck, evaluate_deck_bump
from hero_bot.pool_evaluator import evaluate_pool_effect, evaluate_pool_bump
from draft_env.pack_sampler import sample_pack
from hero_bot.hero_policy import hero_policy_soft


# Policy adapters
def human_policy_bot(pack_cards: List[str], pool_counts: Dict[str, int], seat_idx: int, rng: np.random.Generator) -> str:
    bc = make_bc_policy()
    ranked = bc(pack_cards, pool_counts)
    return ranked[0][0] if ranked else pack_cards[0]


POLICY_MAP: Dict[str, Callable[[List[str], Dict[str, int], int, np.random.Generator], str]] = {
    "hero": hero_policy,
    "hero_soft": hero_policy_soft,
    "hero_distill": hero_policy_distill,
    "human": human_policy_bot,
    "random": random_policy,
}


@dataclass
class PickLog:
    pod_id: int
    seat_idx: int
    policy_name: str
    pack_number: int
    pick_number: int
    fmt: str
    pack_card_ids: List[str]
    pool_before: Dict[str, int]
    chosen_card: str
    pool_after: Dict[str, int]
    hero_value_before: float | None = None
    hero_value_after: float | None = None
    hero_delta: float | None = None


def run_pod(
    pod_id: int,
    human_seat: int,
    bot_policies: List[str],
    set_code: str,
    seed: int,
    auto_human: bool = False,
    base_p: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    # policies list length 7 for bots; insert human placeholder
    bot_funcs = [POLICY_MAP.get(p, hero_policy) for p in bot_policies]
    policies: List[Callable] = []
    b_iter = iter(bot_funcs)
    for seat in range(8):
        if seat == human_seat:
            policies.append(None)
        else:
            policies.append(next(b_iter))

    env = DraftEnv(seed=seed, set_code=set_code)
    pools: List[Dict[str, int]] = [dict() for _ in range(8)]
    pick_logs: List[PickLog] = []

    # pack 1 -> pass left, pack 2 -> pass right, pack 3 -> pass left
    for pack_num in range(env.packs_per_player):
        direction = 1 if pack_num % 2 == 0 else -1
        current_packs = [sample_pack(env.rng, env.pack_size, set_code=env.set_code) for _ in range(env.num_seats)]
        for pick in range(env.pack_size):
            new_packs = [None] * env.num_seats
            for seat in range(env.num_seats):
                pack_cards = current_packs[seat]
                if not pack_cards:
                    continue
                pool_before = dict(pools[seat])
                pool_effect_before, pool_bump_before = evaluate_pool_effect(pool_before, base_p=base_p)
                pool_bump_before_val = evaluate_pool_bump(pool_before, base_p=base_p)[0]
                if seat == human_seat:
                    if auto_human:
                        chosen = pack_cards[0]
                    else:
                        chosen = prompt_human(
                            pack_cards,
                            pools[seat],
                            pod_id,
                            seat,
                            pack_num + 1,
                            pick + 1,
                            base_p=base_p,
                        )
                else:
                    chosen = policies[seat](pack_cards, pools[seat], seat, rng)
                if chosen not in pack_cards:
                    chosen = rng.choice(pack_cards)
                pools[seat][chosen] = pools[seat].get(chosen, 0) + 1
                pool_effect_after, pool_bump_after = evaluate_pool_effect(pools[seat], base_p=base_p)
                pool_bump_after_val = evaluate_pool_bump(pools[seat], base_p=base_p)[0]
                pick_logs.append(
                    PickLog(
                        pod_id=pod_id,
                        seat_idx=seat,
                        policy_name="human" if seat == human_seat else bot_policies[seat if seat < human_seat else seat - 1],
                        pack_number=pack_num + 1,
                        pick_number=pick + 1,
                        fmt=set_code,
                        pack_card_ids=list(pack_cards),
                        pool_before=pool_before,
                        chosen_card=chosen,
                        pool_after=dict(pools[seat]),
                        hero_value_before=pool_effect_before,
                        hero_value_after=pool_effect_after,
                        hero_delta=pool_effect_after - pool_effect_before,
                    )
                )
                pack_cards.remove(chosen)
                new_packs[seat] = pack_cards
            # pass packs
            passed = [None] * env.num_seats
            for seat in range(env.num_seats):
                target = (seat + direction) % env.num_seats
                passed[target] = new_packs[seat]
            current_packs = passed
    results = []
    for seat_idx, pool in enumerate(pools):
        deck = build_deck(pool)
        deck_effect = evaluate_deck(deck)
        deck_bump = evaluate_deck_bump(deck)
        pool_effect, pool_effect_delta = evaluate_pool_effect(pool, base_p=base_p)
        pool_bump, pool_bump_delta = evaluate_pool_bump(pool, base_p=base_p)
        results.append(
            {
                "pod_id": pod_id,
                "seat_idx": seat_idx,
                "policy": "human" if seat_idx == human_seat else bot_policies[seat_idx if seat_idx < human_seat else seat_idx - 1],
                "pool": pool,
                "deck": deck,
                "deck_effect": deck_effect,
                "deck_bump": deck_bump,
                "pool_effect": pool_effect,
                "pool_effect_delta": pool_effect_delta,
                "pool_bump": pool_bump,
                "pool_bump_delta": pool_bump_delta,
            }
        )
    return results, [log.__dict__ for log in pick_logs]


def prompt_human(
    pack_cards: List[str],
    pool: Dict[str, int],
    pod_id: int,
    seat_idx: int,
    pack_number: int,
    pick_number: int,
    base_p: float,
) -> str:
    print(f"Pod {pod_id} | Seat {seat_idx} | Pack {pack_number} | Pick {pick_number}")
    print(f"Pool size: {sum(pool.values())}")
    try:
        cur_eff, cur_eff_delta = evaluate_pool_effect(pool, base_p=base_p)
        cur_bump, cur_bump_delta = evaluate_pool_bump(pool, base_p=base_p)
        print(f"Pool value -> effect={cur_eff:.3f} (delta {cur_eff_delta:+.3f}) | bump={cur_bump:.3f} (delta {cur_bump_delta:+.3f})")
    except Exception:
        pass
    projections = []
    for c in pack_cards:
        pool_plus = dict(pool)
        pool_plus[c] = pool_plus.get(c, 0) + 1
        try:
            eff, eff_delta = evaluate_pool_effect(pool_plus, base_p=base_p)
            bump, bump_delta = evaluate_pool_bump(pool_plus, base_p=base_p)
            projections.append((c, eff, eff_delta, bump, bump_delta))
        except Exception:
            projections.append((c, None, None, None, None))

    for i, c in enumerate(pack_cards, 1):
        proj = projections[i - 1]
        if proj[1] is not None:
            print(f"  [{i}] {c} -> effect={proj[1]:.3f} (Δ{proj[2]:+.3f}) | bump={proj[3]:.3f} (Δ{proj[4]:+.3f})")
        else:
            print(f"  [{i}] {c}")
    while True:
        s = input("Enter pick index (or q to quit): ").strip()
        if s.lower() == "q":
            print("Quitting draft.")
            sys.exit(0)
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(pack_cards):
                return pack_cards[idx - 1]
        print("Invalid input.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pods", type=int, default=1)
    parser.add_argument("--human_seat", type=int, default=0)
    parser.add_argument("--bot_policies", type=str, required=True, help="Comma-separated 7 policies (hero|human|random)")
    parser.add_argument("--format", type=str, default="FIN")
    parser.add_argument("--output", type=str, default="data/processed/human_pods.parquet")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--auto_human", action="store_true", help="Auto-pick first card for human (for smoke tests)")
    parser.add_argument("--base_p", type=float, default=0.5, help="Baseline skill probability for pool models")
    args = parser.parse_args()

    bots = args.bot_policies.split(",")
    if len(bots) != 7:
        raise ValueError("bot_policies must have 7 entries (for non-human seats)")
    if not (0 <= args.human_seat <= 7):
        raise ValueError("human_seat must be in [0,7]")

    all_results = []
    all_picks = []
    for pod in range(args.num_pods):
        res, picks = run_pod(
            pod_id=pod,
            human_seat=args.human_seat,
            bot_policies=bots,
            set_code=args.format,
            seed=args.seed + pod,
            auto_human=args.auto_human,
            base_p=args.base_p,
        )
        all_results.extend(res)
        all_picks.extend(picks)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_parquet(out_path, index=False)
    picks_path = out_path.with_name(out_path.stem + "_picks.parquet")
    pd.DataFrame(all_picks).to_parquet(picks_path, index=False)
    print(f"Saved results to {out_path} and picks to {picks_path}")


if __name__ == "__main__":
    main()
def sample_pack_for_seat(env: DraftEnv, pack_size: int) -> List[str]:
    return sample_pack(env.rng, pack_size, set_code=env.set_code)
