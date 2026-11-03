import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from pong_env import numTrainEpisodes, randomSeed
from train import train, episodes_to_reach
from evaluate import evaluate

RESULTS_DIR = Path("results")
SUMMARY_FILE = RESULTS_DIR / "summary.csv"
NUM_EVAL_GAMES = 3000       # same for every experiment so results are comparable
CURVE_WINDOW = 5000         # window for "average bounces during training"

# name -> settings. "seed" changes the training run's randomness; every other
# key is passed to QAgent. Anything left out uses the defaults in pong_env.py.
EXPERIMENTS = {
    "baseline": {},
    # Same settings, different seeds: measures run-to-run noise
    "baseline_seed1": {"seed": 1},
    "baseline_seed2": {"seed": 2},
    # Learning-rate constant C (baseline 50)
    "c10": {"learningRate": 10},
    "c200": {"learningRate": 200},
    # Discount factor (baseline 0.95)
    "gamma80": {"discountFactor": 0.80},
    "gamma90": {"discountFactor": 0.90},
    "gamma99": {"discountFactor": 0.99},
    # Epsilon decay per episode (baseline 0.99995)
    "decay9998": {"epsilonDecay": 0.9998},
    "decay9999": {"epsilonDecay": 0.9999},
}


def episodes_to_reach(curve, target, window=CURVE_WINDOW):
    """First episode where the trailing-window average reaches target.

    Returns None if it never does.
    """
    window = min(window, len(curve))
    sums = np.cumsum(np.insert(curve, 0, 0))
    moving_avg = (sums[window:] - sums[:-window]) / window
    hits = np.flatnonzero(moving_avg >= target)
    return int(hits[0] + window) if hits.size else None


def run_experiment(name, settings, num_episodes):
    """Train + evaluate one config, save its files, and log a summary row."""
    original = dict(settings)
    settings = dict(settings)
    seed = settings.pop("seed", randomSeed)

    start = time.time()
    agent, curve = train(numEpisode=num_episodes, seed=seed, verbose=False, **settings)
    train_seconds = time.time() - start

    bounces, hit_cap = evaluate(agent, numGames=NUM_EVAL_GAMES)

    RESULTS_DIR.mkdir(exist_ok=True)
    agent.save(RESULTS_DIR / f"{name}_q_table.npy")
    np.save(RESULTS_DIR / f"{name}_curve.npy", curve)

    row = {
        "name": name,
        "settings": json.dumps(original),
        "train_episodes": num_episodes,
        "train_seconds": round(train_seconds),
        "final_train_avg": round(float(curve[-CURVE_WINDOW:].mean()), 2),
        "episodes_to_9": episodes_to_reach(curve, 9),
        "episodes_to_12": episodes_to_reach(curve, 12),
        "eval_mean": round(float(bounces.mean()), 2),
        "eval_std": round(float(bounces.std()), 2),
        "eval_stderr": round(float(bounces.std() / np.sqrt(len(bounces))), 2),
        "eval_median": int(np.median(bounces)),
        "hit_cap": hit_cap,
    }

    # Append to the summary file, writing the header only the first time
    write_header = not SUMMARY_FILE.exists()
    with open(SUMMARY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", metavar="NAME", help="run only these experiments (default: all)")
    parser.add_argument("--episodes", type=int, default=numTrainEpisodes, help="training games per experiment")
    parser.add_argument("--list", action="store_true", help="print experiment names and exit")
    args = parser.parse_args()

    if args.list:
        print("\n".join(EXPERIMENTS))
        raise SystemExit

    names = args.only or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {unknown}")

    for name in names:
        print(f"\n=== {name}: {EXPERIMENTS[name]} ===")
        run_experiment(name, EXPERIMENTS[name], args.episodes)