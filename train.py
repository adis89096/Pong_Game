import argparse
import csv
import time
from pathlib import Path
import numpy as np
from pong_env import PongEnv, numTrainEpisodes, maxStepsPerGame, randomSeed, checkpointFile
from discretizer import stateToIndex
from q_agent import QAgent
from evaluate import evaluate

# print progress every this many episodes
logEvery = 5000
# bounces per episode, for plotting later
curveFile = "learning_curve.npy"
resultFile = Path("results.csv")
numEvalGames = 3000

def train(numEpisode=numTrainEpisodes, seed=randomSeed, verbose=True, **agentKwargs):
    """Train a Q-learning agent. Returns (agent, bounces_per_episode)."""
    env = PongEnv(seed=seed)
    agent = QAgent(seed=seed, **agentKwargs)
    bouncePerEpisode = np.zeros(numEpisode, dtype=np.int32)

    startTime = time.time()
    for episode in range(numEpisode):
        state = env.reset()
        stateIndex = stateToIndex(state)
        done = False
        steps = 0

        while not done and steps < maxStepsPerGame:
            # 1. Agent picks a state index (explores with probability epsilon)
            action = agent.chooseActions(stateIndex)
            # 2. Game advances one tick
            nextState, reward, done = env.step(action)
            nextStateIndex = stateToIndex(nextState)
            # 3. Agent learns from (s, a, r, s'). Old state/action are still in hand.
            agent.update(stateIndex, action, reward, nextStateIndex, done)
            # 4. Move on
            stateIndex = nextStateIndex
            steps += 1
        
        bouncePerEpisode[episode] = env.bounce
        # once per episode
        agent.decay_epsilon()

        # Progress report: average bounces over the most recent window
        if verbose and (episode + 1) % logEvery == 0:
            recent = bouncePerEpisode[episode + 1 - logEvery: episode + 1]
            print(f"Episode {episode + 1:>7} | avg bounces (last {logEvery}): "
                  f"{recent.mean():6.2f} | epsilon: {agent.epsilon:.3f} | "
                  f"time: {time.time() - startTime:6.1f}s")
    
    return agent, bouncePerEpisode

def episodes_to_reach(curve, target, window=logEvery):
    """
    First episode where the trailing-window average reaches target.
    Returns None if it never does.
    """
    window = min(window, len(curve))
    sums = np.cumsum(np.insert(curve, 0, 0))
    moving_avg = (sums[window:] - sums[:-window]) / window
    hits = np.flatnonzero(moving_avg >= target)
    return int(hits[0] + window) if hits.size else None


def report_and_log(name, settings, curve, train_seconds, bounces, hit_cap):
    """Print the key metrics and append one row to results.csv."""
    row = {
        "name": name,
        "settings": str(settings),
        "eval_mean": round(float(bounces.mean()), 2),
        "eval_stderr": round(float(bounces.std() / np.sqrt(len(bounces))), 2),
        "episodes_to_9": episodes_to_reach(curve, 9),
        "episodes_to_12": episodes_to_reach(curve, 12),
        "train_seconds": round(train_seconds),
        "eval_median": int(np.median(bounces)),
        "eval_std": round(float(bounces.std()), 2),
        "hit_cap": hit_cap,
    }

    print(f"\n=== Results for '{name}' ===")
    print(f"Average bounces (eval): {row['eval_mean']} +/- {row['eval_stderr']}")
    print(f"Episodes to reach 9:    {row['episodes_to_9']}")
    print(f"Episodes to reach 12:   {row['episodes_to_12']}")
    print(f"Training time (s):      {row['train_seconds']}")
    print(f"Median / std:           {row['eval_median']} / {row['eval_std']}")
    print(f"Games hitting cap:      {row['hit_cap']}")

    # Append to the summary file, writing the header only the first time
    write_header = not resultFile.exists()
    with open(resultFile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=numTrainEpisodes, help="number of training games (use a small number to test)")
    parser.add_argument("--name", default="baseline", help="label for this run (also used in saved file names)")
    parser.add_argument("--seed", type=int, default=randomSeed)
    parser.add_argument("--c", type=float, help="learning-rate constant C")
    parser.add_argument("--gamma", type=float, help="discount factor")
    parser.add_argument("--decay", type=float, help="epsilon decay per episode")
    args = parser.parse_args()

    # Only pass along the settings you actually gave; the rest use defaults
    agent_kwargs = {}
    if args.c is not None:
        agent_kwargs["learningRate"] = args.c
    if args.gamma is not None:
        agent_kwargs["discountFactor"] = args.gamma
    if args.decay is not None:
        agent_kwargs["epsilonDecay"] = args.decay

    start = time.time()
    agent, curve = train(numEpisode=args.episodes, seed=args.seed, **agent_kwargs)
    train_seconds = time.time() - start

    agent.save(f"q_table_{args.name}.npy")
    np.save(f"curve_{args.name}.npy", curve)

    bounces, hit_cap = evaluate(agent, numGames=numEvalGames)
    settings = dict(agent_kwargs, seed=args.seed)
    report_and_log(args.name, settings, curve, train_seconds, bounces, hit_cap)