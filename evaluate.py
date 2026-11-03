import argparse
import numpy as np
from pong_env import PongEnv, numTestGames, maxStepsPerGame, randomSeed, checkpointFile
from discretizer import stateToIndex
from q_agent import QAgent

def evaluate(agent, numGames=numTestGames, seed=randomSeed + 1):
    """
    Play num_games with the greedy policy (no exploring, no learning).
    Returns (bounces_per_game, num_hit_cap), where num_hit_cap counts games
    stopped by the step limit instead of by a miss.
    """
    # Different seed from training to test games are fresh
    env = PongEnv(seed=seed)
    bouncePerEpisode = np.zeros(numGames, dtype=np.int32)
    numHitCap = 0

    for game in range(numGames):
        state = env.reset()
        stateIndex = stateToIndex(state)
        done = False
        steps = 0

        while not done and steps < maxStepsPerGame:
            # Greedy action only: explore=False. No agent.update() anywhere here.
            action = agent.chooseActions(stateIndex, explore=False)
            state, _, done = env.step(action)
            stateIndex = stateToIndex(state)
            steps += 1
        
        if not done:
            numHitCap += 1
        bouncePerEpisode[game] = env.bounce

    return bouncePerEpisode, numHitCap

def printSummary(bounces, numHitCap):
    """Printing the statistics"""
    print(f"Games played: {len(bounces)}")
    print(f"Average bounces: {bounces.mean():.2f}")
    print(f"Std deviation: {bounces.std():.2f}")
    print(f"Min: {bounces.min()}")
    print(f"Median: {np.median(bounces):.0f}")
    print(f"Max: {bounces.max()}")
    print(f"Games hitting step cap: {numHitCap}")

    # Tiny text histogram: how often each bounce count occurred
    print("\nBounce distribution (bounces: games)")
    counts = np.bincount(bounces)
    for n, c in enumerate(counts):
        if c > 0:
            print(f"  {n:>3}: {c:>5}  {'#' * int(50 * c / counts.max())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=numTestGames, help="number of test games (assignment requires >= 1000)")
    parser.add_argument("--checkpoint", default=checkpointFile, help="Q-table file saved by train.py")
    args = parser.parse_args()

    agent = QAgent()
    agent.load(args.checkpoint)
    bounces, hit_cap = evaluate(agent, numGames=args.games)
    printSummary(bounces, hit_cap)