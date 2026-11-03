import numpy as np
from discretizer import numStates, numActions
from pong_env import randomSeed

# Q-Learning Hyperparameters
learningRateC = 50
discountFactor = 0.95
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 0.99995

class QAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""
    def __init__(self, seed=randomSeed, learningRate=learningRateC, discountFactor=discountFactor, epsilonStart=epsilonStart, epsilonEnd=epsilonEnd, epsilonDecay=epsilonDecay):
        # Hyperparameters default to the constants
        self.learningRateC = learningRate
        self.discountFactor = discountFactor
        self.epsilonEnd = epsilonEnd
        self.epsilonDecay = epsilonDecay
        # Q(s, a): all zeros, as the assignment requires
        self.qTable = np.zeros((numStates, numActions))
        # N(s, a): number of times each state-action pair has been updated
        self.visitCounts = np.zeros((numStates, numActions), dtype=np.int64)
        self.epsilon = epsilonStart
        self.rng = np.random.default_rng(seed)

    def bestAction(self, stateIndex):
        """Return the action with the highest Q-value, breaking ties randomly."""
        qValues = self.qTable[stateIndex]
        best_actions = np.flatnonzero(qValues == qValues.max())
        return int(self.rng.choice(best_actions))
    
    def chooseActions(self, stateIndex, explore=True):
        """
        Epsilon-greedy action selection.
        Set explore=False for evaluation: always pick the greedy action.
        """
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(numActions))
        return self.bestAction(stateIndex)
    
    def update(self, stateIndex, action, reward, nextStateIndex, done):
        """Q-learning update for one observed transition (s, a, r, s')."""
        # Learning rate decays with the number of visits: alpha = C / (C + N)
        n = self.visitCounts[stateIndex, action]
        alpha = self.learningRateC / (self.learningRateC + n)
        self.visitCounts[stateIndex, action] += 1

        # Target: reward plus discounted best future value.
        # If the game ended, there is no future, so the target is just the reward.
        if done:
            target = reward
        else:
            target = reward + self.discountFactor * self.qTable[nextStateIndex].max()
        
        # Move Q(s, a) a fraction alpha of the way toward the target
        current = self.qTable[stateIndex, action]
        self.qTable[stateIndex, action] = current + alpha * (target - current)
    
    def decay_epsilon(self):
        """Call once per episode: shrink epsilon, but never below the floor."""
        self.epsilon = max(self.epsilonEnd, self.epsilon * self.epsilonDecay)

    def save(self, path):
        """Save the Q-table to disk."""
        np.save(path, self.qTable)

    def load(self, path):
        """Load a Q-table from disk."""
        self.qTable = np.load(path)



    