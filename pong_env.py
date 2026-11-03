import random


paddleHeight = 0.2
paddleStep = 0.04
paddle_x = 1.0
paddle_y_min = 0.0
paddle_y_max = 1 - paddleHeight

# ball speed rules
min_abs_velocity_x = 0.03
velocityLimit = 1.0
bounce_noise_x = 0.015
bounce_noise_y = 0.03

# Initial state: (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
initialBallX = 0.5
initialBallY = 0.5
initialVelocityX = 0.03
initialVelocityY = 0.01
initialPaddleY = 0.5 - paddleHeight / 2

# Reward
rewardHit = 1
rewardMiss = -1
rewardStep = 0

# Training/Evaluation Settings
numTrainEpisodes = 100000
numTestGames = 1000
maxStepsPerGame = 5000
randomSeed = 42
checkpointFile = "q_table.npy"


"""
MDP Environment
"""

# Action codes. y = 0 is the TOP of the screen, so "up" makes y smaller.
action_stay = 0
action_up = 1
action_down = 2

class PongEnv:
    """Single-player Pong. The agent controls the paddle on the right (x = 1)."""
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self):
        """Start a new game and return the initial state."""
        self.ballx = initialBallX
        self.bally = initialBallY
        self.velx = initialVelocityX
        self.vely = initialVelocityY
        self.paddley = initialPaddleY
        self.done = False
        # paddle hits this game (key metric)
        self.bounce = 0
        return self.getSelf()
    
    def getSelf(self):
        """Return the current continuous state as a tuple"""
        return (self.ballx, self.bally, self.velx, self.vely, self.paddley)

    def step(self, action):
        """Advance the game one tick. Return (new_state, reward, done)"""
        reward = rewardStep

        # 1. Move the paddle, then clamp it on screen
        if action == action_up:
            self.paddley -= paddleStep
        elif action == action_down:
            self.paddley += paddleStep
        self.paddley = max(paddle_y_min, min(paddle_y_max, self.paddley))

        # 2. Move the ball
        self.ballx += self.velx
        self.bally += self.vely

        # 3. Wall bounce (reflects position, flip velocity)
        #  top wall
        if self.bally < 0:
            self.bally = -self.bally
            self.vely = -self.vely
        # bottom wall
        if self.bally > 1:
            self.bally = 2 - self.bally
            self.vely = -self.vely
        # left wall
        if self.ballx < 0:
            self.ballx = -self.ballx
            self.velx = -self.velx

        # 4. Paddle hit: ball reached x = 1 and is within the paddle's span
        paddleBottom = self.paddley + paddleHeight
        if (self.ballx >= paddle_x and self.paddley <= self.bally <= paddleBottom):
            self.ballx = 2 * paddle_x - self.ballx
            self.velx = -self.velx + self.rng.uniform(-bounce_noise_x, bounce_noise_x)
            self.vely = -self.vely + self.rng.uniform(-bounce_noise_y, bounce_noise_y)

            # Ball now heads left, so vel_x must be negative and at least min_abs_velocity_x in size; also cap the maximum speed
            self.velx = min(self.velx, -min_abs_velocity_x)
            self.velx = max(self.velx, -velocityLimit)
            self.vely = max(-velocityLimit, min(velocityLimit, self.vely))
            reward = rewardHit
            self.bounce += 1

        # 5. Miss: ball got past the paddle without being hit
        elif self.ballx > paddle_x:
            reward = rewardMiss
            self.done = True

        # 6. Report back
        return self.getSelf(), reward, self.done