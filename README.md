# Single-Player Pong with Tabular Q-Learning

A single-player version of Pong in which an agent learns, through Q-learning, to keep the ball in play by bouncing it off a paddle as many times as possible before missing.

## Objective

The goals of this project are to:

1. Formulate single-player Pong as a Markov Decision Process (MDP).
2. Implement tabular Q-learning with an exploration strategy to train an agent on that MDP.
3. Evaluate the trained agent on a large number of test games, and measure how the MDP and hyperparameter choices affect performance.

The target from the assignment is at least 9 consecutive rebounds on average, with 12 to 14 being a strong result.

## Files

| File | Purpose |
|---|---|
| `pong_env.py` | Game simulation: ball physics, wall and paddle bounces, rewards, termination. Also holds the game and training constants. |
| `discretizer.py` | Converts the continuous state into one of 10,369 discrete state indices. |
| `q_agent.py` | Q-table, epsilon-greedy action selection, learning-rate schedule, Q-update, save and load. |
| `train.py` | Training loop. Also a command-line entry point for one training run, which logs a row to `results.csv`. |
| `evaluate.py` | Plays test games with the greedy policy and prints summary statistics and a bounce histogram. |
| `experiments.py` | Runs a set of named hyperparameter configurations and logs each to `results/summary.csv`. |
| `play_gui.py` | pygame window to watch the trained agent play, or to play yourself with the arrow keys. |
| `results.csv` | Log of training runs written by `train.py`, one row per run. |
| `q_table.npy` | Trained Q-table checkpoint used by `evaluate.py` and `play_gui.py` by default. |
| `results/`, `npy/` | Saved Q-tables and learning curves from experiments, and other saved `.npy` arrays. |

## The MDP

**State.** The state is a tuple `(ball_x, ball_y, velocity_x, velocity_y, paddle_y)`. The ball lives in the unit square, with walls at `x = 0`, `y = 0`, and `y = 1`. The paddle sits on the right wall at `x = 1`, is 0.2 tall, and its top edge `paddle_y` lies in `[0, 0.8]`. The ball's horizontal speed is always at least 0.03, so it is always moving.

**Actions.** The agent chooses one of three actions each tick: stay, move the paddle up by 0.04, or move it down by 0.04. The paddle is clamped to stay on screen.

**Rewards.** The agent receives +1 when its paddle hits the ball, −1 when the ball gets past the paddle, and 0 otherwise.

**Termination.** A game ends when the ball's x-coordinate passes the paddle (`ball_x > 1`). The game also stops after 5,000 steps as a safety cap.

**Initial state.** `(0.5, 0.5, 0.03, 0.01, 0.4)`. The ball starts in the centre, moving toward the agent and slightly downward, with the paddle centred.

**Physics.** Each tick, the ball moves by its velocity. It reflects off the top, bottom, and left walls. When it hits the paddle, its horizontal velocity is reversed and both velocity components get a small random perturbation (uniform in ±0.015 for x and ±0.03 for y). The velocity is then clamped to a maximum magnitude of 1.

**Discretization.** A continuous state has to be mapped to a finite index before a Q-table can store it:

- The board is split into a 12 × 12 grid, giving 144 ball-position cells.
- `vel_x` is reduced to its sign: 2 values.
- `vel_y` becomes 3 values: up (`< −0.015`), level (`|vel_y| ≤ 0.015`), or down (`> 0.015`).
- `paddle_y` is mapped to one of 12 bins using `floor(12 · paddle_y / 0.8)`, capped at 11.
- One extra terminal state is used whenever `ball_x > 1`.

Total: `144 × 2 × 3 × 12 + 1 = 10,369` states. The indices are built with mixed-radix arithmetic, so each combination maps to a unique integer.

## Q-Learning

Q-learning estimates the value `Q(s, a)` of taking action `a` in state `s`, and then acts greedily with respect to those estimates. Each observed transition `(s, a, r, s')` updates the table as

$$Q(s,a) \leftarrow Q(s,a) + \alpha \big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \,\big]$$

- The bracketed term is the temporal-difference error: the gap between the new estimate (reward plus discounted best future value) and the old one.
- If the game ended on this step, there is no future, so the target is just `r`.

**Learning rate.** Each state-action pair gets its own learning rate, which shrinks as the pair is visited more often:

$$\alpha = \frac{C}{C + N(s,a)}$$

where `N(s,a)` counts past updates to that pair. Early estimates move quickly, and later ones settle.

**Exploration.** The agent uses epsilon-greedy action selection. With probability ε it picks a random action, otherwise the best-valued one. ε starts at 1.0 (fully random), is multiplied by a decay factor after each episode, and never falls below 0.01. This lets the agent explore broadly at first and then mostly exploit what it has learned.

**Evaluation.** Test games use the greedy policy only (ε is off), and no learning happens. Test games use a different random seed from training.

## Important Features

- **Single-point configuration.** All discretization constants (grid size, bin counts, velocity threshold) are defined at the top of `discretizer.py`, so the state space can be changed in one place.
- **Visit-based learning rate.** The `C / (C + N)` schedule is implemented per state-action pair, so rarely visited pairs keep learning while frequently visited ones stabilise.
- **Random tie-breaking.** When several actions share the highest Q-value (common early on, when every value is zero), one is picked at random. Always choosing the first action would bias the agent.
- **Separate training and test seeds.** Evaluation uses `randomSeed + 1`, so test games are new and not replays of training games.
- **Step cap and hit counter.** Each game is capped at 5,000 steps. `hit_cap` records how many games ended by the cap rather than by a miss, which shows whether a result is inflated by a stuck agent.
- **Command-line control.** `train.py` takes `--c`, `--gamma`, `--decay`, `--seed`, and `--episodes` flags, so experiments don't require editing code.
- **Batch experiments.** `experiments.py` runs named configurations, uses a fixed 3,000-game evaluation for each so results are comparable, and appends each run to a summary CSV.
- **pygame viewer.** `play_gui.py` shows the trained agent playing, or lets you play with the UP and DOWN arrows. Controls: SPACE pauses, N starts a new game, + and − change speed, ESC quits.

## Hyperparameters and Why

| Symbol | Meaning | Baseline | Values tried | Chosen |
|---|---|---|---|---|
| `C` | Learning-rate constant in `α = C / (C + N)` | 50 | 10, 50, 200 | 200 |
| `γ` | Discount factor | 0.95 | 0.90, 0.95, 0.99 | 0.95 |
| `ε₀` | Starting exploration rate | 1.0 | fixed | 1.0 |
| `ε_min` | Exploration floor | 0.01 | fixed | 0.01 |
| `decay` | Multiplier on ε after each episode | 0.99995 | 0.9998, 0.99995, 0.9999 | 0.9999 |
| Training games | Episodes per run | 100,000 | fixed | 100,000 |

**Learning-rate constant C.** A larger `C` keeps `α` high for longer. With `C = 50`, a state-action pair has `α = 0.5` after 50 updates. With `C = 200`, it still has `α = 0.5` after 200. A higher learning rate for longer means the agent averages over more samples before settling, which helps because rewards in Pong are noisy. The cost is slower early progress: `C = 200` reached 9 bounces later than the baseline, but finished with a higher average.

**Discount factor γ.** Pong rewards arrive only at paddle hits and misses, so the agent must connect a current move to a reward that is a rally away. The assignment suggests choosing `γ` so the previous hit's reward has mostly faded by the time the ball returns. At minimum speed, a round trip across the board takes roughly 35 to 70 ticks, and `0.95` discounts a reward to about 3 to 17% over that span. A smaller value (`0.90`) makes the agent too short-sighted and was the worst setting tested (15.70). A larger value (`0.99`) keeps the previous hit influential for about half a rally, which adds noise and did not help (16.70).

**Exploration decay.** Epsilon reaches its floor of 0.01 after about `ln(0.01) / ln(decay)` episodes:

- `0.99995` reaches the floor at about 92,000 episodes, so almost all training is spent exploring.
- `0.9999` reaches it at about 46,000 episodes, leaving more time to exploit the learned policy.
- `0.9998` reaches it at about 23,000 episodes. It learns fastest but commits earlier.

The `0.9999` setting was combined with `C = 200` for the final configuration.

## Results: Baseline Experiments

Each row is one training run of 100,000 games, followed by evaluation on 3,000 greedy test games. "Episodes to 9" and "episodes to 12" are the first training episode at which the trailing 5,000-game average bounce count reaches 9 and 12.

| Run | Setting | Avg bounces (± stderr) | Episodes to 9 | Episodes to 12 | Median | Std | Hit cap |
|---|---|---|---|---|---|---|---|
| baseline | seed 42 | 17.12 ± 0.23 | 39,646 | 49,544 | 14 | 12.45 | 0 |
| baseline_seed1 | seed 1 | 16.97 ± 0.22 | 38,622 | 50,330 | 14 | 12.03 | 0 |
| baseline_seed2 | seed 2 | 16.68 ± 0.21 | 40,475 | 49,937 | 14 | 11.76 | 0 |
| c10 | C = 10 | 16.81 ± 0.23 | 35,556 | 49,155 | 14 | 12.45 | 1 |
| c200 | C = 200 | 17.96 ± 0.24 | 48,038 | 56,035 | 14 | 13.11 | 2 |
| c200_seed1 | C = 200, seed 1 | 17.23 ± 0.23 | 47,292 | 56,446 | 14 | 12.71 | 0 |
| gamma90 | γ = 0.90 | 15.70 ± 0.21 | 41,878 | 54,287 | 13 | 11.28 | 0 |
| gamma99 | γ = 0.99 | 16.70 ± 0.22 | 35,696 | 46,018 | 14 | 11.79 | 1 |
| decay9999 | decay = 0.9999 | 17.43 ± 0.23 | 23,484 | 28,276 | 14 | 12.74 | 0 |
| decay9998 | decay = 0.9998 | 17.77 ± 0.24 | 14,470 | 16,752 | 14 | 12.95 | 1 |
| tuned | C = 200, decay = 0.9999 | **18.01 ± 0.23** | 28,841 | 33,520 | 15 | 12.84 | 1 |

**Average bounces on a large test set.** The tuned configuration averaged **18.01 bounces per game** over 3,000 greedy test games (standard error 0.23), with a median of 15. That is above the assignment's 12 to 14 target.

**Noise.** Running the baseline with three seeds gave 17.12, 16.97, and 16.68. Differences smaller than about 0.5 bounces should therefore be treated as noise. The C = 200 result was checked on two seeds (17.96 on seed 42, 17.23 on seed 1, against 16.97 for the baseline on seed 1), which supports it being a real improvement. The tuned configuration was run with one seed only.

**Convergence.** Across these runs, the agent reached a 9-bounce average between roughly 14,000 and 48,000 training games, and a 12-bounce average between roughly 17,000 and 56,000 games. The choice of decay rate has the largest effect on speed.

**Timing.** Wall-clock training time ranged from 825 to 1,229 seconds. These timings were not gathered under controlled conditions, so they should not be compared closely.

## Changes to the MDP

> **TODO (Part C):** Not yet run. Fill in the following:
>
> - **What was changed.** Describe each change to the discretization (for example, grid resolution, velocity bins) or the reward model, and the constants involved.
> - **Why.** The reasoning behind each change.
> - **Effect on training.** How the change affected the number of states, the number of games needed to converge, and training time.

## Observations for the New MDP

> **TODO (Part C):** Record what changed in behaviour, including anything unexpected.

## New Average Bounces

> **TODO (Part C):** Report the average bounces over at least 1,000 greedy test games for the converged modified configuration, with standard error, and compare it to the baseline (17.12 ± 0.23 on 3,000 games; tuned 18.01 ± 0.23).

## How to Run

Install the dependencies:

```bash
pip install numpy pygame
```

Train a single run with the baseline settings, or with the tuned settings:

```bash
python train.py --name baseline
python train.py --name tuned --c 200 --decay 0.9999
```

Run the named experiment batch (use `--list` to see names):

```bash
python experiments.py --list
python experiments.py --only baseline tuned
```

Evaluate a saved Q-table on 1,000 or more games:

```bash
python evaluate.py --games 1000 --checkpoint q_table.npy
```

Watch the trained agent, or play yourself:

```bash
python play_gui.py --checkpoint q_table.npy
python play_gui.py --human
```

`play_gui.py` and `evaluate.py` both default to `q_table.npy`, so copy or rename the checkpoint you want to use if you train a new model.
