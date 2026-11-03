from pong_env import paddle_y_max, paddle_x

# Discretization Constants
gridSize = 12
paddleBins = 12
velocity_y_threshold = 0.015
numVelocityXBins = 2
numVelocityYBins = 3

# Derived: (12 * 12 * 2 * 3 * 12) + 1
numStates = (gridSize * gridSize * numVelocityXBins * numVelocityYBins * paddleBins) + 1
numActions = 3

TERMINAL_STATE = numStates - 1

def clamp_index(value, num_bins):
    """Force a bin number into the valid range 0 .. num_bins - 1."""
    return max(0, min(num_bins - 1, value))

def stateToIndex(state):
    """Convert a continuous state tuple into one integer in 0 .. numStates - 1.
    state = (ball_x, ball_y, vel_x, vel_y, paddle_y)
    """

    ball_x, ball_y, vel_x, vel_y, paddle_y = state

    # Special state: ball aready past the paddle. Everything else is ignored.
    if ball_x > paddle_x:
        return TERMINAL_STATE
    
    # Ball position: which cell of the GRID_SIZE x GRIDE_SIZE grid?
    # Multiply to stretch [0, 1] over the grid, then round down. Clamp handles ball_x and ball_y being exactly 1.0
    cell_x = clamp_index(int(ball_x * gridSize), gridSize)
    cell_y = clamp_index(int(ball_y * gridSize), gridSize)

    # Horizontal velocity: only the direction matters
    # 0 = moving left, 1 = moving right
    vel_x_bin = 0 if vel_x < 0 else 1

    # Vertical velocity: 0 = up (negative), 1 = roughly level, 2 = down (positive)
    if vel_y < -velocity_y_threshold:
        vel_y_bin = 0
    elif vel_y > velocity_y_threshold:
        vel_y_bin = 2
    else:
        vel_y_bin = 1

    # Paddle position
    paddle_bin = clamp_index(int(paddleBins * paddle_y / paddle_y_max), paddleBins)

    # Combine the "digits" into one number (mixed radix / odometer idea): multiply by each digit's size, then add the digit.
    index = cell_x
    index = index * gridSize + cell_y
    index = index * numVelocityXBins + vel_x_bin
    index = index * numVelocityYBins + vel_y_bin
    index = index * paddleBins + paddle_bin
    return index