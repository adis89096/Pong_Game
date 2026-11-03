"""
play_gui.py - Watch the Q-learning agent play Pong, or play it yourself.

Pygame handles drawing and input only. Game logic comes from the existing
project files (pong_env.py, discretizer.py, q_agent.py), which are not changed.

Controls:
  SPACE     pause / resume
  N         start a new game
  + / -     speed up / slow down (game ticks per frame)
  ESC       quit
"""
import argparse
import os
import sys
import pygame
from pong_env import PongEnv, paddleHeight, paddle_x, checkpointFile, randomSeed, maxStepsPerGame, action_stay, action_up, action_down
from discretizer import stateToIndex
from q_agent import QAgent

# ------ Window Layout (pixels) ------

# the playing field is a square this size
fieldSize = 520
# blank border around the field
margin = 40
# space below the field for text
hudHeight = 100

windowWidth = fieldSize + 2 * margin
windowHeight = fieldSize + 2 * margin + hudHeight

# screen refreshes per second
fps = 60
# show a finished game for about 1 second
pauseFrames = fps
maxTicksPerFrame = 20
ballRadius = 8
paddleThickness = 6


# ------ Colors (R, G, B) ------
black = (0, 0, 0)
white = (255, 255, 255)
gray = (90, 90, 90)
yellow = (240, 210, 80)
red = (230, 90, 90)


def toPixel(x, y):
    """Convert game coordinates (0..1, origin at top-left) to screen pixels."""
    return int(margin + x * fieldSize), int(margin + y * fieldSize)

def humanAction():
    """Read the arrow keys and return the matching paddle action."""
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return action_up
    if keys[pygame.K_DOWN]:
        return action_down
    return action_stay

def drawGame(screen, env, fonts, info):
    """Draw the field, paddle, ball, and text for the current frame."""
    screen.fill(black)

    # The walls: a rectangle around the field
    pygame.draw.rect(screen, gray, (margin, margin, fieldSize, fieldSize), 2)

    # The paddle: a bar on the right edge (x = paddle_x) at height paddley
    right_px = int(margin + paddle_x * fieldSize)
    top_px = int(margin + env.paddley * fieldSize)
    height_px = int(paddleHeight * fieldSize)
    pygame.draw.rect(screen, white, (right_px - paddleThickness, top_px, paddleThickness, height_px))

    # The ball
    ball_px, ball_py = toPixel(env.ballx, env.bally)
    pygame.draw.circle(screen, yellow, (ball_px, ball_py), ballRadius)

    # Text under the field
    base_y = margin + fieldSize + 12
    lines = [
        f"Mode: {info['mode']}     Speed: {info['speed']} tick(s) per frame",
        f"Bounces now: {env.bounce}     Last game: {info['last']}     Best: {info['best']}",
        f"Games played: {info['games']}     Average bounces: {info['average']}",
    ]
    for i, text in enumerate(lines):
        surface = fonts["small"].render(text, True, white)
        screen.blit(surface, (margin, base_y + i * 24))

    # Big message in the middle of the field (PAUSED, or the result of a game)
    if info["message"]:
        surface = fonts["big"].render(info["message"], True, info["message_color"])
        screen.blit(surface, surface.get_rect(center=(windowWidth // 2, windowHeight // 2)))


def main():
    parser = argparse.ArgumentParser(description="Watch or play Pong with pygame.")
    parser.add_argument("--checkpoint", default=checkpointFile, help="Q-table .npy file saved by train.py")
    parser.add_argument("--human", action="store_true", help="play yourself with the UP / DOWN arrows instead of the AI")
    parser.add_argument("--seed", type=int, default=randomSeed + 2, help="random seed for the ball's bounce noise")
    args = parser.parse_args()

    # Load the trained agent (not needed in human mode)
    agent = None
    if not args.human:
        if not os.path.exists(args.checkpoint):
            sys.exit(f"Can't find '{args.checkpoint}'. Run train.py first or pass --checkpoint.")
        agent = QAgent()
        agent.load(args.checkpoint)

    pygame.init()
    screen = pygame.display.set_mode((windowWidth, windowHeight))
    pygame.display.set_caption("Pong - you" if args.human else "Pong - Q-learning agent")
    clock = pygame.time.Clock()
    fonts = {"small": pygame.font.SysFont(None, 24), "big": pygame.font.SysFont(None, 56)}

    env = PongEnv(seed=args.seed)
    state = env.reset()
    state_index = stateToIndex(state)
    steps = 0

    # Statistics across games
    games = 0
    total_bounces = 0
    best = 0
    last = None

    ticks_per_frame = 1
    paused = False
    restart = False
    wait_frames = 0           # > 0 while showing the result of a finished game
    message = ""
    message_color = white

    running = True
    while running:
        # 1. Handle input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    restart = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    ticks_per_frame = min(maxTicksPerFrame, ticks_per_frame + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    ticks_per_frame = max(1, ticks_per_frame - 1)

        # Starting a new game by hand (the N key)
        if restart:
            state = env.reset()
            state_index = stateToIndex(state)
            steps = 0
            wait_frames = 0
            message = ""
            restart = False

        # 2. Update the game (skipped while paused)
        if not paused:
            if wait_frames > 0:
                # Showing the result of the last game; start the next one when the timer ends
                wait_frames -= 1
                if wait_frames == 0:
                    state = env.reset()
                    state_index = stateToIndex(state)
                    steps = 0
                    message = ""
            else:
                for _ in range(ticks_per_frame):
                    # Pick an action: from the keyboard, or from the trained Q-table
                    if args.human:
                        action = humanAction()
                    else:
                        action = agent.chooseActions(state_index, explore=False)

                    # Advance the game one tick
                    state, _, done = env.step(action)
                    state_index = stateToIndex(state)
                    steps += 1

                    if done or steps >= maxStepsPerGame:
                        games += 1
                        total_bounces += env.bounce
                        best = max(best, env.bounce)
                        last = env.bounce
                        if done:
                            message = f"Missed! {env.bounce} bounces"
                            message_color = red
                        else:
                            message = "Step limit reached"
                            message_color = yellow
                        wait_frames = pauseFrames
                        break

        # 3. Draw the frame
        info = {
            "mode": "Human (UP / DOWN)" if args.human else "AI (greedy Q-table)",
            "speed": ticks_per_frame,
            "last": "-" if last is None else last,
            "best": best,
            "games": games,
            "average": "-" if games == 0 else f"{total_bounces / games:.2f}",
            "message": "PAUSED" if paused else message,
            "message_color": yellow if paused else message_color,
        }
        drawGame(screen, env, fonts, info)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()








