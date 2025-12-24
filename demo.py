#!/usr/bin/env python3
"""Watch the Logic agent play Minesweeper."""
import time
import os

from src.game.environment import MinesweeperEnv
from src.game.board import BoardConfig
from src.agents import LogicAgent


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def demo(delay: float = 0.3, games: int = 5, size: int = 9, mines: int = 10):
    """Run demo games with visualization."""
    config = BoardConfig(height=size, width=size, num_mines=mines)
    env = MinesweeperEnv(config=config, render_mode="ansi")
    agent = LogicAgent(size, size)

    print(f"Board: {size}x{size} with {mines} mines ({100*mines/(size*size):.1f}% density)")
    print("Starting in 2 seconds...")
    time.sleep(2)

    wins = 0

    for game in range(games):
        obs, _ = env.reset()
        agent.reset()

        clear_screen()
        print(f"=== Game {game + 1}/{games} ===")
        print(f"Wins so far: {wins}\n")
        print(env.render())
        time.sleep(delay)

        done = False
        step = 0

        while not done:
            valid_actions = env.get_action_mask()
            action = agent.select_action(obs, valid_actions)
            row, col = action // size, action % size

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            clear_screen()
            print(f"=== Game {game + 1}/{games} | Step {step} ===")
            print(f"Wins so far: {wins}")
            print(f"Last move: ({row}, {col})\n")
            print(env.render())

            if done:
                if info.get("game_state") == "WON":
                    wins += 1
                    print(f"\n*** WIN! ***")
                else:
                    print(f"\n*** LOST (hit mine) ***")

            time.sleep(delay)

        time.sleep(1.0)  # Pause between games

    print(f"\n=== Final: {wins}/{games} wins ({100*wins/games:.0f}%) ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between moves")
    parser.add_argument("--games", type=int, default=5, help="Number of games")
    parser.add_argument("--size", type=int, default=9, help="Board size (NxN)")
    parser.add_argument("--mines", type=int, default=None, help="Number of mines (default: ~20%% of cells)")
    args = parser.parse_args()

    # Default mines to ~20% of cells (expert difficulty)
    mines = args.mines if args.mines else int(args.size * args.size * 0.2)

    demo(delay=args.delay, games=args.games, size=args.size, mines=mines)
