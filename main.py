#!/usr/bin/env python3
"""
Minesweeper ML - Main entry point.

Usage:
    python main.py train [--episodes N] [--eval]
    python main.py evaluate [--agent {random,logic,dqn}]
    python main.py compare
"""
import argparse
from pathlib import Path

from src.game.board import BoardConfig
from src.game.environment import MinesweeperEnv
from src.agents import RandomAgent, LogicAgent, DQNAgent, HybridAgent, get_device
from src.training import Trainer, TrainingConfig, Evaluator


def train(args: argparse.Namespace) -> None:
    """Train a DQN agent."""
    print(f"Training DQN agent for {args.episodes} episodes...")
    print(f"Using device: {get_device()}")

    # Create agent
    agent = DQNAgent(
        board_height=9,
        board_width=9,
    )

    # Create training config
    config = TrainingConfig(
        num_episodes=args.episodes,
        eval_frequency=100,
        eval_episodes=50,
        save_frequency=1000,
        log_frequency=10,
    )

    # Train
    trainer = Trainer(agent, config)
    stats = trainer.train()

    print(f"\nTraining complete!")
    print(f"Final win rate: {stats.recent_win_rate:.1%}")
    print(f"Total episodes: {stats.episodes_completed}")
    print(f"Model saved to: checkpoints/latest.pt")

    # Optional evaluation
    if args.eval:
        evaluate_agent(agent, "DQN (trained)")


def train_hybrid(args: argparse.Namespace) -> None:
    """Train a Hybrid agent (AC-3 + neural network for guessing)."""
    import time

    print(f"Training Hybrid agent for {args.episodes} episodes...")
    print(f"Using device: {get_device()}")

    config = BoardConfig(height=9, width=9, num_mines=10)
    env = MinesweeperEnv(config=config)

    agent = HybridAgent(board_height=9, board_width=9)

    wins = 0
    start_time = time.time()

    for episode in range(args.episodes):
        obs, _ = env.reset()
        agent.reset()
        done = False

        while not done:
            valid_actions = env.get_action_mask()
            action = agent.select_action(obs, valid_actions, training=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs

            if done and info.get("game_state") == "WON":
                wins += 1

        # Progress logging
        if (episode + 1) % 100 == 0:
            elapsed = time.time() - start_time
            win_rate = wins / (episode + 1)
            guess_acc = agent.guess_accuracy
            eps_per_sec = (episode + 1) / elapsed

            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"Win Rate: {win_rate:.1%} | "
                f"Guess Acc: {guess_acc:.1%} | "
                f"Guesses: {agent.guesses_made} | "
                f"Speed: {eps_per_sec:.1f} ep/s"
            )

        # Save checkpoint
        if (episode + 1) % 1000 == 0:
            checkpoint_path = Path("checkpoints")
            checkpoint_path.mkdir(exist_ok=True)
            agent.save(str(checkpoint_path / "hybrid_latest.pt"))

    # Final save
    checkpoint_path = Path("checkpoints")
    checkpoint_path.mkdir(exist_ok=True)
    agent.save(str(checkpoint_path / "hybrid_latest.pt"))

    print(f"\nTraining complete!")
    print(f"Final win rate: {wins / args.episodes:.1%}")
    print(f"Guess accuracy: {agent.guess_accuracy:.1%}")
    print(f"Total guesses: {agent.guesses_made}")
    print(f"Certain moves: {agent.certain_moves}")
    print(f"Model saved to: checkpoints/hybrid_latest.pt")

    if args.eval:
        evaluate_agent(agent, "Hybrid (trained)", config, 100)


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a specific agent."""
    config = BoardConfig(height=9, width=9, num_mines=10)

    if args.agent == "random":
        agent = RandomAgent(9, 9)
        name = "Random"
    elif args.agent == "logic":
        agent = LogicAgent(9, 9)
        name = "Logic"
    elif args.agent == "dqn":
        agent = DQNAgent(9, 9)
        checkpoint = Path("checkpoints/latest.pt")
        if checkpoint.exists():
            agent.load(str(checkpoint))
            name = "DQN (trained)"
        else:
            name = "DQN (untrained)"
    elif args.agent == "hybrid":
        agent = HybridAgent(9, 9)
        checkpoint = Path("checkpoints/hybrid_latest.pt")
        if checkpoint.exists():
            agent.load(str(checkpoint))
            name = "Hybrid (trained)"
        else:
            name = "Hybrid (untrained)"
    else:
        print(f"Unknown agent: {args.agent}")
        return

    evaluate_agent(agent, name, config, args.games)


def evaluate_agent(
    agent,
    name: str,
    config: BoardConfig = None,
    num_episodes: int = 100,
) -> None:
    """Evaluate a single agent and print results."""
    config = config or BoardConfig(height=9, width=9, num_mines=10)
    evaluator = Evaluator(config, num_episodes=num_episodes)

    print(f"\nEvaluating {name} over {num_episodes} games...")
    results = evaluator.evaluate(agent)

    print(f"Results for {name}:")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Avg reward: {results['avg_reward']:.2f}")
    print(f"  Avg steps: {results['avg_steps']:.1f}")
    print(f"  Avg revealed: {results['avg_revealed']:.1f} cells")


def compare(args: argparse.Namespace) -> None:
    """Compare all agents."""
    config = BoardConfig(height=9, width=9, num_mines=10)

    agents = {
        "Random": RandomAgent(9, 9),
        "Logic": LogicAgent(9, 9),
    }

    # Add trained DQN if available
    dqn_checkpoint = Path("checkpoints/latest.pt")
    if dqn_checkpoint.exists():
        dqn = DQNAgent(9, 9)
        dqn.load(str(dqn_checkpoint))
        agents["DQN (trained)"] = dqn
    else:
        agents["DQN (untrained)"] = DQNAgent(9, 9)

    # Add trained Hybrid if available
    hybrid_checkpoint = Path("checkpoints/hybrid_latest.pt")
    if hybrid_checkpoint.exists():
        hybrid = HybridAgent(9, 9)
        hybrid.load(str(hybrid_checkpoint))
        agents["Hybrid (trained)"] = hybrid
    else:
        agents["Hybrid (untrained)"] = HybridAgent(9, 9)

    evaluator = Evaluator(config, num_episodes=args.games)
    results = evaluator.compare(agents)

    print("\n" + "=" * 50)
    print("Agent Comparison Results")
    print("=" * 50)
    print(f"{'Agent':<20} {'Win Rate':<12} {'Avg Reward':<12} {'Avg Steps':<10}")
    print("-" * 50)

    for name, metrics in results.items():
        print(
            f"{name:<20} {metrics['win_rate']:>10.1%} "
            f"{metrics['avg_reward']:>10.2f} "
            f"{metrics['avg_steps']:>10.1f}"
        )


def main() -> None:
    """Parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(
        description="Minesweeper ML - Train and evaluate AI agents"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train DQN command
    train_parser = subparsers.add_parser("train", help="Train a DQN agent")
    train_parser.add_argument(
        "--episodes", type=int, default=5000, help="Number of training episodes"
    )
    train_parser.add_argument(
        "--eval", action="store_true", help="Evaluate after training"
    )

    # Train Hybrid command
    train_hybrid_parser = subparsers.add_parser(
        "train-hybrid", help="Train a Hybrid agent (AC-3 + neural guessing)"
    )
    train_hybrid_parser.add_argument(
        "--episodes", type=int, default=10000, help="Number of training episodes"
    )
    train_hybrid_parser.add_argument(
        "--eval", action="store_true", help="Evaluate after training"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an agent")
    eval_parser.add_argument(
        "--agent",
        choices=["random", "logic", "dqn", "hybrid"],
        default="logic",
        help="Agent to evaluate",
    )
    eval_parser.add_argument(
        "--games", type=int, default=100, help="Number of games to play"
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare all agents")
    compare_parser.add_argument(
        "--games", type=int, default=100, help="Number of games per agent"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "train-hybrid":
        train_hybrid(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "compare":
        compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
