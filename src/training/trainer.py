"""
Training module for Minesweeper agents.

Provides training loops with logging, checkpointing, and evaluation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import time
import json

import numpy as np

from ..game.environment import MinesweeperEnv
from ..game.board import BoardConfig
from ..agents.base_agent import BaseAgent
from ..agents.dqn_agent import DQNAgent
from ..agents.hybrid_agent import HybridAgent


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training a DQN agent."""

    # Environment settings
    board_height: int = 9
    board_width: int = 9
    num_mines: int = 10

    # Training settings
    num_episodes: int = 10000
    max_steps_per_episode: int = 100

    # Evaluation settings
    eval_frequency: int = 100
    eval_episodes: int = 50

    # Checkpointing
    save_frequency: int = 1000
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_frequency: int = 10


# ============================================================================
# Training Statistics
# ============================================================================

@dataclass
class EpisodeStats:
    """Statistics for a single episode."""

    total_reward: float = 0.0
    steps: int = 0
    won: bool = False
    revealed_cells: int = 0


@dataclass
class TrainingStats:
    """Accumulated training statistics."""

    episodes_completed: int = 0
    total_steps: int = 0
    wins: int = 0
    losses: int = 0
    episode_rewards: List[float] = field(default_factory=list)
    win_history: List[bool] = field(default_factory=list)
    win_rates: List[float] = field(default_factory=list)
    eval_win_rates: List[float] = field(default_factory=list)
    losses_history: List[float] = field(default_factory=list)

    @property
    def recent_win_rate(self) -> float:
        """Win rate over last 100 episodes."""
        if not self.win_history:
            return 0.0
        recent = self.win_history[-100:]
        return sum(recent) / len(recent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "episodes_completed": self.episodes_completed,
            "total_steps": self.total_steps,
            "wins": self.wins,
            "losses": self.losses,
            "recent_win_rate": self.recent_win_rate,
        }


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """
    Training loop for DQN agents.

    Features:
        - Episode-based training with experience replay
        - Periodic evaluation against fresh environments
        - Checkpointing for model persistence
        - Progress logging and statistics
    """

    def __init__(
        self,
        agent: DQNAgent,
        config: TrainingConfig,
        callback: Optional[Callable[[TrainingStats], None]] = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            agent: DQN agent to train.
            config: Training configuration.
            callback: Optional callback for progress updates.
        """
        self.agent = agent
        self.config = config
        self.callback = callback
        self.stats = TrainingStats()

        # Create board config
        self.board_config = BoardConfig(
            height=config.board_height,
            width=config.board_width,
            num_mines=config.num_mines,
        )

        # Create checkpoint directory
        self.checkpoint_path = Path(config.checkpoint_dir)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    def train(self) -> TrainingStats:
        """
        Run the full training loop.

        Returns:
            Final training statistics.
        """
        env = MinesweeperEnv(config=self.board_config)

        start_time = time.time()

        for episode in range(self.config.num_episodes):
            episode_stats = self._run_episode(env)
            self._update_stats(episode_stats)

            # Logging
            if (episode + 1) % self.config.log_frequency == 0:
                self._log_progress(episode + 1, start_time)

            # Evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_win_rate = self._evaluate()
                self.stats.eval_win_rates.append(eval_win_rate)

            # Checkpointing
            if (episode + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(episode + 1)

            # Callback
            if self.callback:
                self.callback(self.stats)

        # Final save
        self._save_checkpoint(self.config.num_episodes)
        self._save_stats()

        return self.stats

    def _run_episode(self, env: MinesweeperEnv) -> EpisodeStats:
        """Run a single training episode."""
        stats = EpisodeStats()

        observation, _ = env.reset()
        self.agent.reset()

        for _ in range(self.config.max_steps_per_episode):
            valid_actions = env.get_action_mask()
            action = self.agent.select_action(
                observation, valid_actions, training=True
            )

            next_observation, reward, terminated, truncated, info = env.step(
                action
            )

            # Update agent with experience
            self.agent.update(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=terminated or truncated,
            )

            stats.total_reward += reward
            stats.steps += 1
            stats.revealed_cells = info.get("revealed", 0)

            observation = next_observation

            if terminated or truncated:
                stats.won = info.get("game_state") == "WON"
                break

        return stats

    def _update_stats(self, episode_stats: EpisodeStats) -> None:
        """Update training statistics with episode results."""
        self.stats.episodes_completed += 1
        self.stats.total_steps += episode_stats.steps
        self.stats.episode_rewards.append(episode_stats.total_reward)
        self.stats.win_history.append(episode_stats.won)

        if episode_stats.won:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        self.stats.win_rates.append(self.stats.recent_win_rate)

        # Track loss history
        if self.agent.losses:
            self.stats.losses_history.append(self.agent.losses[-1])

    def _evaluate(self) -> float:
        """
        Evaluate agent without exploration.

        Returns:
            Win rate over evaluation episodes.
        """
        env = MinesweeperEnv(config=self.board_config)
        wins = 0

        for _ in range(self.config.eval_episodes):
            observation, _ = env.reset()
            self.agent.reset()

            for _ in range(self.config.max_steps_per_episode):
                valid_actions = env.get_action_mask()
                action = self.agent.select_action(
                    observation, valid_actions, training=False
                )

                observation, _, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    if info.get("game_state") == "WON":
                        wins += 1
                    break

        return wins / self.config.eval_episodes

    def _log_progress(self, episode: int, start_time: float) -> None:
        """Log training progress."""
        elapsed = time.time() - start_time
        eps_per_sec = episode / elapsed if elapsed > 0 else 0

        print(
            f"Episode {episode}/{self.config.num_episodes} | "
            f"Win Rate: {self.stats.recent_win_rate:.1%} | "
            f"Epsilon: {self.agent.epsilon:.3f} | "
            f"Speed: {eps_per_sec:.1f} ep/s | "
            f"Device: {self.agent.device_name}"
        )

    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint."""
        checkpoint_file = self.checkpoint_path / f"checkpoint_{episode}.pt"
        self.agent.save(str(checkpoint_file))

        # Also save as latest
        latest_file = self.checkpoint_path / "latest.pt"
        self.agent.save(str(latest_file))

    def _save_stats(self) -> None:
        """Save training statistics to JSON."""
        stats_file = self.checkpoint_path / "training_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats.to_dict(), f, indent=2)


# ============================================================================
# Agent Evaluator
# ============================================================================

class Evaluator:
    """
    Evaluate and compare multiple agents.

    Provides standardized evaluation across different agent types.
    """

    def __init__(
        self,
        board_config: Optional[BoardConfig] = None,
        num_episodes: int = 100,
        max_steps: int = 100,
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            board_config: Board configuration for evaluation.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
        """
        self.board_config = board_config or BoardConfig()
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def evaluate(self, agent: BaseAgent) -> Dict[str, float]:
        """
        Evaluate a single agent.

        Args:
            agent: Agent to evaluate.

        Returns:
            Dictionary with evaluation metrics.
        """
        env = MinesweeperEnv(config=self.board_config)

        wins = 0
        total_reward = 0.0
        total_steps = 0
        total_revealed = 0

        for _ in range(self.num_episodes):
            observation, _ = env.reset()
            agent.reset()
            episode_reward = 0.0

            for _ in range(self.max_steps):
                valid_actions = env.get_action_mask()

                # Handle different agent signatures
                if isinstance(agent, (DQNAgent, HybridAgent)):
                    action = agent.select_action(
                        observation, valid_actions, training=False
                    )
                else:
                    action = agent.select_action(observation, valid_actions)

                observation, reward, terminated, truncated, info = env.step(
                    action
                )

                episode_reward += reward
                total_steps += 1

                if terminated or truncated:
                    if info.get("game_state") == "WON":
                        wins += 1
                    total_revealed += info.get("revealed", 0)
                    break

            total_reward += episode_reward

        return {
            "win_rate": wins / self.num_episodes,
            "avg_reward": total_reward / self.num_episodes,
            "avg_steps": total_steps / self.num_episodes,
            "avg_revealed": total_revealed / self.num_episodes,
        }

    def compare(
        self, agents: Dict[str, BaseAgent]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple agents.

        Args:
            agents: Dictionary of agent_name -> agent.

        Returns:
            Dictionary of agent_name -> evaluation metrics.
        """
        results = {}
        for name, agent in agents.items():
            print(f"Evaluating {name}...")
            results[name] = self.evaluate(agent)
        return results
