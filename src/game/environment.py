"""
Gymnasium environment wrapper for Minesweeper.

Provides a standard RL interface for training agents.
"""
from typing import Any, Dict, Optional, Tuple, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .board import Board, BoardConfig, GameState


# ============================================================================
# Minesweeper Environment
# ============================================================================

class MinesweeperEnv(gym.Env):
    """
    Gymnasium environment for Minesweeper.

    Observation:
        2D array where:
        - -1 = hidden cell
        - -2 = flagged cell
        - 0-8 = revealed cell with adjacent mine count

    Actions:
        Discrete action space of size width * height.
        Action i corresponds to cell at (i // width, i % width).

    Rewards:
        - +1 for revealing a safe cell
        - +10 for winning the game
        - -10 for hitting a mine
        - -0.1 for invalid action (already revealed/flagged)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[BoardConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the Minesweeper environment.

        Args:
            config: Board configuration (default: 9x9 with 10 mines).
            render_mode: How to render the environment.
        """
        super().__init__()

        self.config = config or BoardConfig()
        self.board = Board(self.config)
        self.render_mode = render_mode

        # Define observation space
        self.observation_space = spaces.Box(
            low=-2,
            high=9,
            shape=(self.config.height, self.config.width),
            dtype=np.int8,
        )

        # Define action space (one action per cell)
        self.action_space = spaces.Discrete(
            self.config.height * self.config.width
        )

        # Track steps for info
        self._steps = 0
        self._total_safe_cells = (
            self.config.width * self.config.height - self.config.num_mines
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)
        self.board.reset()
        self._steps = 0

        observation = self.board.get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Args:
            action: Cell index to reveal (row * width + col).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        row, col = self._action_to_position(action)
        self._steps += 1

        # Calculate reward based on action result
        reward = self._calculate_reward(row, col)

        # Get new observation
        observation = self.board.get_observation()

        # Check if episode is done
        terminated = not self.board.is_playing
        truncated = False

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _action_to_position(self, action: int) -> Tuple[int, int]:
        """Convert flat action index to (row, col) position."""
        row = action // self.config.width
        col = action % self.config.width
        return row, col

    def _calculate_reward(self, row: int, col: int) -> float:
        """
        Calculate reward for revealing a cell.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Reward value.
        """
        cell = self.board.get_cell(row, col)

        # Invalid action (already revealed or flagged)
        if cell is None or not cell.is_hidden:
            return -0.1

        # Perform the reveal
        self.board.reveal(row, col)

        # Check game state
        if self.board.is_won:
            return 10.0
        if self.board.is_lost:
            return -10.0

        # Successful safe reveal
        return 1.0

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for current state."""
        revealed = sum(
            1 for r in range(self.config.height)
            for c in range(self.config.width)
            if self.board.get_cell(r, c).is_revealed
        )

        return {
            "steps": self._steps,
            "revealed": revealed,
            "total_safe": self._total_safe_cells,
            "game_state": self.board.game_state.name,
            "valid_actions": len(self.board.get_valid_actions()),
        }

    def render(self) -> Optional[str]:
        """Render the current board state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            print(self._render_ansi())
        return None

    def _render_ansi(self) -> str:
        """Render board as ASCII string."""
        lines = []
        obs = self.board.get_observation()

        for row in range(self.config.height):
            row_str = ""
            for col in range(self.config.width):
                val = obs[row, col]
                if val == -1:
                    row_str += "."
                elif val == -2:
                    row_str += "F"
                elif val == 9:
                    row_str += "*"
                elif val == 0:
                    row_str += " "
                else:
                    row_str += str(val)
                row_str += " "
            lines.append(row_str)

        return "\n".join(lines)

    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of valid actions.

        Returns:
            Boolean array where True = valid action.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        for row, col in self.board.get_valid_actions():
            action = row * self.config.width + col
            mask[action] = True
        return mask


# ============================================================================
# Vectorized Environment Factory
# ============================================================================

def make_vec_env(
    n_envs: int = 4,
    config: Optional[BoardConfig] = None,
) -> gym.vector.VectorEnv:
    """
    Create vectorized environment for parallel training.

    Args:
        n_envs: Number of parallel environments.
        config: Board configuration.

    Returns:
        Vectorized environment.
    """
    def make_env() -> MinesweeperEnv:
        return MinesweeperEnv(config=config)

    return gym.vector.AsyncVectorEnv([make_env for _ in range(n_envs)])
