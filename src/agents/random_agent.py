"""
Random agent for Minesweeper.

Serves as a baseline by selecting random valid actions.
"""
from typing import Optional

import numpy as np

from .base_agent import BaseAgent


# ============================================================================
# Random Agent
# ============================================================================

class RandomAgent(BaseAgent):
    """
    Agent that selects actions uniformly at random.

    This provides a baseline for comparing more sophisticated agents.
    Expected win rate on beginner: ~10-15%
    """

    def __init__(
        self,
        board_height: int = 9,
        board_width: int = 9,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the random agent.

        Args:
            board_height: Number of rows in the board.
            board_width: Number of columns in the board.
            seed: Random seed for reproducibility.
        """
        super().__init__(board_height, board_width)
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Optional[np.ndarray] = None,
    ) -> int:
        """
        Select a random valid action.

        Args:
            observation: 2D array of cell states.
            valid_actions: Optional mask of valid actions.

        Returns:
            Random action index from valid actions.
        """
        if valid_actions is None:
            valid_actions = self.get_valid_actions_from_obs(observation)

        valid_indices = np.where(valid_actions)[0]

        if len(valid_indices) == 0:
            # No valid actions, return any action (will be invalid)
            return 0

        return self.rng.choice(valid_indices)

    def reset(self) -> None:
        """Reset is a no-op for random agent."""
        pass
