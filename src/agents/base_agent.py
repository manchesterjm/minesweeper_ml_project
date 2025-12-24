"""
Base agent interface for Minesweeper AI.

Defines the abstract interface that all agents must implement.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


# ============================================================================
# Base Agent Interface
# ============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for Minesweeper agents.

    All agents must implement the select_action method to choose
    which cell to reveal based on the current observation.
    """

    def __init__(self, board_height: int, board_width: int) -> None:
        """
        Initialize the agent.

        Args:
            board_height: Number of rows in the board.
            board_width: Number of columns in the board.
        """
        self.board_height = board_height
        self.board_width = board_width
        self.total_cells = board_height * board_width

    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Optional[np.ndarray] = None,
    ) -> int:
        """
        Select an action based on the current observation.

        Args:
            observation: 2D array of cell states.
            valid_actions: Optional mask of valid actions.

        Returns:
            Action index (row * width + col).
        """
        pass

    def action_to_position(self, action: int) -> Tuple[int, int]:
        """Convert flat action index to (row, col) position."""
        row = action // self.board_width
        col = action % self.board_width
        return row, col

    def position_to_action(self, row: int, col: int) -> int:
        """Convert (row, col) position to flat action index."""
        return row * self.board_width + col

    def get_valid_actions_from_obs(self, observation: np.ndarray) -> np.ndarray:
        """
        Get valid actions mask from observation.

        Args:
            observation: 2D array of cell states.

        Returns:
            Boolean mask where True = valid action.
        """
        # Hidden cells (value -1) are valid actions
        flat_obs = observation.flatten()
        return flat_obs == -1

    def reset(self) -> None:
        """Reset agent state for new episode."""
        pass

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update agent with experience (for learning agents).

        Args:
            observation: State before action.
            action: Action taken.
            reward: Reward received.
            next_observation: State after action.
            done: Whether episode ended.
        """
        pass
