"""
Board module for Minesweeper game.

Implements the game board with mine placement, cell revealing,
and game state management.
"""
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Set, Optional

import numpy as np

from .cell import Cell, CellState


# ============================================================================
# Constants
# ============================================================================

class GameState(Enum):
    """Possible states of the game."""

    PLAYING = auto()
    WON = auto()
    LOST = auto()


@dataclass
class BoardConfig:
    """
    Configuration for a Minesweeper board.

    Attributes:
        width: Number of columns.
        height: Number of rows.
        num_mines: Total mines to place.
    """

    width: int = 9
    height: int = 9
    num_mines: int = 10

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Ensure configuration values are valid."""
        if self.width < 1 or self.height < 1:
            raise ValueError("Board dimensions must be positive")
        if self.num_mines < 0:
            raise ValueError("Number of mines cannot be negative")
        max_mines = self.width * self.height - 1
        if self.num_mines > max_mines:
            raise ValueError(f"Too many mines (max {max_mines})")


# Preset difficulty levels
BEGINNER = BoardConfig(9, 9, 10)
INTERMEDIATE = BoardConfig(16, 16, 40)
EXPERT = BoardConfig(30, 16, 99)


# ============================================================================
# Board Class
# ============================================================================

@dataclass
class Board:
    """
    Minesweeper game board.

    Manages the grid of cells, mine placement, revealing logic,
    and win/lose conditions.
    """

    config: BoardConfig = field(default_factory=lambda: BoardConfig())
    _grid: List[List[Cell]] = field(default_factory=list, repr=False)
    _game_state: GameState = GameState.PLAYING
    _first_click: bool = True
    _cells_revealed: int = 0

    def __post_init__(self) -> None:
        """Initialize the grid after dataclass creation."""
        self._init_grid()

    # ========================================================================
    # Grid Initialization (Low-level)
    # ========================================================================

    def _init_grid(self) -> None:
        """Create empty grid of cells."""
        self._grid = [
            [Cell() for _ in range(self.config.width)]
            for _ in range(self.config.height)
        ]

    def _place_mines(self, exclude: Tuple[int, int]) -> None:
        """
        Place mines randomly, excluding a specific cell.

        Args:
            exclude: (row, col) position to keep mine-free.
        """
        positions = self._get_valid_mine_positions(exclude)
        mine_positions = random.sample(positions, self.config.num_mines)
        for row, col in mine_positions:
            self._grid[row][col].is_mine = True

    def _get_valid_mine_positions(
        self, exclude: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Get all valid positions for mine placement."""
        positions = []
        for row in range(self.config.height):
            for col in range(self.config.width):
                if (row, col) != exclude:
                    positions.append((row, col))
        return positions

    def _calculate_adjacent_mines(self) -> None:
        """Calculate adjacent mine counts for all cells."""
        for row in range(self.config.height):
            for col in range(self.config.width):
                if not self._grid[row][col].is_mine:
                    count = self._count_adjacent_mines(row, col)
                    self._grid[row][col].adjacent_mines = count

    def _count_adjacent_mines(self, row: int, col: int) -> int:
        """Count mines adjacent to a specific cell."""
        count = 0
        for neighbor_row, neighbor_col in self._get_neighbors(row, col):
            if self._grid[neighbor_row][neighbor_col].is_mine:
                count += 1
        return count

    # ========================================================================
    # Neighbor Utilities (Low-level)
    # ========================================================================

    def _get_neighbors(
        self, row: int, col: int
    ) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cell positions.

        Args:
            row: Row index of center cell.
            col: Column index of center cell.

        Returns:
            List of (row, col) tuples for valid neighbors.
        """
        neighbors = []
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                new_row = row + delta_row
                new_col = col + delta_col
                if self._is_valid_position(new_row, new_col):
                    neighbors.append((new_row, new_col))
        return neighbors

    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= row < self.config.height and 0 <= col < self.config.width

    # ========================================================================
    # Game Actions (Mid-level)
    # ========================================================================

    def reveal(self, row: int, col: int) -> bool:
        """
        Reveal a cell at the given position.

        On first click, places mines avoiding this cell.
        If cell is empty (0 adjacent mines), reveals neighbors recursively.
        If cell is a mine, game is lost.

        Args:
            row: Row index to reveal.
            col: Column index to reveal.

        Returns:
            True if reveal was successful, False otherwise.
        """
        if not self._can_reveal(row, col):
            return False

        if self._first_click:
            self._handle_first_click(row, col)

        return self._reveal_cell(row, col)

    def _can_reveal(self, row: int, col: int) -> bool:
        """Check if a cell can be revealed."""
        if self._game_state != GameState.PLAYING:
            return False
        if not self._is_valid_position(row, col):
            return False
        cell = self._grid[row][col]
        return cell.state == CellState.HIDDEN

    def _handle_first_click(self, row: int, col: int) -> None:
        """Handle first click: place mines and calculate counts."""
        self._first_click = False
        self._place_mines((row, col))
        self._calculate_adjacent_mines()

    def _reveal_cell(self, row: int, col: int) -> bool:
        """Reveal a single cell and handle consequences."""
        cell = self._grid[row][col]
        if not cell.reveal():
            return False

        self._cells_revealed += 1

        if cell.is_mine:
            self._game_state = GameState.LOST
            return True

        if cell.adjacent_mines == 0:
            self._reveal_neighbors(row, col)

        self._check_win_condition()
        return True

    def _reveal_neighbors(self, row: int, col: int) -> None:
        """Recursively reveal neighbors of an empty cell."""
        for neighbor_row, neighbor_col in self._get_neighbors(row, col):
            neighbor = self._grid[neighbor_row][neighbor_col]
            if neighbor.state == CellState.HIDDEN:
                self._reveal_cell(neighbor_row, neighbor_col)

    def _check_win_condition(self) -> None:
        """Check if all non-mine cells are revealed."""
        total_cells = self.config.width * self.config.height
        non_mine_cells = total_cells - self.config.num_mines
        if self._cells_revealed >= non_mine_cells:
            self._game_state = GameState.WON

    def flag(self, row: int, col: int) -> bool:
        """
        Toggle flag on a cell.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if flag was toggled, False otherwise.
        """
        if self._game_state != GameState.PLAYING:
            return False
        if not self._is_valid_position(row, col):
            return False
        return self._grid[row][col].toggle_flag()

    def chord(self, row: int, col: int) -> bool:
        """
        Chord action: reveal all unflagged neighbors if flag count matches.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if chord was performed, False otherwise.
        """
        if not self._can_chord(row, col):
            return False

        revealed_any = False
        for neighbor_row, neighbor_col in self._get_neighbors(row, col):
            neighbor = self._grid[neighbor_row][neighbor_col]
            if neighbor.state == CellState.HIDDEN:
                self._reveal_cell(neighbor_row, neighbor_col)
                revealed_any = True

        return revealed_any

    def _can_chord(self, row: int, col: int) -> bool:
        """Check if chord action is valid."""
        if self._game_state != GameState.PLAYING:
            return False
        if not self._is_valid_position(row, col):
            return False
        cell = self._grid[row][col]
        if not cell.is_revealed or cell.adjacent_mines == 0:
            return False
        flag_count = self._count_adjacent_flags(row, col)
        return flag_count == cell.adjacent_mines

    def _count_adjacent_flags(self, row: int, col: int) -> int:
        """Count flagged cells adjacent to position."""
        count = 0
        for neighbor_row, neighbor_col in self._get_neighbors(row, col):
            if self._grid[neighbor_row][neighbor_col].is_flagged:
                count += 1
        return count

    # ========================================================================
    # State Accessors (High-level)
    # ========================================================================

    @property
    def game_state(self) -> GameState:
        """Get current game state."""
        return self._game_state

    @property
    def is_playing(self) -> bool:
        """Check if game is still in progress."""
        return self._game_state == GameState.PLAYING

    @property
    def is_won(self) -> bool:
        """Check if game was won."""
        return self._game_state == GameState.WON

    @property
    def is_lost(self) -> bool:
        """Check if game was lost."""
        return self._game_state == GameState.LOST

    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """Get cell at position, or None if invalid."""
        if not self._is_valid_position(row, col):
            return None
        return self._grid[row][col]

    def get_observation(self) -> np.ndarray:
        """
        Get board state as numpy array for ML agent.

        Returns:
            2D numpy array where:
                -1 = hidden
                -2 = flagged
                0-8 = revealed with adjacent count
                9 = revealed mine
        """
        obs = np.zeros((self.config.height, self.config.width), dtype=np.int8)
        for row in range(self.config.height):
            for col in range(self.config.width):
                obs[row, col] = self._grid[row][col].to_observation()
        return obs

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """
        Get list of valid cells to reveal.

        Returns:
            List of (row, col) positions that can be revealed.
        """
        actions = []
        for row in range(self.config.height):
            for col in range(self.config.width):
                if self._grid[row][col].state == CellState.HIDDEN:
                    actions.append((row, col))
        return actions

    def reset(self) -> None:
        """Reset board to initial state for new game."""
        self._init_grid()
        self._game_state = GameState.PLAYING
        self._first_click = True
        self._cells_revealed = 0
