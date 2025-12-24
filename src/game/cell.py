"""
Cell module for Minesweeper game.

Represents individual cells on the game board with their state
(hidden/revealed/flagged) and content (mine/number).
"""
from enum import Enum, auto
from dataclasses import dataclass


# ============================================================================
# Constants
# ============================================================================

class CellState(Enum):
    """Possible visual states of a cell."""

    HIDDEN = auto()
    REVEALED = auto()
    FLAGGED = auto()


# ============================================================================
# Cell Data Class
# ============================================================================

@dataclass
class Cell:
    """
    Represents a single cell in the Minesweeper grid.

    Attributes:
        is_mine: Whether this cell contains a mine.
        adjacent_mines: Count of mines in neighboring cells (0-8).
        state: Current visual state (hidden, revealed, or flagged).
    """

    is_mine: bool = False
    adjacent_mines: int = 0
    state: CellState = CellState.HIDDEN

    def reveal(self) -> bool:
        """
        Reveal this cell.

        Returns:
            True if cell was successfully revealed, False if already
            revealed or flagged.
        """
        if self.state != CellState.HIDDEN:
            return False
        self.state = CellState.REVEALED
        return True

    def toggle_flag(self) -> bool:
        """
        Toggle flag on this cell.

        Returns:
            True if flag was toggled, False if cell is revealed.
        """
        if self.state == CellState.REVEALED:
            return False
        if self.state == CellState.HIDDEN:
            self.state = CellState.FLAGGED
        else:
            self.state = CellState.HIDDEN
        return True

    @property
    def is_hidden(self) -> bool:
        """Check if cell is hidden."""
        return self.state == CellState.HIDDEN

    @property
    def is_revealed(self) -> bool:
        """Check if cell is revealed."""
        return self.state == CellState.REVEALED

    @property
    def is_flagged(self) -> bool:
        """Check if cell is flagged."""
        return self.state == CellState.FLAGGED

    def to_observation(self) -> int:
        """
        Convert cell to observation value for ML agent.

        Returns:
            -1: Hidden cell
            -2: Flagged cell
            0-8: Revealed cell with adjacent mine count
            9: Revealed mine (game over state)
        """
        if self.state == CellState.HIDDEN:
            return -1
        if self.state == CellState.FLAGGED:
            return -2
        if self.is_mine:
            return 9
        return self.adjacent_mines
