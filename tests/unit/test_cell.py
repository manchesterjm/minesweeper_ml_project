"""
Unit tests for Cell class.

Tests cell state management, reveal/flag behavior, and observation conversion.
"""
import pytest
from game import Cell, CellState


# ============================================================================
# Cell Initialization Tests
# ============================================================================

class TestCellInitialization:
    """Test cell creation and default values."""

    def test_default_cell_is_not_mine(self) -> None:
        """New cell should not be a mine by default."""
        cell = Cell()
        assert cell.is_mine is False

    def test_default_cell_is_hidden(self) -> None:
        """New cell should be hidden by default."""
        cell = Cell()
        assert cell.state == CellState.HIDDEN
        assert cell.is_hidden is True

    def test_default_cell_has_zero_adjacent_mines(self) -> None:
        """New cell should have 0 adjacent mines by default."""
        cell = Cell()
        assert cell.adjacent_mines == 0

    def test_mine_cell_creation(self) -> None:
        """Can create a cell that is a mine."""
        cell = Cell(is_mine=True)
        assert cell.is_mine is True

    def test_cell_with_adjacent_mines(self) -> None:
        """Can create a cell with adjacent mine count."""
        cell = Cell(adjacent_mines=5)
        assert cell.adjacent_mines == 5


# ============================================================================
# Cell Reveal Tests
# ============================================================================

class TestCellReveal:
    """Test cell reveal behavior."""

    def test_reveal_hidden_cell_returns_true(self, hidden_cell: Cell) -> None:
        """Revealing a hidden cell should succeed."""
        result = hidden_cell.reveal()
        assert result is True

    def test_reveal_changes_state_to_revealed(self, hidden_cell: Cell) -> None:
        """Revealing a cell should change its state."""
        hidden_cell.reveal()
        assert hidden_cell.state == CellState.REVEALED
        assert hidden_cell.is_revealed is True

    def test_reveal_already_revealed_returns_false(
        self, hidden_cell: Cell
    ) -> None:
        """Revealing an already revealed cell should fail."""
        hidden_cell.reveal()
        result = hidden_cell.reveal()
        assert result is False

    def test_reveal_flagged_cell_returns_false(self, hidden_cell: Cell) -> None:
        """Cannot reveal a flagged cell."""
        hidden_cell.toggle_flag()
        result = hidden_cell.reveal()
        assert result is False


# ============================================================================
# Cell Flag Tests
# ============================================================================

class TestCellFlag:
    """Test cell flagging behavior."""

    def test_flag_hidden_cell_returns_true(self, hidden_cell: Cell) -> None:
        """Flagging a hidden cell should succeed."""
        result = hidden_cell.toggle_flag()
        assert result is True

    def test_flag_changes_state_to_flagged(self, hidden_cell: Cell) -> None:
        """Flagging a cell should change its state."""
        hidden_cell.toggle_flag()
        assert hidden_cell.state == CellState.FLAGGED
        assert hidden_cell.is_flagged is True

    def test_unflag_returns_to_hidden(self, hidden_cell: Cell) -> None:
        """Unflagging a cell should return it to hidden."""
        hidden_cell.toggle_flag()
        hidden_cell.toggle_flag()
        assert hidden_cell.state == CellState.HIDDEN
        assert hidden_cell.is_hidden is True

    def test_flag_revealed_cell_returns_false(self, hidden_cell: Cell) -> None:
        """Cannot flag a revealed cell."""
        hidden_cell.reveal()
        result = hidden_cell.toggle_flag()
        assert result is False


# ============================================================================
# Cell Observation Tests
# ============================================================================

class TestCellObservation:
    """Test cell observation values for ML agent."""

    def test_hidden_cell_observation_is_negative_one(
        self, hidden_cell: Cell
    ) -> None:
        """Hidden cell should return -1 for observation."""
        assert hidden_cell.to_observation() == -1

    def test_flagged_cell_observation_is_negative_two(
        self, hidden_cell: Cell
    ) -> None:
        """Flagged cell should return -2 for observation."""
        hidden_cell.toggle_flag()
        assert hidden_cell.to_observation() == -2

    def test_revealed_empty_cell_observation_is_zero(
        self, hidden_cell: Cell
    ) -> None:
        """Revealed cell with 0 adjacent mines returns 0."""
        hidden_cell.reveal()
        assert hidden_cell.to_observation() == 0

    @pytest.mark.parametrize("count", range(1, 9))
    def test_revealed_cell_observation_matches_adjacent_count(
        self, count: int
    ) -> None:
        """Revealed cell returns its adjacent mine count."""
        cell = Cell(adjacent_mines=count)
        cell.reveal()
        assert cell.to_observation() == count

    def test_revealed_mine_observation_is_nine(self, mine_cell: Cell) -> None:
        """Revealed mine should return 9 for observation."""
        mine_cell.reveal()
        assert mine_cell.to_observation() == 9
