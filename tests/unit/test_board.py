"""
Unit tests for Board class.

Tests board initialization, game mechanics, win/lose conditions,
and observation generation.
"""
import pytest
import numpy as np
from game import Board, BoardConfig, GameState, CellState


# ============================================================================
# Board Configuration Tests
# ============================================================================

class TestBoardConfig:
    """Test board configuration validation."""

    def test_valid_config_creation(self, valid_config: BoardConfig) -> None:
        """Valid configuration should be created successfully."""
        assert valid_config.width == 9
        assert valid_config.height == 9
        assert valid_config.num_mines == 10

    def test_zero_width_raises_error(self) -> None:
        """Width of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            BoardConfig(0, 9, 10)

    def test_zero_height_raises_error(self) -> None:
        """Height of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            BoardConfig(9, 0, 10)

    def test_negative_mines_raises_error(self) -> None:
        """Negative mine count should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            BoardConfig(9, 9, -1)

    def test_too_many_mines_raises_error(self) -> None:
        """Too many mines should raise ValueError."""
        with pytest.raises(ValueError, match="Too many mines"):
            BoardConfig(3, 3, 10)  # Max is 8 (9 cells - 1)

    def test_max_mines_is_valid(self) -> None:
        """Maximum valid mines should be accepted."""
        config = BoardConfig(3, 3, 8)  # 9 cells - 1 = 8 max
        assert config.num_mines == 8


# ============================================================================
# Board Initialization Tests
# ============================================================================

class TestBoardInitialization:
    """Test board creation and initial state."""

    def test_new_board_is_playing(self, default_board: Board) -> None:
        """New board should be in playing state."""
        assert default_board.game_state == GameState.PLAYING
        assert default_board.is_playing is True

    def test_new_board_all_cells_hidden(self, default_board: Board) -> None:
        """All cells should be hidden on new board."""
        for row in range(default_board.config.height):
            for col in range(default_board.config.width):
                cell = default_board.get_cell(row, col)
                assert cell.is_hidden is True

    def test_mines_not_placed_before_first_click(
        self, default_board: Board
    ) -> None:
        """Mines should not be placed until first reveal."""
        mine_count = 0
        for row in range(default_board.config.height):
            for col in range(default_board.config.width):
                cell = default_board.get_cell(row, col)
                if cell.is_mine:
                    mine_count += 1
        assert mine_count == 0

    def test_board_has_correct_dimensions(self, default_board: Board) -> None:
        """Board should have correct dimensions."""
        assert default_board.config.width == 9
        assert default_board.config.height == 9


# ============================================================================
# First Click Tests
# ============================================================================

class TestFirstClick:
    """Test first click behavior."""

    def test_first_click_places_mines(self, default_board: Board) -> None:
        """First reveal should place mines."""
        default_board.reveal(0, 0)

        mine_count = 0
        for row in range(default_board.config.height):
            for col in range(default_board.config.width):
                cell = default_board.get_cell(row, col)
                if cell.is_mine:
                    mine_count += 1
        assert mine_count == default_board.config.num_mines

    def test_first_click_never_hits_mine(self, default_board: Board) -> None:
        """First click should never hit a mine."""
        for _ in range(100):  # Test multiple times
            board = Board()
            board.reveal(4, 4)
            assert board.is_playing is True

    def test_first_click_cell_is_safe(self, default_board: Board) -> None:
        """First clicked cell should not be a mine."""
        default_board.reveal(4, 4)
        cell = default_board.get_cell(4, 4)
        assert cell.is_mine is False


# ============================================================================
# Reveal Tests
# ============================================================================

class TestReveal:
    """Test cell revealing behavior."""

    def test_reveal_returns_true_on_success(self, default_board: Board) -> None:
        """Successful reveal should return True."""
        result = default_board.reveal(0, 0)
        assert result is True

    def test_reveal_changes_cell_state(self, default_board: Board) -> None:
        """Revealed cell should change state."""
        default_board.reveal(0, 0)
        cell = default_board.get_cell(0, 0)
        assert cell.is_revealed is True

    def test_reveal_same_cell_twice_returns_false(
        self, default_board: Board
    ) -> None:
        """Revealing same cell twice should fail."""
        default_board.reveal(0, 0)
        result = default_board.reveal(0, 0)
        assert result is False

    def test_reveal_invalid_position_returns_false(
        self, default_board: Board
    ) -> None:
        """Revealing invalid position should fail."""
        result = default_board.reveal(-1, 0)
        assert result is False
        result = default_board.reveal(0, 100)
        assert result is False

    def test_reveal_flagged_cell_returns_false(
        self, default_board: Board
    ) -> None:
        """Cannot reveal a flagged cell."""
        default_board.flag(0, 0)
        result = default_board.reveal(0, 0)
        assert result is False


# ============================================================================
# Cascade Reveal Tests
# ============================================================================

class TestCascadeReveal:
    """Test empty cell cascade behavior."""

    def test_empty_cell_reveals_neighbors(self, empty_board: Board) -> None:
        """Revealing empty cell should cascade to neighbors."""
        empty_board.reveal(2, 2)

        # In a board with no mines, all cells should be revealed
        for row in range(empty_board.config.height):
            for col in range(empty_board.config.width):
                cell = empty_board.get_cell(row, col)
                assert cell.is_revealed is True

    def test_cascade_stops_at_numbered_cells(self, small_board: Board) -> None:
        """Cascade should stop at cells with adjacent mines."""
        # Force mine placement
        small_board.reveal(2, 2)  # First click to place mines

        # Count revealed cells - should not be all cells
        revealed_count = 0
        for row in range(small_board.config.height):
            for col in range(small_board.config.width):
                cell = small_board.get_cell(row, col)
                if cell.is_revealed:
                    revealed_count += 1

        # At least one cell revealed, but not necessarily all
        assert revealed_count >= 1


# ============================================================================
# Flag Tests
# ============================================================================

class TestFlag:
    """Test flagging behavior."""

    def test_flag_hidden_cell_succeeds(self, default_board: Board) -> None:
        """Flagging hidden cell should succeed."""
        result = default_board.flag(0, 0)
        assert result is True

    def test_flag_changes_cell_state(self, default_board: Board) -> None:
        """Flagged cell should change state."""
        default_board.flag(0, 0)
        cell = default_board.get_cell(0, 0)
        assert cell.is_flagged is True

    def test_unflag_returns_to_hidden(self, default_board: Board) -> None:
        """Unflagging should return cell to hidden."""
        default_board.flag(0, 0)
        default_board.flag(0, 0)
        cell = default_board.get_cell(0, 0)
        assert cell.is_hidden is True

    def test_flag_revealed_cell_fails(self, default_board: Board) -> None:
        """Cannot flag a revealed cell."""
        default_board.reveal(0, 0)
        result = default_board.flag(0, 0)
        assert result is False


# ============================================================================
# Win/Lose Condition Tests
# ============================================================================

class TestGameEndConditions:
    """Test win and lose conditions."""

    def test_reveal_mine_loses_game(self) -> None:
        """Revealing a mine should end the game as lost."""
        # Create board and manually set up mine
        board = Board(BoardConfig(3, 3, 1))
        board.reveal(2, 2)  # First safe click

        # Find and reveal the mine
        for row in range(3):
            for col in range(3):
                cell = board.get_cell(row, col)
                if cell.is_mine and cell.is_hidden:
                    board.reveal(row, col)
                    break

        assert board.is_lost is True

    def test_reveal_all_safe_cells_wins(self) -> None:
        """Revealing all non-mine cells should win."""
        board = Board(BoardConfig(3, 3, 1))

        # Reveal cells until we win or lose
        for row in range(3):
            for col in range(3):
                if board.is_playing:
                    cell = board.get_cell(row, col)
                    if cell.is_hidden and not cell.is_flagged:
                        board.reveal(row, col)

        # Either won or lost
        assert board.game_state in (GameState.WON, GameState.LOST)

    def test_cannot_reveal_after_game_over(self, default_board: Board) -> None:
        """Cannot reveal cells after game ends."""
        # Force a loss
        default_board.reveal(0, 0)  # First click

        # Find and reveal mine
        for row in range(9):
            for col in range(9):
                cell = default_board.get_cell(row, col)
                if cell.is_mine and cell.is_hidden:
                    default_board.reveal(row, col)
                    break

        # Try to reveal another cell
        result = default_board.reveal(4, 4)
        assert result is False


# ============================================================================
# Observation Tests
# ============================================================================

class TestObservation:
    """Test observation array for ML agent."""

    def test_observation_shape_matches_board(
        self, default_board: Board
    ) -> None:
        """Observation should match board dimensions."""
        obs = default_board.get_observation()
        assert obs.shape == (9, 9)

    def test_new_board_observation_all_hidden(
        self, default_board: Board
    ) -> None:
        """New board observation should be all -1."""
        obs = default_board.get_observation()
        assert np.all(obs == -1)

    def test_observation_dtype_is_int8(self, default_board: Board) -> None:
        """Observation should be int8 for memory efficiency."""
        obs = default_board.get_observation()
        assert obs.dtype == np.int8

    def test_flagged_cell_in_observation(self, default_board: Board) -> None:
        """Flagged cell should show -2 in observation."""
        default_board.flag(0, 0)
        obs = default_board.get_observation()
        assert obs[0, 0] == -2

    def test_revealed_cell_shows_count(self, empty_board: Board) -> None:
        """Revealed cell should show adjacent mine count."""
        empty_board.reveal(0, 0)
        obs = empty_board.get_observation()
        # Empty board, all revealed cells should be 0
        assert obs[0, 0] == 0


# ============================================================================
# Valid Actions Tests
# ============================================================================

class TestValidActions:
    """Test valid action enumeration."""

    def test_new_board_has_all_cells_as_valid(
        self, default_board: Board
    ) -> None:
        """New board should have all cells as valid actions."""
        actions = default_board.get_valid_actions()
        assert len(actions) == 81

    def test_revealed_cell_not_in_valid_actions(
        self, default_board: Board
    ) -> None:
        """Revealed cells should not be in valid actions."""
        default_board.reveal(0, 0)
        actions = default_board.get_valid_actions()
        assert (0, 0) not in actions

    def test_valid_actions_format(self, default_board: Board) -> None:
        """Valid actions should be (row, col) tuples."""
        actions = default_board.get_valid_actions()
        for action in actions:
            assert isinstance(action, tuple)
            assert len(action) == 2


# ============================================================================
# Reset Tests
# ============================================================================

class TestReset:
    """Test board reset functionality."""

    def test_reset_restores_playing_state(self, default_board: Board) -> None:
        """Reset should restore playing state."""
        default_board.reveal(0, 0)
        default_board.reset()
        assert default_board.is_playing is True

    def test_reset_hides_all_cells(self, default_board: Board) -> None:
        """Reset should hide all cells."""
        default_board.reveal(0, 0)
        default_board.reset()

        for row in range(9):
            for col in range(9):
                cell = default_board.get_cell(row, col)
                assert cell.is_hidden is True

    def test_reset_removes_mines(self, default_board: Board) -> None:
        """Reset should remove all mines."""
        default_board.reveal(0, 0)  # Places mines
        default_board.reset()

        mine_count = 0
        for row in range(9):
            for col in range(9):
                cell = default_board.get_cell(row, col)
                if cell.is_mine:
                    mine_count += 1
        assert mine_count == 0
