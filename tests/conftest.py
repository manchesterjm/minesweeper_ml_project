"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from game import Board, BoardConfig, Cell, CellState, GameState


# ============================================================================
# Board Fixtures
# ============================================================================

@pytest.fixture
def default_board() -> Board:
    """Create a default 9x9 board with 10 mines."""
    return Board()


@pytest.fixture
def beginner_board() -> Board:
    """Create a beginner difficulty board."""
    return Board(BoardConfig(9, 9, 10))


@pytest.fixture
def small_board() -> Board:
    """Create a small 3x3 board with 1 mine for testing."""
    return Board(BoardConfig(3, 3, 1))


@pytest.fixture
def empty_board() -> Board:
    """Create a board with no mines for cascade testing."""
    return Board(BoardConfig(5, 5, 0))


# ============================================================================
# Cell Fixtures
# ============================================================================

@pytest.fixture
def hidden_cell() -> Cell:
    """Create a hidden cell."""
    return Cell()


@pytest.fixture
def mine_cell() -> Cell:
    """Create a cell containing a mine."""
    return Cell(is_mine=True)


@pytest.fixture
def numbered_cell() -> Cell:
    """Create a revealed cell with adjacent mines."""
    cell = Cell(adjacent_mines=3)
    cell.reveal()
    return cell


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def valid_config() -> BoardConfig:
    """Create a valid board configuration."""
    return BoardConfig(9, 9, 10)


@pytest.fixture
def beginner_config() -> BoardConfig:
    """Beginner difficulty configuration."""
    return BoardConfig(9, 9, 10)


@pytest.fixture
def intermediate_config() -> BoardConfig:
    """Intermediate difficulty configuration."""
    return BoardConfig(16, 16, 40)


@pytest.fixture
def expert_config() -> BoardConfig:
    """Expert difficulty configuration."""
    return BoardConfig(30, 16, 99)
