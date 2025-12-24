"""
Minesweeper game module.

Provides core game logic including board management and cell state.
"""
from .cell import Cell, CellState
from .board import Board, BoardConfig, GameState, BEGINNER, INTERMEDIATE, EXPERT
from .environment import MinesweeperEnv, make_vec_env

__all__ = [
    "Cell",
    "CellState",
    "Board",
    "BoardConfig",
    "GameState",
    "BEGINNER",
    "INTERMEDIATE",
    "EXPERT",
    "MinesweeperEnv",
    "make_vec_env",
]
