"""
Training module for Minesweeper agents.

Provides training loops, evaluation, and checkpointing.
"""
from .trainer import (
    TrainingConfig,
    EpisodeStats,
    TrainingStats,
    Trainer,
    Evaluator,
)

__all__ = [
    "TrainingConfig",
    "EpisodeStats",
    "TrainingStats",
    "Trainer",
    "Evaluator",
]
