"""
Hybrid agent combining AC-3 constraint propagation with neural network guessing.

Uses logical deduction when possible, falls back to a trained neural network
for uncertain situations where guessing is required.
"""
from typing import Optional, Set, Tuple, Dict, List, FrozenSet
from dataclasses import dataclass
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent
from .dqn_agent import get_device


# ============================================================================
# Guess Network - Predicts safety probability for each cell
# ============================================================================

class GuessNetwork(nn.Module):
    """
    Neural network that predicts safety probability for hidden cells.

    Takes board state as input, outputs probability that each cell is safe.
    Used only when AC-3 constraint propagation finds no certain moves.
    """

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        n_cells = height * width

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Output layer predicts safety score for each cell
        self.conv_out = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Board state tensor (batch, height, width)

        Returns:
            Safety scores for each cell (batch, height * width)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Output safety scores
        x = self.conv_out(x)
        x = x.view(x.size(0), -1)

        return x


# ============================================================================
# Experience buffer for guess situations only
# ============================================================================

@dataclass
class GuessExperience:
    """Experience from a guessing situation."""
    state: np.ndarray
    action: int
    was_safe: bool  # True if the guess was safe, False if hit mine


class GuessBuffer:
    """Buffer storing only guessing situations for training."""

    def __init__(self, capacity: int = 50000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, was_safe: bool) -> None:
        self.buffer.append(GuessExperience(state, action, was_safe))

    def sample(self, batch_size: int) -> List[GuessExperience]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# Constraint types (copied from logic_agent for self-containment)
# ============================================================================

@dataclass
class Constraint:
    """Constraint: sum of cells equals mine_count."""
    cells: FrozenSet[Tuple[int, int]]
    mine_count: int

    def __hash__(self) -> int:
        return hash((self.cells, self.mine_count))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constraint):
            return False
        return self.cells == other.cells and self.mine_count == other.mine_count


# ============================================================================
# Hybrid Agent
# ============================================================================

class HybridAgent(BaseAgent):
    """
    Hybrid agent: AC-3 logic + neural network for guessing.

    Strategy:
        1. Use AC-3 constraint propagation to find definite safe/mine cells
        2. If certain moves exist, take one
        3. If no certain moves (forced guess), use trained neural network
        4. Network predicts which hidden cell is most likely safe

    The neural network is trained only on "forced guess" situations,
    learning patterns that indicate safety beyond what pure logic can deduce.
    """

    def __init__(
        self,
        board_height: int = 9,
        board_width: int = 9,
        learning_rate: float = 1e-3,
        buffer_size: int = 50000,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(board_height, board_width)

        self.device = device or get_device()
        self.batch_size = batch_size
        self._first_move = True

        # Neural network for guessing
        self.guess_net = GuessNetwork(board_height, board_width).to(self.device)
        self.optimizer = optim.Adam(self.guess_net.parameters(), lr=learning_rate)

        # Experience buffer for guess situations
        self.guess_buffer = GuessBuffer(buffer_size)

        # Track current guess for delayed feedback
        self._pending_guess: Optional[Tuple[np.ndarray, int]] = None

        # Stats
        self.guesses_made = 0
        self.guesses_correct = 0
        self.certain_moves = 0
        self.losses: List[float] = []

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Optional[np.ndarray] = None,
        training: bool = True,
    ) -> int:
        """Select action using AC-3 logic, falling back to neural network."""
        if valid_actions is None:
            valid_actions = self.get_valid_actions_from_obs(observation)

        valid_indices = np.where(valid_actions)[0]

        if len(valid_indices) == 0:
            return 0

        # First move: prefer corners
        if self._first_move:
            self._first_move = False
            return self._select_corner(valid_indices)

        # Use AC-3 to find certain moves
        safe_cells, mine_cells = self._solve_constraints(observation)

        # If we found definitely safe cells, take one
        if safe_cells:
            for row, col in safe_cells:
                action = self.position_to_action(row, col)
                if action in valid_indices:
                    self.certain_moves += 1
                    return action

        # No certain moves - must guess using neural network
        self.guesses_made += 1

        # Store state for learning (we'll get feedback after the move)
        if training:
            self._pending_guess = (observation.copy(), None)

        action = self._neural_guess(observation, valid_indices, mine_cells, training)

        if training and self._pending_guess is not None:
            self._pending_guess = (self._pending_guess[0], action)

        return action

    def _select_corner(self, valid_indices: np.ndarray) -> int:
        """Select random corner for first move (corners are statistically safer)."""
        corners = [
            self.position_to_action(0, 0),
            self.position_to_action(0, self.board_width - 1),
            self.position_to_action(self.board_height - 1, 0),
            self.position_to_action(self.board_height - 1, self.board_width - 1),
        ]
        # Randomize corner selection to avoid always hitting same unlucky corner
        random.shuffle(corners)
        for corner in corners:
            if corner in valid_indices:
                return corner
        return random.choice(valid_indices)

    def _neural_guess(
        self,
        observation: np.ndarray,
        valid_indices: np.ndarray,
        known_mines: Set[Tuple[int, int]],
        training: bool,
    ) -> int:
        """Use neural network to pick best guess."""
        # Small exploration during training
        if training and random.random() < 0.1:
            # Filter out known mines
            safe_indices = [
                idx for idx in valid_indices
                if self.action_to_position(idx) not in known_mines
            ]
            if safe_indices:
                return random.choice(safe_indices)
            return random.choice(valid_indices)

        # Get neural network predictions
        with torch.no_grad():
            state = self._preprocess(observation)
            safety_scores = self.guess_net(state).cpu().numpy()[0]

        # Mask invalid actions and known mines
        masked_scores = np.full_like(safety_scores, -np.inf)
        for idx in valid_indices:
            row, col = self.action_to_position(idx)
            if (row, col) not in known_mines:
                masked_scores[idx] = safety_scores[idx]

        # Pick highest safety score
        if np.all(np.isinf(masked_scores)):
            return valid_indices[0]

        return int(np.argmax(masked_scores))

    def _preprocess(self, observation: np.ndarray) -> torch.Tensor:
        """Convert observation to tensor."""
        obs_normalized = observation.astype(np.float32) / 9.0
        tensor = torch.from_numpy(obs_normalized).unsqueeze(0)
        return tensor.to(self.device)

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Process feedback from environment."""
        # Check if we had a pending guess
        if self._pending_guess is not None:
            state, guess_action = self._pending_guess

            if guess_action is not None:
                # Determine if guess was safe (didn't lose immediately)
                # If reward is very negative, we hit a mine
                was_safe = reward > -5.0

                if was_safe:
                    self.guesses_correct += 1

                # Store experience
                self.guess_buffer.push(state, guess_action, was_safe)

            self._pending_guess = None

        # Train if we have enough experiences
        if len(self.guess_buffer) >= self.batch_size:
            self._train_step()

    def _train_step(self) -> None:
        """Train the guess network on a mini-batch."""
        batch = self.guess_buffer.sample(self.batch_size)

        # Prepare tensors
        states = torch.stack([
            self._preprocess(e.state).squeeze(0) for e in batch
        ])
        actions = torch.tensor(
            [e.action for e in batch], dtype=torch.long, device=self.device
        )
        targets = torch.tensor(
            [1.0 if e.was_safe else 0.0 for e in batch],
            dtype=torch.float32, device=self.device
        )

        # Forward pass
        safety_scores = self.guess_net(states)
        predicted = safety_scores.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(predicted, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.guess_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())

    # ========================================================================
    # AC-3 Constraint Solving (from LogicAgent)
    # ========================================================================

    def _solve_constraints(
        self, observation: np.ndarray
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Use AC-3 constraint propagation to find definite safe/mine cells."""
        safe_cells: Set[Tuple[int, int]] = set()
        mine_cells: Set[Tuple[int, int]] = set()

        constraints = self._build_constraints(observation)

        changed = True
        iterations = 0

        while changed and iterations < 100:
            changed = False
            iterations += 1

            new_constraints = []
            for constraint in constraints:
                remaining_cells = constraint.cells - safe_cells - mine_cells
                remaining_mines = constraint.mine_count - len(constraint.cells & mine_cells)

                if not remaining_cells:
                    continue

                if remaining_mines == 0:
                    safe_cells.update(remaining_cells)
                    changed = True
                    continue

                if remaining_mines == len(remaining_cells):
                    mine_cells.update(remaining_cells)
                    changed = True
                    continue

                new_constraints.append(Constraint(
                    cells=frozenset(remaining_cells),
                    mine_count=remaining_mines
                ))

            constraints = new_constraints

            # Subset reduction
            subset_safe, subset_mines, constraints = self._subset_reduction(constraints)
            if subset_safe or subset_mines:
                safe_cells.update(subset_safe)
                mine_cells.update(subset_mines)
                changed = True

        return safe_cells, mine_cells

    def _build_constraints(self, observation: np.ndarray) -> List[Constraint]:
        """Build constraints from revealed cells."""
        constraints = []

        for row in range(self.board_height):
            for col in range(self.board_width):
                value = observation[row, col]

                if value < 1 or value > 8:
                    continue

                hidden_neighbors, flagged_count = self._get_neighbors(observation, row, col)

                if not hidden_neighbors:
                    continue

                remaining_mines = int(value) - flagged_count

                if remaining_mines < 0 or remaining_mines > len(hidden_neighbors):
                    continue

                constraints.append(Constraint(
                    cells=frozenset(hidden_neighbors),
                    mine_count=remaining_mines
                ))

        return constraints

    def _get_neighbors(
        self, observation: np.ndarray, row: int, col: int
    ) -> Tuple[Set[Tuple[int, int]], int]:
        """Get hidden neighbors and flagged count for a cell."""
        hidden: Set[Tuple[int, int]] = set()
        flagged = 0

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.board_height and 0 <= nc < self.board_width:
                    val = observation[nr, nc]
                    if val == -1:
                        hidden.add((nr, nc))
                    elif val == -2:
                        flagged += 1

        return hidden, flagged

    def _subset_reduction(
        self, constraints: List[Constraint]
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], List[Constraint]]:
        """Apply subset reduction for advanced deductions."""
        safe_cells: Set[Tuple[int, int]] = set()
        mine_cells: Set[Tuple[int, int]] = set()
        new_constraints: List[Constraint] = []

        for i, c1 in enumerate(constraints):
            for j, c2 in enumerate(constraints):
                if i >= j:
                    continue

                if c1.cells < c2.cells:
                    diff_cells = c2.cells - c1.cells
                    diff_mines = c2.mine_count - c1.mine_count

                    if diff_mines == 0:
                        safe_cells.update(diff_cells)
                    elif diff_mines == len(diff_cells):
                        mine_cells.update(diff_cells)
                    elif 0 < diff_mines < len(diff_cells):
                        new_constraints.append(Constraint(
                            cells=frozenset(diff_cells),
                            mine_count=diff_mines
                        ))

                elif c2.cells < c1.cells:
                    diff_cells = c1.cells - c2.cells
                    diff_mines = c1.mine_count - c2.mine_count

                    if diff_mines == 0:
                        safe_cells.update(diff_cells)
                    elif diff_mines == len(diff_cells):
                        mine_cells.update(diff_cells)
                    elif 0 < diff_mines < len(diff_cells):
                        new_constraints.append(Constraint(
                            cells=frozenset(diff_cells),
                            mine_count=diff_mines
                        ))

        seen = set()
        result = []
        for c in constraints + new_constraints:
            if c not in seen:
                seen.add(c)
                result.append(c)

        return safe_cells, mine_cells, result

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            "guess_net": self.guess_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "guesses_made": self.guesses_made,
            "guesses_correct": self.guesses_correct,
            "certain_moves": self.certain_moves,
        }, path)

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.guess_net.load_state_dict(checkpoint["guess_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.guesses_made = checkpoint.get("guesses_made", 0)
        self.guesses_correct = checkpoint.get("guesses_correct", 0)
        self.certain_moves = checkpoint.get("certain_moves", 0)

    def reset(self) -> None:
        """Reset for new game."""
        self._first_move = True
        self._pending_guess = None

    @property
    def guess_accuracy(self) -> float:
        """Return accuracy of guesses."""
        if self.guesses_made == 0:
            return 0.0
        return self.guesses_correct / self.guesses_made

    @property
    def device_name(self) -> str:
        return str(self.device)
