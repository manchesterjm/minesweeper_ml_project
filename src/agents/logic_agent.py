"""
Logic-based agent for Minesweeper.

Uses constraint satisfaction with AC-3 arc consistency
to make optimal deductions without guessing when possible.
"""
from typing import Optional, Set, Tuple, Dict, List, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, deque

import numpy as np

from .base_agent import BaseAgent


# ============================================================================
# Constraint Types
# ============================================================================

@dataclass
class Constraint:
    """
    A constraint representing: sum of cells in 'cells' == mine_count.

    For example, if a revealed "2" has 3 hidden neighbors and 0 flagged,
    the constraint is: cells={A, B, C}, mine_count=2
    """
    cells: FrozenSet[Tuple[int, int]]
    mine_count: int

    def __hash__(self) -> int:
        return hash((self.cells, self.mine_count))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constraint):
            return False
        return self.cells == other.cells and self.mine_count == other.mine_count


@dataclass
class CellInfo:
    """Information about a revealed cell for constraint analysis."""

    row: int
    col: int
    adjacent_mines: int
    hidden_neighbors: Set[Tuple[int, int]]
    flagged_neighbors: Set[Tuple[int, int]]

    @property
    def remaining_mines(self) -> int:
        """Mines still to be found among hidden neighbors."""
        return self.adjacent_mines - len(self.flagged_neighbors)


# ============================================================================
# Logic Agent with AC-3 Constraint Propagation
# ============================================================================

class LogicAgent(BaseAgent):
    """
    Agent that uses AC-3 arc consistency for constraint propagation.

    Strategy:
        1. Build constraints from all revealed numbered cells
        2. Use AC-3 to propagate constraints and find definite safe/mine cells
        3. Apply subset reduction for advanced deductions
        4. If no certain moves, calculate mine probabilities and pick safest
        5. Prefer corner and edge cells for first move (better cascades)

    The AC-3 algorithm iteratively enforces arc consistency:
    - For each constraint, check if any cell must be safe or mine
    - When a cell is determined, update all related constraints
    - Repeat until no more deductions can be made

    Expected win rate: ~40-50% on beginner with proper constraint propagation.
    """

    def __init__(
        self,
        board_height: int = 9,
        board_width: int = 9,
    ) -> None:
        """
        Initialize the logic agent.

        Args:
            board_height: Number of rows in the board.
            board_width: Number of columns in the board.
        """
        super().__init__(board_height, board_width)
        self._first_move = True

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Optional[np.ndarray] = None,
    ) -> int:
        """
        Select the best action using AC-3 constraint propagation.

        Args:
            observation: 2D array of cell states.
            valid_actions: Optional mask of valid actions.

        Returns:
            Best action index based on analysis.
        """
        if valid_actions is None:
            valid_actions = self.get_valid_actions_from_obs(observation)

        valid_indices = np.where(valid_actions)[0]

        if len(valid_indices) == 0:
            return 0

        # First move: prefer corners for better cascade potential
        if self._first_move:
            self._first_move = False
            return self._select_first_move(valid_indices)

        # Use AC-3 constraint propagation to find safe/mine cells
        safe_cells, mine_cells = self._solve_constraints(observation)

        # If we found definitely safe cells, reveal one
        if safe_cells:
            for row, col in safe_cells:
                action = self.position_to_action(row, col)
                if action in valid_indices:
                    return action

        # If no certain moves, use probability
        return self._select_by_probability(observation, valid_indices, mine_cells)

    def _select_first_move(self, valid_indices: np.ndarray) -> int:
        """Select random corner for first move (corners are statistically safer)."""
        import random
        corners = [
            self.position_to_action(0, 0),
            self.position_to_action(0, self.board_width - 1),
            self.position_to_action(self.board_height - 1, 0),
            self.position_to_action(self.board_height - 1, self.board_width - 1),
        ]
        # Randomize corner selection
        random.shuffle(corners)
        for corner in corners:
            if corner in valid_indices:
                return corner
        return random.choice(valid_indices)

    def _build_constraints(
        self, observation: np.ndarray
    ) -> List[Constraint]:
        """
        Build constraints from revealed numbered cells.

        Each revealed number N with hidden neighbors creates a constraint:
        "exactly (N - flagged_count) of these hidden cells are mines"
        """
        constraints = []

        for row in range(self.board_height):
            for col in range(self.board_width):
                value = observation[row, col]

                # Only process revealed numbered cells (1-8)
                if value < 1 or value > 8:
                    continue

                info = self._get_cell_info(observation, row, col)

                # Skip if no hidden neighbors (constraint is satisfied)
                if not info.hidden_neighbors:
                    continue

                # Skip invalid constraints (more mines needed than cells available)
                if info.remaining_mines < 0:
                    continue
                if info.remaining_mines > len(info.hidden_neighbors):
                    continue

                constraints.append(Constraint(
                    cells=frozenset(info.hidden_neighbors),
                    mine_count=info.remaining_mines
                ))

        return constraints

    def _solve_constraints(
        self, observation: np.ndarray
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Use AC-3 style constraint propagation to find definite safe/mine cells.

        Returns:
            Tuple of (safe_cells, mine_cells) sets.
        """
        safe_cells: Set[Tuple[int, int]] = set()
        mine_cells: Set[Tuple[int, int]] = set()

        # Build initial constraints
        constraints = self._build_constraints(observation)

        # Iteratively propagate until fixpoint
        changed = True
        iterations = 0
        max_iterations = 100  # Prevent infinite loops

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            # Apply basic constraint rules
            new_constraints = []
            for constraint in constraints:
                # Remove known cells from constraint
                remaining_cells = constraint.cells - safe_cells - mine_cells
                remaining_mines = constraint.mine_count - len(constraint.cells & mine_cells)

                # Skip empty constraints
                if not remaining_cells:
                    continue

                # All remaining cells are safe (no mines left to find)
                if remaining_mines == 0:
                    safe_cells.update(remaining_cells)
                    changed = True
                    continue

                # All remaining cells are mines
                if remaining_mines == len(remaining_cells):
                    mine_cells.update(remaining_cells)
                    changed = True
                    continue

                # Keep constraint for further processing
                new_constraints.append(Constraint(
                    cells=frozenset(remaining_cells),
                    mine_count=remaining_mines
                ))

            constraints = new_constraints

            # Apply subset reduction for more advanced deductions
            subset_safe, subset_mines, constraints = self._subset_reduction(constraints)
            if subset_safe or subset_mines:
                safe_cells.update(subset_safe)
                mine_cells.update(subset_mines)
                changed = True

        return safe_cells, mine_cells

    def _subset_reduction(
        self, constraints: List[Constraint]
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], List[Constraint]]:
        """
        Apply subset reduction to find additional deductions.

        If constraint A's cells are a subset of constraint B's cells:
        - The difference (B - A) has (B.mines - A.mines) mines
        - This may reveal new safe or mine cells

        Example:
            A: {X, Y} has 1 mine
            B: {X, Y, Z} has 1 mine
            → Z must be safe (B - A = {Z} has 0 mines)
        """
        safe_cells: Set[Tuple[int, int]] = set()
        mine_cells: Set[Tuple[int, int]] = set()
        new_constraints: List[Constraint] = []

        # Check each pair of constraints
        for i, c1 in enumerate(constraints):
            for j, c2 in enumerate(constraints):
                if i >= j:
                    continue

                # Check if c1 is subset of c2
                if c1.cells < c2.cells:
                    diff_cells = c2.cells - c1.cells
                    diff_mines = c2.mine_count - c1.mine_count

                    if diff_mines == 0:
                        # All cells in difference are safe
                        safe_cells.update(diff_cells)
                    elif diff_mines == len(diff_cells):
                        # All cells in difference are mines
                        mine_cells.update(diff_cells)
                    elif diff_mines > 0 and diff_mines < len(diff_cells):
                        # Create new reduced constraint
                        new_constraints.append(Constraint(
                            cells=frozenset(diff_cells),
                            mine_count=diff_mines
                        ))

                # Check if c2 is subset of c1
                elif c2.cells < c1.cells:
                    diff_cells = c1.cells - c2.cells
                    diff_mines = c1.mine_count - c2.mine_count

                    if diff_mines == 0:
                        safe_cells.update(diff_cells)
                    elif diff_mines == len(diff_cells):
                        mine_cells.update(diff_cells)
                    elif diff_mines > 0 and diff_mines < len(diff_cells):
                        new_constraints.append(Constraint(
                            cells=frozenset(diff_cells),
                            mine_count=diff_mines
                        ))

        # Deduplicate constraints
        seen = set()
        result_constraints = []
        for c in constraints + new_constraints:
            if c not in seen:
                seen.add(c)
                result_constraints.append(c)

        return safe_cells, mine_cells, result_constraints

    def _get_cell_info(
        self, observation: np.ndarray, row: int, col: int
    ) -> CellInfo:
        """Get analysis info for a revealed cell."""
        adjacent_mines = int(observation[row, col])
        hidden_neighbors: Set[Tuple[int, int]] = set()
        flagged_neighbors: Set[Tuple[int, int]] = set()

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.board_height and 0 <= nc < self.board_width:
                    val = observation[nr, nc]
                    if val == -1:  # Hidden
                        hidden_neighbors.add((nr, nc))
                    elif val == -2:  # Flagged
                        flagged_neighbors.add((nr, nc))

        return CellInfo(
            row=row,
            col=col,
            adjacent_mines=adjacent_mines,
            hidden_neighbors=hidden_neighbors,
            flagged_neighbors=flagged_neighbors,
        )

    def _select_by_probability(
        self,
        observation: np.ndarray,
        valid_indices: np.ndarray,
        known_mines: Set[Tuple[int, int]],
    ) -> int:
        """
        Select cell with lowest estimated mine probability.

        Uses constraint-based probability estimation.
        """
        probabilities = self._estimate_mine_probabilities(observation, known_mines)

        best_action = valid_indices[0]
        best_prob = 1.0

        for action in valid_indices:
            row, col = self.action_to_position(action)

            # Skip known mines
            if (row, col) in known_mines:
                continue

            prob = probabilities.get((row, col), 0.5)
            if prob < best_prob:
                best_prob = prob
                best_action = action

        return best_action

    def _estimate_mine_probabilities(
        self,
        observation: np.ndarray,
        known_mines: Set[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], float]:
        """
        Estimate mine probability for each hidden cell.

        Returns:
            Dict mapping (row, col) to probability of being a mine.
        """
        probabilities: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        for row in range(self.board_height):
            for col in range(self.board_width):
                value = observation[row, col]

                if value < 1 or value > 8:
                    continue

                info = self._get_cell_info(observation, row, col)

                if not info.hidden_neighbors:
                    continue

                # Adjust for known mines
                unknown_neighbors = info.hidden_neighbors - known_mines
                remaining = info.remaining_mines - len(info.hidden_neighbors & known_mines)

                if not unknown_neighbors or remaining < 0:
                    continue

                # Probability = remaining_mines / unknown_neighbors
                prob = remaining / len(unknown_neighbors)

                for neighbor in unknown_neighbors:
                    probabilities[neighbor].append(prob)

        # Combine probabilities from multiple constraints
        result: Dict[Tuple[int, int], float] = {}
        for cell, probs in probabilities.items():
            # Take maximum (most conservative estimate)
            result[cell] = max(probs)

        return result

    def reset(self) -> None:
        """Reset for new game."""
        self._first_move = True
