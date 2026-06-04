#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Sequence Pair + Tabu Search

Algorithm: Sequence Pair representation with Tabu Search optimization
Key Features:
  - Sequence Pair: Two permutations (Γ+, Γ-) encode relative positions
  - Tabu Search: Avoid revisiting recent solutions
  - Constraint-aware: Prioritize moves that satisfy soft constraints
  - Fast decoding: O(n²) time to compute positions from sequence pair
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Set
from collections import deque

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)


# =============================================================================
# SEQUENCE PAIR DATA STRUCTURE
# =============================================================================

class SequencePair:
    """
    Sequence Pair representation for floorplanning.

    Two permutations (Γ+, Γ-) encode the relative positions of blocks:
    - If block i appears before block j in both Γ+ and Γ-, then i is left of j
    - If block i appears before block j in Γ+ but after in Γ-, then i is below j
    """

    def __init__(self, n_blocks: int, widths: List[float], heights: List[float]):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        # Initialize with random permutations
        self.gamma_plus = list(range(n_blocks))
        self.gamma_minus = list(range(n_blocks))
        random.shuffle(self.gamma_plus)
        random.shuffle(self.gamma_minus)

    def pack(self) -> List[Tuple[float, float, float, float]]:
        """
        Decode sequence pair to positions using longest path algorithm.
        Time complexity: O(n²)
        """
        n = self.n

        # Build position indices for fast lookup
        pos_plus = {block: i for i, block in enumerate(self.gamma_plus)}
        pos_minus = {block: i for i, block in enumerate(self.gamma_minus)}

        # Compute x-coordinates using horizontal constraint graph
        x_coords = [0.0] * n
        for i in range(n):
            block_i = self.gamma_plus[i]
            max_x = 0.0
            # Check all blocks that must be to the left of block_i
            for j in range(i):
                block_j = self.gamma_plus[j]
                # block_j is left of block_i if it appears before in both sequences
                if pos_minus[block_j] < pos_minus[block_i]:
                    max_x = max(max_x, x_coords[block_j] + self.widths[block_j])
            x_coords[block_i] = max_x

        # Compute y-coordinates using vertical constraint graph
        y_coords = [0.0] * n
        for i in range(n):
            block_i = self.gamma_plus[i]
            max_y = 0.0
            # Check all blocks that must be below block_i
            for j in range(i):
                block_j = self.gamma_plus[j]
                # block_j is below block_i if it appears before in Γ+ but after in Γ-
                if pos_minus[block_j] > pos_minus[block_i]:
                    max_y = max(max_y, y_coords[block_j] + self.heights[block_j])
            y_coords[block_i] = max_y

        # Build result
        positions = []
        for i in range(n):
            positions.append((x_coords[i], y_coords[i], self.widths[i], self.heights[i]))

        return positions

    def copy(self) -> 'SequencePair':
        """Create a deep copy."""
        new = SequencePair.__new__(SequencePair)
        new.n = self.n
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.gamma_plus = self.gamma_plus.copy()
        new.gamma_minus = self.gamma_minus.copy()
        return new

    def get_hash(self) -> Tuple:
        """Get a hashable representation for tabu list."""
        return (tuple(self.gamma_plus), tuple(self.gamma_minus))


# =============================================================================
# TABU SEARCH OPTIMIZER
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    Sequence Pair + Tabu Search optimizer with constraint awareness.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.max_iterations = 50  # Reduced for faster runtime
        self.tabu_tenure = 15  # How long a move stays in tabu list
        self.aspiration_factor = 0.95  # Accept tabu move if cost < best * factor
        self.diversification_interval = 20  # Restart if no improvement
        self.neighbor_sample_size = 10  # Sample size for neighborhood

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None
    ) -> List[Tuple[float, float, float, float]]:
        """
        Sequence Pair + Tabu Search optimization.
        """
        # Store problem data
        self.block_count = block_count
        self.area_targets = area_targets
        self.b2b_connectivity = b2b_connectivity
        self.p2b_connectivity = p2b_connectivity
        self.pins_pos = pins_pos
        self.constraints = constraints
        self.target_positions = target_positions

        # Initialize dimensions
        widths, heights = self._initialize_dimensions()

        # Create initial sequence pair
        sp = SequencePair(block_count, widths, heights)
        current_positions = sp.pack()
        current_positions = self._fix_constraints(current_positions)
        current_cost = self._evaluate(current_positions)

        best_sp = sp.copy()
        best_positions = current_positions
        best_cost = current_cost

        # Tabu list: stores recent moves
        tabu_list = deque(maxlen=self.tabu_tenure)

        # Tabu search
        no_improvement_count = 0
        for iteration in range(self.max_iterations):
            # Generate neighborhood
            neighbors = self._generate_neighbors(sp)

            # Find best non-tabu neighbor
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_neighbor_positions = None

            for neighbor_sp, move_type in neighbors:
                # Check if move is tabu
                neighbor_hash = neighbor_sp.get_hash()
                is_tabu = neighbor_hash in tabu_list

                # Evaluate neighbor
                neighbor_positions = neighbor_sp.pack()
                neighbor_positions = self._fix_constraints(neighbor_positions)
                neighbor_cost = self._evaluate(neighbor_positions)

                # Aspiration criterion: accept tabu move if it's better than best
                if is_tabu and neighbor_cost >= best_cost * self.aspiration_factor:
                    continue

                # Update best neighbor
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor_sp
                    best_neighbor_cost = neighbor_cost
                    best_neighbor_positions = neighbor_positions

            # Move to best neighbor
            if best_neighbor is not None:
                sp = best_neighbor
                current_positions = best_neighbor_positions
                current_cost = best_neighbor_cost

                # Add to tabu list
                tabu_list.append(sp.get_hash())

                # Update global best
                if current_cost < best_cost:
                    best_sp = sp.copy()
                    best_positions = current_positions
                    best_cost = current_cost
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

            # Diversification: restart if stuck
            if no_improvement_count >= self.diversification_interval:
                sp = self._diversify(best_sp)
                current_positions = sp.pack()
                current_positions = self._fix_constraints(current_positions)
                current_cost = self._evaluate(current_positions)
                no_improvement_count = 0
                tabu_list.clear()

        return best_positions

    def _initialize_dimensions(self) -> Tuple[List[float], List[float]]:
        """Initialize block dimensions based on constraints."""
        widths, heights = [], []
        for i in range(self.block_count):
            if (self.target_positions is not None and
                    self.target_positions[i, 2] != -1 and
                    self.target_positions[i, 3] != -1):
                # Fixed-shape or preplaced: use target dimensions
                w = float(self.target_positions[i, 2])
                h = float(self.target_positions[i, 3])
            else:
                # Soft block: start with square
                area = float(self.area_targets[i]) if self.area_targets[i] > 0 else 1.0
                w = h = math.sqrt(area)
            widths.append(w)
            heights.append(h)
        return widths, heights

    def _generate_neighbors(self, sp: SequencePair) -> List[Tuple[SequencePair, str]]:
        """
        Generate neighborhood by applying different move operators.
        Returns list of (neighbor, move_type) tuples.
        """
        neighbors = []
        n = self.block_count
        sample_size = min(self.neighbor_sample_size, n)

        # Move 1: Swap two adjacent blocks in Γ+
        for _ in range(sample_size // 4):
            idx = random.randint(0, n - 2)
            neighbor = sp.copy()
            neighbor.gamma_plus[idx], neighbor.gamma_plus[idx + 1] = \
                neighbor.gamma_plus[idx + 1], neighbor.gamma_plus[idx]
            neighbors.append((neighbor, 'swap_plus'))

        # Move 2: Swap two adjacent blocks in Γ-
        for _ in range(sample_size // 4):
            idx = random.randint(0, n - 2)
            neighbor = sp.copy()
            neighbor.gamma_minus[idx], neighbor.gamma_minus[idx + 1] = \
                neighbor.gamma_minus[idx + 1], neighbor.gamma_minus[idx]
            neighbors.append((neighbor, 'swap_minus'))

        # Move 3: Rotate a block (swap width/height)
        for _ in range(sample_size // 4):
            block = random.randint(0, n - 1)
            if self._can_rotate(block):
                neighbor = sp.copy()
                neighbor.widths[block], neighbor.heights[block] = \
                    neighbor.heights[block], neighbor.widths[block]
                neighbors.append((neighbor, 'rotate'))

        # Move 4: Move a block to a different position in both sequences
        for _ in range(sample_size // 4):
            block_idx = random.randint(0, n - 1)
            new_pos = random.randint(0, n - 1)
            if block_idx != new_pos:
                neighbor = sp.copy()
                # Move in Γ+
                block = neighbor.gamma_plus.pop(block_idx)
                neighbor.gamma_plus.insert(new_pos, block)
                neighbors.append((neighbor, 'move'))

        return neighbors

    def _diversify(self, sp: SequencePair) -> SequencePair:
        """Create a diversified solution by perturbing the current best."""
        new_sp = sp.copy()
        # Randomly shuffle a portion of the sequences
        n = self.block_count
        shuffle_size = max(3, n // 4)

        start = random.randint(0, n - shuffle_size)
        segment_plus = new_sp.gamma_plus[start:start + shuffle_size]
        segment_minus = new_sp.gamma_minus[start:start + shuffle_size]

        random.shuffle(segment_plus)
        random.shuffle(segment_minus)

        new_sp.gamma_plus[start:start + shuffle_size] = segment_plus
        new_sp.gamma_minus[start:start + shuffle_size] = segment_minus

        return new_sp

    def _can_rotate(self, block: int) -> bool:
        """Check if a block can be rotated."""
        if self.constraints is None:
            return True
        is_fixed = self.constraints[block, 0] == 1
        is_preplaced = self.constraints[block, 1] == 1
        return not (is_fixed or is_preplaced)

    def _is_preplaced(self, block: int) -> bool:
        """Check if a block is preplaced."""
        if self.constraints is None:
            return False
        return self.constraints[block, 1] == 1

    def _fix_constraints(self, positions: List[Tuple[float, float, float, float]]) \
            -> List[Tuple[float, float, float, float]]:
        """Fix positions to satisfy hard constraints."""
        fixed_positions = list(positions)

        # Fix preplaced blocks
        if self.target_positions is not None and self.constraints is not None:
            for i in range(self.block_count):
                if self.constraints[i, 1] == 1:  # Preplaced
                    x = float(self.target_positions[i, 0])
                    y = float(self.target_positions[i, 1])
                    w = float(self.target_positions[i, 2])
                    h = float(self.target_positions[i, 3])
                    fixed_positions[i] = (x, y, w, h)

        # Fix fixed-shape blocks
        if self.target_positions is not None and self.constraints is not None:
            for i in range(self.block_count):
                if self.constraints[i, 0] == 1 and self.constraints[i, 1] == 0:
                    x, y, _, _ = fixed_positions[i]
                    w = float(self.target_positions[i, 2])
                    h = float(self.target_positions[i, 3])
                    fixed_positions[i] = (x, y, w, h)

        # Adjust soft blocks for area constraint
        for i in range(self.block_count):
            if self.constraints is not None:
                if self.constraints[i, 0] == 1 or self.constraints[i, 1] == 1:
                    continue

            x, y, w, h = fixed_positions[i]
            target_area = float(self.area_targets[i])
            current_area = w * h

            if abs(current_area - target_area) / target_area > 0.01:
                h = target_area / w if w > 0 else math.sqrt(target_area)
                fixed_positions[i] = (x, y, w, h)

        # Remove overlaps
        fixed_positions = self._remove_overlaps(fixed_positions)

        return fixed_positions

    def _remove_overlaps(self, positions: List[Tuple[float, float, float, float]],
                         max_iterations: int = 10) -> List[Tuple[float, float, float, float]]:
        """Iteratively remove overlaps."""
        fixed_positions = list(positions)

        for _ in range(max_iterations):
            has_overlap = False

            for i in range(self.block_count):
                for j in range(i + 1, self.block_count):
                    x1, y1, w1, h1 = fixed_positions[i]
                    x2, y2, w2, h2 = fixed_positions[j]

                    overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                    overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)

                    if overlap_x > 1e-6 and overlap_y > 1e-6:
                        has_overlap = True

                        i_preplaced = self._is_preplaced(i)
                        j_preplaced = self._is_preplaced(j)

                        if i_preplaced and j_preplaced:
                            continue
                        elif i_preplaced:
                            if overlap_x < overlap_y:
                                fixed_positions[j] = (x1 + w1, y2, w2, h2)
                            else:
                                fixed_positions[j] = (x2, y1 + h1, w2, h2)
                        elif j_preplaced:
                            if overlap_x < overlap_y:
                                fixed_positions[i] = (x2 + w2, y1, w1, h1)
                            else:
                                fixed_positions[i] = (x1, y2 + h2, w1, h1)
                        else:
                            if overlap_x < overlap_y:
                                fixed_positions[j] = (x1 + w1, y2, w2, h2)
                            else:
                                fixed_positions[j] = (x2, y1 + h1, w2, h2)

            if not has_overlap:
                break

        return fixed_positions

    def _evaluate(self, positions: List[Tuple[float, float, float, float]]) -> float:
        """Evaluate solution quality."""
        hpwl_b2b = calculate_hpwl_b2b(positions, self.b2b_connectivity)
        hpwl_p2b = calculate_hpwl_p2b(positions, self.p2b_connectivity, self.pins_pos)
        area = calculate_bbox_area(positions)

        # Add soft constraint penalty
        penalty = self._compute_soft_constraint_penalty(positions)

        return hpwl_b2b + hpwl_p2b + area * 0.01 + penalty * 100

    def _compute_soft_constraint_penalty(self, positions: List[Tuple[float, float, float, float]]) -> float:
        """Compute penalty for soft constraint violations."""
        penalty = 0.0

        # TODO: Add grouping, MIB, boundary constraint penalties
        # For now, return 0 to focus on hard constraints

        return penalty
