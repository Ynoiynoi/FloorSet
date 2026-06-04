#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - B*-tree with Soft Constraint Handling

Improvements over my_optimizer.py:
  - Soft constraint detection and penalty
  - Constraint-aware neighborhood operations
  - Better evaluation function
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Set

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)


# =============================================================================
# B*-TREE DATA STRUCTURE (copied from original implementation)
# =============================================================================

class BStarTree:
    """
    B*-tree for overlap-free floorplanning.

    Left child: placed to the RIGHT of parent
    Right child: placed ABOVE parent (same x)
    """

    def __init__(self, n_blocks: int, widths: List[float], heights: List[float]):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left = [-1] * n_blocks
        self.right = [-1] * n_blocks
        self.root = 0
        self._build_random_tree()

    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n

        order = list(range(self.n))
        random.shuffle(order)
        self.root = order[0]

        for i in range(1, self.n):
            block = order[i]
            existing = order[random.randint(0, i - 1)]
            if random.random() < 0.5:
                if self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                elif self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
            else:
                if self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                elif self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)

    def _insert_at_leaf(self, block: int, start: int):
        current = start
        while True:
            if random.random() < 0.5:
                if self.left[current] == -1:
                    self.left[current] = block
                    self.parent[block] = current
                    return
                current = self.left[current]
            else:
                if self.right[current] == -1:
                    self.right[current] = block
                    self.parent[block] = current
                    return
                current = self.right[current]

    def pack(self) -> List[Tuple[float, float, float, float]]:
        """Compute (x, y, w, h) from tree structure."""
        positions = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0:
            return positions

        contour = [(0.0, 0.0)]

        def get_contour_y(x_start: float, x_end: float) -> float:
            max_y = 0.0
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y

        def update_contour(x_start: float, x_end: float, y_top: float):
            nonlocal contour
            new_contour = []

            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0

                if cx_end <= x_start:
                    new_contour.append((cx_end, cy_top))
                elif cx_start >= x_end:
                    new_contour.append((cx_end, cy_top))
                else:
                    if cx_start < x_start:
                        new_contour.append((x_start, cy_top))
                    if cx_end > x_end:
                        new_contour.append((cx_end, cy_top))

            insert_pos = 0
            for i, (cx_end, _) in enumerate(new_contour):
                if cx_end <= x_start:
                    insert_pos = i + 1
            new_contour.insert(insert_pos, (x_end, y_top))

            new_contour.sort(key=lambda x: x[0])

            merged = []
            for x_end, y_top in new_contour:
                if merged and merged[-1][1] == y_top:
                    merged[-1] = (x_end, y_top)
                else:
                    merged.append((x_end, y_top))

            contour = merged if merged else [(x_end, 0.0)]

        def dfs(node: int, parent_right_edge: float):
            if node == -1:
                return

            w, h = self.widths[node], self.heights[node]

            if node == self.root:
                x = 0.0
                y = 0.0
            else:
                x = parent_right_edge
                y = get_contour_y(x, x + w)

            positions[node] = (x, y, w, h)
            update_contour(x, x + w, y + h)

            dfs(self.left[node], x + w)
            dfs(self.right[node], x)

        dfs(self.root, 0.0)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                x1, y1, w1, h1 = positions[i]
                x2, y2, w2, h2 = positions[j]
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                if overlap_x > 1e-6 and overlap_y > 1e-6:
                    positions[j] = (x2, max(y1 + h1, y2), w2, h2)

        return positions

    def copy(self) -> 'BStarTree':
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        return new

    def move_rotate(self, block: int):
        """Swap width/height (90° rotation, preserves area)."""
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]

    def move_swap(self, b1: int, b2: int):
        """Swap two blocks' dimensions."""
        self.widths[b1], self.widths[b2] = self.widths[b2], self.widths[b1]
        self.heights[b1], self.heights[b2] = self.heights[b2], self.heights[b1]

    def move_delete_insert(self, block: int):
        """Delete and reinsert block at random position."""
        if self.n <= 1:
            return
        w, h = self.widths[block], self.heights[block]
        self._delete_node(block)
        target = random.randint(0, self.n - 1)
        while target == block:
            target = random.randint(0, self.n - 1)
        self._insert_node(block, target, random.choice([True, False]))
        self.widths[block], self.heights[block] = w, h

    def _delete_node(self, node: int):
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]

        if left_child == -1 and right_child == -1:
            replacement = -1
        elif left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            rightmost = left_child
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = right_child
            self.parent[right_child] = rightmost

        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement

        if replacement != -1:
            self.parent[replacement] = parent

        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1

    def _insert_node(self, node: int, target: int, as_left: bool):
        if as_left:
            old_child = self.left[target]
            self.left[target] = node
        else:
            old_child = self.right[target]
            self.right[target] = node
        self.parent[node] = target
        if old_child != -1:
            self.left[node] = old_child
            self.parent[old_child] = node


class MyOptimizer(FloorplanOptimizer):
    """
    B*-tree SA with soft constraint handling.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.initial_temp = 100.0
        self.final_temp = 1.0
        self.cooling_rate = 0.9
        self.moves_per_temp = 20
        self.constraints = None
        self.target_positions = None
        self.area_targets = None

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
        """B*-tree SA with soft constraint optimization."""

        # Store problem data
        self.block_count = block_count
        self.area_targets = area_targets
        self.b2b_connectivity = b2b_connectivity
        self.p2b_connectivity = p2b_connectivity
        self.pins_pos = pins_pos
        self.constraints = constraints
        self.target_positions = target_positions

        # Parse constraints
        self._parse_constraints()

        # Initialize dimensions
        widths, heights = [], []
        for i in range(block_count):
            if (target_positions is not None and
                    target_positions[i, 2] != -1 and target_positions[i, 3] != -1):
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
            else:
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                w = h = math.sqrt(area)
            widths.append(w)
            heights.append(h)

        # Build B*-tree with smart initialization
        tree = self._smart_initialization(widths, heights)
        current_positions = tree.pack()
        current_positions = self._fix_constraints(current_positions)
        current_cost = self._evaluate(current_positions)

        best_tree = tree.copy()
        best_positions = current_positions
        best_cost = current_cost

        # Simulated Annealing
        temp = self.initial_temp
        iteration = 0
        while temp > self.final_temp:
            for _ in range(self.moves_per_temp):
                old_tree = tree.copy()

                # Choose move based on constraints
                move_type = self._choose_move()

                if move_type == 'rotate':
                    block = random.randint(0, block_count - 1)
                    if self._can_rotate(block):
                        tree.move_rotate(block)
                elif move_type == 'delete_insert':
                    block = random.randint(0, block_count - 1)
                    if not self._is_preplaced(block):
                        tree.move_delete_insert(block)
                elif move_type == 'swap':
                    # Swap two blocks (new move for better exploration)
                    b1, b2 = random.sample(range(block_count), 2)
                    if self._can_swap(b1, b2):
                        tree.move_swap(b1, b2)

                new_positions = tree.pack()
                new_positions = self._fix_constraints(new_positions)
                new_cost = self._evaluate(new_positions)

                # Accept/reject
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_positions = new_positions
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_positions = new_positions
                        best_tree = tree.copy()
                else:
                    tree = old_tree

                iteration += 1

            temp *= self.cooling_rate

        return best_positions

    def _parse_constraints(self):
        """Parse constraint tensor to identify groups."""
        # constraints: [n, 5] (fixed, preplaced, MIB, cluster/grouping, boundary)
        self.grouping_groups = {}
        self.mib_groups = {}
        self.boundary_blocks = {}

        if self.constraints is None:
            return

        # Parse grouping constraints (column 3)
        for i in range(self.block_count):
            group_id = int(self.constraints[i, 3])
            if group_id > 0:
                if group_id not in self.grouping_groups:
                    self.grouping_groups[group_id] = []
                self.grouping_groups[group_id].append(i)

        # Parse MIB constraints (column 2)
        for i in range(self.block_count):
            mib_id = int(self.constraints[i, 2])
            if mib_id > 0:
                if mib_id not in self.mib_groups:
                    self.mib_groups[mib_id] = []
                self.mib_groups[mib_id].append(i)

        # Parse boundary constraints (column 4)
        for i in range(self.block_count):
            boundary_type = int(self.constraints[i, 4])
            if boundary_type > 0:
                self.boundary_blocks[i] = boundary_type

    def _smart_initialization(self, widths: List[float], heights: List[float]) -> BStarTree:
        """
        Smart initialization considering connectivity.
        Place highly connected blocks close together.
        """
        tree = BStarTree(self.block_count, widths, heights)

        # If we have connectivity info, use it for initialization
        if self.b2b_connectivity is not None and len(self.b2b_connectivity) > 0:
            # Calculate total connectivity for each block
            connectivity_score = [0.0] * self.block_count
            for edge in self.b2b_connectivity:
                i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
                connectivity_score[i] += weight
                connectivity_score[j] += weight

            # Sort blocks by connectivity (high to low)
            sorted_blocks = sorted(range(self.block_count),
                                 key=lambda x: connectivity_score[x],
                                 reverse=True)

            # Build tree with sorted order
            tree.root = sorted_blocks[0]
            tree.parent = [-1] * self.block_count
            tree.left = [-1] * self.block_count
            tree.right = [-1] * self.block_count

            for i in range(1, min(len(sorted_blocks), self.block_count)):
                block = sorted_blocks[i]
                parent = sorted_blocks[random.randint(0, i - 1)]
                if random.random() < 0.5 and tree.left[parent] == -1:
                    tree.left[parent] = block
                    tree.parent[block] = parent
                elif tree.right[parent] == -1:
                    tree.right[parent] = block
                    tree.parent[block] = parent
                else:
                    # Find a leaf
                    current = parent
                    while tree.left[current] != -1 or tree.right[current] != -1:
                        if tree.left[current] != -1 and tree.right[current] != -1:
                            current = random.choice([tree.left[current], tree.right[current]])
                        elif tree.left[current] != -1:
                            current = tree.left[current]
                        else:
                            current = tree.right[current]

                    if random.random() < 0.5:
                        tree.left[current] = block
                    else:
                        tree.right[current] = block
                    tree.parent[block] = current

        return tree

    def _choose_move(self) -> str:
        """Choose move type based on current state."""
        # Weighted random choice
        moves = ['rotate', 'delete_insert', 'swap']
        weights = [0.3, 0.5, 0.2]  # Favor delete_insert for topology changes
        return random.choices(moves, weights=weights)[0]

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

    def _can_swap(self, b1: int, b2: int) -> bool:
        """Check if two blocks can be swapped."""
        # Don't swap if either is preplaced or fixed
        if self._is_preplaced(b1) or self._is_preplaced(b2):
            return False
        if self.constraints is not None:
            if self.constraints[b1, 0] == 1 or self.constraints[b2, 0] == 1:
                return False
        return True

    def _fix_constraints(self, positions: List[Tuple[float, float, float, float]]) \
            -> List[Tuple[float, float, float, float]]:
        """Fix positions to satisfy hard constraints."""
        fixed_positions = list(positions)

        # Fix preplaced blocks
        if self.target_positions is not None and self.constraints is not None:
            for i in range(self.block_count):
                if self.constraints[i, 1] == 1:
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

        # Enforce MIB constraints (same dimensions)
        for group_id, blocks in self.mib_groups.items():
            if len(blocks) > 1:
                # Use average dimensions
                avg_w = sum(fixed_positions[b][2] for b in blocks) / len(blocks)
                avg_h = sum(fixed_positions[b][3] for b in blocks) / len(blocks)

                for block in blocks:
                    if not self._is_preplaced(block):
                        x, y, _, _ = fixed_positions[block]
                        # Adjust to maintain area
                        target_area = float(self.area_targets[block])
                        scale = math.sqrt(target_area / (avg_w * avg_h))
                        fixed_positions[block] = (x, y, avg_w * scale, avg_h * scale)

        # Remove overlaps
        fixed_positions = self._remove_overlaps(fixed_positions)

        # Try to satisfy boundary constraints
        fixed_positions = self._adjust_for_boundary(fixed_positions)

        return fixed_positions

    def _adjust_for_boundary(self, positions: List[Tuple[float, float, float, float]]) \
            -> List[Tuple[float, float, float, float]]:
        """Adjust positions to satisfy boundary constraints."""
        if not self.boundary_blocks:
            return positions

        adjusted = list(positions)

        # Calculate current bounding box
        if len(positions) == 0:
            return adjusted

        min_x = min(p[0] for p in positions)
        max_x = max(p[0] + p[2] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] + p[3] for p in positions)

        W = max_x - min_x
        H = max_y - min_y

        for block, boundary_type in self.boundary_blocks.items():
            if self._is_preplaced(block):
                continue

            x, y, w, h = adjusted[block]

            # Boundary types: 1=left, 2=right, 4=top, 8=bottom
            # Corners: 5=top-left, 6=top-right, 9=bottom-left, 10=bottom-right

            if boundary_type == 1:  # Left
                adjusted[block] = (min_x, y, w, h)
            elif boundary_type == 2:  # Right
                adjusted[block] = (max_x - w, y, w, h)
            elif boundary_type == 4:  # Top
                adjusted[block] = (x, max_y - h, w, h)
            elif boundary_type == 8:  # Bottom
                adjusted[block] = (x, min_y, w, h)
            elif boundary_type == 5:  # Top-left
                adjusted[block] = (min_x, max_y - h, w, h)
            elif boundary_type == 6:  # Top-right
                adjusted[block] = (max_x - w, max_y - h, w, h)
            elif boundary_type == 9:  # Bottom-left
                adjusted[block] = (min_x, min_y, w, h)
            elif boundary_type == 10:  # Bottom-right
                adjusted[block] = (max_x - w, min_y, w, h)

        return adjusted

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
        """Evaluate solution with soft constraint penalties."""
        # Base cost: HPWL + area
        hpwl_b2b = calculate_hpwl_b2b(positions, self.b2b_connectivity)
        hpwl_p2b = calculate_hpwl_p2b(positions, self.p2b_connectivity, self.pins_pos)
        area = calculate_bbox_area(positions)

        base_cost = hpwl_b2b + hpwl_p2b + area * 0.01

        # Soft constraint penalties
        grouping_penalty = self._compute_grouping_penalty(positions)
        mib_penalty = self._compute_mib_penalty(positions)
        boundary_penalty = self._compute_boundary_penalty(positions)

        # Total penalty (weighted)
        total_penalty = grouping_penalty * 50 + mib_penalty * 30 + boundary_penalty * 20

        return base_cost + total_penalty

    def _compute_grouping_penalty(self, positions: List[Tuple[float, float, float, float]]) -> float:
        """Compute penalty for grouping constraint violations."""
        if not self.grouping_groups:
            return 0.0

        penalty = 0.0
        for group_id, blocks in self.grouping_groups.items():
            if len(blocks) <= 1:
                continue

            # Check if blocks are adjacent (share edges)
            # For simplicity, count disconnected components
            adjacent = set()
            for i, b1 in enumerate(blocks):
                for b2 in blocks[i+1:]:
                    if self._are_adjacent(positions[b1], positions[b2]):
                        adjacent.add((min(b1, b2), max(b1, b2)))

            # Penalty = number of blocks - 1 - number of adjacencies
            # Ideally, n blocks should have n-1 adjacencies (tree structure)
            expected_adjacencies = len(blocks) - 1
            actual_adjacencies = len(adjacent)
            penalty += max(0, expected_adjacencies - actual_adjacencies)

        return penalty

    def _are_adjacent(self, pos1: Tuple[float, float, float, float],
                     pos2: Tuple[float, float, float, float],
                     tolerance: float = 1e-3) -> bool:
        """Check if two blocks share an edge."""
        x1, y1, w1, h1 = pos1
        x2, y2, w2, h2 = pos2

        # Check if they share a vertical edge
        if abs(x1 + w1 - x2) < tolerance or abs(x2 + w2 - x1) < tolerance:
            # Check y overlap
            y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
            if y_overlap > tolerance:
                return True

        # Check if they share a horizontal edge
        if abs(y1 + h1 - y2) < tolerance or abs(y2 + h2 - y1) < tolerance:
            # Check x overlap
            x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
            if x_overlap > tolerance:
                return True

        return False

    def _compute_mib_penalty(self, positions: List[Tuple[float, float, float, float]]) -> float:
        """Compute penalty for MIB constraint violations."""
        if not self.mib_groups:
            return 0.0

        penalty = 0.0
        for group_id, blocks in self.mib_groups.items():
            if len(blocks) <= 1:
                continue

            # Check if all blocks have the same dimensions
            dimensions = [(positions[b][2], positions[b][3]) for b in blocks]
            unique_dims = set()
            for w, h in dimensions:
                # Round to avoid floating point issues
                unique_dims.add((round(w, 3), round(h, 3)))

            # Penalty = number of unique dimensions - 1
            penalty += len(unique_dims) - 1

        return penalty

    def _compute_boundary_penalty(self, positions: List[Tuple[float, float, float, float]]) -> float:
        """Compute penalty for boundary constraint violations."""
        if not self.boundary_blocks:
            return 0.0

        # Calculate bounding box
        if len(positions) == 0:
            return 0.0

        min_x = min(p[0] for p in positions)
        max_x = max(p[0] + p[2] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] + p[3] for p in positions)

        penalty = 0.0
        tolerance = 1e-3

        for block, boundary_type in self.boundary_blocks.items():
            x, y, w, h = positions[block]

            # Check if block touches required boundary
            touches_left = abs(x - min_x) < tolerance
            touches_right = abs(x + w - max_x) < tolerance
            touches_top = abs(y + h - max_y) < tolerance
            touches_bottom = abs(y - min_y) < tolerance

            satisfied = False
            if boundary_type == 1 and touches_left:
                satisfied = True
            elif boundary_type == 2 and touches_right:
                satisfied = True
            elif boundary_type == 4 and touches_top:
                satisfied = True
            elif boundary_type == 8 and touches_bottom:
                satisfied = True
            elif boundary_type == 5 and touches_left and touches_top:
                satisfied = True
            elif boundary_type == 6 and touches_right and touches_top:
                satisfied = True
            elif boundary_type == 9 and touches_left and touches_bottom:
                satisfied = True
            elif boundary_type == 10 and touches_right and touches_bottom:
                satisfied = True

            if not satisfied:
                penalty += 1.0

        return penalty
