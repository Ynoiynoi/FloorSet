#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Layered Optimization with Constraint Propagation

Algorithm: Multi-layer optimization approach
  Layer 1: Hard constraint satisfaction (preplaced, fixed-shape, area)
  Layer 2: Soft constraint optimization (grouping, MIB, boundary)
  Layer 3: Wirelength optimization (force-directed placement)
  Layer 4: Area compaction (longest path compression)

Key Features:
  - Constraint propagation to reduce search space
  - Hierarchical optimization from hard to soft constraints
  - Force-directed algorithm for wirelength minimization
  - Compaction for area reduction
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)


class MyOptimizer(FloorplanOptimizer):
    """
    Layered optimization with constraint propagation.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.verbose = verbose

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
        Layered optimization approach.
        """
        self.block_count = block_count
        self.area_targets = area_targets
        self.b2b_connectivity = b2b_connectivity
        self.p2b_connectivity = p2b_connectivity
        self.pins_pos = pins_pos
        self.constraints = constraints
        self.target_positions = target_positions

        if self.verbose:
            print(f"\n=== Layered Optimization: {block_count} blocks ===")

        # Parse constraints
        self._parse_constraints()

        # Layer 1: Initialize dimensions and satisfy hard constraints
        if self.verbose:
            print("Layer 1: Hard constraint satisfaction...")
        widths, heights = self._layer1_hard_constraints()

        # Layer 2: Initial placement with soft constraint awareness
        if self.verbose:
            print("Layer 2: Soft constraint optimization...")
        positions = self._layer2_soft_constraints(widths, heights)

        # Layer 3: Wirelength optimization
        if self.verbose:
            print("Layer 3: Wirelength optimization...")
        positions = self._layer3_wirelength_optimization(positions)

        # Layer 4: Area compaction
        if self.verbose:
            print("Layer 4: Area compaction...")
        positions = self._layer4_area_compaction(positions)

        # Final cleanup
        positions = self._fix_hard_constraints(positions, widths, heights)
        positions = self._remove_overlaps(positions)

        if self.verbose:
            cost = self._evaluate(positions)
            print(f"Final cost: {cost:.4f}")

        return positions

    def _parse_constraints(self):
        """Parse constraint tensor."""
        self.fixed_blocks = set()
        self.preplaced_blocks = set()
        self.grouping_groups = defaultdict(list)
        self.mib_groups = defaultdict(list)
        self.boundary_blocks = {}

        if self.constraints is None:
            return

        for i in range(self.block_count):
            # Fixed shape (column 0)
            if self.constraints[i, 0] == 1:
                self.fixed_blocks.add(i)

            # Preplaced (column 1)
            if self.constraints[i, 1] == 1:
                self.preplaced_blocks.add(i)

            # MIB (column 2)
            mib_id = int(self.constraints[i, 2])
            if mib_id > 0:
                self.mib_groups[mib_id].append(i)

            # Grouping (column 3)
            group_id = int(self.constraints[i, 3])
            if group_id > 0:
                self.grouping_groups[group_id].append(i)

            # Boundary (column 4)
            boundary_type = int(self.constraints[i, 4])
            if boundary_type > 0:
                self.boundary_blocks[i] = boundary_type

    def _layer1_hard_constraints(self) -> Tuple[List[float], List[float]]:
        """
        Layer 1: Determine dimensions satisfying hard constraints.
        - Preplaced: use target dimensions
        - Fixed-shape: use target dimensions
        - MIB: unify dimensions within groups
        - Others: start with square
        """
        widths = []
        heights = []

        for i in range(self.block_count):
            if i in self.preplaced_blocks or i in self.fixed_blocks:
                # Use target dimensions
                w = float(self.target_positions[i, 2])
                h = float(self.target_positions[i, 3])
            else:
                # Start with square
                area = float(self.area_targets[i]) if self.area_targets[i] > 0 else 1.0
                w = h = math.sqrt(area)

            widths.append(w)
            heights.append(h)

        # Unify MIB group dimensions
        for group_id, blocks in self.mib_groups.items():
            if len(blocks) <= 1:
                continue

            # Find reference dimensions (prefer fixed/preplaced)
            ref_block = None
            for b in blocks:
                if b in self.fixed_blocks or b in self.preplaced_blocks:
                    ref_block = b
                    break

            if ref_block is not None:
                ref_w, ref_h = widths[ref_block], heights[ref_block]
            else:
                # Use average area
                avg_area = sum(float(self.area_targets[b]) for b in blocks) / len(blocks)
                ref_w = ref_h = math.sqrt(avg_area)

            # Apply to all blocks in group
            for b in blocks:
                if b not in self.preplaced_blocks:
                    # Scale to maintain area
                    target_area = float(self.area_targets[b])
                    scale = math.sqrt(target_area / (ref_w * ref_h))
                    widths[b] = ref_w * scale
                    heights[b] = ref_h * scale

        return widths, heights

    def _layer2_soft_constraints(self, widths: List[float], heights: List[float]) \
            -> List[Tuple[float, float, float, float]]:
        """
        Layer 2: Initial placement considering soft constraints.
        - Group blocks in same cluster together
        - Place boundary blocks near boundaries
        - Use connectivity for initial positioning
        """
        positions = [(0.0, 0.0, widths[i], heights[i]) for i in range(self.block_count)]

        # Calculate connectivity-based preferred positions
        preferred_positions = self._calculate_preferred_positions(widths, heights)

        # Place preplaced blocks first
        for i in self.preplaced_blocks:
            x = float(self.target_positions[i, 0])
            y = float(self.target_positions[i, 1])
            positions[i] = (x, y, widths[i], heights[i])

        # Place grouping clusters
        placed = set(self.preplaced_blocks)
        for group_id, blocks in self.grouping_groups.items():
            if any(b in placed for b in blocks):
                continue

            # Place cluster as a connected group
            self._place_cluster(blocks, positions, widths, heights, preferred_positions)
            placed.update(blocks)

        # Place remaining blocks
        for i in range(self.block_count):
            if i in placed:
                continue

            # Use preferred position or grid placement
            if preferred_positions[i] is not None:
                x, y = preferred_positions[i]
            else:
                # Grid placement
                grid_size = math.ceil(math.sqrt(self.block_count))
                row = i // grid_size
                col = i % grid_size
                x = col * 100.0
                y = row * 100.0

            positions[i] = (x, y, widths[i], heights[i])

        return positions

    def _calculate_preferred_positions(self, widths: List[float], heights: List[float]) \
            -> List[Optional[Tuple[float, float]]]:
        """Calculate preferred positions based on connectivity."""
        preferred = [None] * self.block_count

        # Use pin positions as hints
        if self.p2b_connectivity is not None and len(self.p2b_connectivity) > 0:
            for edge in self.p2b_connectivity:
                pin_idx, block_idx, weight = int(edge[0]), int(edge[1]), float(edge[2])
                if 0 <= block_idx < self.block_count and 0 <= pin_idx < len(self.pins_pos):
                    px, py = float(self.pins_pos[pin_idx, 0]), float(self.pins_pos[pin_idx, 1])
                    if px >= 0 and py >= 0:
                        if preferred[block_idx] is None:
                            preferred[block_idx] = (px, py)
                        else:
                            # Average with existing
                            old_x, old_y = preferred[block_idx]
                            preferred[block_idx] = ((old_x + px) / 2, (old_y + py) / 2)

        return preferred

    def _place_cluster(self, blocks: List[int], positions: List[Tuple[float, float, float, float]],
                      widths: List[float], heights: List[float],
                      preferred_positions: List[Optional[Tuple[float, float]]]):
        """Place a cluster of blocks together."""
        if not blocks:
            return

        # Find anchor position (average of preferred positions)
        anchor_x, anchor_y = 0.0, 0.0
        count = 0
        for b in blocks:
            if preferred_positions[b] is not None:
                px, py = preferred_positions[b]
                anchor_x += px
                anchor_y += py
                count += 1

        if count > 0:
            anchor_x /= count
            anchor_y /= count
        else:
            anchor_x = anchor_y = 100.0

        # Place blocks horizontally
        x_cursor = anchor_x
        for b in blocks:
            positions[b] = (x_cursor, anchor_y, widths[b], heights[b])
            x_cursor += widths[b]

    def _layer3_wirelength_optimization(self, positions: List[Tuple[float, float, float, float]]) \
            -> List[Tuple[float, float, float, float]]:
        """
        Layer 3: Optimize wirelength using force-directed placement.
        """
        if self.b2b_connectivity is None or len(self.b2b_connectivity) == 0:
            return positions

        # Force-directed iterations
        max_iterations = 20
        step_size = 10.0
        damping = 0.9

        current_positions = list(positions)

        for iteration in range(max_iterations):
            forces = [(0.0, 0.0) for _ in range(self.block_count)]

            # Calculate attractive forces from connections
            for edge in self.b2b_connectivity:
                i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
                if i >= self.block_count or j >= self.block_count:
                    continue

                xi, yi, wi, hi = current_positions[i]
                xj, yj, wj, hj = current_positions[j]

                # Centers
                ci_x, ci_y = xi + wi / 2, yi + hi / 2
                cj_x, cj_y = xj + wj / 2, yj + hj / 2

                # Force direction
                dx = cj_x - ci_x
                dy = cj_y - ci_y
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6

                # Attractive force proportional to distance and weight
                force_mag = weight * dist * 0.01

                fx = force_mag * dx / dist
                fy = force_mag * dy / dist

                forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
                forces[j] = (forces[j][0] - fx, forces[j][1] - fy)

            # Apply forces (skip preplaced blocks)
            for i in range(self.block_count):
                if i in self.preplaced_blocks:
                    continue

                x, y, w, h = current_positions[i]
                fx, fy = forces[i]

                # Update position
                x += fx * step_size
                y += fy * step_size

                # Keep positive
                x = max(0, x)
                y = max(0, y)

                current_positions[i] = (x, y, w, h)

            step_size *= damping

        return current_positions

    def _layer4_area_compaction(self, positions: List[Tuple[float, float, float, float]]) \
            -> List[Tuple[float, float, float, float]]:
        """
        Layer 4: Compact layout to reduce bounding box area.
        """
        # Shift all blocks to origin
        if not positions:
            return positions

        min_x = min(p[0] for p in positions)
        min_y = min(p[1] for p in positions)

        compacted = []
        for x, y, w, h in positions:
            compacted.append((x - min_x, y - min_y, w, h))

        return compacted

    def _fix_hard_constraints(self, positions: List[Tuple[float, float, float, float]],
                             widths: List[float], heights: List[float]) \
            -> List[Tuple[float, float, float, float]]:
        """Enforce hard constraints."""
        fixed = list(positions)

        # Fix preplaced blocks
        for i in self.preplaced_blocks:
            x = float(self.target_positions[i, 0])
            y = float(self.target_positions[i, 1])
            w = float(self.target_positions[i, 2])
            h = float(self.target_positions[i, 3])
            fixed[i] = (x, y, w, h)

        # Fix fixed-shape blocks
        for i in self.fixed_blocks:
            if i not in self.preplaced_blocks:
                x, y, _, _ = fixed[i]
                w = float(self.target_positions[i, 2])
                h = float(self.target_positions[i, 3])
                fixed[i] = (x, y, w, h)

        # Adjust area for soft blocks
        for i in range(self.block_count):
            if i in self.fixed_blocks or i in self.preplaced_blocks:
                continue

            x, y, w, h = fixed[i]
            target_area = float(self.area_targets[i])
            current_area = w * h

            if abs(current_area - target_area) / target_area > 0.01:
                h = target_area / w if w > 0 else math.sqrt(target_area)
                fixed[i] = (x, y, w, h)

        return fixed

    def _remove_overlaps(self, positions: List[Tuple[float, float, float, float]],
                        max_iterations: int = 10) -> List[Tuple[float, float, float, float]]:
        """Remove overlaps iteratively."""
        fixed = list(positions)

        for _ in range(max_iterations):
            has_overlap = False

            for i in range(self.block_count):
                for j in range(i + 1, self.block_count):
                    x1, y1, w1, h1 = fixed[i]
                    x2, y2, w2, h2 = fixed[j]

                    overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                    overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)

                    if overlap_x > 1e-6 and overlap_y > 1e-6:
                        has_overlap = True

                        # Decide which to move
                        if i in self.preplaced_blocks and j in self.preplaced_blocks:
                            continue
                        elif i in self.preplaced_blocks:
                            # Move j
                            if overlap_x < overlap_y:
                                fixed[j] = (x1 + w1, y2, w2, h2)
                            else:
                                fixed[j] = (x2, y1 + h1, w2, h2)
                        elif j in self.preplaced_blocks:
                            # Move i
                            if overlap_x < overlap_y:
                                fixed[i] = (x2 + w2, y1, w1, h1)
                            else:
                                fixed[i] = (x1, y2 + h2, w1, h1)
                        else:
                            # Move j (higher index)
                            if overlap_x < overlap_y:
                                fixed[j] = (x1 + w1, y2, w2, h2)
                            else:
                                fixed[j] = (x2, y1 + h1, w2, h2)

            if not has_overlap:
                break

        return fixed

    def _evaluate(self, positions: List[Tuple[float, float, float, float]]) -> float:
        """Evaluate solution quality."""
        hpwl_b2b = calculate_hpwl_b2b(positions, self.b2b_connectivity)
        hpwl_p2b = calculate_hpwl_p2b(positions, self.p2b_connectivity, self.pins_pos)
        area = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.01
