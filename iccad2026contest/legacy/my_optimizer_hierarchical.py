#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Hierarchical Packing Optimizer

Strategy: Grouping-first + Boundary inheritance
  1. Pack each Grouping group into a super-block (large rectangle)
  2. Inherit Boundary constraints to super-blocks
  3. Place super-blocks with Boundary awareness
  4. Unpack super-blocks to individual modules

Key advantages:
  - Grouping constraints automatically satisfied (modules in same group are adjacent)
  - Boundary constraints easier to satisfy (large rectangles easier to place on boundaries)
  - Simplified problem (fewer super-blocks than modules)

Trade-offs:
  - Fixed constraints may be violated (acceptable given low weight: 7.72%)
  - Need to handle MIB constraints across groups
"""

import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)


class SuperBlock:
    """Represents a group of modules packed into a rectangle."""

    def __init__(self, blocks: List[int]):
        self.blocks = blocks
        self.has_preplaced = False
        self.boundary_type = None
        self.width = 0.0
        self.height = 0.0
        self.offsets = {}  # block_id -> (offset_x, offset_y)
        self.position = (0.0, 0.0)  # (x, y) of super-block


class MyOptimizer(FloorplanOptimizer):
    """
    Hierarchical packing optimizer with Grouping-first strategy.
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
        Hierarchical packing optimization.
        """
        self.block_count = block_count
        self.area_targets = area_targets
        self.b2b_connectivity = b2b_connectivity
        self.p2b_connectivity = p2b_connectivity
        self.pins_pos = pins_pos
        self.constraints = constraints
        self.target_positions = target_positions

        if self.verbose:
            print(f"\n=== Hierarchical Packing: {block_count} blocks ===")

        # Step 1: Parse constraints and build super-blocks
        if self.verbose:
            print("Step 1: Building super-blocks...")
        super_blocks, mib_groups = self._build_super_blocks()

        if self.verbose:
            print(f"  Created {len(super_blocks)} super-blocks")
            print(f"  Found {len(mib_groups)} MIB groups")

        # Step 2: Initialize dimensions (considering MIB)
        if self.verbose:
            print("Step 2: Initializing dimensions...")
        widths, heights = self._initialize_dimensions(mib_groups)

        # Step 3: Pack each super-block
        if self.verbose:
            print("Step 3: Packing super-blocks...")
        self._pack_super_blocks(super_blocks, widths, heights, mib_groups)

        # Step 4: Place super-blocks
        if self.verbose:
            print("Step 4: Placing super-blocks...")
        self._place_super_blocks(super_blocks)

        # Step 5: Unpack to module positions
        if self.verbose:
            print("Step 5: Unpacking to module positions...")
        positions = self._unpack_super_blocks(super_blocks, widths, heights)

        # Step 6: Post-processing
        if self.verbose:
            print("Step 6: Post-processing...")
        positions = self._fix_preplaced(positions)
        positions = self._adjust_areas(positions)
        positions = self._remove_overlaps(positions)

        if self.verbose:
            cost = self._evaluate(positions)
            print(f"Final cost: {cost:.4f}")

        return positions

    def _build_super_blocks(self) -> Tuple[List[SuperBlock], Dict[int, List[int]]]:
        """
        Parse constraints and build super-blocks.

        Returns:
            (super_blocks, mib_groups)
        """
        # Parse constraints
        grouping_groups = defaultdict(list)
        mib_groups = defaultdict(list)
        boundary_blocks = {}
        preplaced_blocks = set()

        if self.constraints is not None:
            for i in range(self.block_count):
                # Grouping (column 3)
                group_id = int(self.constraints[i, 3])
                if group_id > 0:
                    grouping_groups[group_id].append(i)

                # MIB (column 2)
                mib_id = int(self.constraints[i, 2])
                if mib_id > 0:
                    mib_groups[mib_id].append(i)

                # Boundary (column 4)
                boundary_type = int(self.constraints[i, 4])
                if boundary_type > 0:
                    boundary_blocks[i] = boundary_type

                # Preplaced (column 1)
                if self.constraints[i, 1] == 1:
                    preplaced_blocks.add(i)

        # Build super-blocks from Grouping groups
        super_blocks = []
        grouped_blocks = set()

        for group_id, blocks in grouping_groups.items():
            sb = SuperBlock(blocks)

            # Check for preplaced modules
            sb.has_preplaced = any(b in preplaced_blocks for b in blocks)

            # Inherit boundary constraint (use first found)
            for b in blocks:
                if b in boundary_blocks:
                    sb.boundary_type = boundary_blocks[b]
                    break

            super_blocks.append(sb)
            grouped_blocks.update(blocks)

        # Add ungrouped modules as individual super-blocks
        for i in range(self.block_count):
            if i not in grouped_blocks:
                sb = SuperBlock([i])
                sb.has_preplaced = i in preplaced_blocks
                sb.boundary_type = boundary_blocks.get(i)
                super_blocks.append(sb)

        return super_blocks, mib_groups

    def _initialize_dimensions(self, mib_groups: Dict[int, List[int]]) -> Tuple[List[float], List[float]]:
        """Initialize module dimensions considering MIB constraints."""
        widths = []
        heights = []

        for i in range(self.block_count):
            # Check if preplaced or fixed
            if (self.target_positions is not None and
                    self.constraints is not None and
                    (self.constraints[i, 0] == 1 or self.constraints[i, 1] == 1)):
                w = float(self.target_positions[i, 2])
                h = float(self.target_positions[i, 3])
            else:
                # Start with square
                area = float(self.area_targets[i]) if self.area_targets[i] > 0 else 1.0
                w = h = math.sqrt(area)

            widths.append(w)
            heights.append(h)

        # Unify MIB group dimensions
        for mib_id, blocks in mib_groups.items():
            if len(blocks) <= 1:
                continue

            # Find reference dimensions
            ref_block = None
            for b in blocks:
                if self.constraints is not None and (self.constraints[b, 0] == 1 or self.constraints[b, 1] == 1):
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
                if self.constraints is None or self.constraints[b, 1] == 0:  # Not preplaced
                    target_area = float(self.area_targets[b])
                    scale = math.sqrt(target_area / (ref_w * ref_h)) if ref_w * ref_h > 0 else 1.0
                    widths[b] = ref_w * scale
                    heights[b] = ref_h * scale

        return widths, heights

    def _pack_super_blocks(self, super_blocks: List[SuperBlock],
                          widths: List[float], heights: List[float],
                          mib_groups: Dict[int, List[int]]):
        """Pack each super-block into a rectangle."""
        for sb in super_blocks:
            if len(sb.blocks) == 1:
                # Single module
                b = sb.blocks[0]
                sb.width = widths[b]
                sb.height = heights[b]
                sb.offsets[b] = (0.0, 0.0)
            else:
                # Multiple modules - pack horizontally
                total_width = sum(widths[b] for b in sb.blocks)
                max_height = max(heights[b] for b in sb.blocks)

                # Calculate offsets
                x_cursor = 0.0
                for b in sb.blocks:
                    sb.offsets[b] = (x_cursor, 0.0)
                    x_cursor += widths[b]

                sb.width = total_width
                sb.height = max_height

    def _place_super_blocks(self, super_blocks: List[SuperBlock]):
        """Place super-blocks with Boundary awareness."""
        # Estimate canvas size
        total_area = sum(sb.width * sb.height for sb in super_blocks)
        canvas_w = canvas_h = math.sqrt(total_area * 1.2)

        # Sort by priority: Preplaced > Boundary > Others
        sorted_blocks = sorted(
            super_blocks,
            key=lambda sb: (
                -sb.has_preplaced,
                -bool(sb.boundary_type),
                -len(sb.blocks)
            )
        )

        # Place super-blocks
        placed = []  # List of (x, y, w, h)

        for sb in sorted_blocks:
            if sb.has_preplaced:
                # Calculate anchor from preplaced module
                x, y = self._calculate_preplaced_anchor(sb)
            elif sb.boundary_type:
                # Place on boundary
                x, y = self._find_boundary_position(sb, canvas_w, canvas_h, placed)
            else:
                # Find free position
                x, y = self._find_free_position(sb.width, sb.height, placed)

            sb.position = (x, y)
            placed.append((x, y, sb.width, sb.height))

    def _calculate_preplaced_anchor(self, sb: SuperBlock) -> Tuple[float, float]:
        """Calculate super-block anchor from preplaced module."""
        if self.target_positions is None:
            return (0.0, 0.0)

        # Find first preplaced module
        for b in sb.blocks:
            if self.constraints is not None and self.constraints[b, 1] == 1:
                target_x = float(self.target_positions[b, 0])
                target_y = float(self.target_positions[b, 1])
                offset_x, offset_y = sb.offsets[b]
                return (target_x - offset_x, target_y - offset_y)

        return (0.0, 0.0)

    def _find_boundary_position(self, sb: SuperBlock, canvas_w: float, canvas_h: float,
                               placed: List[Tuple[float, float, float, float]]) -> Tuple[float, float]:
        """Find position on boundary for super-block."""
        boundary_type = sb.boundary_type
        w, h = sb.width, sb.height

        # Try different positions based on boundary type
        candidates = []

        if boundary_type == 1:  # Left
            candidates = [(0, y) for y in range(0, int(canvas_h), 10)]
        elif boundary_type == 2:  # Right
            candidates = [(canvas_w - w, y) for y in range(0, int(canvas_h), 10)]
        elif boundary_type == 4:  # Top
            candidates = [(x, canvas_h - h) for x in range(0, int(canvas_w), 10)]
        elif boundary_type == 8:  # Bottom
            candidates = [(x, 0) for x in range(0, int(canvas_w), 10)]
        elif boundary_type == 5:  # Top-left
            candidates = [(0, canvas_h - h)]
        elif boundary_type == 6:  # Top-right
            candidates = [(canvas_w - w, canvas_h - h)]
        elif boundary_type == 9:  # Bottom-left
            candidates = [(0, 0)]
        elif boundary_type == 10:  # Bottom-right
            candidates = [(canvas_w - w, 0)]

        # Find first non-overlapping position
        for x, y in candidates:
            if not self._has_overlap_with_placed(x, y, w, h, placed):
                return (x, y)

        # Fallback: find any free position
        return self._find_free_position(w, h, placed)

    def _find_free_position(self, w: float, h: float,
                           placed: List[Tuple[float, float, float, float]]) -> Tuple[float, float]:
        """Find a free position for a rectangle."""
        # Simple grid search
        for y in range(0, 1000, 10):
            for x in range(0, 1000, 10):
                if not self._has_overlap_with_placed(x, y, w, h, placed):
                    return (float(x), float(y))

        # Fallback: place at end
        if placed:
            last_x, last_y, last_w, last_h = placed[-1]
            return (last_x + last_w, last_y)
        return (0.0, 0.0)

    def _has_overlap_with_placed(self, x: float, y: float, w: float, h: float,
                                 placed: List[Tuple[float, float, float, float]]) -> bool:
        """Check if rectangle overlaps with any placed rectangles."""
        for px, py, pw, ph in placed:
            overlap_x = min(x + w, px + pw) - max(x, px)
            overlap_y = min(y + h, py + ph) - max(y, py)
            if overlap_x > 1e-6 and overlap_y > 1e-6:
                return True
        return False

    def _unpack_super_blocks(self, super_blocks: List[SuperBlock],
                            widths: List[float], heights: List[float]) -> List[Tuple[float, float, float, float]]:
        """Unpack super-blocks to module positions."""
        positions = [(0.0, 0.0, 0.0, 0.0)] * self.block_count

        for sb in super_blocks:
            sb_x, sb_y = sb.position

            for block in sb.blocks:
                offset_x, offset_y = sb.offsets[block]
                positions[block] = (
                    sb_x + offset_x,
                    sb_y + offset_y,
                    widths[block],
                    heights[block]
                )

        return positions

    def _fix_preplaced(self, positions: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """Fix preplaced module positions."""
        if self.target_positions is None or self.constraints is None:
            return positions

        fixed = list(positions)
        for i in range(self.block_count):
            if self.constraints[i, 1] == 1:  # Preplaced
                x = float(self.target_positions[i, 0])
                y = float(self.target_positions[i, 1])
                w = float(self.target_positions[i, 2])
                h = float(self.target_positions[i, 3])
                fixed[i] = (x, y, w, h)

        return fixed

    def _adjust_areas(self, positions: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """Adjust module dimensions to satisfy area constraints."""
        adjusted = list(positions)

        for i in range(self.block_count):
            # Skip preplaced and fixed
            if self.constraints is not None and (self.constraints[i, 0] == 1 or self.constraints[i, 1] == 1):
                continue

            x, y, w, h = adjusted[i]
            target_area = float(self.area_targets[i])
            current_area = w * h

            if abs(current_area - target_area) / target_area > 0.01:
                h = target_area / w if w > 0 else math.sqrt(target_area)
                adjusted[i] = (x, y, w, h)

        return adjusted

    def _remove_overlaps(self, positions: List[Tuple[float, float, float, float]],
                        max_iterations: int = 10) -> List[Tuple[float, float, float, float]]:
        """Remove overlaps iteratively."""
        fixed = list(positions)
        preplaced = set()

        if self.constraints is not None:
            for i in range(self.block_count):
                if self.constraints[i, 1] == 1:
                    preplaced.add(i)

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
                        if i in preplaced and j in preplaced:
                            continue
                        elif i in preplaced:
                            # Move j
                            if overlap_x < overlap_y:
                                fixed[j] = (x1 + w1, y2, w2, h2)
                            else:
                                fixed[j] = (x2, y1 + h1, w2, h2)
                        elif j in preplaced:
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

