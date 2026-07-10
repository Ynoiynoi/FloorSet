#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - fast deterministic optimizer.

This version is tuned for runtime first. It avoids simulated annealing and
repeated repacking, and instead performs one legal shelf-style placement pass:
  - fixed-shape and preplaced blocks keep their exact required geometry
  - clustered movable blocks are packed as abutting horizontal units
  - remaining blocks are packed in weighted order on shelves
  - when no preplaced anchors exist, the whole layout is shifted toward the
    weighted pin centroid to keep p2b HPWL reasonable
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import FloorplanOptimizer


Placement = Tuple[float, float, float, float]
Size = Tuple[float, float]


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[Placement]:
        if block_count <= 0:
            return []

        cluster_ids = [0] * block_count
        boundary_codes = [0] * block_count
        if constraints is not None and len(constraints) >= block_count:
            ncols = constraints.shape[1]
            if ncols > 3:
                cluster_ids = [int(constraints[i, 3].item()) for i in range(block_count)]
            if ncols > 4:
                boundary_codes = [int(constraints[i, 4].item()) for i in range(block_count)]

        sizes: List[Size] = []
        placements: List[Placement] = [(0.0, 0.0, 0.0, 0.0) for _ in range(block_count)]
        preplaced_indices: List[int] = []
        movable_indices: List[int] = []

        for i in range(block_count):
            width, height = self._block_size(i, area_targets, target_positions)
            sizes.append((width, height))

            if self._is_preplaced(i, constraints, target_positions):
                x = float(target_positions[i, 0].item())
                y = float(target_positions[i, 1].item())
                placements[i] = (x, y, width, height)
                preplaced_indices.append(i)
            else:
                movable_indices.append(i)

        if not movable_indices:
            return placements

        weighted_scores = self._block_scores(block_count, b2b_connectivity, p2b_connectivity)
        units = self._build_units(
            movable_indices, cluster_ids, boundary_codes, sizes, weighted_scores
        )
        start_x, start_y, target_width = self._packing_window(
            units, sizes, placements, preplaced_indices
        )

        cursor_x = start_x
        cursor_y = start_y
        row_height = 0.0

        for unit in units:
            unit_width = sum(sizes[idx][0] for idx in unit)
            unit_height = max(sizes[idx][1] for idx in unit)

            if cursor_x > start_x and cursor_x + unit_width > start_x + target_width:
                cursor_x = start_x
                cursor_y += row_height
                row_height = 0.0

            local_x = cursor_x
            for idx in unit:
                width, height = sizes[idx]
                placements[idx] = (local_x, cursor_y, width, height)
                local_x += width

            cursor_x = local_x
            row_height = max(row_height, unit_height)

        if not preplaced_indices:
            placements = self._shift_toward_pins(placements, p2b_connectivity, pins_pos)

        return placements

    def _block_size(
        self,
        block_idx: int,
        area_targets: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> Size:
        if target_positions is not None and block_idx < len(target_positions):
            target_w = float(target_positions[block_idx, 2].item())
            target_h = float(target_positions[block_idx, 3].item())
            if target_w != -1.0 and target_h != -1.0:
                return target_w, target_h

        area = float(area_targets[block_idx].item()) if block_idx < len(area_targets) else 1.0
        if area <= 0.0:
            area = 1.0
        side = math.sqrt(area)
        return side, side

    def _is_preplaced(
        self,
        block_idx: int,
        constraints: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> bool:
        if constraints is not None and len(constraints) > block_idx and constraints.shape[1] > 1:
            return bool(constraints[block_idx, 1].item() != 0)

        if target_positions is None or block_idx >= len(target_positions):
            return False

        target_x = float(target_positions[block_idx, 0].item())
        target_y = float(target_positions[block_idx, 1].item())
        return target_x != -1.0 and target_y != -1.0

    def _block_scores(
        self,
        block_count: int,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
    ) -> List[float]:
        scores = [0.0] * block_count

        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if int(edge[0].item()) == -1:
                    continue
                src = int(edge[0].item())
                dst = int(edge[1].item())
                weight = float(edge[2].item())
                if 0 <= src < block_count:
                    scores[src] += weight
                if 0 <= dst < block_count:
                    scores[dst] += weight

        if p2b_connectivity is not None:
            for edge in p2b_connectivity:
                if int(edge[0].item()) == -1:
                    continue
                block_idx = int(edge[1].item())
                weight = float(edge[2].item())
                if 0 <= block_idx < block_count:
                    scores[block_idx] += weight

        return scores

    def _build_units(
        self,
        movable_indices: Sequence[int],
        cluster_ids: Sequence[int],
        boundary_codes: Sequence[int],
        sizes: Sequence[Size],
        weighted_scores: Sequence[float],
    ) -> List[List[int]]:
        grouped: Dict[int, List[int]] = {}
        singles: List[int] = []

        for idx in movable_indices:
            cluster_id = cluster_ids[idx]
            if cluster_id > 0:
                grouped.setdefault(cluster_id, []).append(idx)
            else:
                singles.append(idx)

        units: List[List[int]] = []

        for members in grouped.values():
            members.sort(
                key=lambda idx: (
                    0 if boundary_codes[idx] != 0 else 1,
                    -weighted_scores[idx],
                    -(sizes[idx][0] * sizes[idx][1]),
                    idx,
                )
            )
            units.append(members)

        singles.sort(
            key=lambda idx: (
                0 if boundary_codes[idx] != 0 else 1,
                -weighted_scores[idx],
                -(sizes[idx][0] * sizes[idx][1]),
                idx,
            )
        )
        units.extend([[idx] for idx in singles])

        units.sort(
            key=lambda unit: (
                0 if any(boundary_codes[idx] != 0 for idx in unit) else 1,
                -sum(weighted_scores[idx] for idx in unit),
                -sum(sizes[idx][0] * sizes[idx][1] for idx in unit),
                min(unit),
            )
        )
        return units

    def _packing_window(
        self,
        units: Sequence[Sequence[int]],
        sizes: Sequence[Size],
        placements: Sequence[Placement],
        preplaced_indices: Sequence[int],
    ) -> Tuple[float, float, float]:
        total_area = sum(
            sizes[idx][0] * sizes[idx][1]
            for unit in units
            for idx in unit
        )
        max_unit_width = max(sum(sizes[idx][0] for idx in unit) for unit in units)
        target_width = max(max_unit_width, math.sqrt(max(total_area, 1.0)) * 1.15)

        if not preplaced_indices:
            return 0.0, 0.0, target_width

        min_x = min(placements[idx][0] for idx in preplaced_indices)
        min_y = min(placements[idx][1] for idx in preplaced_indices)
        max_x = max(placements[idx][0] + placements[idx][2] for idx in preplaced_indices)
        max_y = max(placements[idx][1] + placements[idx][3] for idx in preplaced_indices)

        anchor_width = max(max_x - min_x, 1.0)
        anchor_height = max(max_y - min_y, 1.0)
        est_height = total_area / max(target_width, 1e-6)

        area_if_right = (anchor_width + target_width) * max(anchor_height, est_height)
        area_if_top = max(anchor_width, target_width) * (anchor_height + est_height)

        if area_if_right <= area_if_top:
            return max_x, min_y, target_width

        return min_x, max_y, max(target_width, anchor_width)

    def _shift_toward_pins(
        self,
        placements: Sequence[Placement],
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> List[Placement]:
        if p2b_connectivity is None or pins_pos is None:
            return list(placements)

        total_weight = 0.0
        pin_x_sum = 0.0
        pin_y_sum = 0.0
        block_x_sum = 0.0
        block_y_sum = 0.0

        for edge in p2b_connectivity:
            if int(edge[0].item()) == -1:
                continue

            pin_idx = int(edge[0].item())
            block_idx = int(edge[1].item())
            weight = float(edge[2].item())
            if not (0 <= pin_idx < len(pins_pos) and 0 <= block_idx < len(placements)):
                continue

            pin_x = float(pins_pos[pin_idx, 0].item())
            pin_y = float(pins_pos[pin_idx, 1].item())
            block_x, block_y, width, height = placements[block_idx]
            center_x = block_x + width * 0.5
            center_y = block_y + height * 0.5

            total_weight += weight
            pin_x_sum += weight * pin_x
            pin_y_sum += weight * pin_y
            block_x_sum += weight * center_x
            block_y_sum += weight * center_y

        if total_weight <= 0.0:
            return list(placements)

        dx = pin_x_sum / total_weight - block_x_sum / total_weight
        dy = pin_y_sum / total_weight - block_y_sum / total_weight

        return [
            (x + dx, y + dy, width, height)
            for x, y, width, height in placements
        ]
