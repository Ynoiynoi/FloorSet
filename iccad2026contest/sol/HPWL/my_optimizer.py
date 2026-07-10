#!/usr/bin/env python3
"""
HPWL-focused floorplanner for the simplified hard-constraint-only setting.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # When the evaluator runs as a script, its classes live in __main__.
    from __main__ import (  # type: ignore  # noqa: E402
        FloorplanOptimizer,
        calculate_bbox_area,
        calculate_hpwl_b2b,
        calculate_hpwl_p2b,
        check_overlap,
    )
except ImportError:  # Fallback for direct imports and local debugging.
    from iccad2026_evaluate import (  # noqa: E402
        FloorplanOptimizer,
        calculate_bbox_area,
        calculate_hpwl_b2b,
        calculate_hpwl_p2b,
        check_overlap,
    )

Position = Tuple[float, float, float, float]


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.lambda_bbox = 0.0045
        self.min_ratio = 0.40
        self.max_ratio = 2.50
        self.seed_limit = 7
        self.local_passes = 2
        self.max_starts = 3

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Position]:
        n = block_count
        if n == 0:
            return []

        cons = self._parse_constraints(constraints, n)
        b2b_adj, p2b_adj, total_weight, anchor_weight = self._build_graph(
            n, b2b_connectivity, p2b_connectivity, pins_pos, cons["preplaced"], target_positions
        )
        ideal_centers = self._solve_ideal_centers(
            n, b2b_connectivity, p2b_connectivity, pins_pos, cons["preplaced"], target_positions
        )
        ideal_centers = self._median_refine_centers(
            ideal_centers, b2b_adj, p2b_adj, cons["preplaced"], target_positions
        )

        shape_options = self._build_shape_options(
            n, area_targets, cons, target_positions, ideal_centers, b2b_adj, p2b_adj
        )
        orders = self._build_orders(n, total_weight, anchor_weight, area_targets, ideal_centers, cons)

        best_positions: Optional[List[Position]] = None
        best_score = float("inf")

        for order in orders[: self.max_starts]:
            positions = self._construct_placement(
                n,
                order,
                cons,
                target_positions,
                shape_options,
                ideal_centers,
                b2b_adj,
                p2b_adj,
                pins_pos,
            )
            positions = self._improve_positions(
                positions,
                shape_options,
                cons,
                ideal_centers,
                b2b_adj,
                p2b_adj,
                pins_pos,
                total_weight,
            )
            if check_overlap(positions) != 0:
                continue
            score = self._objective(positions, b2b_connectivity, p2b_connectivity, pins_pos)
            if score < best_score:
                best_score = score
                best_positions = positions

        if best_positions is None:
            best_positions = self._fallback_stack(
                n, cons, target_positions, shape_options
            )

        return best_positions

    def _parse_constraints(self, constraints: torch.Tensor, n: int) -> Dict[str, List[bool]]:
        cols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0

        def col(idx: int) -> List[bool]:
            if idx >= cols:
                return [False] * n
            return [bool(float(constraints[i, idx])) for i in range(n)]

        return {
            "fixed": col(0),
            "preplaced": col(1),
        }

    def _build_graph(
        self,
        n: int,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        preplaced: Sequence[bool],
        target_positions: Optional[torch.Tensor],
    ) -> Tuple[List[List[Tuple[int, float]]], List[List[Tuple[float, float, float]]], List[float], List[float]]:
        b2b_adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        p2b_adj: List[List[Tuple[float, float, float]]] = [[] for _ in range(n)]
        total_weight = [0.0] * n
        anchor_weight = [0.0] * n

        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if int(edge[0]) == -1:
                    continue
                i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
                if 0 <= i < n and 0 <= j < n and w > 0.0:
                    b2b_adj[i].append((j, w))
                    b2b_adj[j].append((i, w))
                    total_weight[i] += w
                    total_weight[j] += w
                    if preplaced[i]:
                        anchor_weight[j] += w
                    if preplaced[j]:
                        anchor_weight[i] += w

        if p2b_connectivity is not None:
            for edge in p2b_connectivity:
                if int(edge[0]) == -1:
                    continue
                pin_idx, block_idx, w = int(edge[0]), int(edge[1]), float(edge[2])
                if 0 <= block_idx < n and 0 <= pin_idx < len(pins_pos) and w > 0.0:
                    px = float(pins_pos[pin_idx][0])
                    py = float(pins_pos[pin_idx][1])
                    p2b_adj[block_idx].append((px, py, w))
                    total_weight[block_idx] += w
                    anchor_weight[block_idx] += w

        if target_positions is not None:
            for i in range(n):
                if preplaced[i]:
                    anchor_weight[i] += 10.0

        return b2b_adj, p2b_adj, total_weight, anchor_weight

    def _solve_ideal_centers(
        self,
        n: int,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        preplaced: Sequence[bool],
        target_positions: Optional[torch.Tensor],
    ) -> List[Tuple[float, float]]:
        centers = [(0.0, 0.0)] * n
        anchor_points: List[Tuple[float, float, float]] = []
        if pins_pos is not None:
            for pin in pins_pos:
                px = float(pin[0])
                if px == -1:
                    continue
                anchor_points.append((float(pin[0]), float(pin[1]), 1.0))
        if target_positions is not None:
            for i in range(n):
                if preplaced[i]:
                    tx = float(target_positions[i, 0] + target_positions[i, 2] * 0.5)
                    ty = float(target_positions[i, 1] + target_positions[i, 3] * 0.5)
                    anchor_points.append((tx, ty, 4.0))
                    centers[i] = (tx, ty)

        if anchor_points:
            total = sum(w for _, _, w in anchor_points)
            ref_x = sum(x * w for x, _, w in anchor_points) / total
            ref_y = sum(y * w for _, y, w in anchor_points) / total
        else:
            ref_x = 0.0
            ref_y = 0.0

        movable = [i for i in range(n) if not preplaced[i]]
        if not movable:
            return centers

        index = {block: idx for idx, block in enumerate(movable)}
        m = len(movable)
        ax = np.zeros((m, m), dtype=np.float64)
        bx = np.zeros(m, dtype=np.float64)
        by = np.zeros(m, dtype=np.float64)

        reg = 1e-3
        for i in movable:
            ii = index[i]
            ax[ii, ii] += reg
            bx[ii] += reg * ref_x
            by[ii] += reg * ref_y

        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if int(edge[0]) == -1:
                    continue
                i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
                if not (0 <= i < n and 0 <= j < n and w > 0.0):
                    continue
                i_movable = i in index
                j_movable = j in index
                if i_movable and j_movable:
                    ii = index[i]
                    jj = index[j]
                    ax[ii, ii] += w
                    ax[jj, jj] += w
                    ax[ii, jj] -= w
                    ax[jj, ii] -= w
                elif i_movable and preplaced[j] and target_positions is not None:
                    ii = index[i]
                    cx = float(target_positions[j, 0] + target_positions[j, 2] * 0.5)
                    cy = float(target_positions[j, 1] + target_positions[j, 3] * 0.5)
                    ax[ii, ii] += w
                    bx[ii] += w * cx
                    by[ii] += w * cy
                elif j_movable and preplaced[i] and target_positions is not None:
                    jj = index[j]
                    cx = float(target_positions[i, 0] + target_positions[i, 2] * 0.5)
                    cy = float(target_positions[i, 1] + target_positions[i, 3] * 0.5)
                    ax[jj, jj] += w
                    bx[jj] += w * cx
                    by[jj] += w * cy

        if p2b_connectivity is not None:
            for edge in p2b_connectivity:
                if int(edge[0]) == -1:
                    continue
                pin_idx, block_idx, w = int(edge[0]), int(edge[1]), float(edge[2])
                if block_idx not in index or pin_idx >= len(pins_pos) or w <= 0.0:
                    continue
                ii = index[block_idx]
                px = float(pins_pos[pin_idx][0])
                py = float(pins_pos[pin_idx][1])
                ax[ii, ii] += w
                bx[ii] += w * px
                by[ii] += w * py

        try:
            xs = np.linalg.solve(ax, bx)
            ys = np.linalg.solve(ax, by)
        except np.linalg.LinAlgError:
            ax += np.eye(m, dtype=np.float64) * 1e-2
            xs = np.linalg.solve(ax, bx)
            ys = np.linalg.solve(ax, by)

        for block, ii in index.items():
            centers[block] = (float(xs[ii]), float(ys[ii]))

        return centers

    def _median_refine_centers(
        self,
        centers: List[Tuple[float, float]],
        b2b_adj: Sequence[Sequence[Tuple[int, float]]],
        p2b_adj: Sequence[Sequence[Tuple[float, float, float]]],
        preplaced: Sequence[bool],
        target_positions: Optional[torch.Tensor],
    ) -> List[Tuple[float, float]]:
        refined = list(centers)
        movable = [i for i in range(len(refined)) if not preplaced[i]]
        for _ in range(3):
            for i in movable:
                xs: List[Tuple[float, float]] = []
                ys: List[Tuple[float, float]] = []
                for j, w in b2b_adj[i]:
                    cx, cy = refined[j]
                    xs.append((cx, w))
                    ys.append((cy, w))
                for px, py, w in p2b_adj[i]:
                    xs.append((px, w))
                    ys.append((py, w))
                if xs:
                    mx = self._weighted_median(xs)
                    my = self._weighted_median(ys)
                    old_x, old_y = refined[i]
                    refined[i] = (0.65 * mx + 0.35 * old_x, 0.65 * my + 0.35 * old_y)
        if target_positions is not None:
            for i in range(len(refined)):
                if preplaced[i]:
                    refined[i] = (
                        float(target_positions[i, 0] + target_positions[i, 2] * 0.5),
                        float(target_positions[i, 1] + target_positions[i, 3] * 0.5),
                    )
        return refined

    def _build_shape_options(
        self,
        n: int,
        area_targets: torch.Tensor,
        cons: Dict[str, List[bool]],
        target_positions: Optional[torch.Tensor],
        ideal_centers: Sequence[Tuple[float, float]],
        b2b_adj: Sequence[Sequence[Tuple[int, float]]],
        p2b_adj: Sequence[Sequence[Tuple[float, float, float]]],
    ) -> List[List[Tuple[float, float]]]:
        options: List[List[Tuple[float, float]]] = [[] for _ in range(n)]

        for i in range(n):
            if target_positions is not None and (cons["fixed"][i] or cons["preplaced"][i]):
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
                options[i] = [(w, h)]
                continue

            area = max(float(area_targets[i]), 1e-3)
            base = math.sqrt(area)
            square = (base, base)
            ratio = self._shape_ratio_from_anchors(i, area, ideal_centers, b2b_adj, p2b_adj)
            rw = math.sqrt(area * ratio)
            rh = area / rw
            candidates = [square, (rw, rh)]
            dedup: List[Tuple[float, float]] = []
            for w, h in candidates:
                key = (round(w, 4), round(h, 4))
                if not any(round(dw, 4) == key[0] and round(dh, 4) == key[1] for dw, dh in dedup):
                    dedup.append((w, h))
            options[i] = dedup

        return options

    def _shape_ratio_from_anchors(
        self,
        block: int,
        area: float,
        ideal_centers: Sequence[Tuple[float, float]],
        b2b_adj: Sequence[Sequence[Tuple[int, float]]],
        p2b_adj: Sequence[Sequence[Tuple[float, float, float]]],
    ) -> float:
        cx, cy = ideal_centers[block]
        h_pull = 1e-6
        v_pull = 1e-6
        for j, w in b2b_adj[block]:
            ox, oy = ideal_centers[j]
            h_pull += w * abs(cx - ox)
            v_pull += w * abs(cy - oy)
        for px, py, w in p2b_adj[block]:
            h_pull += w * abs(cx - px)
            v_pull += w * abs(cy - py)
        ratio = math.sqrt(h_pull / v_pull)
        return min(self.max_ratio, max(self.min_ratio, ratio))

    def _build_orders(
        self,
        n: int,
        total_weight: Sequence[float],
        anchor_weight: Sequence[float],
        area_targets: torch.Tensor,
        ideal_centers: Sequence[Tuple[float, float]],
        cons: Dict[str, List[bool]],
    ) -> List[List[int]]:
        movable = [i for i in range(n) if not cons["preplaced"][i]]

        def area_of(i: int) -> float:
            return math.sqrt(max(float(area_targets[i]), 1e-3))

        orders = [
            sorted(
                movable,
                key=lambda i: (-anchor_weight[i], -total_weight[i], -area_of(i), ideal_centers[i][1], ideal_centers[i][0]),
            ),
            sorted(
                movable,
                key=lambda i: (-total_weight[i], ideal_centers[i][1], ideal_centers[i][0]),
            ),
            sorted(
                movable,
                key=lambda i: (ideal_centers[i][1], ideal_centers[i][0], -total_weight[i]),
            ),
        ]
        return orders

    def _construct_placement(
        self,
        n: int,
        order: Sequence[int],
        cons: Dict[str, List[bool]],
        target_positions: Optional[torch.Tensor],
        shape_options: Sequence[Sequence[Tuple[float, float]]],
        ideal_centers: Sequence[Tuple[float, float]],
        b2b_adj: Sequence[Sequence[Tuple[int, float]]],
        p2b_adj: Sequence[Sequence[Tuple[float, float, float]]],
        pins_pos: torch.Tensor,
    ) -> List[Position]:
        positions: List[Optional[Position]] = [None] * n
        placed_mask = [False] * n

        for i in range(n):
            if cons["preplaced"][i]:
                pos = (
                    float(target_positions[i, 0]),
                    float(target_positions[i, 1]),
                    float(target_positions[i, 2]),
                    float(target_positions[i, 3]),
                )
                positions[i] = pos
                placed_mask[i] = True

        for block in order:
            if cons["preplaced"][block]:
                continue

            best_choice = None
            best_score = float("inf")
            target_center = ideal_centers[block]

            for w, h in shape_options[block]:
                occupied = [positions[j] for j in range(n) if positions[j] is not None and j != block]
                candidate = self._search_best_position(
                    block,
                    w,
                    h,
                    target_center,
                    positions,
                    ideal_centers,
                    occupied,
                    b2b_adj,
                    p2b_adj,
                )
                score = self._surrogate_cost(
                    block,
                    candidate,
                    positions,
                    ideal_centers,
                    occupied,
                    b2b_adj,
                    p2b_adj,
                )
                if score < best_score:
                    best_score = score
                    best_choice = candidate

            positions[block] = best_choice
            placed_mask[block] = True

        return [self._ensure_position(p) for p in positions]

    def _search_best_position(
        self,
        block: int,
        w: float,
        h: float,
        target_center: Tuple[float, float],
        positions: Sequence[Optional[Position]],
        ideal_centers: Sequence[Tuple[float, float]],
        occupied: Sequence[Optional[Position]],
        b2b_adj: Sequence[Sequence[Tuple[int, float]]],
        p2b_adj: Sequence[Sequence[Tuple[float, float, float]]],
    ) -> Position:
        desired_x = target_center[0] - 0.5 * w
        desired_y = target_center[1] - 0.5 * h

        x_values = [0.0, max(0.0, desired_x)]
        y_values = [0.0, max(0.0, desired_y)]
        bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = self._bbox_of(occupied)

        for rect in occupied:
            if rect is None:
                continue
            rx, ry, rw, rh = rect
            x_values.extend([rx, rx + rw, max(0.0, rx - w)])
            y_values.extend([ry, ry + rh, max(0.0, ry - h)])

        x_values.extend([bbox_min_x, bbox_max_x, max(0.0, bbox_max_x - w)])
        y_values.extend([bbox_min_y, bbox_max_y, max(0.0, bbox_max_y - h)])

        x_seeds = self._nearest_unique(x_values, max(0.0, desired_x), self.seed_limit)
        y_seeds = self._nearest_unique(y_values, max(0.0, desired_y), self.seed_limit)

        best: Optional[Position] = None
        best_score = float("inf")
        seen = set()

        for seed_x in x_seeds:
            for seed_y in y_seeds:
                x, y = self._legalize_from_seed(
                    seed_x,
                    seed_y,
                    w,
                    h,
                    occupied,
                    desired_x,
                    desired_y,
                    bbox_max_x,
                    bbox_max_y,
                )
                key = (round(x, 4), round(y, 4))
                if key in seen:
                    continue
                seen.add(key)
                candidate = (x, y, w, h)
                if self._overlaps_any(candidate, occupied):
                    continue
                score = self._surrogate_cost(
                    block,
                    candidate,
                    positions,
                    ideal_centers,
                    occupied,
                    b2b_adj,
                    p2b_adj,
                )
                if score < best_score:
                    best_score = score
                    best = candidate

        if best is not None:
            return best

        fallback_seeds = [
            (bbox_max_x, bbox_min_y),
            (bbox_min_x, bbox_max_y),
            (bbox_max_x, bbox_max_y),
            (0.0, bbox_max_y),
            (bbox_max_x, 0.0),
        ]
        for seed_x, seed_y in fallback_seeds:
            x, y = self._legalize_from_seed(
                seed_x,
                seed_y,
                w,
                h,
                occupied,
                desired_x,
                desired_y,
                bbox_max_x,
                bbox_max_y,
            )
            candidate = (x, y, w, h)
            if not self._overlaps_any(candidate, occupied):
                return candidate

        return (max(0.0, bbox_max_x), max(0.0, bbox_max_y), w, h)

    def _surrogate_cost(
        self,
        block: int,
        candidate: Position,
        positions: Sequence[Optional[Position]],
        ideal_centers: Sequence[Tuple[float, float]],
        occupied: Sequence[Optional[Position]],
        b2b_adj: Sequence[Sequence[Tuple[int, float]]],
        p2b_adj: Sequence[Sequence[Tuple[float, float, float]]],
    ) -> float:
        cx = candidate[0] + candidate[2] * 0.5
        cy = candidate[1] + candidate[3] * 0.5
        wire = 0.0
        for j, w in b2b_adj[block]:
            if positions[j] is not None:
                ocx = positions[j][0] + positions[j][2] * 0.5
                ocy = positions[j][1] + positions[j][3] * 0.5
            else:
                ocx, ocy = ideal_centers[j]
            wire += w * (abs(cx - ocx) + abs(cy - ocy))
        for px, py, w in p2b_adj[block]:
            wire += w * (abs(cx - px) + abs(cy - py))

        min_x, min_y, max_x, max_y = self._bbox_of(occupied)
        if occupied:
            min_x = min(min_x, candidate[0])
            min_y = min(min_y, candidate[1])
            max_x = max(max_x, candidate[0] + candidate[2])
            max_y = max(max_y, candidate[1] + candidate[3])
        else:
            min_x = candidate[0]
            min_y = candidate[1]
            max_x = candidate[0] + candidate[2]
            max_y = candidate[1] + candidate[3]

        bbox_area = (max_x - min_x) * (max_y - min_y)
        return wire + self.lambda_bbox * bbox_area

    def _improve_positions(
        self,
        positions: List[Position],
        shape_options: Sequence[Sequence[Tuple[float, float]]],
        cons: Dict[str, List[bool]],
        ideal_centers: Sequence[Tuple[float, float]],
        b2b_adj: Sequence[Sequence[Tuple[int, float]]],
        p2b_adj: Sequence[Sequence[Tuple[float, float, float]]],
        pins_pos: torch.Tensor,
        total_weight: Sequence[float],
    ) -> List[Position]:
        movable = [i for i in range(len(positions)) if not cons["preplaced"][i]]
        order = sorted(movable, key=lambda i: -total_weight[i])
        current = list(positions)

        for _ in range(self.local_passes):
            changed = False
            for block in order:
                target_center = self._weighted_target_center(block, current, ideal_centers, b2b_adj, p2b_adj)
                occupied = [current[j] for j in range(len(current)) if j != block]
                base_metric = self._incident_cost(block, current, b2b_adj, p2b_adj) + self.lambda_bbox * calculate_bbox_area(current)
                best_pos = current[block]
                best_metric = base_metric
                candidate_shapes = [current[block][2:4]] + list(shape_options[block])

                for dims in candidate_shapes:
                    w, h = dims
                    candidate = self._search_best_position(
                        block,
                        w,
                        h,
                        target_center,
                        current,
                        ideal_centers,
                        occupied,
                        b2b_adj,
                        p2b_adj,
                    )
                    trial = list(current)
                    trial[block] = candidate
                    if check_overlap(trial) != 0:
                        continue
                    metric = self._incident_cost(block, trial, b2b_adj, p2b_adj) + self.lambda_bbox * calculate_bbox_area(trial)
                    if metric + 1e-9 < best_metric:
                        best_metric = metric
                        best_pos = candidate

                if best_pos != current[block]:
                    current[block] = best_pos
                    changed = True
            if not changed:
                break

        return current

    def _weighted_target_center(
        self,
        block: int,
        positions: Sequence[Position],
        ideal_centers: Sequence[Tuple[float, float]],
        b2b_adj: Sequence[Sequence[Tuple[int, float]]],
        p2b_adj: Sequence[Sequence[Tuple[float, float, float]]],
    ) -> Tuple[float, float]:
        xs: List[Tuple[float, float]] = []
        ys: List[Tuple[float, float]] = []
        for j, w in b2b_adj[block]:
            ocx = positions[j][0] + positions[j][2] * 0.5
            ocy = positions[j][1] + positions[j][3] * 0.5
            xs.append((ocx, w))
            ys.append((ocy, w))
        for px, py, w in p2b_adj[block]:
            xs.append((px, w))
            ys.append((py, w))
        if not xs:
            return ideal_centers[block]
        return (self._weighted_median(xs), self._weighted_median(ys))

    def _incident_cost(
        self,
        block: int,
        positions: Sequence[Position],
        b2b_adj: Sequence[Sequence[Tuple[int, float]]],
        p2b_adj: Sequence[Sequence[Tuple[float, float, float]]],
    ) -> float:
        cx = positions[block][0] + positions[block][2] * 0.5
        cy = positions[block][1] + positions[block][3] * 0.5
        total = 0.0
        for j, w in b2b_adj[block]:
            ocx = positions[j][0] + positions[j][2] * 0.5
            ocy = positions[j][1] + positions[j][3] * 0.5
            total += w * (abs(cx - ocx) + abs(cy - ocy))
        for px, py, w in p2b_adj[block]:
            total += w * (abs(cx - px) + abs(cy - py))
        return total

    def _objective(
        self,
        positions: Sequence[Position],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> float:
        hpwl = calculate_hpwl_b2b(positions, b2b_connectivity) + calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)
        bbox = calculate_bbox_area(positions)
        return hpwl + self.lambda_bbox * bbox

    def _fallback_stack(
        self,
        n: int,
        cons: Dict[str, List[bool]],
        target_positions: Optional[torch.Tensor],
        shape_options: Sequence[Sequence[Tuple[float, float]]],
    ) -> List[Position]:
        positions: List[Optional[Position]] = [None] * n
        top = 0.0
        for i in range(n):
            if cons["preplaced"][i]:
                positions[i] = (
                    float(target_positions[i, 0]),
                    float(target_positions[i, 1]),
                    float(target_positions[i, 2]),
                    float(target_positions[i, 3]),
                )
        _, _, max_x, max_y = self._bbox_of(positions)
        top = max_y
        for i in range(n):
            if positions[i] is not None:
                continue
            w, h = shape_options[i][0]
            positions[i] = (0.0 if max_x == 0.0 else max_x, top, w, h)
            top += h
        return [self._ensure_position(p) for p in positions]

    def _legalize_from_seed(
        self,
        seed_x: float,
        seed_y: float,
        w: float,
        h: float,
        occupied: Sequence[Optional[Position]],
        desired_x: float,
        desired_y: float,
        bbox_max_x: float,
        bbox_max_y: float,
    ) -> Tuple[float, float]:
        x = max(0.0, seed_x)
        y = max(0.0, seed_y)
        max_iters = len(occupied) * 6 + 16

        for _ in range(max_iters):
            overlaps = []
            for rect in occupied:
                if rect is None:
                    continue
                if self._overlap_xywh(x, y, w, h, rect):
                    overlaps.append(rect)

            if not overlaps:
                return x, y

            right_x = max(rx + rw for rx, ry, rw, rh in overlaps)
            up_y = max(ry + rh for rx, ry, rw, rh in overlaps)

            right_score = abs(right_x - desired_x) + 0.5 * abs(y - desired_y) + 0.02 * max(0.0, right_x + w - bbox_max_x)
            up_score = 0.5 * abs(x - desired_x) + abs(up_y - desired_y) + 0.02 * max(0.0, up_y + h - bbox_max_y)

            if right_score <= up_score:
                x = max(x, right_x)
            else:
                y = max(y, up_y)

        return x, y

    def _nearest_unique(self, values: Sequence[float], target: float, limit: int) -> List[float]:
        uniq = []
        seen = set()
        for value in values:
            v = max(0.0, float(value))
            key = round(v, 4)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(v)
        uniq.sort(key=lambda v: (abs(v - target), v))
        return uniq[:limit]

    def _bbox_of(self, positions: Sequence[Optional[Position]]) -> Tuple[float, float, float, float]:
        active = [p for p in positions if p is not None]
        if not active:
            return 0.0, 0.0, 0.0, 0.0
        min_x = min(p[0] for p in active)
        min_y = min(p[1] for p in active)
        max_x = max(p[0] + p[2] for p in active)
        max_y = max(p[1] + p[3] for p in active)
        return min_x, min_y, max_x, max_y

    def _ensure_position(self, pos: Optional[Position]) -> Position:
        if pos is None:
            return (0.0, 0.0, 1.0, 1.0)
        return pos

    def _overlaps_any(self, candidate: Position, occupied: Sequence[Optional[Position]]) -> bool:
        for rect in occupied:
            if rect is not None and self._overlap_xywh(candidate[0], candidate[1], candidate[2], candidate[3], rect):
                return True
        return False

    def _overlap_xywh(self, x: float, y: float, w: float, h: float, rect: Position) -> bool:
        rx, ry, rw, rh = rect
        overlap_x = min(x + w, rx + rw) - max(x, rx)
        if overlap_x <= 1e-6:
            return False
        overlap_y = min(y + h, ry + rh) - max(y, ry)
        return overlap_y > 1e-6

    def _weighted_median(self, pairs: Sequence[Tuple[float, float]]) -> float:
        ordered = sorted((float(v), float(w)) for v, w in pairs if w > 0.0)
        if not ordered:
            return 0.0
        total_weight = sum(w for _, w in ordered)
        cutoff = 0.5 * total_weight
        prefix = 0.0
        for value, weight in ordered:
            prefix += weight
            if prefix >= cutoff:
                return value
        return ordered[-1][0]
