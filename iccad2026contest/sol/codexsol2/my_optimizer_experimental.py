#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge optimizer.

Traditional non-ML heuristic:
1. choose legal block shapes up front;
2. collapse each cluster group into a connected mini-floorplan;
3. build an item-level connectivity graph from b2b nets;
4. compute coarse barycentric targets from pins, nets, and boundary hints;
5. legalize with corner-based greedy placement plus a short local refinement;
6. run a boundary-focused post-pass and keep it when the proxy cost improves.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from iccad2026_evaluate import FloorplanOptimizer
from iccad2026_evaluate import calculate_bbox_area, calculate_hpwl_b2b, calculate_hpwl_p2b


EPS = 1e-7


@dataclass
class Item:
    blocks: List[int]
    offsets: Dict[int, Tuple[float, float]]
    width: float
    height: float
    fixed_anchor: Optional[Tuple[float, float]] = None
    boundary_code: int = 0
    preferred_center: Optional[Tuple[float, float]] = None
    pin_weight: float = 0.0
    is_cluster: bool = False


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
    ) -> List[Tuple[float, float, float, float]]:
        n = block_count
        area = [float(area_targets[i]) if float(area_targets[i]) > 0 else 1.0 for i in range(n)]
        cons = self._constraints_to_lists(constraints, n)

        widths, heights = self._choose_dimensions(n, area, cons, target_positions)
        pin_centers, pin_weights = self._pin_targets(n, p2b_connectivity, pins_pos)
        items, block_to_item = self._build_items(
            n,
            widths,
            heights,
            cons,
            target_positions,
            pin_centers,
            pin_weights,
        )
        outline_w, outline_h = self._estimate_outline(n, area, items, pins_pos, target_positions, cons)
        item_edges, item_degree = self._build_item_graph(len(items), block_to_item, b2b_connectivity)
        coarse_centers = self._relax_item_centers(items, item_edges, item_degree, outline_w, outline_h)

        anchors = self._place_items(items, outline_w, outline_h, item_edges, item_degree, coarse_centers)
        anchors = self._refine_anchors(items, anchors, outline_w, outline_h, item_edges, item_degree, coarse_centers)

        positions, item_boxes = self._materialize_positions(items, anchors, widths, heights, n)

        # Hard constraints always win over every heuristic.
        if target_positions is not None:
            for i in range(n):
                if cons["preplaced"][i]:
                    positions[i] = (
                        float(target_positions[i, 0]),
                        float(target_positions[i, 1]),
                        float(target_positions[i, 2]),
                        float(target_positions[i, 3]),
                    )
                elif cons["fixed"][i]:
                    x, y, _, _ = positions[i]
                    positions[i] = (
                        x,
                        y,
                        float(target_positions[i, 2]),
                        float(target_positions[i, 3]),
                    )

        if self._has_overlap(positions):
            positions = self._repair_overlaps(positions, cons)
            item_boxes = self._item_boxes_from_positions(items, positions)

        boundary_positions = self._postprocess_boundary_grouped(
            positions,
            cons,
            items,
            item_boxes,
            block_to_item,
        )
        if (
            not self._has_overlap(boundary_positions)
            and self._accept_boundary_candidate(
                positions,
                boundary_positions,
                cons,
                b2b_connectivity,
                p2b_connectivity,
                pins_pos,
            )
        ):
            return boundary_positions
        return positions

    def _constraints_to_lists(self, constraints: torch.Tensor, n: int) -> Dict[str, List[int]]:
        if constraints is None or constraints.numel() == 0:
            cols = 0
        else:
            cols = constraints.shape[1] if constraints.dim() > 1 else 0

        def col(idx: int) -> List[int]:
            if cols <= idx:
                return [0] * n
            return [int(float(constraints[i, idx])) for i in range(n)]

        return {
            "fixed": col(0),
            "preplaced": col(1),
            "mib": col(2),
            "cluster": col(3),
            "boundary": col(4),
        }

    def _choose_dimensions(
        self,
        n: int,
        area: List[float],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
    ) -> Tuple[List[float], List[float]]:
        widths = [math.sqrt(max(a, 1.0)) for a in area]
        heights = [math.sqrt(max(a, 1.0)) for a in area]

        if target_positions is not None:
            for i in range(n):
                if cons["fixed"][i] or cons["preplaced"][i]:
                    widths[i] = float(target_positions[i, 2])
                    heights[i] = float(target_positions[i, 3])

        for members in self._groups(cons["mib"]).values():
            fixed_ref = next((i for i in members if cons["fixed"][i] or cons["preplaced"][i]), None)
            if fixed_ref is not None:
                w, h = widths[fixed_ref], heights[fixed_ref]
            else:
                side = math.sqrt(max(area[members[0]], 1.0))
                w = h = side
            for i in members:
                if not (cons["fixed"][i] or cons["preplaced"][i]):
                    widths[i] = w
                    heights[i] = h

        return widths, heights

    def _pin_targets(
        self,
        n: int,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> Tuple[List[Optional[Tuple[float, float]]], List[float]]:
        sums = [[0.0, 0.0, 0.0] for _ in range(n)]
        if p2b_connectivity is not None:
            for edge in p2b_connectivity:
                if len(edge) < 3 or int(edge[0]) < 0:
                    continue
                pin_idx, block_idx, weight = int(edge[0]), int(edge[1]), float(edge[2])
                if 0 <= block_idx < n and 0 <= pin_idx < len(pins_pos) and weight > 0:
                    px = float(pins_pos[pin_idx, 0])
                    py = float(pins_pos[pin_idx, 1])
                    if px >= 0 and py >= 0:
                        sums[block_idx][0] += weight * px
                        sums[block_idx][1] += weight * py
                        sums[block_idx][2] += weight

        centers: List[Optional[Tuple[float, float]]] = []
        weights: List[float] = []
        for sx, sy, sw in sums:
            centers.append((sx / sw, sy / sw) if sw > 0 else None)
            weights.append(sw)
        return centers, weights

    def _build_items(
        self,
        n: int,
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
        pin_centers: List[Optional[Tuple[float, float]]],
        pin_weights: List[float],
    ) -> Tuple[List[Item], Dict[int, int]]:
        assigned = set()
        items: List[Item] = []
        block_to_item: Dict[int, int] = {}

        for _, members in sorted(self._groups(cons["cluster"]).items()):
            members = self._cluster_order(members, cons, target_positions, pin_centers)
            offsets, item_w, item_h, bcode = self._cluster_layout(members, widths, heights, cons, pin_centers)

            fixed_anchor = None
            if target_positions is not None:
                pre = next((b for b in members if cons["preplaced"][b]), None)
                if pre is not None:
                    ox, oy = offsets[pre]
                    fixed_anchor = (float(target_positions[pre, 0]) - ox, float(target_positions[pre, 1]) - oy)

            pref, weight = self._weighted_average_centers(
                [pin_centers[b] for b in members],
                [pin_weights[b] for b in members],
            )
            item_idx = len(items)
            items.append(Item(members, offsets, item_w, item_h, fixed_anchor, bcode, pref, weight, True))
            for b in members:
                assigned.add(b)
                block_to_item[b] = item_idx

        for b in range(n):
            if b in assigned:
                continue
            fixed_anchor = None
            if target_positions is not None and cons["preplaced"][b]:
                fixed_anchor = (float(target_positions[b, 0]), float(target_positions[b, 1]))
            item_idx = len(items)
            items.append(
                Item(
                    [b],
                    {b: (0.0, 0.0)},
                    widths[b],
                    heights[b],
                    fixed_anchor,
                    cons["boundary"][b],
                    pin_centers[b],
                    pin_weights[b],
                    False,
                )
            )
            block_to_item[b] = item_idx

        return items, block_to_item

    def _cluster_order(
        self,
        members: List[int],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
        pin_centers: List[Optional[Tuple[float, float]]],
    ) -> List[int]:
        def key(b: int) -> Tuple[int, float, float, float]:
            code = cons["boundary"][b]
            corner = 0 if code in (5, 6, 9, 10) else 1
            pre = 0 if cons["preplaced"][b] else 1
            if target_positions is not None and cons["preplaced"][b]:
                return (pre, corner, float(target_positions[b, 1]), float(target_positions[b, 0]))
            center = pin_centers[b]
            py = center[1] if center is not None else 0.0
            px = center[0] if center is not None else 0.0
            return (pre, corner, py, px)

        return sorted(members, key=key)

    def _cluster_layout(
        self,
        members: List[int],
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        pin_centers: List[Optional[Tuple[float, float]]],
    ) -> Tuple[Dict[int, Tuple[float, float]], float, float, int]:
        code = 0
        for b in members:
            code |= cons["boundary"][b]

        need_left = bool(code & 1)
        need_right = bool(code & 2)
        need_top = bool(code & 4)
        need_bottom = bool(code & 8)

        if (need_left or need_right) and (need_top or need_bottom):
            offsets = self._cluster_corner_layout(members, widths, heights, cons, pin_centers)
        elif need_left or need_right:
            offsets = self._cluster_vertical_layout(members, widths, heights, cons, pin_centers)
        elif need_top or need_bottom:
            offsets = self._cluster_horizontal_layout(members, widths, heights, cons, pin_centers)
        else:
            offsets = self._cluster_row_layout(members, widths, heights, pin_centers)

        min_x = min(offset[0] for offset in offsets.values())
        min_y = min(offset[1] for offset in offsets.values())
        if abs(min_x) > EPS or abs(min_y) > EPS:
            offsets = {b: (x - min_x, y - min_y) for b, (x, y) in offsets.items()}

        width = max(offsets[b][0] + widths[b] for b in members)
        height = max(offsets[b][1] + heights[b] for b in members)

        if need_right:
            offsets = {b: (width - widths[b] - x, y) for b, (x, y) in offsets.items()}
        if need_top:
            offsets = {b: (x, height - heights[b] - y) for b, (x, y) in offsets.items()}

        width = max(offsets[b][0] + widths[b] for b in members)
        height = max(offsets[b][1] + heights[b] for b in members)
        return offsets, width, height, code

    def _cluster_row_layout(
        self,
        members: List[int],
        widths: List[float],
        heights: List[float],
        pin_centers: List[Optional[Tuple[float, float]]],
    ) -> Dict[int, Tuple[float, float]]:
        ordered = self._sort_members_by_axis(members, pin_centers, axis=0)
        offsets: Dict[int, Tuple[float, float]] = {}
        x_cursor = 0.0
        for b in ordered:
            offsets[b] = (x_cursor, 0.0)
            x_cursor += widths[b]
        return offsets

    def _cluster_vertical_layout(
        self,
        members: List[int],
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        pin_centers: List[Optional[Tuple[float, float]]],
    ) -> Dict[int, Tuple[float, float]]:
        boundary_members = [b for b in members if cons["boundary"][b] != 0]
        free_members = [b for b in members if cons["boundary"][b] == 0]
        ordered_boundary = self._sort_members_by_axis(boundary_members, pin_centers, axis=1)
        offsets: Dict[int, Tuple[float, float]] = {}
        y_cursor = 0.0
        left_width = max((widths[b] for b in ordered_boundary), default=0.0)
        for b in ordered_boundary:
            offsets[b] = (0.0, y_cursor)
            y_cursor += heights[b]

        x_cursor = left_width
        for b in self._sort_members_by_axis(free_members, pin_centers, axis=0):
            offsets[b] = (x_cursor, 0.0)
            x_cursor += widths[b]
        return offsets

    def _cluster_horizontal_layout(
        self,
        members: List[int],
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        pin_centers: List[Optional[Tuple[float, float]]],
    ) -> Dict[int, Tuple[float, float]]:
        boundary_members = [b for b in members if cons["boundary"][b] != 0]
        free_members = [b for b in members if cons["boundary"][b] == 0]
        ordered_boundary = self._sort_members_by_axis(boundary_members, pin_centers, axis=0)
        offsets: Dict[int, Tuple[float, float]] = {}
        x_cursor = 0.0
        bottom_height = max((heights[b] for b in ordered_boundary), default=0.0)
        for b in ordered_boundary:
            offsets[b] = (x_cursor, 0.0)
            x_cursor += widths[b]

        y_cursor = bottom_height
        for b in self._sort_members_by_axis(free_members, pin_centers, axis=1):
            offsets[b] = (0.0, y_cursor)
            y_cursor += heights[b]
        return offsets

    def _cluster_corner_layout(
        self,
        members: List[int],
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        pin_centers: List[Optional[Tuple[float, float]]],
    ) -> Dict[int, Tuple[float, float]]:
        codes = {b: cons["boundary"][b] for b in members}
        corner_members = [b for b in members if codes[b] in (5, 6, 9, 10)]
        pivot = corner_members[0] if corner_members else max(members, key=lambda b: widths[b] * heights[b])

        vertical_members = [
            b for b in members
            if b != pivot and codes[b] != 0 and (codes[b] & (1 | 2)) and not (codes[b] & (4 | 8))
        ]
        horizontal_members = [
            b for b in members
            if b != pivot and codes[b] != 0 and (codes[b] & (4 | 8)) and not (codes[b] & (1 | 2))
        ]
        free_members = [b for b in members if b not in vertical_members and b not in horizontal_members and b != pivot]

        offsets: Dict[int, Tuple[float, float]] = {pivot: (0.0, 0.0)}
        y_cursor = heights[pivot]
        for b in self._sort_members_by_axis(vertical_members, pin_centers, axis=1):
            offsets[b] = (0.0, y_cursor)
            y_cursor += heights[b]

        x_cursor = widths[pivot]
        for b in self._sort_members_by_axis(horizontal_members, pin_centers, axis=0):
            offsets[b] = (x_cursor, 0.0)
            x_cursor += widths[b]

        free_x = x_cursor
        for b in self._sort_members_by_axis(free_members, pin_centers, axis=0):
            offsets[b] = (free_x, 0.0)
            free_x += widths[b]
        return offsets

    def _sort_members_by_axis(
        self,
        members: Iterable[int],
        pin_centers: List[Optional[Tuple[float, float]]],
        axis: int,
    ) -> List[int]:
        return sorted(
            members,
            key=lambda b: (
                pin_centers[b][axis] if pin_centers[b] is not None else 0.0,
                pin_centers[b][1 - axis] if pin_centers[b] is not None else 0.0,
                b,
            ),
        )

    def _weighted_average_centers(
        self,
        centers: List[Optional[Tuple[float, float]]],
        weights: List[float],
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        sx = 0.0
        sy = 0.0
        sw = 0.0
        for center, weight in zip(centers, weights):
            if center is None or weight <= 0:
                continue
            sx += center[0] * weight
            sy += center[1] * weight
            sw += weight
        if sw <= 0:
            return None, 0.0
        return (sx / sw, sy / sw), sw

    def _estimate_outline(
        self,
        n: int,
        area: List[float],
        items: List[Item],
        pins_pos: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        cons: Dict[str, List[int]],
    ) -> Tuple[float, float]:
        total_area = sum(area)
        width = height = 0.0
        if pins_pos is not None and pins_pos.numel() > 0:
            valid = pins_pos[(pins_pos[:, 0] >= 0) & (pins_pos[:, 1] >= 0)]
            if len(valid) > 0:
                width = float(valid[:, 0].max()) / 1.05
                height = float(valid[:, 1].max()) / 1.05

        if width <= 0 or height <= 0:
            side = math.sqrt(max(total_area, 1.0))
            width = side
            height = side

        if target_positions is not None:
            for i in range(n):
                if cons["preplaced"][i]:
                    width = max(width, float(target_positions[i, 0] + target_positions[i, 2]))
                    height = max(height, float(target_positions[i, 1] + target_positions[i, 3]))

        left_stack = sum(item.height for item in items if item.boundary_code & 1)
        right_stack = sum(item.height for item in items if item.boundary_code & 2)
        top_span = sum(item.width for item in items if item.boundary_code & 4)
        bottom_span = sum(item.width for item in items if item.boundary_code & 8)

        width = max(width, max((item.width for item in items), default=1.0), top_span, bottom_span, 1.0)
        height = max(height, max((item.height for item in items), default=1.0), left_stack, right_stack, 1.0)

        required = max(total_area * 1.12, sum(item.width * item.height for item in items) * 1.04)
        if width * height < required:
            scale = math.sqrt(required / max(width * height, 1.0))
            width *= scale
            height *= scale
        return width, height

    def _build_item_graph(
        self,
        item_count: int,
        block_to_item: Dict[int, int],
        b2b_connectivity: torch.Tensor,
    ) -> Tuple[List[Dict[int, float]], List[float]]:
        edges = [dict() for _ in range(item_count)]
        degree = [0.0] * item_count
        if b2b_connectivity is None:
            return edges, degree

        for edge in b2b_connectivity:
            if len(edge) < 3 or int(edge[0]) < 0:
                continue
            a = int(edge[0])
            b = int(edge[1])
            weight = float(edge[2])
            if weight <= 0:
                continue
            ia = block_to_item.get(a)
            ib = block_to_item.get(b)
            if ia is None or ib is None or ia == ib:
                continue
            edges[ia][ib] = edges[ia].get(ib, 0.0) + weight
            edges[ib][ia] = edges[ib].get(ia, 0.0) + weight
            degree[ia] += weight
            degree[ib] += weight

        return edges, degree

    def _relax_item_centers(
        self,
        items: List[Item],
        item_edges: List[Dict[int, float]],
        item_degree: List[float],
        outline_w: float,
        outline_h: float,
    ) -> List[Tuple[float, float]]:
        centers: List[Tuple[float, float]] = []
        for item in items:
            if item.fixed_anchor is not None:
                ax, ay = item.fixed_anchor
                center = (ax + item.width / 2.0, ay + item.height / 2.0)
            elif item.preferred_center is not None:
                center = item.preferred_center
            else:
                center = (outline_w / 2.0, outline_h / 2.0)
            centers.append(self._project_boundary_center(item, center, outline_w, outline_h))

        for _ in range(8):
            new_centers = list(centers)
            for i, item in enumerate(items):
                if item.fixed_anchor is not None:
                    continue
                sx = 0.0
                sy = 0.0
                sw = 0.0

                if item.preferred_center is not None:
                    weight = max(3.0, item.pin_weight)
                    sx += weight * item.preferred_center[0]
                    sy += weight * item.preferred_center[1]
                    sw += weight

                for j, weight in item_edges[i].items():
                    cx, cy = centers[j]
                    sx += weight * cx
                    sy += weight * cy
                    sw += weight

                bx, by, bweight = self._boundary_target(item, item_degree[i], outline_w, outline_h)
                if bweight > 0:
                    sx += bweight * bx
                    sy += bweight * by
                    sw += bweight

                sx += 0.35 * centers[i][0]
                sy += 0.35 * centers[i][1]
                sw += 0.35

                center = (sx / sw, sy / sw) if sw > 0 else centers[i]
                new_centers[i] = self._project_boundary_center(item, center, outline_w, outline_h)
            centers = new_centers

        return centers

    def _boundary_target(
        self,
        item: Item,
        degree: float,
        outline_w: float,
        outline_h: float,
    ) -> Tuple[float, float, float]:
        if item.boundary_code == 0:
            return 0.0, 0.0, 0.0

        cx = item.width / 2.0
        cy = item.height / 2.0
        if item.boundary_code & 2:
            cx = outline_w - item.width / 2.0
        elif not (item.boundary_code & 1):
            cx = outline_w / 2.0

        if item.boundary_code & 4:
            cy = outline_h - item.height / 2.0
        elif not (item.boundary_code & 8):
            cy = outline_h / 2.0

        weight = 6.0 + 0.18 * degree + 0.025 * item.width * item.height
        return cx, cy, weight

    def _project_boundary_center(
        self,
        item: Item,
        center: Tuple[float, float],
        outline_w: float,
        outline_h: float,
    ) -> Tuple[float, float]:
        cx, cy = center
        min_x = item.width / 2.0
        max_x = max(min_x, outline_w - item.width / 2.0)
        min_y = item.height / 2.0
        max_y = max(min_y, outline_h - item.height / 2.0)

        if item.boundary_code & 1:
            cx = min_x
        elif item.boundary_code & 2:
            cx = max_x
        else:
            cx = min(max(cx, min_x), max_x)

        if item.boundary_code & 8:
            cy = min_y
        elif item.boundary_code & 4:
            cy = max_y
        else:
            cy = min(max(cy, min_y), max_y)
        return cx, cy

    def _place_items(
        self,
        items: List[Item],
        outline_w: float,
        outline_h: float,
        item_edges: List[Dict[int, float]],
        item_degree: List[float],
        coarse_centers: List[Tuple[float, float]],
    ) -> Dict[int, Tuple[float, float]]:
        placed: List[Tuple[float, float, float, float, int]] = []
        placed_map: Dict[int, Tuple[float, float, float, float, int]] = {}
        anchors: Dict[int, Tuple[float, float]] = {}

        fixed_order = [i for i, item in enumerate(items) if item.fixed_anchor is not None]
        free_order = [i for i, item in enumerate(items) if item.fixed_anchor is None]
        free_order.sort(
            key=lambda i: self._priority(items[i], item_degree[i], coarse_centers[i]),
            reverse=True,
        )

        for idx in fixed_order:
            item = items[idx]
            ax, ay = item.fixed_anchor or (0.0, 0.0)
            anchors[idx] = (ax, ay)
            rect = (ax, ay, item.width, item.height, idx)
            placed.append(rect)
            placed_map[idx] = rect

        for idx in free_order:
            item = items[idx]
            ax, ay = self._find_position(
                idx,
                item,
                placed,
                placed_map,
                outline_w,
                outline_h,
                item_edges,
                coarse_centers,
            )
            anchors[idx] = (ax, ay)
            rect = (ax, ay, item.width, item.height, idx)
            placed.append(rect)
            placed_map[idx] = rect

        return anchors

    def _priority(
        self,
        item: Item,
        degree: float,
        coarse_center: Tuple[float, float],
    ) -> float:
        boundary = 8_000.0 if item.boundary_code else 0.0
        pin_bias = 40.0 * item.pin_weight
        shape = item.width * item.height
        center_pull = abs(coarse_center[0]) + abs(coarse_center[1])
        return boundary + 18.0 * degree + pin_bias + shape + 12.0 * len(item.blocks) - 0.01 * center_pull

    def _find_position(
        self,
        idx: int,
        item: Item,
        placed: List[Tuple[float, float, float, float, int]],
        placed_map: Dict[int, Tuple[float, float, float, float, int]],
        outline_w: float,
        outline_h: float,
        item_edges: List[Dict[int, float]],
        coarse_centers: List[Tuple[float, float]],
        current_anchor: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        desired_center = self._desired_center(idx, item, placed_map, item_edges, coarse_centers, outline_w, outline_h)
        xs = {0.0, max(0.0, desired_center[0] - item.width / 2.0)}
        ys = {0.0, max(0.0, desired_center[1] - item.height / 2.0)}

        if current_anchor is not None:
            xs.add(current_anchor[0])
            ys.add(current_anchor[1])

        for x, y, w, h, _ in placed:
            xs.add(x + w)
            xs.add(max(0.0, x - item.width))
            xs.add(max(0.0, x + 0.5 * (w - item.width)))
            ys.add(y + h)
            ys.add(max(0.0, y - item.height))
            ys.add(max(0.0, y + 0.5 * (h - item.height)))

        for nbr in item_edges[idx]:
            if nbr not in placed_map:
                continue
            x, y, w, h, _ = placed_map[nbr]
            xs.add(x + w)
            xs.add(max(0.0, x - item.width))
            xs.add(max(0.0, x + 0.5 * (w - item.width)))
            ys.add(y + h)
            ys.add(max(0.0, y - item.height))
            ys.add(max(0.0, y + 0.5 * (h - item.height)))

        if item.boundary_code & 1:
            xs = {0.0}
        elif item.boundary_code & 2:
            xs = {max(0.0, outline_w - item.width)}
        else:
            xs.add(max(0.0, outline_w - item.width))

        if item.boundary_code & 8:
            ys = {0.0}
        elif item.boundary_code & 4:
            ys = {max(0.0, outline_h - item.height)}
        else:
            ys.add(max(0.0, outline_h - item.height))

        x_values = sorted(xs)
        y_values = sorted(ys)
        if len(x_values) > 96:
            x_values = self._trim_axis_candidates(x_values, desired_center[0] - item.width / 2.0)
        if len(y_values) > 96:
            y_values = self._trim_axis_candidates(y_values, desired_center[1] - item.height / 2.0)

        candidates: List[Tuple[float, float]] = []
        if item.boundary_code & (1 | 2) and item.boundary_code & (4 | 8):
            candidates.append((next(iter(xs)), next(iter(ys))))
        elif item.boundary_code & (4 | 8):
            fixed_y = next(iter(ys))
            candidates.extend((x, fixed_y) for x in x_values)
        elif item.boundary_code & (1 | 2):
            fixed_x = next(iter(xs))
            candidates.extend((fixed_x, y) for y in y_values)
        else:
            candidates.extend((x, self._lowest_nonoverlap_y(x, item.width, item.height, placed)) for x in x_values)

        if item.boundary_code == 0 and len(placed) < 96:
            candidates.extend((x, y) for x in x_values[:28] for y in y_values[:28])

        seen = set()
        unique_candidates = []
        for x, y in candidates:
            key = (round(x, 6), round(y, 6))
            if key not in seen:
                seen.add(key)
                unique_candidates.append((x, y))

        best = None
        best_score = float("inf")
        for x, y in unique_candidates:
            rect = (x, y, item.width, item.height)
            if self._rect_overlaps_any(rect, placed):
                continue
            score = self._placement_score(
                idx,
                item,
                rect,
                placed,
                placed_map,
                desired_center,
                item_edges,
                outline_w,
                outline_h,
            )
            if score < best_score:
                best_score = score
                best = (x, y)

        if best is not None:
            return best

        y = 0.0 if not placed else max(py + ph for _, py, _, ph, _ in placed)
        x = 0.0
        if item.boundary_code & 2:
            x = max(0.0, outline_w - item.width)
        if item.boundary_code & 4:
            y = max(y, outline_h - item.height)
        return x, y

    def _desired_center(
        self,
        idx: int,
        item: Item,
        placed_map: Dict[int, Tuple[float, float, float, float, int]],
        item_edges: List[Dict[int, float]],
        coarse_centers: List[Tuple[float, float]],
        outline_w: float,
        outline_h: float,
    ) -> Tuple[float, float]:
        sx = 0.0
        sy = 0.0
        sw = 0.0

        if item.preferred_center is not None:
            weight = max(2.0, item.pin_weight)
            sx += weight * item.preferred_center[0]
            sy += weight * item.preferred_center[1]
            sw += weight

        for nbr, weight in item_edges[idx].items():
            if nbr not in placed_map:
                continue
            x, y, w, h, _ = placed_map[nbr]
            sx += weight * (x + w / 2.0)
            sy += weight * (y + h / 2.0)
            sw += weight

        cx, cy = coarse_centers[idx]
        sx += 0.8 * cx
        sy += 0.8 * cy
        sw += 0.8

        center = (sx / sw, sy / sw) if sw > 0 else (cx, cy)
        return self._project_boundary_center(item, center, outline_w, outline_h)

    def _trim_axis_candidates(self, values: List[float], target: float) -> List[float]:
        keep = set(values[:18])
        keep.update(values[-12:])
        ranked = sorted(values, key=lambda v: abs(v - target))
        keep.update(ranked[:56])
        return sorted(keep)

    def _lowest_nonoverlap_y(
        self,
        x: float,
        width: float,
        height: float,
        placed: List[Tuple[float, float, float, float, int]],
    ) -> float:
        y = 0.0
        for _ in range(len(placed) + 1):
            next_y = y
            for px, py, pw, ph, _ in placed:
                if min(x + width, px + pw) - max(x, px) > 1e-6:
                    if min(y + height, py + ph) - max(y, py) > 1e-6:
                        next_y = max(next_y, py + ph)
            if next_y <= y + 1e-9:
                return y
            y = next_y
        return y

    def _placement_score(
        self,
        idx: int,
        item: Item,
        rect: Tuple[float, float, float, float],
        placed: List[Tuple[float, float, float, float, int]],
        placed_map: Dict[int, Tuple[float, float, float, float, int]],
        desired_center: Tuple[float, float],
        item_edges: List[Dict[int, float]],
        outline_w: float,
        outline_h: float,
    ) -> float:
        x, y, w, h = rect
        cx = x + w / 2.0
        cy = y + h / 2.0
        overflow = max(0.0, x + w - outline_w) + max(0.0, y + h - outline_h)
        pref = abs(cx - desired_center[0]) + abs(cy - desired_center[1])

        wire = 0.0
        total_weight = 0.0
        for nbr, weight in item_edges[idx].items():
            if nbr not in placed_map:
                continue
            px, py, pw, ph, _ = placed_map[nbr]
            pcx = px + pw / 2.0
            pcy = py + ph / 2.0
            wire += weight * (abs(cx - pcx) + abs(cy - pcy))
            total_weight += weight
        if total_weight > 0:
            wire /= total_weight

        all_rects = placed + [(x, y, w, h, -1)]
        xmin = min(r[0] for r in all_rects)
        ymin = min(r[1] for r in all_rects)
        xmax = max(r[0] + r[2] for r in all_rects)
        ymax = max(r[1] + r[3] for r in all_rects)
        bbox_area = (xmax - xmin) * (ymax - ymin)

        boundary_bonus = -80.0 if item.boundary_code else 0.0
        return (
            overflow * 1_000_000.0
            + 0.44 * wire
            + 0.22 * pref
            + 0.035 * bbox_area
            + 0.05 * y
            + 0.006 * x
            + boundary_bonus
        )

    def _rect_overlaps_any(
        self,
        rect: Tuple[float, float, float, float],
        placed: List[Tuple[float, float, float, float, int]],
    ) -> bool:
        x, y, w, h = rect
        for px, py, pw, ph, _ in placed:
            if min(x + w, px + pw) - max(x, px) > 1e-6 and min(y + h, py + ph) - max(y, py) > 1e-6:
                return True
        return False

    def _refine_anchors(
        self,
        items: List[Item],
        anchors: Dict[int, Tuple[float, float]],
        outline_w: float,
        outline_h: float,
        item_edges: List[Dict[int, float]],
        item_degree: List[float],
        coarse_centers: List[Tuple[float, float]],
    ) -> Dict[int, Tuple[float, float]]:
        movable = [i for i, item in enumerate(items) if item.fixed_anchor is None]
        if not movable:
            return anchors

        rects = {
            i: (anchors[i][0], anchors[i][1], items[i].width, items[i].height, i)
            for i in range(len(items))
        }
        order = sorted(
            movable,
            key=lambda i: self._priority(items[i], item_degree[i], coarse_centers[i]),
            reverse=True,
        )

        for _ in range(2):
            improved = False
            for idx in order:
                current = rects.pop(idx)
                placed = list(rects.values())
                ax, ay = self._find_position(
                    idx,
                    items[idx],
                    placed,
                    rects,
                    outline_w,
                    outline_h,
                    item_edges,
                    coarse_centers,
                    current_anchor=(current[0], current[1]),
                )
                desired_center = self._desired_center(
                    idx,
                    items[idx],
                    rects,
                    item_edges,
                    coarse_centers,
                    outline_w,
                    outline_h,
                )
                old_score = self._placement_score(
                    idx,
                    items[idx],
                    current[:4],
                    placed,
                    rects,
                    desired_center,
                    item_edges,
                    outline_w,
                    outline_h,
                )
                new_rect = (ax, ay, items[idx].width, items[idx].height)
                new_score = self._placement_score(
                    idx,
                    items[idx],
                    new_rect,
                    placed,
                    rects,
                    desired_center,
                    item_edges,
                    outline_w,
                    outline_h,
                )
                if new_score + 1e-6 < old_score:
                    rects[idx] = (ax, ay, items[idx].width, items[idx].height, idx)
                    improved = True
                else:
                    rects[idx] = current
            if not improved:
                break

        return {i: (rects[i][0], rects[i][1]) for i in rects}

    def _materialize_positions(
        self,
        items: List[Item],
        anchors: Dict[int, Tuple[float, float]],
        widths: List[float],
        heights: List[float],
        n: int,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
        positions = [(0.0, 0.0, widths[i], heights[i]) for i in range(n)]
        item_boxes = []
        for item_idx, item in enumerate(items):
            ax, ay = anchors[item_idx]
            item_boxes.append((ax, ay, item.width, item.height))
            for b in item.blocks:
                ox, oy = item.offsets[b]
                positions[b] = (ax + ox, ay + oy, widths[b], heights[b])
        return positions, item_boxes

    def _item_boxes_from_positions(
        self,
        items: List[Item],
        positions: List[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        item_boxes = []
        for item in items:
            xs = [positions[b][0] for b in item.blocks]
            ys = [positions[b][1] for b in item.blocks]
            xe = [positions[b][0] + positions[b][2] for b in item.blocks]
            ye = [positions[b][1] + positions[b][3] for b in item.blocks]
            item_boxes.append((min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)))
        return item_boxes

    def _groups(self, values: List[int]) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for i, g in enumerate(values):
            if g:
                groups.setdefault(g, []).append(i)
        return groups

    def _has_overlap(self, positions: List[Tuple[float, float, float, float]]) -> bool:
        n = len(positions)
        for i in range(n):
            x1, y1, w1, h1 = positions[i]
            for j in range(i + 1, n):
                x2, y2, w2, h2 = positions[j]
                if min(x1 + w1, x2 + w2) - max(x1, x2) > 1e-6 and min(y1 + h1, y2 + h2) - max(y1, y2) > 1e-6:
                    return True
        return False

    def _postprocess_boundary_grouped(
        self,
        positions: List[Tuple[float, float, float, float]],
        cons: Dict[str, List[int]],
        items: List[Item],
        item_boxes: List[Tuple[float, float, float, float]],
        block_to_item: Dict[int, int],
    ) -> List[Tuple[float, float, float, float]]:
        item_boundary: Dict[int, int] = {}
        for b, code in enumerate(cons["boundary"]):
            if code == 0 or cons["preplaced"][b]:
                continue
            item_idx = block_to_item.get(b)
            if item_idx is None:
                continue
            item_boundary[item_idx] = item_boundary.get(item_idx, 0) | code

        if not item_boundary:
            return positions

        xmin = min(x for x, _, _, _ in positions)
        ymin = min(y for _, y, _, _ in positions)
        xmax = max(x + w for x, _, w, _ in positions)
        ymax = max(y + h for _, y, _, h in positions)
        base_w = xmax - xmin
        base_h = ymax - ymin

        left_items = [idx for idx, code in item_boundary.items() if code & 1]
        right_items = [idx for idx, code in item_boundary.items() if code & 2]
        top_items = [idx for idx, code in item_boundary.items() if code & 4]
        bottom_items = [idx for idx, code in item_boundary.items() if code & 8]

        left_w = max((item_boxes[i][2] for i in left_items), default=0.0)
        right_w = max((item_boxes[i][2] for i in right_items), default=0.0)
        top_h = max((item_boxes[i][3] for i in top_items), default=0.0)
        bottom_h = max((item_boxes[i][3] for i in bottom_items), default=0.0)

        min_h = max(
            base_h + top_h + bottom_h,
            sum(item_boxes[i][3] for i in left_items),
            sum(item_boxes[i][3] for i in right_items),
            1.0,
        )
        min_w = max(
            base_w + left_w + right_w,
            sum(item_boxes[i][2] for i in top_items),
            sum(item_boxes[i][2] for i in bottom_items),
            1.0,
        )

        new_xmin = xmin - left_w
        new_ymin = ymin - bottom_h
        new_xmax = new_xmin + min_w
        new_ymax = new_ymin + min_h

        item_anchor = {i: (item_boxes[i][0], item_boxes[i][1]) for i in range(len(items))}
        assigned = set()

        def place_item(item_idx: int, ax: float, ay: float) -> None:
            item_anchor[item_idx] = (ax, ay)
            assigned.add(item_idx)

        for item_idx in left_items:
            code = item_boundary[item_idx]
            _, _, iw, ih = item_boxes[item_idx]
            if code & 4:
                place_item(item_idx, new_xmin, new_ymax - ih)
            elif code & 8:
                place_item(item_idx, new_xmin, new_ymin)
        for item_idx in right_items:
            code = item_boundary[item_idx]
            _, _, iw, ih = item_boxes[item_idx]
            if code & 4:
                place_item(item_idx, new_xmax - iw, new_ymax - ih)
            elif code & 8:
                place_item(item_idx, new_xmax - iw, new_ymin)

        y_cursor = new_ymin
        for item_idx in sorted(left_items, key=lambda i: item_boxes[i][3], reverse=True):
            if item_idx in assigned:
                continue
            _, _, iw, ih = item_boxes[item_idx]
            place_item(item_idx, new_xmin, y_cursor)
            y_cursor += ih

        y_cursor = new_ymin
        for item_idx in sorted(right_items, key=lambda i: item_boxes[i][3], reverse=True):
            if item_idx in assigned:
                continue
            _, _, iw, ih = item_boxes[item_idx]
            place_item(item_idx, new_xmax - iw, y_cursor)
            y_cursor += ih

        x_start = new_xmin + left_w
        x_end = new_xmax - right_w
        x_cursor = x_start
        for item_idx in sorted(top_items, key=lambda i: item_boxes[i][2], reverse=True):
            if item_idx in assigned:
                continue
            _, _, iw, ih = item_boxes[item_idx]
            if x_cursor + iw > x_end:
                x_cursor = x_start
            place_item(item_idx, x_cursor, new_ymax - ih)
            x_cursor += iw

        x_cursor = x_start
        for item_idx in sorted(bottom_items, key=lambda i: item_boxes[i][2], reverse=True):
            if item_idx in assigned:
                continue
            _, _, iw, ih = item_boxes[item_idx]
            if x_cursor + iw > x_end:
                x_cursor = x_start
            place_item(item_idx, x_cursor, new_ymin)
            x_cursor += iw

        result = list(positions)
        for item_idx, (ax, ay) in item_anchor.items():
            item = items[item_idx]
            ox0, oy0, _, _ = item_boxes[item_idx]
            dx = ax - ox0
            dy = ay - oy0
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            for b in item.blocks:
                x, y, w, h = result[b]
                result[b] = (x + dx, y + dy, w, h)

        if self._has_overlap(result):
            return positions
        return result

    def _proxy_cost(
        self,
        positions: List[Tuple[float, float, float, float]],
        cons: Dict[str, List[int]],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> float:
        hpwl = calculate_hpwl_b2b(positions, b2b_connectivity) + calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)
        area = calculate_bbox_area(positions)
        boundary, grouping, mib, n_soft = self._soft_violations_fast(positions, cons)
        vrel = (boundary + grouping + mib) / max(n_soft, 1)
        return (1.0 + 0.002 * hpwl + 0.01 * area) * math.exp(2.0 * vrel)

    def _accept_boundary_candidate(
        self,
        before: List[Tuple[float, float, float, float]],
        after: List[Tuple[float, float, float, float]],
        cons: Dict[str, List[int]],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> bool:
        b0, g0, m0, n0 = self._soft_violations_fast(before, cons)
        b1, g1, m1, n1 = self._soft_violations_fast(after, cons)
        soft0 = b0 + g0 + m0
        soft1 = b1 + g1 + m1
        if soft1 > soft0:
            return False

        proxy0 = self._proxy_cost(before, cons, b2b_connectivity, p2b_connectivity, pins_pos)
        proxy1 = self._proxy_cost(after, cons, b2b_connectivity, p2b_connectivity, pins_pos)

        if soft1 < soft0:
            if proxy1 <= proxy0 * 1.08:
                return True
            area0 = calculate_bbox_area(before)
            area1 = calculate_bbox_area(after)
            area_ratio = area1 / max(area0, 1.0)
            gain = soft0 - soft1
            return gain >= max(3, int(0.10 * max(n0, n1))) and area_ratio <= 1.60

        return proxy1 < proxy0

    def _soft_violations_fast(
        self,
        positions: List[Tuple[float, float, float, float]],
        cons: Dict[str, List[int]],
    ) -> Tuple[int, int, int, int]:
        n = len(positions)
        boundary = 0
        grouping = 0
        mib = 0
        n_soft = sum(1 for c in cons["boundary"][:n] if c)

        for members in self._groups(cons["mib"][:n]).values():
            n_soft += max(0, len(members) - 1)
            shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in members}
            mib += max(0, len(shapes) - 1)

        for members in self._groups(cons["cluster"][:n]).values():
            n_soft += max(0, len(members) - 1)
            grouping += max(0, self._component_count(members, positions) - 1)

        xmin = min(x for x, _, _, _ in positions)
        ymin = min(y for _, y, _, _ in positions)
        xmax = max(x + w for x, _, w, _ in positions)
        ymax = max(y + h for _, y, _, h in positions)
        for i, code in enumerate(cons["boundary"][:n]):
            if not code:
                continue
            x, y, w, h = positions[i]
            touches = {
                1: abs(x - xmin) < 1e-6,
                2: abs(x + w - xmax) < 1e-6,
                4: abs(y + h - ymax) < 1e-6,
                8: abs(y - ymin) < 1e-6,
            }
            if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                boundary += 1

        return boundary, grouping, mib, n_soft

    def _component_count(
        self,
        members: List[int],
        positions: List[Tuple[float, float, float, float]],
    ) -> int:
        parent = {i: i for i in members}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for idx, a in enumerate(members):
            ax, ay, aw, ah = positions[a]
            for b in members[idx + 1:]:
                bx, by, bw, bh = positions[b]
                vertical_touch = abs(ax + aw - bx) < 1e-6 or abs(bx + bw - ax) < 1e-6
                vertical_overlap = min(ay + ah, by + bh) - max(ay, by) > 1e-6
                horizontal_touch = abs(ay + ah - by) < 1e-6 or abs(by + bh - ay) < 1e-6
                horizontal_overlap = min(ax + aw, bx + bw) - max(ax, bx) > 1e-6
                if (vertical_touch and vertical_overlap) or (horizontal_touch and horizontal_overlap):
                    union(a, b)

        return len({find(i) for i in members})

    def _repair_overlaps(
        self,
        positions: List[Tuple[float, float, float, float]],
        cons: Dict[str, List[int]],
    ) -> List[Tuple[float, float, float, float]]:
        repaired = list(positions)
        for _ in range(20):
            changed = False
            for i in range(len(repaired)):
                x1, y1, w1, h1 = repaired[i]
                for j in range(i + 1, len(repaired)):
                    x2, y2, w2, h2 = repaired[j]
                    ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                    oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                    if ox > 1e-6 and oy > 1e-6:
                        movable = j if not cons["preplaced"][j] else i
                        if cons["preplaced"][movable]:
                            continue
                        mx, my, mw, mh = repaired[movable]
                        repaired[movable] = (mx, max(y1 + h1, y2 + h2), mw, mh)
                        changed = True
            if not changed:
                break
        return repaired
