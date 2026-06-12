#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge optimizer.

This implementation favors fast, deterministic construction over long local
search.  It enforces all hard constraints by construction, preserves MIB shapes
when possible, packs clustering constraints as connected mini-floorplans, and
places boundary-constrained items on their requested outline edge.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

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
        items = self._build_items(n, widths, heights, cons, target_positions, p2b_connectivity, pins_pos)
        outline_w, outline_h = self._estimate_outline(n, area, items, pins_pos, target_positions, cons)

        anchors = self._place_items(items, outline_w, outline_h)

        positions = [(0.0, 0.0, widths[i], heights[i]) for i in range(n)]
        block_to_item = {}
        item_boxes = []
        for item_idx, item in enumerate(items):
            ax, ay = anchors[item_idx]
            item_boxes.append((ax, ay, item.width, item.height))
            for b in item.blocks:
                ox, oy = item.offsets[b]
                positions[b] = (ax + ox, ay + oy, widths[b], heights[b])
                block_to_item[b] = item_idx

        # Exact hard constraints win over every heuristic.
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

        boundary_positions = self._postprocess_boundary_grouped(
            positions, cons, items, item_boxes, block_to_item
        )
        if not self._has_overlap(boundary_positions):
            boundary_positions = self._repair_overlaps(boundary_positions, cons)
        if not self._has_overlap(boundary_positions) and self._accept_boundary_candidate(
            positions, boundary_positions, cons, b2b_connectivity, p2b_connectivity, pins_pos
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

        groups = self._groups(cons["mib"])
        for members in groups.values():
            fixed_ref = next((i for i in members if cons["fixed"][i] or cons["preplaced"][i]), None)
            if fixed_ref is not None:
                w, h = widths[fixed_ref], heights[fixed_ref]
            else:
                # MIB members in FloorSet-Lite use the same target area.
                a = max(area[members[0]], 1.0)
                w = h = math.sqrt(a)
            for i in members:
                if not (cons["fixed"][i] or cons["preplaced"][i]):
                    widths[i], heights[i] = w, h

        return widths, heights

    def _build_items(
        self,
        n: int,
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> List[Item]:
        assigned = set()
        items: List[Item] = []
        pin_centers = self._pin_preferred_centers(n, p2b_connectivity, pins_pos)

        for _, members in sorted(self._groups(cons["cluster"]).items()):
            members = self._cluster_order(members, cons, target_positions)
            bcode = self._item_boundary_code(members, cons)
            offsets: Dict[int, Tuple[float, float]] = {}
            x_cursor = 0.0
            max_h = 0.0
            for b in members:
                offsets[b] = (x_cursor, 0.0)
                x_cursor += widths[b]
                max_h = max(max_h, heights[b])

            fixed_anchor = None
            if target_positions is not None:
                pre = next((b for b in members if cons["preplaced"][b]), None)
                if pre is not None:
                    ox, oy = offsets[pre]
                    fixed_anchor = (float(target_positions[pre, 0]) - ox, float(target_positions[pre, 1]) - oy)

            pref = self._average_center([pin_centers[b] for b in members if pin_centers[b] is not None])
            items.append(Item(members, offsets, x_cursor, max_h, fixed_anchor, bcode, pref, True))
            assigned.update(members)

        for b in range(n):
            if b in assigned:
                continue
            fixed_anchor = None
            if target_positions is not None and cons["preplaced"][b]:
                fixed_anchor = (float(target_positions[b, 0]), float(target_positions[b, 1]))
            items.append(
                Item(
                    [b],
                    {b: (0.0, 0.0)},
                    widths[b],
                    heights[b],
                    fixed_anchor,
                    cons["boundary"][b],
                    pin_centers[b],
                    False,
                )
            )

        return items

    def _cluster_order(
        self,
        members: List[int],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
    ) -> List[int]:
        if target_positions is None:
            return members
        pre = [b for b in members if cons["preplaced"][b]]
        rest = [b for b in members if not cons["preplaced"][b]]
        # Keep preplaced blocks first so their cluster anchor rarely goes negative.
        pre.sort(key=lambda b: (float(target_positions[b, 0]), float(target_positions[b, 1])))
        return pre + rest

    def _item_boundary_code(self, members: Iterable[int], cons: Dict[str, List[int]]) -> int:
        codes = [cons["boundary"][b] for b in members if cons["boundary"][b] != 0]
        if not codes:
            return 0
        # If a group asks for several different edges, choosing the most common
        # one usually satisfies at least one block without sacrificing grouping.
        return max(set(codes), key=codes.count)

    def _pin_preferred_centers(
        self,
        n: int,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> List[Optional[Tuple[float, float]]]:
        sums = [[0.0, 0.0, 0.0] for _ in range(n)]
        if p2b_connectivity is not None:
            for edge in p2b_connectivity:
                if len(edge) < 3 or int(edge[0]) < 0:
                    continue
                pin_idx, block_idx, weight = int(edge[0]), int(edge[1]), float(edge[2])
                if 0 <= block_idx < n and 0 <= pin_idx < len(pins_pos) and weight > 0:
                    px, py = float(pins_pos[pin_idx, 0]), float(pins_pos[pin_idx, 1])
                    if px >= 0 and py >= 0:
                        sums[block_idx][0] += weight * px
                        sums[block_idx][1] += weight * py
                        sums[block_idx][2] += weight

        centers: List[Optional[Tuple[float, float]]] = []
        for sx, sy, sw in sums:
            centers.append((sx / sw, sy / sw) if sw > 0 else None)
        return centers

    def _average_center(self, centers: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not centers:
            return None
        return (
            sum(c[0] for c in centers) / len(centers),
            sum(c[1] for c in centers) / len(centers),
        )

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
        pin_w = pin_h = 0.0
        if pins_pos is not None and pins_pos.numel() > 0:
            valid = pins_pos[(pins_pos[:, 0] >= 0) & (pins_pos[:, 1] >= 0)]
            if len(valid) > 0:
                pin_w = float(valid[:, 0].max()) / 1.10
                pin_h = float(valid[:, 1].max()) / 1.10

        if pin_w <= 0 or pin_h <= 0:
            side = math.sqrt(max(total_area, 1.0))
            pin_w = side
            pin_h = side

        if target_positions is not None:
            for i in range(n):
                if cons["preplaced"][i]:
                    pin_w = max(pin_w, float(target_positions[i, 0] + target_positions[i, 2]))
                    pin_h = max(pin_h, float(target_positions[i, 1] + target_positions[i, 3]))

        width = max(pin_w, max(item.width for item in items), 1.0)
        height = max(pin_h, max(item.height for item in items), 1.0)
        required = max(total_area * 1.08, sum(item.width * item.height for item in items) * 1.02)
        if width * height < required:
            scale = math.sqrt(required / max(width * height, 1.0))
            width *= scale
            height *= scale
        return width, height

    def _place_items(self, items: List[Item], outline_w: float, outline_h: float) -> Dict[int, Tuple[float, float]]:
        placed: List[Tuple[float, float, float, float, int]] = []
        anchors: Dict[int, Tuple[float, float]] = {}

        fixed_order = [i for i, item in enumerate(items) if item.fixed_anchor is not None]
        free_order = [i for i, item in enumerate(items) if item.fixed_anchor is None]
        free_order.sort(key=lambda i: self._priority(items[i]), reverse=True)

        for idx in fixed_order:
            item = items[idx]
            ax, ay = item.fixed_anchor or (0.0, 0.0)
            anchors[idx] = (ax, ay)
            placed.append((ax, ay, item.width, item.height, idx))

        for idx in free_order:
            item = items[idx]
            ax, ay = self._find_position(item, placed, outline_w, outline_h)
            anchors[idx] = (ax, ay)
            placed.append((ax, ay, item.width, item.height, idx))

        return anchors

    def _priority(self, item: Item) -> float:
        boundary = 10_000.0 if item.boundary_code else 0.0
        return boundary + item.width * item.height + 10.0 * len(item.blocks)

    def _find_position(
        self,
        item: Item,
        placed: List[Tuple[float, float, float, float, int]],
        outline_w: float,
        outline_h: float,
    ) -> Tuple[float, float]:
        xs = {0.0}
        ys = {0.0}
        for x, y, w, h, _ in placed:
            xs.add(x + w)
            xs.add(max(0.0, x - item.width))
            ys.add(y + h)
            ys.add(max(0.0, y - item.height))

        if item.preferred_center is not None:
            cx, cy = item.preferred_center
            xs.add(max(0.0, cx - item.width / 2.0))
            ys.add(max(0.0, cy - item.height / 2.0))

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
        if len(x_values) > 80:
            x_values = self._trim_axis_candidates(x_values, item, axis=0)
        if len(y_values) > 80:
            y_values = self._trim_axis_candidates(y_values, item, axis=1)

        candidates = []
        if item.boundary_code & (4 | 8):
            # Top/bottom constraints fix y; scan likely x anchors along that edge.
            fixed_y = next(iter(ys))
            candidates.extend((x, fixed_y) for x in x_values)
        elif item.boundary_code & (1 | 2):
            # Left/right constraints fix x; scan likely y anchors along that edge.
            fixed_x = next(iter(xs))
            candidates.extend((fixed_x, y) for y in y_values)
        else:
            candidates.extend((x, self._lowest_nonoverlap_y(x, item.width, item.height, placed)) for x in x_values)

        # A few 2-D candidates help with dense preplaced obstacles without going
        # back to the expensive full Cartesian product.
        if item.boundary_code == 0 and len(placed) < 80:
            candidates.extend((x, y) for x in x_values[:24] for y in y_values[:24])

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
            score = self._placement_score(item, rect, placed, outline_w, outline_h)
            if score < best_score:
                best_score = score
                best = (x, y)

        if best is not None:
            return best

        # Guaranteed fallback: append above the current layout.  This keeps the
        # solution feasible even for dense preplaced obstacles.
        y = 0.0 if not placed else max(py + ph for _, py, _, ph, _ in placed)
        x = 0.0
        if item.boundary_code & 2:
            x = max(0.0, outline_w - item.width)
        if item.boundary_code & 4:
            y = max(y, outline_h - item.height)
        return x, y

    def _trim_axis_candidates(self, values: List[float], item: Item, axis: int) -> List[float]:
        keep = set(values[:20])
        keep.update(values[-12:])
        if item.preferred_center is not None:
            target = item.preferred_center[axis] - (item.width if axis == 0 else item.height) / 2.0
            ranked = sorted(values, key=lambda v: abs(v - target))
            keep.update(ranked[:48])
        else:
            keep.update(values[:68])
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
        item: Item,
        rect: Tuple[float, float, float, float],
        placed: List[Tuple[float, float, float, float, int]],
        outline_w: float,
        outline_h: float,
    ) -> float:
        x, y, w, h = rect
        overflow = max(0.0, x + w - outline_w) + max(0.0, y + h - outline_h)
        pref = 0.0
        if item.preferred_center is not None:
            cx, cy = item.preferred_center
            pref = abs((x + w / 2.0) - cx) + abs((y + h / 2.0) - cy)

        all_rects = placed + [(x, y, w, h, -1)]
        xmin = min(r[0] for r in all_rects)
        ymin = min(r[1] for r in all_rects)
        xmax = max(r[0] + r[2] for r in all_rects)
        ymax = max(r[1] + r[3] for r in all_rects)
        bbox_area = (xmax - xmin) * (ymax - ymin)
        edge_bonus = 0.0
        if item.boundary_code:
            edge_bonus -= 100.0
        return overflow * 1_000_000.0 + pref * 0.18 + bbox_area * 0.04 + y * 0.1 + x * 0.01 + edge_bonus

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

    def _postprocess_boundary(
        self,
        positions: List[Tuple[float, float, float, float]],
        cons: Dict[str, List[int]],
    ) -> List[Tuple[float, float, float, float]]:
        boundary_blocks = [
            i for i, code in enumerate(cons["boundary"])
            if code != 0 and not cons["preplaced"][i]
        ]
        if not boundary_blocks:
            return positions

        xmin = min(x for x, _, _, _ in positions)
        ymin = min(y for _, y, _, _ in positions)
        xmax = max(x + w for x, _, w, _ in positions)
        ymax = max(y + h for _, y, _, h in positions)
        base_w = xmax - xmin
        base_h = ymax - ymin

        left_blocks = [i for i in boundary_blocks if cons["boundary"][i] & 1]
        right_blocks = [i for i in boundary_blocks if cons["boundary"][i] & 2]
        top_blocks = [i for i in boundary_blocks if cons["boundary"][i] & 4]
        bottom_blocks = [i for i in boundary_blocks if cons["boundary"][i] & 8]

        left_w = max((positions[i][2] for i in left_blocks), default=0.0)
        right_w = max((positions[i][2] for i in right_blocks), default=0.0)
        top_h = max((positions[i][3] for i in top_blocks), default=0.0)
        bottom_h = max((positions[i][3] for i in bottom_blocks), default=0.0)

        min_h = max(
            base_h + top_h + bottom_h,
            sum(positions[i][3] for i in left_blocks),
            sum(positions[i][3] for i in right_blocks),
            1.0,
        )
        min_w = max(
            base_w + left_w + right_w,
            sum(positions[i][2] for i in top_blocks),
            sum(positions[i][2] for i in bottom_blocks),
            1.0,
        )

        new_xmin = xmin - left_w
        new_ymin = ymin - bottom_h
        new_xmax = new_xmin + min_w
        new_ymax = new_ymin + min_h

        result = list(positions)
        assigned = set()

        def reserve_corner(blocks: List[int], x: float, y_func) -> float:
            if not blocks:
                return 0.0
            blocks.sort(key=lambda b: positions[b][2] * positions[b][3], reverse=True)
            first = blocks[0]
            _, _, w, h = positions[first]
            result[first] = (x if x is not None else result[first][0], y_func(h), w, h)
            assigned.add(first)
            return h

        left_top = [i for i in left_blocks if cons["boundary"][i] & 4]
        left_bottom = [i for i in left_blocks if cons["boundary"][i] & 8]
        right_top = [i for i in right_blocks if cons["boundary"][i] & 4]
        right_bottom = [i for i in right_blocks if cons["boundary"][i] & 8]

        lt_h = reserve_corner(left_top, new_xmin, lambda h: new_ymax - h)
        lb_h = reserve_corner(left_bottom, new_xmin, lambda h: new_ymin)
        rt_h = reserve_corner(right_top, None, lambda h: new_ymax - h)
        if right_top:
            b = right_top[0]
            x, y, w, h = result[b]
            result[b] = (new_xmax - w, y, w, h)
        rb_h = reserve_corner(right_bottom, None, lambda h: new_ymin)
        if right_bottom:
            b = right_bottom[0]
            x, y, w, h = result[b]
            result[b] = (new_xmax - w, y, w, h)

        y_cursor = new_ymin + lb_h
        y_limit = new_ymax - lt_h
        for i in sorted(left_blocks, key=lambda b: positions[b][3], reverse=True):
            if i in assigned:
                continue
            _, _, w, h = positions[i]
            if y_cursor + h > y_limit:
                y_cursor = new_ymin
            result[i] = (new_xmin, y_cursor, w, h)
            assigned.add(i)
            y_cursor += h

        y_cursor = new_ymin + rb_h
        y_limit = new_ymax - rt_h
        for i in sorted(right_blocks, key=lambda b: positions[b][3], reverse=True):
            if i in assigned:
                continue
            _, _, w, h = positions[i]
            if y_cursor + h > y_limit:
                y_cursor = new_ymin
            result[i] = (new_xmax - w, y_cursor, w, h)
            assigned.add(i)
            y_cursor += h

        x_start = new_xmin + left_w
        x_end = new_xmax - right_w

        x_cursor = x_start
        for i in sorted(top_blocks, key=lambda b: positions[b][2], reverse=True):
            if i in assigned:
                continue
            _, _, w, h = positions[i]
            if x_cursor + w > x_end:
                x_cursor = x_start
            result[i] = (x_cursor, new_ymax - h, w, h)
            assigned.add(i)
            x_cursor += w

        x_cursor = x_start
        for i in sorted(bottom_blocks, key=lambda b: positions[b][2], reverse=True):
            if i in assigned:
                continue
            _, _, w, h = positions[i]
            if x_cursor + w > x_end:
                x_cursor = x_start
            result[i] = (x_cursor, new_ymin, w, h)
            assigned.add(i)
            x_cursor += w

        if self._has_overlap(result):
            return positions
        return result

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

        # Corners first.
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
        if soft1 >= soft0:
            return False

        area0 = calculate_bbox_area(before)
        area1 = calculate_bbox_area(after)
        area_ratio = area1 / max(area0, 1.0)
        if area_ratio > 1.22 and soft0 - soft1 < max(3, int(0.12 * max(n0, n1))):
            return False

        # Fall back to a cheap weighted proxy for close calls.
        return self._proxy_cost(after, cons, b2b_connectivity, p2b_connectivity, pins_pos) < (
            self._proxy_cost(before, cons, b2b_connectivity, p2b_connectivity, pins_pos) * 0.97
        )

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

        mib_groups = self._groups(cons["mib"][:n])
        for members in mib_groups.values():
            n_soft += max(0, len(members) - 1)
            shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in members}
            mib += max(0, len(shapes) - 1)

        cluster_groups = self._groups(cons["cluster"][:n])
        for members in cluster_groups.values():
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
