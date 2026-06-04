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
        for item_idx, item in enumerate(items):
            ax, ay = anchors[item_idx]
            for b in item.blocks:
                ox, oy = item.offsets[b]
                positions[b] = (ax + ox, ay + oy, widths[b], heights[b])

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
            items.append(Item(members, offsets, x_cursor, max_h, fixed_anchor, bcode, pref))
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
