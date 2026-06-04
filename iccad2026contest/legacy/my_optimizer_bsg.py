#!/usr/bin/env python3
"""
BSG-style grid optimizer for ICCAD 2026 FloorSet.

This is an experimental alternative to the constructive bottom-left optimizer.
It packs cluster superblocks into horizontal grid rows, uses boundary-aware
row ordering, and runs a short deterministic local search over item order.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_bbox_area,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
)


@dataclass
class GridItem:
    blocks: List[int]
    offsets: Dict[int, Tuple[float, float]]
    width: float
    height: float
    boundary_code: int = 0
    fixed_anchor: Optional[Tuple[float, float]] = None
    preferred_center: Optional[Tuple[float, float]] = None


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.random_seed = 20260325

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
        random.seed(self.random_seed + block_count)
        n = block_count
        areas = [float(area_targets[i]) if float(area_targets[i]) > 0 else 1.0 for i in range(n)]
        cons = self._constraints_to_lists(constraints, n)
        widths, heights = self._choose_dimensions(n, areas, cons, target_positions)
        items = self._build_items(n, widths, heights, cons, target_positions, p2b_connectivity, pins_pos)

        rows = self._initial_rows(items, sum(areas))
        rows = self._local_search(rows, items, widths, heights, cons, target_positions, b2b_connectivity, p2b_connectivity, pins_pos)
        positions = self._pack_rows(rows, items, widths, heights, cons, target_positions)
        positions = self._finalize_hard_constraints(positions, cons, target_positions)
        if self._has_overlap(positions):
            positions = self._repair_overlaps(positions, cons)
        return positions

    def _constraints_to_lists(self, constraints: torch.Tensor, n: int) -> Dict[str, List[int]]:
        cols = constraints.shape[1] if constraints is not None and constraints.numel() > 0 and constraints.dim() > 1 else 0

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
        areas: List[float],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
    ) -> Tuple[List[float], List[float]]:
        widths = [math.sqrt(max(a, 1.0)) for a in areas]
        heights = [math.sqrt(max(a, 1.0)) for a in areas]

        if target_positions is not None:
            for i in range(n):
                if cons["fixed"][i] or cons["preplaced"][i]:
                    widths[i] = float(target_positions[i, 2])
                    heights[i] = float(target_positions[i, 3])

        for members in self._groups(cons["mib"]).values():
            ref = next((i for i in members if cons["fixed"][i] or cons["preplaced"][i]), None)
            if ref is not None:
                w, h = widths[ref], heights[ref]
            else:
                w = h = math.sqrt(max(areas[members[0]], 1.0))
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
    ) -> List[GridItem]:
        pin_centers = self._pin_preferred_centers(n, p2b_connectivity, pins_pos)
        items: List[GridItem] = []
        assigned = set()

        for _, members in sorted(self._groups(cons["cluster"]).items()):
            members = self._cluster_order(members, cons, target_positions)
            offsets: Dict[int, Tuple[float, float]] = {}
            x = 0.0
            h = 0.0
            for b in members:
                offsets[b] = (x, 0.0)
                x += widths[b]
                h = max(h, heights[b])

            fixed_anchor = None
            if target_positions is not None:
                pre = next((b for b in members if cons["preplaced"][b]), None)
                if pre is not None:
                    ox, oy = offsets[pre]
                    fixed_anchor = (float(target_positions[pre, 0]) - ox, float(target_positions[pre, 1]) - oy)

            pref = self._average_center([pin_centers[b] for b in members if pin_centers[b] is not None])
            items.append(GridItem(members, offsets, x, h, self._item_boundary_code(members, cons), fixed_anchor, pref))
            assigned.update(members)

        for b in range(n):
            if b in assigned:
                continue
            fixed_anchor = None
            if target_positions is not None and cons["preplaced"][b]:
                fixed_anchor = (float(target_positions[b, 0]), float(target_positions[b, 1]))
            items.append(GridItem([b], {b: (0.0, 0.0)}, widths[b], heights[b], cons["boundary"][b], fixed_anchor, pin_centers[b]))

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
        pre.sort(key=lambda b: (float(target_positions[b, 0]), float(target_positions[b, 1])))
        return pre + rest

    def _item_boundary_code(self, members: Iterable[int], cons: Dict[str, List[int]]) -> int:
        codes = [cons["boundary"][b] for b in members if cons["boundary"][b]]
        return max(set(codes), key=codes.count) if codes else 0

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
        return [(sx / sw, sy / sw) if sw > 0 else None for sx, sy, sw in sums]

    def _average_center(self, centers: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not centers:
            return None
        return (sum(x for x, _ in centers) / len(centers), sum(y for _, y in centers) / len(centers))

    def _initial_rows(self, items: List[GridItem], total_area: float) -> List[List[int]]:
        target_width = max(math.sqrt(max(total_area, 1.0)) * 1.05, max(item.width for item in items))
        top: List[int] = []
        bottom: List[int] = []
        middle_items: List[int] = []

        for idx, item in enumerate(items):
            code = item.boundary_code
            if code & 4:
                top.append(idx)
            elif code & 8:
                bottom.append(idx)
            else:
                middle_items.append(idx)

        def sort_key(i: int) -> Tuple[int, float]:
            item = items[i]
            left_score = 0 if item.boundary_code & 1 else 1
            right_score = 2 if item.boundary_code & 2 else 1
            xpref = item.preferred_center[0] if item.preferred_center is not None else 0.0
            return (left_score + right_score, xpref)

        top.sort(key=sort_key)
        bottom.sort(key=sort_key)
        middle_items.sort(key=lambda i: (
            items[i].preferred_center[1] if items[i].preferred_center is not None else 0.0,
            items[i].preferred_center[0] if items[i].preferred_center is not None else 0.0,
        ))

        rows: List[List[int]] = []
        if bottom:
            rows.append(bottom)

        current: List[int] = []
        current_w = 0.0
        for idx in middle_items:
            item = items[idx]
            if current and current_w + item.width > target_width:
                rows.append(self._order_row(current, items))
                current = []
                current_w = 0.0
            current.append(idx)
            current_w += item.width
        if current:
            rows.append(self._order_row(current, items))

        if top:
            rows.append(top)
        return rows or [[]]

    def _order_row(self, row: List[int], items: List[GridItem]) -> List[int]:
        left = [i for i in row if items[i].boundary_code & 1]
        right = [i for i in row if items[i].boundary_code & 2 and i not in left]
        mid = [i for i in row if i not in left and i not in right]
        mid.sort(key=lambda i: items[i].preferred_center[0] if items[i].preferred_center is not None else 0.0)
        return left + mid + right

    def _local_search(
        self,
        rows: List[List[int]],
        items: List[GridItem],
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> List[List[int]]:
        best_rows = [row[:] for row in rows]
        best_pos = self._pack_rows(best_rows, items, widths, heights, cons, target_positions)
        best_score = self._proxy_cost(best_pos, cons, b2b_connectivity, p2b_connectivity, pins_pos)
        current_rows = [row[:] for row in best_rows]
        current_score = best_score

        iterations = min(120, 35 + len(items))
        for step in range(iterations):
            candidate = [row[:] for row in current_rows]
            move = step % 4
            nonempty = [r for r, row in enumerate(candidate) if row]
            if len(nonempty) < 1:
                break

            if move == 0 and len(nonempty) >= 2:
                r1, r2 = random.sample(nonempty, 2)
                i1 = random.randrange(len(candidate[r1]))
                i2 = random.randrange(len(candidate[r2]))
                candidate[r1][i1], candidate[r2][i2] = candidate[r2][i2], candidate[r1][i1]
            elif move == 1 and len(nonempty) >= 2:
                r1, r2 = random.sample(nonempty, 2)
                i1 = random.randrange(len(candidate[r1]))
                item = candidate[r1].pop(i1)
                insert_at = random.randrange(len(candidate[r2]) + 1)
                candidate[r2].insert(insert_at, item)
            elif move == 2:
                r = random.choice(nonempty)
                candidate[r] = self._order_row(candidate[r], items)
            else:
                if len(candidate) > 2:
                    r1, r2 = sorted(random.sample(range(len(candidate)), 2))
                    # Keep top/bottom boundary rows mostly stable.
                    if r1 not in (0, len(candidate) - 1) and r2 not in (0, len(candidate) - 1):
                        candidate[r1], candidate[r2] = candidate[r2], candidate[r1]

            candidate = [self._order_row(row, items) for row in candidate if row]
            pos = self._pack_rows(candidate, items, widths, heights, cons, target_positions)
            score = self._proxy_cost(pos, cons, b2b_connectivity, p2b_connectivity, pins_pos)

            temp = max(0.02, 1.0 - step / max(iterations, 1))
            if score < current_score or random.random() < math.exp((current_score - score) / max(1.0, current_score) / temp):
                current_rows = candidate
                current_score = score
                if score < best_score:
                    best_rows = [row[:] for row in candidate]
                    best_score = score

        return best_rows

    def _pack_rows(
        self,
        rows: List[List[int]],
        items: List[GridItem],
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
    ) -> List[Tuple[float, float, float, float]]:
        row_heights = [max((items[i].height for i in row), default=0.0) for row in rows]
        raw_widths = [sum(items[i].width for i in row) for row in rows]
        grid_w = max(raw_widths + [1.0])

        anchors: Dict[int, Tuple[float, float]] = {}
        y = 0.0
        for ridx, row in enumerate(rows):
            row_h = row_heights[ridx]
            left = [i for i in row if items[i].boundary_code & 1]
            right = [i for i in row if items[i].boundary_code & 2 and i not in left]
            mid = [i for i in row if i not in left and i not in right]

            x = 0.0
            for idx in left + mid:
                item = items[idx]
                iy = self._row_item_y(item, y, row_h)
                anchors[idx] = (x, iy)
                x += item.width

            x = grid_w - sum(items[i].width for i in right)
            for idx in right:
                item = items[idx]
                iy = self._row_item_y(item, y, row_h)
                anchors[idx] = (x, iy)
                x += item.width

            y += row_h

        positions = [(0.0, 0.0, widths[i], heights[i]) for i in range(len(widths))]
        for idx, item in enumerate(items):
            ax, ay = anchors.get(idx, (0.0, 0.0))
            for b in item.blocks:
                ox, oy = item.offsets[b]
                positions[b] = (ax + ox, ay + oy, widths[b], heights[b])
        return positions

    def _row_item_y(self, item: GridItem, row_y: float, row_h: float) -> float:
        if item.boundary_code & 4:
            return row_y + row_h - item.height
        return row_y

    def _finalize_hard_constraints(
        self,
        positions: List[Tuple[float, float, float, float]],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
    ) -> List[Tuple[float, float, float, float]]:
        if target_positions is None:
            return positions
        result = list(positions)
        for i in range(len(result)):
            if cons["preplaced"][i]:
                result[i] = (
                    float(target_positions[i, 0]),
                    float(target_positions[i, 1]),
                    float(target_positions[i, 2]),
                    float(target_positions[i, 3]),
                )
            elif cons["fixed"][i]:
                x, y, _, _ = result[i]
                result[i] = (x, y, float(target_positions[i, 2]), float(target_positions[i, 3]))
        return result

    def _proxy_cost(
        self,
        positions: List[Tuple[float, float, float, float]],
        cons: Dict[str, List[int]],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> float:
        area = calculate_bbox_area(positions)
        boundary, grouping, mib, n_soft = self._soft_violations_fast(positions, cons)
        vrel = (boundary + grouping + mib) / max(n_soft, 1)
        overlap = 1_000_000.0 if self._has_overlap(positions) else 0.0
        return overlap + 0.04 * area + 1500.0 * vrel

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

    def _component_count(self, members: List[int], positions: List[Tuple[float, float, float, float]]) -> int:
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

    def _groups(self, values: List[int]) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for i, g in enumerate(values):
            if g:
                groups.setdefault(g, []).append(i)
        return groups

    def _has_overlap(self, positions: List[Tuple[float, float, float, float]]) -> bool:
        for i in range(len(positions)):
            x1, y1, w1, h1 = positions[i]
            for j in range(i + 1, len(positions)):
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
        for _ in range(25):
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
                        if ox < oy:
                            repaired[movable] = (max(x1 + w1, x2 + w2), my, mw, mh)
                        else:
                            repaired[movable] = (mx, max(y1 + h1, y2 + h2), mw, mh)
                        changed = True
            if not changed:
                break
        return repaired
