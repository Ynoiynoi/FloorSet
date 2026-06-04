#!/usr/bin/env python3
"""
Slicing-tree-based ICCAD 2026 optimizer.

This optimizer reuses the contest repo's mature hard-constraint handling and
repair logic, but replaces the core placement stage with a lightweight
slicing-tree construction:

- cluster groups are compacted by a recursive slicing layout
- free items are placed by a global slicing tree
- preplaced dimensions are preserved, but their absolute positions are
  intentionally ignored during construction to avoid anchor-driven blowups
"""

import importlib.util
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch


THIS_DIR = Path(__file__).resolve().parent
CONTEST_DIR = THIS_DIR.parents[1]

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(CONTEST_DIR))


def _load_base_optimizer():
    base_path = CONTEST_DIR / "my_optimizer.py"
    spec = importlib.util.spec_from_file_location("contest_base_optimizer", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base optimizer from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_optimizer()
_BaseOptimizer = _BASE.MyOptimizer
_Item = _BASE.Item


class MyOptimizer(_BaseOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        # User-requested experiment mode: ignore preplaced absolute locations
        # during optimization to reduce anchor-driven whitespace.
        self.ignore_preplaced_position = True

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
        anchors = self._compact_item_anchors(items, anchors)

        positions = [(0.0, 0.0, widths[i], heights[i]) for i in range(n)]
        for item_idx, item in enumerate(items):
            ax, ay = anchors[item_idx]
            for b in item.blocks:
                ox, oy = item.offsets[b]
                positions[b] = (ax + ox, ay + oy, widths[b], heights[b])

        # Preserve exact dimensions for fixed/preplaced blocks.  In this
        # experiment mode we intentionally do not snap preplaced blocks back to
        # their original (x, y), because those anchors were the primary source
        # of bbox blow-up.
        if target_positions is not None:
            for i in range(n):
                if cons["fixed"][i] or cons["preplaced"][i]:
                    x, y, _, _ = positions[i]
                    positions[i] = (
                        x,
                        y,
                        float(target_positions[i, 2]),
                        float(target_positions[i, 3]),
                    )

        repair_cons = dict(cons)
        if self.ignore_preplaced_position:
            repair_cons["preplaced"] = [0] * n

        positions = self._compact_block_positions(positions, repair_cons)

        if self._has_overlap(positions):
            positions = self._repair_overlaps(positions, repair_cons)
            positions = self._compact_block_positions(positions, repair_cons)

        boundary_positions = self._postprocess_boundary(positions, cons)
        boundary_positions = self._compact_block_positions(boundary_positions, repair_cons)
        if self._has_overlap(boundary_positions):
            boundary_positions = self._repair_overlaps(boundary_positions, repair_cons)
            boundary_positions = self._compact_block_positions(boundary_positions, repair_cons)

        if (not self._has_overlap(boundary_positions) and
            self._accept_boundary_candidate(
                positions, boundary_positions, cons, b2b_connectivity, p2b_connectivity, pins_pos
            )):
            return boundary_positions
        return positions

    def _build_items(
        self,
        n: int,
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> List[_Item]:
        assigned = set()
        items: List[_Item] = []
        pin_centers = self._pin_preferred_centers(n, p2b_connectivity, pins_pos)

        for _, members in sorted(self._groups(cons["cluster"]).items()):
            members = self._cluster_order(members, cons, target_positions)
            offsets, item_w, item_h = self._build_cluster_slicing_layout(
                members, widths, heights, cons, target_positions, pin_centers
            )

            fixed_anchor = None
            if target_positions is not None and not self.ignore_preplaced_position:
                pre = next((b for b in members if cons["preplaced"][b]), None)
                if pre is not None:
                    ox, oy = offsets[pre]
                    fixed_anchor = (float(target_positions[pre, 0]) - ox, float(target_positions[pre, 1]) - oy)

            item_boundary = 0
            for b in members:
                item_boundary |= cons["boundary"][b]
            pref = self._average_center([pin_centers[b] for b in members if pin_centers[b] is not None])
            items.append(_Item(members, offsets, item_w, item_h, fixed_anchor, item_boundary, pref))
            assigned.update(members)

        for b in range(n):
            if b in assigned:
                continue
            fixed_anchor = None
            if target_positions is not None and cons["preplaced"][b] and not self.ignore_preplaced_position:
                fixed_anchor = (float(target_positions[b, 0]), float(target_positions[b, 1]))
            items.append(
                _Item(
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
        if target_positions is None or self.ignore_preplaced_position:
            return members
        pre = [b for b in members if cons["preplaced"][b]]
        fixed = [b for b in members if cons["fixed"][b] and not cons["preplaced"][b]]
        rest = [b for b in members if not cons["preplaced"][b] and not cons["fixed"][b]]
        pre.sort(key=lambda b: (float(target_positions[b, 0]), float(target_positions[b, 1])))
        return pre + fixed + rest

    def _build_cluster_slicing_layout(
        self,
        members: List[int],
        widths: List[float],
        heights: List[float],
        cons: Dict[str, List[int]],
        target_positions: Optional[torch.Tensor],
        pin_centers: List[Optional[Tuple[float, float]]],
    ) -> Tuple[Dict[int, Tuple[float, float]], float, float]:
        if len(members) == 1:
            b = members[0]
            return {b: (0.0, 0.0)}, widths[b], heights[b]

        if any(cons["preplaced"][b] for b in members) and not self.ignore_preplaced_position:
            return self._build_cluster_row_layout(members, widths, heights)

        size_fn = lambda b: (widths[b], heights[b])
        boundary_fn = lambda b: cons["boundary"][b]
        center_fn = lambda b: pin_centers[b]
        offsets, total_w, total_h = self._build_slicing_layout(members, size_fn, boundary_fn, center_fn)
        offsets, total_w, total_h = self._compact_relative_layout(offsets, members, size_fn)
        return offsets, total_w, total_h

    def _build_cluster_row_layout(
        self,
        members: List[int],
        widths: List[float],
        heights: List[float],
    ) -> Tuple[Dict[int, Tuple[float, float]], float, float]:
        offsets: Dict[int, Tuple[float, float]] = {}
        x_cursor = 0.0
        max_h = 0.0
        for b in members:
            offsets[b] = (x_cursor, 0.0)
            x_cursor += widths[b]
            max_h = max(max_h, heights[b])
        return offsets, x_cursor, max_h

    def _place_items(self, items: List[_Item], outline_w: float, outline_h: float) -> Dict[int, Tuple[float, float]]:
        anchors: Dict[int, Tuple[float, float]] = {}
        placed: List[Tuple[float, float, float, float, int]] = []

        anchored = [i for i, item in enumerate(items) if item.fixed_anchor is not None]
        free = [i for i, item in enumerate(items) if item.fixed_anchor is None]

        for idx in anchored:
            item = items[idx]
            ax, ay = item.fixed_anchor or (0.0, 0.0)
            anchors[idx] = (ax, ay)
            placed.append((ax, ay, item.width, item.height, idx))

        if not free:
            return anchors

        size_fn = lambda idx: (items[idx].width, items[idx].height)
        boundary_fn = lambda idx: items[idx].boundary_code
        center_fn = lambda idx: items[idx].preferred_center
        relative_anchors, free_w, free_h = self._build_slicing_layout(free, size_fn, boundary_fn, center_fn)
        relative_anchors, free_w, free_h = self._compact_relative_layout(relative_anchors, free, size_fn)

        translation = self._choose_tree_translation(
            items, free, relative_anchors, free_w, free_h, placed, outline_w, outline_h
        )
        if translation is None:
            return super()._place_items(items, outline_w, outline_h)

        tx, ty = translation
        for idx in free:
            rx, ry = relative_anchors[idx]
            anchors[idx] = (tx + rx, ty + ry)

        if self._item_layout_has_overlap(items, anchors):
            return super()._place_items(items, outline_w, outline_h)
        return anchors

    def _build_slicing_layout(
        self,
        node_ids: List[int],
        size_fn: Callable[[int], Tuple[float, float]],
        boundary_fn: Callable[[int], int],
        center_fn: Callable[[int], Optional[Tuple[float, float]]],
    ) -> Tuple[Dict[int, Tuple[float, float]], float, float]:
        if len(node_ids) == 1:
            node = node_ids[0]
            w, h = size_fn(node)
            return {node: (0.0, 0.0)}, w, h

        axis = self._choose_tree_axis(node_ids, size_fn, boundary_fn, center_fn)
        ordered = self._order_nodes(node_ids, axis, boundary_fn, center_fn)
        left_ids, right_ids = self._balanced_split(ordered, size_fn)

        left_layout, left_w, left_h = self._build_slicing_layout(left_ids, size_fn, boundary_fn, center_fn)
        right_layout, right_w, right_h = self._build_slicing_layout(right_ids, size_fn, boundary_fn, center_fn)

        merged = dict(left_layout)
        if axis == "V":
            for node, (x, y) in right_layout.items():
                merged[node] = (x + left_w, y)
            return merged, left_w + right_w, max(left_h, right_h)

        for node, (x, y) in right_layout.items():
            merged[node] = (x, y + left_h)
        return merged, max(left_w, right_w), left_h + right_h

    def _choose_tree_axis(
        self,
        node_ids: Iterable[int],
        size_fn: Callable[[int], Tuple[float, float]],
        boundary_fn: Callable[[int], int],
        center_fn: Callable[[int], Optional[Tuple[float, float]]],
    ) -> str:
        left_right = 0
        top_bottom = 0
        x_values: List[float] = []
        y_values: List[float] = []

        for node in node_ids:
            code = boundary_fn(node)
            if code & (1 | 2):
                left_right += 1
            if code & (4 | 8):
                top_bottom += 1
            center = center_fn(node)
            if center is not None:
                x_values.append(center[0])
                y_values.append(center[1])

        if left_right > top_bottom + 1:
            return "V"
        if top_bottom > left_right + 1:
            return "H"

        if len(x_values) >= 2 and len(y_values) >= 2:
            span_x = max(x_values) - min(x_values)
            span_y = max(y_values) - min(y_values)
            if abs(span_x - span_y) > 1e-6:
                return "V" if span_x >= span_y else "H"

        total_w = sum(size_fn(node)[0] for node in node_ids)
        total_h = sum(size_fn(node)[1] for node in node_ids)
        return "V" if total_w >= total_h else "H"

    def _order_nodes(
        self,
        node_ids: List[int],
        axis: str,
        boundary_fn: Callable[[int], int],
        center_fn: Callable[[int], Optional[Tuple[float, float]]],
    ) -> List[int]:
        if axis == "V":
            def key(node: int) -> Tuple[int, float]:
                code = boundary_fn(node)
                if code & 1:
                    group = 0
                elif code & 2:
                    group = 2
                else:
                    group = 1
                center = center_fn(node)
                pos = center[0] if center is not None else 0.0
                return group, pos
        else:
            def key(node: int) -> Tuple[int, float]:
                code = boundary_fn(node)
                if code & 8:
                    group = 0
                elif code & 4:
                    group = 2
                else:
                    group = 1
                center = center_fn(node)
                pos = center[1] if center is not None else 0.0
                return group, pos

        return sorted(node_ids, key=key)

    def _balanced_split(
        self,
        ordered: List[int],
        size_fn: Callable[[int], Tuple[float, float]],
    ) -> Tuple[List[int], List[int]]:
        if len(ordered) == 2:
            return [ordered[0]], [ordered[1]]

        areas = [size_fn(node)[0] * size_fn(node)[1] for node in ordered]
        total = sum(areas)
        prefix = 0.0
        best_idx = 1
        best_gap = float("inf")
        for idx in range(1, len(ordered)):
            prefix += areas[idx - 1]
            gap = abs(total - 2.0 * prefix)
            if gap < best_gap:
                best_gap = gap
                best_idx = idx
        return ordered[:best_idx], ordered[best_idx:]

    def _choose_tree_translation(
        self,
        items: List[_Item],
        free_ids: List[int],
        relative_anchors: Dict[int, Tuple[float, float]],
        free_w: float,
        free_h: float,
        placed: List[Tuple[float, float, float, float, int]],
        outline_w: float,
        outline_h: float,
    ) -> Optional[Tuple[float, float]]:
        xs = {0.0, max(0.0, outline_w - free_w)}
        ys = {0.0, max(0.0, outline_h - free_h)}

        pref_center = self._average_center([items[idx].preferred_center for idx in free_ids if items[idx].preferred_center is not None])
        if pref_center is not None:
            xs.add(max(0.0, pref_center[0] - free_w / 2.0))
            ys.add(max(0.0, pref_center[1] - free_h / 2.0))

        if any(items[idx].boundary_code & 1 for idx in free_ids):
            xs.add(0.0)
        if any(items[idx].boundary_code & 8 for idx in free_ids):
            ys.add(0.0)
        if any(items[idx].boundary_code & 2 for idx in free_ids):
            xs.add(max(0.0, outline_w - free_w))
        if any(items[idx].boundary_code & 4 for idx in free_ids):
            ys.add(max(0.0, outline_h - free_h))

        if placed:
            max_x = max(px + pw for px, _, pw, _, _ in placed)
            max_y = max(py + ph for _, py, _, ph, _ in placed)
            xs.update({max_x, max(0.0, max_x - free_w)})
            ys.update({max_y, max(0.0, max_y - free_h)})
            for px, py, pw, ph, _ in placed:
                xs.update({max(0.0, px - free_w), px + pw})
                ys.update({max(0.0, py - free_h), py + ph})

        x_values = sorted({round(x, 6) for x in xs})
        y_values = sorted({round(y, 6) for y in ys})

        if len(x_values) > 24:
            x_values = x_values[:12] + x_values[-12:]
        if len(y_values) > 24:
            y_values = y_values[:12] + y_values[-12:]

        best = None
        best_score = float("inf")
        for x in x_values:
            for y in y_values:
                if self._translated_layout_overlaps(items, free_ids, relative_anchors, x, y, placed):
                    continue
                score = self._translated_layout_score(
                    items, free_ids, relative_anchors, x, y, free_w, free_h, placed, outline_w, outline_h
                )
                if score < best_score:
                    best_score = score
                    best = (x, y)

        if best is not None:
            return best

        if placed:
            top_y = max(py + ph for _, py, _, ph, _ in placed)
            fallback = (0.0, top_y)
            if not self._translated_layout_overlaps(items, free_ids, relative_anchors, fallback[0], fallback[1], placed):
                return fallback

        return (0.0, 0.0) if not self._translated_layout_overlaps(items, free_ids, relative_anchors, 0.0, 0.0, placed) else None

    def _translated_layout_overlaps(
        self,
        items: List[_Item],
        free_ids: List[int],
        relative_anchors: Dict[int, Tuple[float, float]],
        tx: float,
        ty: float,
        placed: List[Tuple[float, float, float, float, int]],
    ) -> bool:
        for idx in free_ids:
            rx, ry = relative_anchors[idx]
            rect = (tx + rx, ty + ry, items[idx].width, items[idx].height)
            if self._rect_overlaps_any(rect, placed):
                return True
        return False

    def _translated_layout_score(
        self,
        items: List[_Item],
        free_ids: List[int],
        relative_anchors: Dict[int, Tuple[float, float]],
        tx: float,
        ty: float,
        free_w: float,
        free_h: float,
        placed: List[Tuple[float, float, float, float, int]],
        outline_w: float,
        outline_h: float,
    ) -> float:
        overflow = max(0.0, tx + free_w - outline_w) + max(0.0, ty + free_h - outline_h)
        pref_penalty = 0.0
        edge_bonus = 0.0

        rects = [(tx + relative_anchors[idx][0], ty + relative_anchors[idx][1], items[idx].width, items[idx].height)
                 for idx in free_ids]
        all_rects = [(x, y, w, h) for x, y, w, h, _ in placed] + rects

        xmin = min(r[0] for r in all_rects)
        ymin = min(r[1] for r in all_rects)
        xmax = max(r[0] + r[2] for r in all_rects)
        ymax = max(r[1] + r[3] for r in all_rects)
        bbox_area = (xmax - xmin) * (ymax - ymin)

        for idx in free_ids:
            rx, ry = relative_anchors[idx]
            item = items[idx]
            if item.preferred_center is not None:
                cx, cy = item.preferred_center
                pref_penalty += abs(tx + rx + item.width / 2.0 - cx) + abs(ty + ry + item.height / 2.0 - cy)
            if item.boundary_code:
                edge_bonus -= 40.0

        return overflow * 1_000_000.0 + pref_penalty * 0.16 + bbox_area * 0.04 + tx * 0.01 + ty * 0.1 + edge_bonus

    def _item_layout_has_overlap(
        self,
        items: List[_Item],
        anchors: Dict[int, Tuple[float, float]],
    ) -> bool:
        rects = []
        for idx, item in enumerate(items):
            if idx not in anchors:
                return True
            x, y = anchors[idx]
            rects.append((x, y, item.width, item.height))

        for i in range(len(rects)):
            x1, y1, w1, h1 = rects[i]
            for j in range(i + 1, len(rects)):
                x2, y2, w2, h2 = rects[j]
                if min(x1 + w1, x2 + w2) - max(x1, x2) > 1e-6 and min(y1 + h1, y2 + h2) - max(y1, y2) > 1e-6:
                    return True
        return False

    def _compact_relative_layout(
        self,
        anchors: Dict[int, Tuple[float, float]],
        node_ids: List[int],
        size_fn: Callable[[int], Tuple[float, float]],
        passes: int = 2,
    ) -> Tuple[Dict[int, Tuple[float, float]], float, float]:
        compacted = dict(anchors)
        for _ in range(passes):
            compacted = self._compact_relative_axis(compacted, node_ids, size_fn, axis=0)
            compacted = self._compact_relative_axis(compacted, node_ids, size_fn, axis=1)

        max_w = 0.0
        max_h = 0.0
        for node in node_ids:
            x, y = compacted[node]
            w, h = size_fn(node)
            max_w = max(max_w, x + w)
            max_h = max(max_h, y + h)
        return compacted, max_w, max_h

    def _compact_relative_axis(
        self,
        anchors: Dict[int, Tuple[float, float]],
        node_ids: List[int],
        size_fn: Callable[[int], Tuple[float, float]],
        axis: int,
    ) -> Dict[int, Tuple[float, float]]:
        compacted = dict(anchors)
        if axis == 0:
            order = sorted(node_ids, key=lambda node: (compacted[node][0], compacted[node][1]))
        else:
            order = sorted(node_ids, key=lambda node: (compacted[node][1], compacted[node][0]))

        for node in order:
            x, y = compacted[node]
            w, h = size_fn(node)
            candidate = 0.0
            for other in node_ids:
                if other == node:
                    continue
                ox, oy = compacted[other]
                ow, oh = size_fn(other)
                if axis == 0:
                    if oy < y + h - 1e-6 and oy + oh > y + 1e-6 and ox < x + 1e-9:
                        candidate = max(candidate, ox + ow)
                else:
                    if ox < x + w - 1e-6 and ox + ow > x + 1e-6 and oy < y + 1e-9:
                        candidate = max(candidate, oy + oh)
            if axis == 0:
                compacted[node] = (candidate, y)
            else:
                compacted[node] = (x, candidate)
        return compacted

    def _compact_item_anchors(
        self,
        items: List[_Item],
        anchors: Dict[int, Tuple[float, float]],
        passes: int = 3,
    ) -> Dict[int, Tuple[float, float]]:
        compacted = dict(anchors)
        movable = [idx for idx, item in enumerate(items) if item.fixed_anchor is None]
        if not movable:
            return compacted

        for _ in range(passes):
            compacted = self._compact_item_axis(items, compacted, movable, axis=0)
            compacted = self._compact_item_axis(items, compacted, movable, axis=1)
        return compacted

    def _compact_item_axis(
        self,
        items: List[_Item],
        anchors: Dict[int, Tuple[float, float]],
        movable: List[int],
        axis: int,
    ) -> Dict[int, Tuple[float, float]]:
        compacted = dict(anchors)
        if axis == 0:
            order = sorted(movable, key=lambda idx: (compacted[idx][0], compacted[idx][1]))
        else:
            order = sorted(movable, key=lambda idx: (compacted[idx][1], compacted[idx][0]))

        for idx in order:
            x, y = compacted[idx]
            w, h = items[idx].width, items[idx].height
            candidate = 0.0
            for other in range(len(items)):
                if other == idx:
                    continue
                ox, oy = compacted[other]
                ow, oh = items[other].width, items[other].height
                if axis == 0:
                    if oy < y + h - 1e-6 and oy + oh > y + 1e-6 and ox < x + 1e-9:
                        candidate = max(candidate, ox + ow)
                else:
                    if ox < x + w - 1e-6 and ox + ow > x + 1e-6 and oy < y + 1e-9:
                        candidate = max(candidate, oy + oh)
            if axis == 0:
                compacted[idx] = (candidate, y)
            else:
                compacted[idx] = (x, candidate)
        return compacted

    def _compact_block_positions(
        self,
        positions: List[Tuple[float, float, float, float]],
        cons: Dict[str, List[int]],
        passes: int = 3,
    ) -> List[Tuple[float, float, float, float]]:
        compacted = list(positions)
        movable = [i for i in range(len(positions)) if not cons["preplaced"][i]]
        if not movable:
            return compacted

        for _ in range(passes):
            compacted = self._compact_block_axis(compacted, movable, axis=0)
            compacted = self._compact_block_axis(compacted, movable, axis=1)
        return compacted

    def _compact_block_axis(
        self,
        positions: List[Tuple[float, float, float, float]],
        movable: List[int],
        axis: int,
    ) -> List[Tuple[float, float, float, float]]:
        compacted = list(positions)
        if axis == 0:
            order = sorted(movable, key=lambda i: (compacted[i][0], compacted[i][1]))
        else:
            order = sorted(movable, key=lambda i: (compacted[i][1], compacted[i][0]))

        for i in order:
            x, y, w, h = compacted[i]
            candidate = 0.0
            for j, (ox, oy, ow, oh) in enumerate(compacted):
                if i == j:
                    continue
                if axis == 0:
                    if oy < y + h - 1e-6 and oy + oh > y + 1e-6 and ox < x + 1e-9:
                        candidate = max(candidate, ox + ow)
                else:
                    if ox < x + w - 1e-6 and ox + ow > x + 1e-6 and oy < y + 1e-9:
                        candidate = max(candidate, oy + oh)
            if axis == 0:
                compacted[i] = (candidate, y, w, h)
            else:
                compacted[i] = (x, candidate, w, h)
        return compacted


if __name__ == "__main__":
    print("This file is meant to be loaded by iccad2026_evaluate.py")
