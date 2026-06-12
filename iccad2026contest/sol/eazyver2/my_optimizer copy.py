#!/usr/bin/env python3
"""
Grouping-as-item + boundary-post-placement heuristic.

Main rules for this version:
1. Grouping subproblems are built only from non-preplaced, non-boundary blocks.
2. Preplaced blocks ignore all soft constraints.
3. Boundary blocks are placed after the core layout.
4. MIB is still ignored.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

CONTEST_ROOT = Path(__file__).resolve().parents[2]
if str(CONTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEST_ROOT))

from iccad2026_evaluate import FloorplanOptimizer, calculate_bbox_area

EPS = 1e-9
GRID_EPS = 1e-7


class BlockSpec:
    def __init__(
        self,
        block_id: int,
        area: float,
        fixed: bool,
        preplaced: bool,
        width: Optional[float],
        height: Optional[float],
        x: Optional[float],
        y: Optional[float],
        group_id: int,
        boundary_code: int,
    ):
        self.block_id = block_id
        self.area = area
        self.fixed = fixed
        self.preplaced = preplaced
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.group_id = group_id
        self.boundary_code = boundary_code


class LayoutItem:
    def __init__(
        self,
        item_id: int,
        width: float,
        height: float,
        local_rects: Dict[int, Tuple[float, float, float, float]],
    ):
        self.item_id = item_id
        self.width = width
        self.height = height
        self.local_rects = local_rects
        self.area = width * height


class FreeRect:
    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def area(self) -> float:
        return self.w * self.h


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.beam_width = 22
        self.state_candidate_limit = 20
        self.local_passes = 3

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
        blocks = self._build_blocks(block_count, area_targets, constraints, target_positions)
        block_map = {block.block_id: block for block in blocks}

        group_items, grouped_ids = self._build_group_items(blocks)

        fixed_positions: Dict[int, Tuple[float, float, float, float]] = {}
        core_items: List[LayoutItem] = []
        soft_single_ids: List[int] = []
        boundary_ids: List[int] = []

        for block in blocks:
            if block.preplaced:
                fixed_positions[block.block_id] = (block.x, block.y, block.width, block.height)
                continue

            if block.boundary_code != 0:
                boundary_ids.append(block.block_id)
                continue

            if block.block_id in grouped_ids:
                continue

            if block.fixed:
                core_items.append(
                    LayoutItem(
                        item_id=block.block_id,
                        width=block.width,
                        height=block.height,
                        local_rects={block.block_id: (0.0, 0.0, block.width, block.height)},
                    )
                )
            else:
                soft_single_ids.append(block.block_id)

        core_items.extend(group_items)

        best_positions: Optional[List[Tuple[float, float, float, float]]] = None
        best_area = float("inf")

        if not core_items and not fixed_positions and soft_single_ids:
            core_positions = self._solve_soft_only([block_map[idx] for idx in soft_single_ids])
            core_rects = {block.block_id: core_positions[block.block_id] for block in [block_map[idx] for idx in soft_single_ids]}
            candidate = self._finalize_layout(blocks, core_rects, boundary_ids)
            best_positions = candidate
            best_area = calculate_bbox_area(candidate)
        else:
            item_layouts = self._search_item_layouts(core_items, fixed_positions)
            if not item_layouts:
                item_layouts = [{}]

            for item_layout in item_layouts:
                core_positions = dict(fixed_positions)
                for item in core_items:
                    rect = item_layout.get(item.item_id)
                    if rect is None:
                        continue
                    ix, iy, _, _ = rect
                    for block_id, (lx, ly, w, h) in item.local_rects.items():
                        core_positions[block_id] = (ix + lx, iy + ly, w, h)

                core_positions = self._fill_soft_singles(blocks, core_positions, soft_single_ids)
                candidate = self._finalize_layout(blocks, core_positions, boundary_ids)
                area = calculate_bbox_area(candidate)
                if area + EPS < best_area:
                    best_area = area
                    best_positions = candidate

        if best_positions is None:
            fallback_positions = dict(fixed_positions)
            fallback_positions = self._fill_soft_singles(blocks, fallback_positions, soft_single_ids)
            best_positions = self._finalize_layout(blocks, fallback_positions, boundary_ids)

        if not any(block.preplaced for block in blocks):
            best_positions = self._shift_to_origin(best_positions)

        return best_positions

    # ------------------------------------------------------------------
    # Input parsing
    # ------------------------------------------------------------------
    def _build_blocks(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
    ) -> List[BlockSpec]:
        blocks: List[BlockSpec] = []
        ncols = int(constraints.shape[1]) if constraints is not None and constraints.ndim == 2 else 0
        for i in range(block_count):
            area = float(area_targets[i]) if float(area_targets[i]) > 0 else 1.0
            is_fixed = bool(ncols > 0 and constraints[i, 0] != 0)
            is_preplaced = bool(ncols > 1 and constraints[i, 1] != 0)
            group_id = int(constraints[i, 3].item()) if ncols > 3 else 0
            boundary_code = int(constraints[i, 4].item()) if ncols > 4 else 0

            width = height = x = y = None
            if target_positions is not None and i < len(target_positions):
                tx, ty, tw, th = [float(v) for v in target_positions[i]]
                if is_preplaced:
                    x, y, width, height = tx, ty, tw, th
                elif is_fixed:
                    width, height = tw, th

            blocks.append(
                BlockSpec(
                    block_id=i,
                    area=area,
                    fixed=is_fixed,
                    preplaced=is_preplaced,
                    width=width,
                    height=height,
                    x=x,
                    y=y,
                    group_id=group_id,
                    boundary_code=boundary_code,
                )
            )
        return blocks

    # ------------------------------------------------------------------
    # Grouping subproblems
    # ------------------------------------------------------------------
    def _build_group_items(
        self,
        blocks: Sequence[BlockSpec],
    ) -> Tuple[List[LayoutItem], set]:
        groups: Dict[int, List[BlockSpec]] = {}
        for block in blocks:
            if block.group_id <= 0:
                continue
            if block.preplaced:
                continue
            if block.boundary_code != 0:
                continue
            groups.setdefault(block.group_id, []).append(block)

        items: List[LayoutItem] = []
        grouped_ids = set()
        next_item_id = len(blocks) + 1

        for group_id, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            item = self._make_group_item(next_item_id, members)
            next_item_id += 1
            items.append(item)
            grouped_ids.update(block.block_id for block in members)

        return items, grouped_ids

    def _make_group_item(
        self,
        item_id: int,
        members: Sequence[BlockSpec],
    ) -> LayoutItem:
        total_area = sum(block.area for block in members)
        all_soft = all(not block.fixed for block in members)
        candidates: List[Tuple[float, float, float, Dict[int, Tuple[float, float, float, float]]]] = []

        if all_soft:
            h = math.sqrt(max(total_area, 1.0))
            x = 0.0
            local_rects: Dict[int, Tuple[float, float, float, float]] = {}
            for block in sorted(members, key=lambda b: (-b.area, b.block_id)):
                w = block.area / h
                local_rects[block.block_id] = (x, 0.0, w, h)
                x += w
            candidates.append((x * h, abs(x - h), 0.0, local_rects))
        else:
            max_fixed_h = max((block.height for block in members if block.fixed), default=0.0)
            max_fixed_w = max((block.width for block in members if block.fixed), default=0.0)
            orders = [
                sorted(members, key=lambda b: (-b.area, b.block_id)),
                sorted(members, key=lambda b: (0 if b.fixed else 1, -b.area, b.block_id)),
                sorted(members, key=lambda b: (-((b.width or 0.0) * (b.height or 0.0)) if b.fixed else -b.area, b.block_id)),
            ]

            for order in orders:
                h = max(max_fixed_h, math.sqrt(max(total_area, 1.0)))
                x = 0.0
                local_rects_h: Dict[int, Tuple[float, float, float, float]] = {}
                for block in order:
                    if block.fixed:
                        w, bh = block.width, block.height
                    else:
                        bh = h
                        w = block.area / h
                    local_rects_h[block.block_id] = (x, 0.0, w, bh)
                    x += w
                candidates.append((x * h, abs(x - h), 0.0, local_rects_h))

                w = max(max_fixed_w, math.sqrt(max(total_area, 1.0)))
                y = 0.0
                local_rects_v: Dict[int, Tuple[float, float, float, float]] = {}
                for block in order:
                    if block.fixed:
                        bw, hh = block.width, block.height
                    else:
                        bw = w
                        hh = block.area / w
                    local_rects_v[block.block_id] = (0.0, y, bw, hh)
                    y += hh
                candidates.append((w * y, abs(w - y), 1.0, local_rects_v))

        best_local = min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]
        width, height = self._local_bbox(best_local.values())
        normalized = {
            block_id: (lx, ly, w, h)
            for block_id, (lx, ly, w, h) in best_local.items()
        }
        return LayoutItem(item_id=item_id, width=width, height=height, local_rects=normalized)

    # ------------------------------------------------------------------
    # Pure soft fallback
    # ------------------------------------------------------------------
    def _solve_soft_only(
        self,
        blocks: Sequence[BlockSpec],
    ) -> List[Tuple[float, float, float, float]]:
        total_area = sum(block.area for block in blocks)
        height = math.sqrt(max(total_area, 1.0))
        positions: List[Tuple[float, float, float, float]] = [(0.0, 0.0, 1.0, 1.0) for _ in range(max(block.block_id for block in blocks) + 1)]
        x = 0.0
        for block in sorted(blocks, key=lambda b: (-b.area, b.block_id)):
            w = block.area / height
            positions[block.block_id] = (x, 0.0, w, height)
            x += w
        return positions

    # ------------------------------------------------------------------
    # Item-level rigid search
    # ------------------------------------------------------------------
    def _search_item_layouts(
        self,
        items: Sequence[LayoutItem],
        fixed_positions: Dict[int, Tuple[float, float, float, float]],
    ) -> List[Dict[int, Tuple[float, float, float, float]]]:
        if not items:
            return [{}]

        item_map = {item.item_id: item for item in items}
        fixed_rects = list(fixed_positions.values())
        orders = self._make_item_orders(items, fixed_rects)
        best_states: List[Tuple[Tuple[float, float, float], Dict[int, Tuple[float, float, float, float]]]] = []

        for order in orders:
            states = [{}]
            for idx, item_id in enumerate(order):
                item = item_map[item_id]
                new_states: List[Tuple[Tuple[float, float, float], Dict[int, Tuple[float, float, float, float]]]] = []
                for state in states:
                    candidates = self._rank_item_candidates(
                        fixed_rects,
                        state,
                        item,
                        allow_origin=(idx == 0 and not state and not fixed_rects),
                    )
                    for x, y in candidates:
                        rect = (x, y, item.width, item.height)
                        if self._overlaps_any(rect, list(state.values()) + fixed_rects):
                            continue
                        new_state = dict(state)
                        new_state[item_id] = rect
                        new_states.append((self._state_score_occ(fixed_rects, new_state.values(), None), new_state))
                if not new_states:
                    fallback_state = dict(states[0])
                    rect = self._fallback_item_place(fixed_rects, fallback_state, item)
                    fallback_state[item_id] = rect
                    states = [fallback_state]
                else:
                    new_states.sort(key=lambda item_state: item_state[0])
                    states = [state for _, state in new_states[: self.beam_width]]

            for state in states[: min(8, len(states))]:
                improved = self._local_refine_items(fixed_rects, dict(state), item_map, order)
                best_states.append((self._state_score_occ(fixed_rects, improved.values(), None), improved))

        best_states.sort(key=lambda item_state: item_state[0])
        layouts: List[Dict[int, Tuple[float, float, float, float]]] = []
        seen = set()
        for _, layout in best_states:
            key = tuple((item_id, round(layout[item_id][0], 5), round(layout[item_id][1], 5)) for item_id in sorted(layout))
            if key in seen:
                continue
            seen.add(key)
            layouts.append(layout)
            if len(layouts) >= 10:
                break
        return layouts

    def _make_item_orders(
        self,
        items: Sequence[LayoutItem],
        fixed_rects: Sequence[Tuple[float, float, float, float]],
    ) -> List[List[int]]:
        fixed_bbox = self._bbox_xyxy(fixed_rects) if fixed_rects else (0.0, 0.0, 0.0, 0.0)

        def center_metric(item: LayoutItem) -> float:
            cx = 0.5 * (fixed_bbox[0] + fixed_bbox[2])
            cy = 0.5 * (fixed_bbox[1] + fixed_bbox[3])
            return abs(cx) + abs(cy) + abs(item.width - item.height)

        orders = [
            sorted(items, key=lambda item: (-item.area, -max(item.width, item.height), item.item_id)),
            sorted(items, key=lambda item: (-max(item.width, item.height), -item.area, item.item_id)),
            sorted(items, key=lambda item: (center_metric(item), -item.area, item.item_id)),
        ]

        unique_orders: List[List[int]] = []
        seen = set()
        for order in orders:
            key = tuple(item.item_id for item in order)
            if key not in seen:
                seen.add(key)
                unique_orders.append(list(key))
        return unique_orders

    def _rank_item_candidates(
        self,
        fixed_rects: Sequence[Tuple[float, float, float, float]],
        state: Dict[int, Tuple[float, float, float, float]],
        item: LayoutItem,
        allow_origin: bool = False,
    ) -> List[Tuple[float, float]]:
        if allow_origin:
            return [(0.0, 0.0)]

        occupied = list(fixed_rects) + list(state.values())
        if not occupied:
            return [(0.0, 0.0)]

        w, h = item.width, item.height
        x0, y0, x1, y1 = self._bbox_xyxy(occupied)
        xs = {x0, x1, x0 - w, x1 - w}
        ys = {y0, y1, y0 - h, y1 - h}
        for rx, ry, rw, rh in occupied:
            xs.update([rx - w, rx, rx + rw - w, rx + rw])
            ys.update([ry - h, ry, ry + rh - h, ry + rh])

        candidates = []
        seen = set()
        for x in xs:
            for y in ys:
                key = (round(x, 6), round(y, 6))
                if key in seen:
                    continue
                seen.add(key)
                rect = (x, y, w, h)
                if self._overlaps_any(rect, occupied):
                    continue
                score = self._state_score_occ(fixed_rects, state.values(), rect)
                candidates.append((score, x, y))

        candidates.sort(key=lambda item_state: item_state[0])
        return [(x, y) for _, x, y in candidates[: self.state_candidate_limit]]

    def _local_refine_items(
        self,
        fixed_rects: Sequence[Tuple[float, float, float, float]],
        state: Dict[int, Tuple[float, float, float, float]],
        item_map: Dict[int, LayoutItem],
        order: Sequence[int],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        for _ in range(self.local_passes):
            improved = False
            for item_id in order:
                current = state.pop(item_id)
                item = item_map[item_id]
                best_rect = current
                best_score = self._state_score_occ(fixed_rects, state.values(), current)
                candidates = [(current[0], current[1])] + self._rank_item_candidates(
                    fixed_rects,
                    state,
                    item,
                    allow_origin=(not state and not fixed_rects),
                )
                seen = set()
                for x, y in candidates:
                    key = (round(x, 6), round(y, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    rect = (x, y, item.width, item.height)
                    if self._overlaps_any(rect, list(state.values()) + list(fixed_rects)):
                        continue
                    score = self._state_score_occ(fixed_rects, state.values(), rect)
                    if score < best_score:
                        best_score = score
                        best_rect = rect
                state[item_id] = best_rect
                if best_rect != current:
                    improved = True
            if not improved:
                break
        return state

    def _fallback_item_place(
        self,
        fixed_rects: Sequence[Tuple[float, float, float, float]],
        state: Dict[int, Tuple[float, float, float, float]],
        item: LayoutItem,
    ) -> Tuple[float, float, float, float]:
        occupied = list(fixed_rects) + list(state.values())
        if not occupied:
            return (0.0, 0.0, item.width, item.height)

        x0, y0, x1, y1 = self._bbox_xyxy(occupied)
        steps = [
            (x1, y0),
            (x0 - item.width, y0),
            (x0, y1),
            (x0, y0 - item.height),
            (x1, y1),
            (x0 - item.width, y0 - item.height),
        ]
        best_rect = (x1, y0, item.width, item.height)
        best_score = self._state_score_occ(fixed_rects, state.values(), best_rect)
        for x, y in steps:
            rect = (x, y, item.width, item.height)
            if self._overlaps_any(rect, occupied):
                continue
            score = self._state_score_occ(fixed_rects, state.values(), rect)
            if score < best_score:
                best_score = score
                best_rect = rect
        return best_rect

    # ------------------------------------------------------------------
    # Core soft fill
    # ------------------------------------------------------------------
    def _fill_soft_singles(
        self,
        blocks: Sequence[BlockSpec],
        positions: Dict[int, Tuple[float, float, float, float]],
        soft_ids: Sequence[int],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        if not soft_ids:
            return positions

        block_map = {block.block_id: block for block in blocks}
        if not positions:
            total_area = sum(block_map[bid].area for bid in soft_ids)
            height = math.sqrt(max(total_area, 1.0))
            cursor_x = 0.0
            for block_id in sorted(soft_ids, key=lambda bid: (-block_map[bid].area, bid)):
                area = block_map[block_id].area
                width = area / height
                positions[block_id] = (cursor_x, 0.0, width, height)
                cursor_x += width
            return positions

        free_rects = self._extract_free_rectangles(positions.values())
        bins = [FreeRect(*rect) for rect in free_rects]
        unplaced: List[int] = []
        for block_id in sorted(soft_ids, key=lambda bid: (-block_map[bid].area, bid)):
            area = block_map[block_id].area
            chosen_idx = self._choose_bin_for_area(bins, area)
            if chosen_idx is None:
                unplaced.append(block_id)
                continue
            rect = bins[chosen_idx]
            placed, leftover = self._slice_rect(rect, area)
            positions[block_id] = placed
            if leftover is None or leftover.w <= EPS or leftover.h <= EPS:
                bins.pop(chosen_idx)
            else:
                bins[chosen_idx] = leftover

        if unplaced:
            self._pack_remaining_soft_strip(positions, block_map, unplaced)
        return positions

    def _extract_free_rectangles(
        self,
        rects: Iterable[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        rect_list = list(rects)
        if not rect_list:
            return []

        x0, y0, x1, y1 = self._bbox_xyxy(rect_list)
        xs = sorted({x0, x1, *[rx for rx, _, _, _ in rect_list], *[rx + rw for rx, _, rw, _ in rect_list]})
        ys = sorted({y0, y1, *[ry for _, ry, _, _ in rect_list], *[ry + rh for _, ry, _, rh in rect_list]})
        if len(xs) < 2 or len(ys) < 2:
            return []

        empty = [[False for _ in range(len(xs) - 1)] for _ in range(len(ys) - 1)]
        for j in range(len(ys) - 1):
            cy0, cy1 = ys[j], ys[j + 1]
            if cy1 - cy0 <= EPS:
                continue
            my = 0.5 * (cy0 + cy1)
            for i in range(len(xs) - 1):
                cx0, cx1 = xs[i], xs[i + 1]
                if cx1 - cx0 <= EPS:
                    continue
                mx = 0.5 * (cx0 + cx1)
                occupied = False
                for rx, ry, rw, rh in rect_list:
                    if (rx + GRID_EPS) < mx < (rx + rw - GRID_EPS) and (ry + GRID_EPS) < my < (ry + rh - GRID_EPS):
                        occupied = True
                        break
                empty[j][i] = not occupied

        strips: List[FreeRect] = []
        for j in range(len(ys) - 1):
            cy0, cy1 = ys[j], ys[j + 1]
            i = 0
            while i < len(xs) - 1:
                if not empty[j][i]:
                    i += 1
                    continue
                start = i
                while i < len(xs) - 1 and empty[j][i]:
                    i += 1
                strips.append(FreeRect(xs[start], cy0, xs[i] - xs[start], cy1 - cy0))

        strips.sort(key=lambda rect: (round(rect.x, 8), round(rect.w, 8), round(rect.y, 8)))
        merged: List[FreeRect] = []
        for rect in strips:
            if (
                merged
                and abs(merged[-1].x - rect.x) <= EPS
                and abs(merged[-1].w - rect.w) <= EPS
                and abs(merged[-1].y + merged[-1].h - rect.y) <= EPS
            ):
                merged[-1].h += rect.h
            else:
                merged.append(rect)
        merged = [rect for rect in merged if rect.w > EPS and rect.h > EPS]
        merged.sort(key=lambda rect: (-rect.area, rect.x, rect.y))
        return [(rect.x, rect.y, rect.w, rect.h) for rect in merged]

    def _choose_bin_for_area(self, bins: Sequence[FreeRect], area: float) -> Optional[int]:
        best_idx = None
        best_leftover = float("inf")
        for idx, rect in enumerate(bins):
            if rect.area + EPS < area:
                continue
            leftover = rect.area - area
            if leftover < best_leftover:
                best_leftover = leftover
                best_idx = idx
        return best_idx

    def _slice_rect(
        self,
        rect: FreeRect,
        area: float,
    ) -> Tuple[Tuple[float, float, float, float], Optional[FreeRect]]:
        if rect.w >= rect.h:
            height = area / rect.w
            placed = (rect.x, rect.y, rect.w, height)
            rem_h = rect.h - height
            leftover = None if rem_h <= EPS else FreeRect(rect.x, rect.y + height, rect.w, rem_h)
            return placed, leftover

        width = area / rect.h
        placed = (rect.x, rect.y, width, rect.h)
        rem_w = rect.w - width
        leftover = None if rem_w <= EPS else FreeRect(rect.x + width, rect.y, rem_w, rect.h)
        return placed, leftover

    def _pack_remaining_soft_strip(
        self,
        positions: Dict[int, Tuple[float, float, float, float]],
        block_map: Dict[int, BlockSpec],
        remaining_ids: Sequence[int],
    ) -> None:
        x0, y0, x1, y1 = self._bbox_xyxy(positions.values())
        width = x1 - x0
        height = y1 - y0
        remain_area = sum(block_map[bid].area for bid in remaining_ids)
        if remain_area <= EPS:
            return

        if width >= height:
            strip_h = remain_area / max(width, EPS)
            cursor_x = x0
            for block_id in remaining_ids:
                area = block_map[block_id].area
                rect_w = area / strip_h
                positions[block_id] = (cursor_x, y1, rect_w, strip_h)
                cursor_x += rect_w
        else:
            strip_w = remain_area / max(height, EPS)
            cursor_y = y0
            for block_id in remaining_ids:
                area = block_map[block_id].area
                rect_h = area / strip_w
                positions[block_id] = (x1, cursor_y, strip_w, rect_h)
                cursor_y += rect_h

    # ------------------------------------------------------------------
    # Boundary post-placement
    # ------------------------------------------------------------------
    def _finalize_layout(
        self,
        blocks: Sequence[BlockSpec],
        core_positions: Dict[int, Tuple[float, float, float, float]],
        boundary_ids: Sequence[int],
    ) -> List[Tuple[float, float, float, float]]:
        positions = dict(core_positions)
        if boundary_ids:
            positions = self._place_boundary_blocks(blocks, positions, boundary_ids)
        return self._dict_to_positions(blocks, positions)

    def _place_boundary_blocks(
        self,
        blocks: Sequence[BlockSpec],
        positions: Dict[int, Tuple[float, float, float, float]],
        boundary_ids: Sequence[int],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        block_map = {block.block_id: block for block in blocks}
        if positions:
            core_x0, core_y0, core_x1, core_y1 = self._bbox_xyxy(positions.values())
        else:
            core_x0 = core_y0 = core_x1 = core_y1 = 0.0

        left_blocks: List[BlockSpec] = []
        right_blocks: List[BlockSpec] = []
        top_blocks: List[BlockSpec] = []
        bottom_blocks: List[BlockSpec] = []
        corners: Dict[int, Optional[BlockSpec]] = {5: None, 6: None, 9: None, 10: None}

        for block_id in boundary_ids:
            block = block_map[block_id]
            code = block.boundary_code
            if code in corners:
                corners[code] = block
            elif code == 1:
                left_blocks.append(block)
            elif code == 2:
                right_blocks.append(block)
            elif code == 4:
                top_blocks.append(block)
            elif code == 8:
                bottom_blocks.append(block)
            else:
                # Unexpected multi-bit combinations are treated conservatively:
                # keep them on the boundary side implied by the first known bit.
                if code & 1:
                    left_blocks.append(block)
                elif code & 2:
                    right_blocks.append(block)
                elif code & 4:
                    top_blocks.append(block)
                elif code & 8:
                    bottom_blocks.append(block)

        core_w = core_x1 - core_x0
        core_h = core_y1 - core_y0
        corner_dims = self._choose_corner_dims(corners, core_w, core_h)

        w_tl, h_tl = corner_dims.get(5, (0.0, 0.0))
        w_tr, h_tr = corner_dims.get(6, (0.0, 0.0))
        w_bl, h_bl = corner_dims.get(9, (0.0, 0.0))
        w_br, h_br = corner_dims.get(10, (0.0, 0.0))

        left_extra = max(w_tl, w_bl)
        right_extra = max(w_tr, w_br)
        top_extra = max(h_tl, h_tr)
        bottom_extra = max(h_bl, h_br)

        L, R, T, B = left_extra, right_extra, top_extra, bottom_extra

        for _ in range(8):
            W = core_w + L + R
            H = core_h + T + B
            gap_left = max(H - h_tl - h_bl, 1e-6)
            gap_right = max(H - h_tr - h_br, 1e-6)
            gap_top = max(W - w_tl - w_tr, 1e-6)
            gap_bottom = max(W - w_bl - w_br, 1e-6)

            L_req, L_over = self._vertical_side_requirements(left_blocks, gap_left)
            R_req, R_over = self._vertical_side_requirements(right_blocks, gap_right)
            T_req, T_over = self._horizontal_side_requirements(top_blocks, gap_top)
            B_req, B_over = self._horizontal_side_requirements(bottom_blocks, gap_bottom)

            newL = max(left_extra, L_req)
            newR = max(right_extra, R_req)
            newT = max(top_extra, T_req)
            newB = max(bottom_extra, B_req)

            vertical_over = max(L_over, R_over)
            horizontal_over = max(T_over, B_over)
            if vertical_over > 0:
                newT += 0.5 * vertical_over
                newB += 0.5 * vertical_over
            if horizontal_over > 0:
                newL += 0.5 * horizontal_over
                newR += 0.5 * horizontal_over

            if (
                abs(newL - L) < 1e-6
                and abs(newR - R) < 1e-6
                and abs(newT - T) < 1e-6
                and abs(newB - B) < 1e-6
            ):
                L, R, T, B = newL, newR, newT, newB
                break
            L, R, T, B = newL, newR, newT, newB

        X0 = core_x0 - L
        X1 = core_x1 + R
        Y0 = core_y0 - B
        Y1 = core_y1 + T

        for code, block in corners.items():
            if block is None:
                continue
            w, h = corner_dims[code]
            if code == 5:
                positions[block.block_id] = (X0, Y1 - h, w, h)
            elif code == 6:
                positions[block.block_id] = (X1 - w, Y1 - h, w, h)
            elif code == 9:
                positions[block.block_id] = (X0, Y0, w, h)
            elif code == 10:
                positions[block.block_id] = (X1 - w, Y0, w, h)

        gap_left = max(Y1 - h_tl - (Y0 + h_bl), 0.0)
        gap_right = max(Y1 - h_tr - (Y0 + h_br), 0.0)
        gap_top = max(X1 - w_tr - (X0 + w_tl), 0.0)
        gap_bottom = max(X1 - w_br - (X0 + w_bl), 0.0)

        self._place_vertical_side(left_blocks, X0, Y0 + h_bl, gap_left, L, positions)
        self._place_vertical_side(right_blocks, X1, Y0 + h_br, gap_right, R, positions, right_align=True)
        self._place_horizontal_side(top_blocks, X0 + w_tl, Y1, gap_top, T, positions, top_align=True)
        self._place_horizontal_side(bottom_blocks, X0 + w_bl, Y0, gap_bottom, B, positions, top_align=False)

        return positions

    def _choose_corner_dims(
        self,
        corners: Dict[int, Optional[BlockSpec]],
        core_w: float,
        core_h: float,
    ) -> Dict[int, Tuple[float, float]]:
        dims: Dict[int, Tuple[float, float]] = {}
        ratio = max(core_w, 1.0) / max(core_h, 1.0)
        for code, block in corners.items():
            if block is None:
                dims[code] = (0.0, 0.0)
                continue
            if block.fixed:
                dims[code] = (block.width, block.height)
            else:
                width = math.sqrt(max(block.area * ratio, 1e-6))
                height = block.area / max(width, EPS)
                dims[code] = (width, height)
        return dims

    def _vertical_side_requirements(
        self,
        blocks: Sequence[BlockSpec],
        gap_h: float,
    ) -> Tuple[float, float]:
        fixed_blocks = [block for block in blocks if block.fixed]
        soft_blocks = [block for block in blocks if not block.fixed]
        fixed_h = sum(block.height for block in fixed_blocks)
        max_fixed_w = max((block.width for block in fixed_blocks), default=0.0)
        soft_area = sum(block.area for block in soft_blocks)

        overflow = max(0.0, fixed_h - gap_h)
        remaining_h = max(gap_h - fixed_h, 1e-4)
        soft_w = soft_area / remaining_h if soft_area > 0 else 0.0
        return max(max_fixed_w, soft_w), overflow

    def _horizontal_side_requirements(
        self,
        blocks: Sequence[BlockSpec],
        gap_w: float,
    ) -> Tuple[float, float]:
        fixed_blocks = [block for block in blocks if block.fixed]
        soft_blocks = [block for block in blocks if not block.fixed]
        fixed_w = sum(block.width for block in fixed_blocks)
        max_fixed_h = max((block.height for block in fixed_blocks), default=0.0)
        soft_area = sum(block.area for block in soft_blocks)

        overflow = max(0.0, fixed_w - gap_w)
        remaining_w = max(gap_w - fixed_w, 1e-4)
        soft_h = soft_area / remaining_w if soft_area > 0 else 0.0
        return max(max_fixed_h, soft_h), overflow

    def _place_vertical_side(
        self,
        blocks: Sequence[BlockSpec],
        side_x: float,
        start_y: float,
        gap_h: float,
        strip_w: float,
        positions: Dict[int, Tuple[float, float, float, float]],
        right_align: bool = False,
    ) -> None:
        cursor_y = start_y
        ordered = sorted(blocks, key=lambda block: (0 if block.fixed else 1, -block.area, block.block_id))
        for block in ordered:
            if block.fixed:
                w, h = block.width, block.height
            else:
                w = strip_w
                h = block.area / max(w, EPS)
            x = side_x - w if right_align else side_x
            positions[block.block_id] = (x, cursor_y, w, h)
            cursor_y += h

    def _place_horizontal_side(
        self,
        blocks: Sequence[BlockSpec],
        start_x: float,
        side_y: float,
        gap_w: float,
        strip_h: float,
        positions: Dict[int, Tuple[float, float, float, float]],
        top_align: bool,
    ) -> None:
        cursor_x = start_x
        ordered = sorted(blocks, key=lambda block: (0 if block.fixed else 1, -block.area, block.block_id))
        for block in ordered:
            if block.fixed:
                w, h = block.width, block.height
            else:
                h = strip_h
                w = block.area / max(h, EPS)
            y = side_y - h if top_align else side_y
            positions[block.block_id] = (cursor_x, y, w, h)
            cursor_x += w

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _local_bbox(
        self,
        rects: Iterable[Tuple[float, float, float, float]],
    ) -> Tuple[float, float]:
        rect_list = list(rects)
        if not rect_list:
            return (0.0, 0.0)
        x0 = min(rect[0] for rect in rect_list)
        y0 = min(rect[1] for rect in rect_list)
        x1 = max(rect[0] + rect[2] for rect in rect_list)
        y1 = max(rect[1] + rect[3] for rect in rect_list)
        return (x1 - x0, y1 - y0)

    def _dict_to_positions(
        self,
        blocks: Sequence[BlockSpec],
        rects: Dict[int, Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        positions: List[Tuple[float, float, float, float]] = [(0.0, 0.0, 1.0, 1.0) for _ in blocks]
        for block in blocks:
            rect = rects.get(block.block_id)
            if rect is None:
                if block.preplaced:
                    rect = (block.x, block.y, block.width, block.height)
                elif block.fixed:
                    rect = (0.0, 0.0, block.width, block.height)
                else:
                    side = math.sqrt(block.area)
                    rect = (0.0, 0.0, side, side)
            positions[block.block_id] = tuple(float(v) for v in rect)
        return positions

    def _bbox_xyxy(
        self,
        rects: Iterable[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float, float]:
        rect_list = list(rects)
        if not rect_list:
            return (0.0, 0.0, 0.0, 0.0)
        x0 = min(rect[0] for rect in rect_list)
        y0 = min(rect[1] for rect in rect_list)
        x1 = max(rect[0] + rect[2] for rect in rect_list)
        y1 = max(rect[1] + rect[3] for rect in rect_list)
        return x0, y0, x1, y1

    def _state_score_occ(
        self,
        fixed_rects: Sequence[Tuple[float, float, float, float]],
        state_rects: Iterable[Tuple[float, float, float, float]],
        extra: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float]:
        rects = list(fixed_rects) + list(state_rects)
        if extra is not None:
            rects.append(extra)
        if not rects:
            return (0.0, 0.0, 0.0)
        x0, y0, x1, y1 = self._bbox_xyxy(rects)
        w = x1 - x0
        h = y1 - y0
        return (w * h, w + h, max(w, h))

    def _overlaps_any(
        self,
        rect: Tuple[float, float, float, float],
        others: Iterable[Tuple[float, float, float, float]],
    ) -> bool:
        x1, y1, w1, h1 = rect
        for x2, y2, w2, h2 in others:
            overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
            overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
            if overlap_x > 1e-6 and overlap_y > 1e-6:
                return True
        return False

    def _shift_to_origin(
        self,
        positions: Sequence[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        if not positions:
            return []
        x0 = min(x for x, _, _, _ in positions)
        y0 = min(y for _, y, _, _ in positions)
        return [(x - x0, y - y0, w, h) for x, y, w, h in positions]
