#!/usr/bin/env python3
"""
Simplified optimizer for the hard-constraint-only + bbox-area-only variant.

Strategy:
1. Keep all preplaced blocks fixed.
2. Search only over movable rigid blocks (fixed-shape, non-preplaced).
3. Fill rigid-bbox holes with soft blocks when possible.
4. Pack remaining soft blocks into one outer strip.

This intentionally ignores HPWL and all soft constraints.
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
    ):
        self.block_id = block_id
        self.area = area
        self.fixed = fixed
        self.preplaced = preplaced
        self.width = width
        self.height = height
        self.x = x
        self.y = y


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
        self.beam_width = 20
        self.state_candidate_limit = 18
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
        rigid_rects: Dict[int, Tuple[float, float, float, float]] = {}
        movable_rigid_ids: List[int] = []
        soft_ids: List[int] = []

        for block in blocks:
            if block.preplaced:
                rigid_rects[block.block_id] = (block.x, block.y, block.width, block.height)
            elif block.fixed:
                movable_rigid_ids.append(block.block_id)
            else:
                soft_ids.append(block.block_id)

        if not rigid_rects and not movable_rigid_ids:
            return self._solve_soft_only(blocks)

        best_positions: Optional[List[Tuple[float, float, float, float]]] = None
        best_area = float("inf")

        rigid_candidates = self._search_rigid_layouts(blocks, rigid_rects, movable_rigid_ids)
        for rigid_layout in rigid_candidates:
            positions = self._build_full_layout(blocks, rigid_layout, soft_ids)
            area = calculate_bbox_area(positions)
            if area + EPS < best_area:
                best_area = area
                best_positions = positions

        if best_positions is None:
            best_positions = self._build_full_layout(blocks, rigid_rects, soft_ids)

        if not any(block.preplaced for block in blocks):
            best_positions = self._shift_to_origin(best_positions)

        return best_positions

    # -------------------------------------------------------------------------
    # Block parsing
    # -------------------------------------------------------------------------
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
                )
            )
        return blocks

    # -------------------------------------------------------------------------
    # Pure soft case: exact lower bound
    # -------------------------------------------------------------------------
    def _solve_soft_only(self, blocks: Sequence[BlockSpec]) -> List[Tuple[float, float, float, float]]:
        total_area = sum(block.area for block in blocks)
        height = math.sqrt(max(total_area, 1.0))
        if height < EPS:
            height = 1.0

        positions: List[Tuple[float, float, float, float]] = [None] * len(blocks)  # type: ignore
        cursor_x = 0.0
        for block in sorted(blocks, key=lambda b: (-b.area, b.block_id)):
            width = block.area / height
            positions[block.block_id] = (cursor_x, 0.0, width, height)
            cursor_x += width
        return positions

    # -------------------------------------------------------------------------
    # Rigid layout search
    # -------------------------------------------------------------------------
    def _search_rigid_layouts(
        self,
        blocks: Sequence[BlockSpec],
        base_rigid_rects: Dict[int, Tuple[float, float, float, float]],
        movable_rigid_ids: Sequence[int],
    ) -> List[Dict[int, Tuple[float, float, float, float]]]:
        if not movable_rigid_ids:
            return [dict(base_rigid_rects)]

        block_map = {block.block_id: block for block in blocks}
        orders = self._make_orders(block_map, movable_rigid_ids, base_rigid_rects)
        best_states: List[Tuple[Tuple[float, float, float], Dict[int, Tuple[float, float, float, float]]]] = []

        for order in orders:
            states = [dict(base_rigid_rects)]
            for idx, block_id in enumerate(order):
                spec = block_map[block_id]
                new_states: List[Tuple[Tuple[float, float, float], Dict[int, Tuple[float, float, float, float]]]] = []
                for state in states:
                    candidates = self._rank_candidates_for_state(state, spec, allow_origin=(idx == 0 and not state))
                    for x, y in candidates:
                        rect = (x, y, spec.width, spec.height)
                        if self._overlaps_any(rect, state.values()):
                            continue
                        new_state = dict(state)
                        new_state[block_id] = rect
                        new_states.append((self._state_score(new_state), new_state))
                if not new_states:
                    fallback_state = dict(states[0])
                    rect = self._fallback_place(fallback_state, spec)
                    fallback_state[block_id] = rect
                    states = [fallback_state]
                else:
                    new_states.sort(key=lambda item: item[0])
                    states = [state for _, state in new_states[: self.beam_width]]

            for state in states[: min(6, len(states))]:
                improved = self._local_refine(dict(state), block_map, movable_rigid_ids)
                best_states.append((self._state_score(improved), improved))

        best_states.sort(key=lambda item: item[0])
        unique_layouts: List[Dict[int, Tuple[float, float, float, float]]] = []
        seen = set()
        for _, layout in best_states:
            key = tuple(
                (bid, round(layout[bid][0], 5), round(layout[bid][1], 5))
                for bid in sorted(layout)
            )
            if key in seen:
                continue
            seen.add(key)
            unique_layouts.append(layout)
            if len(unique_layouts) >= 8:
                break
        return unique_layouts or [dict(base_rigid_rects)]

    def _make_orders(
        self,
        block_map: Dict[int, BlockSpec],
        movable_ids: Sequence[int],
        base_rigid_rects: Dict[int, Tuple[float, float, float, float]],
    ) -> List[List[int]]:
        def center_distance(block_id: int) -> float:
            if not base_rigid_rects:
                return 0.0
            x0, y0, x1, y1 = self._bbox_xyxy(base_rigid_rects.values())
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            block = block_map[block_id]
            return abs(cx) + abs(cy) + abs(block.width - block.height)

        orders = [
            sorted(movable_ids, key=lambda bid: (-block_map[bid].area, -max(block_map[bid].width, block_map[bid].height), bid)),
            sorted(movable_ids, key=lambda bid: (-max(block_map[bid].width, block_map[bid].height), -block_map[bid].area, bid)),
            sorted(movable_ids, key=lambda bid: (-block_map[bid].width, -block_map[bid].height, -block_map[bid].area, bid)),
            sorted(movable_ids, key=lambda bid: (-block_map[bid].height, -block_map[bid].width, -block_map[bid].area, bid)),
            sorted(movable_ids, key=lambda bid: (center_distance(bid), -block_map[bid].area, bid)),
        ]

        unique_orders: List[List[int]] = []
        seen = set()
        for order in orders:
            key = tuple(order)
            if key not in seen:
                seen.add(key)
                unique_orders.append(order)
        return unique_orders

    def _rank_candidates_for_state(
        self,
        state: Dict[int, Tuple[float, float, float, float]],
        spec: BlockSpec,
        allow_origin: bool = False,
    ) -> List[Tuple[float, float]]:
        w, h = spec.width, spec.height
        if allow_origin:
            return [(0.0, 0.0)]

        if not state:
            return [(0.0, 0.0)]

        xs = set()
        ys = set()
        x0, y0, x1, y1 = self._bbox_xyxy(state.values())
        xs.update([x0, x1, x0 - w, x1 - w])
        ys.update([y0, y1, y0 - h, y1 - h])

        for rx, ry, rw, rh in state.values():
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
                if self._overlaps_any(rect, state.values()):
                    continue
                score = self._state_score_with_extra(state.values(), rect)
                candidates.append((score, x, y))

        candidates.sort(key=lambda item: item[0])
        return [(x, y) for _, x, y in candidates[: self.state_candidate_limit]]

    def _local_refine(
        self,
        rects: Dict[int, Tuple[float, float, float, float]],
        block_map: Dict[int, BlockSpec],
        movable_ids: Sequence[int],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        ordered = sorted(movable_ids, key=lambda bid: (-block_map[bid].area, bid))
        for _ in range(self.local_passes):
            improved = False
            for block_id in ordered:
                current = rects.pop(block_id)
                spec = block_map[block_id]
                best_rect = current
                best_score = self._state_score_with_extra(rects.values(), current)
                candidates = self._rank_candidates_for_state(rects, spec, allow_origin=(not rects))
                candidates = [(current[0], current[1])] + candidates
                seen = set()
                for x, y in candidates:
                    key = (round(x, 6), round(y, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    rect = (x, y, spec.width, spec.height)
                    if self._overlaps_any(rect, rects.values()):
                        continue
                    score = self._state_score_with_extra(rects.values(), rect)
                    if score < best_score:
                        best_score = score
                        best_rect = rect
                rects[block_id] = best_rect
                if best_rect != current:
                    improved = True
            if not improved:
                break
        return rects

    def _fallback_place(
        self,
        state: Dict[int, Tuple[float, float, float, float]],
        spec: BlockSpec,
    ) -> Tuple[float, float, float, float]:
        w, h = spec.width, spec.height
        if not state:
            return (0.0, 0.0, w, h)

        x0, y0, x1, y1 = self._bbox_xyxy(state.values())
        steps = [
            (x1, y0),
            (x0 - w, y0),
            (x0, y1),
            (x0, y0 - h),
            (x1, y1),
            (x0 - w, y0 - h),
        ]
        best_rect = (x1, y0, w, h)
        best_score = self._state_score_with_extra(state.values(), best_rect)
        for x, y in steps:
            rect = (x, y, w, h)
            if self._overlaps_any(rect, state.values()):
                continue
            score = self._state_score_with_extra(state.values(), rect)
            if score < best_score:
                best_score = score
                best_rect = rect
        return best_rect

    # -------------------------------------------------------------------------
    # Soft fill
    # -------------------------------------------------------------------------
    def _build_full_layout(
        self,
        blocks: Sequence[BlockSpec],
        rigid_rects: Dict[int, Tuple[float, float, float, float]],
        soft_ids: Sequence[int],
    ) -> List[Tuple[float, float, float, float]]:
        positions: Dict[int, Tuple[float, float, float, float]] = dict(rigid_rects)
        block_map = {block.block_id: block for block in blocks}

        if not soft_ids:
            return self._dict_to_positions(blocks, positions)

        if not rigid_rects:
            return self._solve_soft_only(blocks)

        free_rects = self._extract_free_rectangles(rigid_rects.values())
        remaining = sorted(soft_ids, key=lambda bid: (-block_map[bid].area, bid))
        bins = [FreeRect(*rect) for rect in free_rects]

        unplaced: List[int] = []
        for block_id in remaining:
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

        return self._dict_to_positions(blocks, positions)

    def _extract_free_rectangles(
        self,
        rigid_rects: Iterable[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        rects = list(rigid_rects)
        if not rects:
            return []

        x0, y0, x1, y1 = self._bbox_xyxy(rects)
        xs = sorted({x0, x1, *[rx for rx, _, _, _ in rects], *[rx + rw for rx, _, rw, _ in rects]})
        ys = sorted({y0, y1, *[ry for _, ry, _, _ in rects], *[ry + rh for _, ry, _, rh in rects]})
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
                for rx, ry, rw, rh in rects:
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

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------
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

    def _state_score(
        self,
        rects: Dict[int, Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float]:
        return self._state_score_with_extra(rects.values(), None)

    def _state_score_with_extra(
        self,
        rects: Iterable[Tuple[float, float, float, float]],
        extra: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float]:
        items = list(rects)
        if extra is not None:
            items.append(extra)
        if not items:
            return (0.0, 0.0, 0.0)
        x0, y0, x1, y1 = self._bbox_xyxy(items)
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
