#!/usr/bin/env python3
"""
BF3: grouping-as-item + MIB templates + region PAVA soft packing.

Main rules for this version:
1. Grouping subproblems are built only from non-preplaced, non-boundary blocks;
   boundary+grouping follows boundary and accepts grouping fragmentation.
2. Preplaced blocks ignore all soft constraints and are never moved.
3. Boundary blocks are placed after the core layout.
4. Blocks in the same MIB group share a template only when doing so keeps
   all hard area/fixed/preplaced constraints legal.
5. Standalone soft blocks are assigned to free regions and solved with a
   fixed-order 1-D L1 PAVA placement.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

CONTEST_ROOT = Path(__file__).resolve().parents[2]
if str(CONTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEST_ROOT))

try:
    from __main__ import FloorplanOptimizer, calculate_bbox_area
except ImportError:
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
        mib_id: int,
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
        self.mib_id = mib_id


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
        self.beam_width = 1
        self.state_candidate_limit = 1
        self.local_passes = 1

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
        self._apply_mib_templates(blocks)
        block_map = {block.block_id: block for block in blocks}

        group_items, grouped_ids = self._build_group_items(blocks)

        fixed_positions: Dict[int, Tuple[float, float, float, float]] = {}
        core_items: List[LayoutItem] = []
        soft_single_ids: List[int] = []
        boundary_ids: List[int] = []
        has_preplaced = False

        for block in blocks:
            if block.preplaced:
                has_preplaced = True
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
                if block.mib_id > 0 and block.width is not None and block.height is not None:
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
            soft_blocks = [block_map[idx] for idx in soft_single_ids]
            core_positions = self._solve_soft_only(soft_blocks)
            core_rects = {block.block_id: core_positions[block.block_id] for block in soft_blocks}
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

                core_positions = self._fill_soft_singles(
                    block_map,
                    core_positions,
                    soft_single_ids,
                    b2b_connectivity,
                    p2b_connectivity,
                    pins_pos,
                )
                candidate = self._finalize_layout(blocks, core_positions, boundary_ids)
                area = calculate_bbox_area(candidate)
                if area + EPS < best_area:
                    best_area = area
                    best_positions = candidate

        if best_positions is None:
            fallback_positions = dict(fixed_positions)
            fallback_positions = self._fill_soft_singles(
                block_map,
                fallback_positions,
                soft_single_ids,
                b2b_connectivity,
                p2b_connectivity,
                pins_pos,
            )
            best_positions = self._finalize_layout(blocks, fallback_positions, boundary_ids)

        if not has_preplaced:
            best_positions = self._shift_to_origin(best_positions)

        # Final safety net: resolve any remaining overlaps
        best_positions = self._resolve_all_overlaps(best_positions, blocks)

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
            mib_id = int(constraints[i, 2].item()) if ncols > 2 else 0
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
                    mib_id=mib_id,
                )
            )
        return blocks

    def _apply_mib_templates(
        self,
        blocks: Sequence[BlockSpec],
    ) -> None:
        mib_groups: Dict[int, List[BlockSpec]] = {}
        for block in blocks:
            if block.mib_id > 0:
                mib_groups.setdefault(block.mib_id, []).append(block)

        for _, members in mib_groups.items():
            hard_members = [block for block in members if block.preplaced or block.fixed]
            template_w = template_h = None

            if hard_members:
                # Hard constraints win.  A hard member can only define a MIB
                # template for movable members whose area target is compatible.
                ref = hard_members[0]
                ref_area = ref.width * ref.height
                if all(abs(block.width * block.height - ref_area) <= max(1e-6, 0.01 * ref_area)
                       for block in hard_members):
                    template_w = ref.width
                    template_h = ref.height
            else:
                areas = [block.area for block in members]
                base_area = areas[0]
                if all(abs(area - base_area) <= max(1e-6, 0.01 * base_area) for area in areas):
                    side = math.sqrt(max(base_area, 1e-6))
                    template_w = side
                    template_h = side

            if template_w is None or template_h is None:
                continue

            template_area = template_w * template_h
            for block in members:
                if block.preplaced or block.fixed:
                    continue
                if abs(template_area - block.area) > max(1e-6, 0.01 * block.area):
                    continue
                block.width = template_w
                block.height = template_h
                block.fixed = True

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
        fixed_rects = tuple(fixed_positions.values())
        fixed_bbox = self._bbox_xyxy(fixed_rects) if fixed_rects else None
        orders = self._make_item_orders(items, fixed_rects)
        best_states: List[Tuple[Tuple[float, float, float], Dict[int, Tuple[float, float, float, float]]]] = []

        for order in orders:
            states: List[
                Tuple[
                    Dict[int, Tuple[float, float, float, float]],
                    Tuple[Tuple[float, float, float, float], ...],
                    Optional[Tuple[float, float, float, float]],
                ]
            ] = [({}, (), fixed_bbox)]
            for idx, item_id in enumerate(order):
                item = item_map[item_id]
                new_states: List[
                    Tuple[
                        Tuple[float, float, float],
                        Dict[int, Tuple[float, float, float, float]],
                        Tuple[Tuple[float, float, float, float], ...],
                        Optional[Tuple[float, float, float, float]],
                    ]
                ] = []
                for state, state_rects, state_bbox in states:
                    occupied = fixed_rects + state_rects
                    candidates = self._rank_item_candidates(
                        occupied,
                        state_bbox,
                        item,
                        allow_origin=(idx == 0 and not state_rects and not fixed_rects),
                    )
                    for score, x, y in candidates:
                        rect = (x, y, item.width, item.height)
                        new_state = dict(state)
                        new_state[item_id] = rect
                        new_states.append(
                            (
                                score,
                                new_state,
                                state_rects + (rect,),
                                self._expand_bbox(state_bbox, rect),
                            )
                        )
                if not new_states:
                    fallback_state, fallback_rects, fallback_bbox = states[0]
                    fallback_state = dict(fallback_state)
                    rect = self._fallback_item_place(fixed_rects + fallback_rects, fallback_bbox, item)
                    fallback_state[item_id] = rect
                    states = [
                        (
                            fallback_state,
                            fallback_rects + (rect,),
                            self._expand_bbox(fallback_bbox, rect),
                        )
                    ]
                else:
                    new_states.sort(key=lambda item_state: item_state[0])
                    states = [
                        (state, state_rects, state_bbox)
                        for _, state, state_rects, state_bbox in new_states[: self.beam_width]
                    ]

            for state, _, _ in states[: min(8, len(states))]:
                improved = self._local_refine_items(fixed_rects, fixed_bbox, dict(state), item_map, order)
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
        occupied: Sequence[Tuple[float, float, float, float]],
        occupied_bbox: Optional[Tuple[float, float, float, float]],
        item: LayoutItem,
        allow_origin: bool = False,
    ) -> List[Tuple[Tuple[float, float, float], float, float]]:
        w, h = item.width, item.height
        if allow_origin:
            return [(self._score_bbox_with_rect(occupied_bbox, (0.0, 0.0, w, h)), 0.0, 0.0)]

        if not occupied:
            return [(self._score_bbox_with_rect(None, (0.0, 0.0, w, h)), 0.0, 0.0)]

        x0, y0, x1, y1 = occupied_bbox if occupied_bbox is not None else self._bbox_xyxy(occupied)
        xs = {x0, x1, x0 - w, x1 - w}
        ys = {y0, y1, y0 - h, y1 - h}
        for rx, ry, rw, rh in occupied:
            xs.update([rx - w, rx, rx + rw - w, rx + rw])
            ys.update([ry - h, ry, ry + rh - h, ry + rh])

        candidates = []
        seen = set()
        overlaps_any = self._overlaps_any
        score_bbox_with_rect = self._score_bbox_with_rect
        for x in xs:
            for y in ys:
                key = (round(x, 6), round(y, 6))
                if key in seen:
                    continue
                seen.add(key)
                rect = (x, y, w, h)
                if overlaps_any(rect, occupied):
                    continue
                candidates.append((score_bbox_with_rect(occupied_bbox, rect), x, y))

        candidates.sort(key=lambda item_state: item_state[0])
        return candidates[: self.state_candidate_limit]

    def _local_refine_items(
        self,
        fixed_rects: Sequence[Tuple[float, float, float, float]],
        fixed_bbox: Optional[Tuple[float, float, float, float]],
        state: Dict[int, Tuple[float, float, float, float]],
        item_map: Dict[int, LayoutItem],
        order: Sequence[int],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        for _ in range(self.local_passes):
            improved = False
            for item_id in order:
                current = state.pop(item_id)
                item = item_map[item_id]
                other_rects = tuple(state.values())
                occupied = fixed_rects + other_rects
                occupied_bbox = fixed_bbox if not other_rects else self._bbox_xyxy(occupied)
                best_rect = current
                if self._overlaps_any(current, occupied):
                    best_score = (float("inf"), float("inf"), float("inf"))
                    candidates: List[Tuple[Tuple[float, float, float], float, float]] = []
                else:
                    best_score = self._score_bbox_with_rect(occupied_bbox, current)
                    candidates = [(best_score, current[0], current[1])]
                candidates.extend(
                    self._rank_item_candidates(
                        occupied,
                        occupied_bbox,
                        item,
                        allow_origin=(not occupied),
                    )
                )
                seen = set()
                for score, x, y in candidates:
                    key = (round(x, 6), round(y, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    if score < best_score:
                        best_score = score
                        best_rect = (x, y, item.width, item.height)
                state[item_id] = best_rect
                if best_rect != current:
                    improved = True
            if not improved:
                break
        return state

    def _fallback_item_place(
        self,
        occupied: Sequence[Tuple[float, float, float, float]],
        occupied_bbox: Optional[Tuple[float, float, float, float]],
        item: LayoutItem,
    ) -> Tuple[float, float, float, float]:
        if not occupied:
            return (0.0, 0.0, item.width, item.height)

        x0, y0, x1, y1 = occupied_bbox if occupied_bbox is not None else self._bbox_xyxy(occupied)
        steps = [
            (x1, y0),
            (x0 - item.width, y0),
            (x0, y1),
            (x0, y0 - item.height),
            (x1, y1),
            (x0 - item.width, y0 - item.height),
        ]
        best_rect = (x1, y0, item.width, item.height)
        best_score = self._score_bbox_with_rect(occupied_bbox, best_rect)
        for x, y in steps:
            rect = (x, y, item.width, item.height)
            if self._overlaps_any(rect, occupied):
                continue
            score = self._score_bbox_with_rect(occupied_bbox, rect)
            if score < best_score:
                best_score = score
                best_rect = rect
        return best_rect

    # ------------------------------------------------------------------
    # Core soft fill
    # ------------------------------------------------------------------
    def _fill_soft_singles(
        self,
        block_map: Dict[int, BlockSpec],
        positions: Dict[int, Tuple[float, float, float, float]],
        soft_ids: Sequence[int],
        b2b_connectivity: Optional[torch.Tensor] = None,
        p2b_connectivity: Optional[torch.Tensor] = None,
        pins_pos: Optional[torch.Tensor] = None,
    ) -> Dict[int, Tuple[float, float, float, float]]:
        if not soft_ids:
            return positions

        if not positions:
            total_area = sum(block_map[bid].area for bid in soft_ids)
            height = math.sqrt(max(total_area, 1.0))
            width = total_area / max(height, EPS)
            self._solve_region_with_pava(
                FreeRect(0.0, 0.0, width, height),
                list(soft_ids),
                positions,
                block_map,
                b2b_connectivity,
                p2b_connectivity,
                pins_pos,
            )
            return positions

        free_rects = self._extract_free_rectangles(positions.values())
        bins = [FreeRect(*rect) for rect in free_rects if rect[2] * rect[3] > EPS]
        bins.sort(key=lambda rect: (-rect.area, rect.x, rect.y))
        bins = bins[: min(len(bins), 48)]
        if not bins:
            self._pack_remaining_soft_strip(positions, block_map, soft_ids)
            return positions

        guides, incident_weight = self._precompute_soft_guides(
            soft_ids,
            positions,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
        )
        x_targets = self._precompute_axis_targets(
            soft_ids,
            positions,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            axis=0,
        )
        remaining_area = [rect.area for rect in bins]
        assignments: Dict[int, List[int]] = {idx: [] for idx in range(len(bins))}
        unplaced: List[int] = []
        ordered_soft = sorted(
            soft_ids,
            key=lambda bid: (-incident_weight.get(bid, 0.0), -block_map[bid].area, bid),
        )

        for block_id in ordered_soft:
            area = block_map[block_id].area
            chosen_idx = None
            best_score = (float("inf"), float("inf"), float("inf"))
            for idx, rect in enumerate(bins):
                if remaining_area[idx] + EPS < area:
                    continue
                cx = rect.x + 0.5 * rect.w
                cy = rect.y + 0.5 * rect.h
                gx, gy, gw = guides[block_id]
                proxy = gw * (abs(cx - gx) + abs(cy - gy))
                score = (proxy, remaining_area[idx] - area, -rect.area)
                if score < best_score:
                    best_score = score
                    chosen_idx = idx
            if chosen_idx is None:
                unplaced.append(block_id)
                continue
            assignments[chosen_idx].append(block_id)
            remaining_area[chosen_idx] -= area

        for idx, block_ids in assignments.items():
            if not block_ids:
                continue
            self._solve_region_with_pava(
                bins[idx],
                block_ids,
                positions,
                block_map,
                b2b_connectivity,
                p2b_connectivity,
                pins_pos,
                x_targets,
            )

        if unplaced:
            self._pack_remaining_soft_strip(positions, block_map, unplaced)
        return positions

    def _precompute_soft_guides(
        self,
        soft_ids: Sequence[int],
        positions: Dict[int, Tuple[float, float, float, float]],
        b2b_connectivity: Optional[torch.Tensor],
        p2b_connectivity: Optional[torch.Tensor],
        pins_pos: Optional[torch.Tensor],
    ) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, float]]:
        soft_set = set(soft_ids)
        sums: Dict[int, List[float]] = {bid: [0.0, 0.0, 0.0] for bid in soft_ids}
        incident: Dict[int, float] = {bid: 0.0 for bid in soft_ids}

        if p2b_connectivity is not None and pins_pos is not None:
            for edge in p2b_connectivity:
                if edge[0] == -1:
                    continue
                pin_idx, bid, weight = int(edge[0]), int(edge[1]), abs(float(edge[2]))
                if bid not in soft_set or pin_idx >= len(pins_pos):
                    continue
                px, py = float(pins_pos[pin_idx][0]), float(pins_pos[pin_idx][1])
                sums[bid][0] += px * weight
                sums[bid][1] += py * weight
                sums[bid][2] += weight
                incident[bid] += weight

        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if edge[0] == -1:
                    continue
                i, j, weight = int(edge[0]), int(edge[1]), abs(float(edge[2]))
                if i in soft_set:
                    incident[i] += weight
                    if j in positions:
                        x, y, w, h = positions[j]
                        sums[i][0] += (x + 0.5 * w) * weight
                        sums[i][1] += (y + 0.5 * h) * weight
                        sums[i][2] += weight
                if j in soft_set:
                    incident[j] += weight
                    if i in positions:
                        x, y, w, h = positions[i]
                        sums[j][0] += (x + 0.5 * w) * weight
                        sums[j][1] += (y + 0.5 * h) * weight
                        sums[j][2] += weight

        if positions:
            x0, y0, x1, y1 = self._bbox_xyxy(positions.values())
            fallback = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        else:
            fallback = (0.0, 0.0)

        guides: Dict[int, Tuple[float, float, float]] = {}
        for bid in soft_ids:
            sx, sy, sw = sums[bid]
            if sw <= EPS:
                guides[bid] = (fallback[0], fallback[1], 1.0)
            else:
                guides[bid] = (sx / sw, sy / sw, sw)
        return guides, incident

    def _precompute_axis_targets(
        self,
        soft_ids: Sequence[int],
        positions: Dict[int, Tuple[float, float, float, float]],
        b2b_connectivity: Optional[torch.Tensor],
        p2b_connectivity: Optional[torch.Tensor],
        pins_pos: Optional[torch.Tensor],
        axis: int,
    ) -> Dict[int, List[Tuple[float, float]]]:
        soft_set = set(soft_ids)
        targets: Dict[int, List[Tuple[float, float]]] = {bid: [] for bid in soft_ids}

        if p2b_connectivity is not None and pins_pos is not None:
            for edge in p2b_connectivity:
                if edge[0] == -1:
                    continue
                pin_idx, bid, weight = int(edge[0]), int(edge[1]), abs(float(edge[2]))
                if bid in soft_set and pin_idx < len(pins_pos):
                    targets[bid].append((float(pins_pos[pin_idx][axis]), max(weight, EPS)))

        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if edge[0] == -1:
                    continue
                i, j, weight = int(edge[0]), int(edge[1]), abs(float(edge[2]))
                if i in soft_set and j in positions:
                    rect = positions[j]
                    targets[i].append((rect[axis] + 0.5 * rect[axis + 2], max(weight, EPS)))
                if j in soft_set and i in positions:
                    rect = positions[i]
                    targets[j].append((rect[axis] + 0.5 * rect[axis + 2], max(weight, EPS)))

        return targets

    def _soft_order_key(
        self,
        block_id: int,
        block_map: Dict[int, BlockSpec],
        b2b_connectivity: Optional[torch.Tensor],
        p2b_connectivity: Optional[torch.Tensor],
    ) -> Tuple[float, float, int]:
        weight = 0.0
        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if edge[0] == -1:
                    continue
                i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
                if i == block_id or j == block_id:
                    weight += abs(w)
        if p2b_connectivity is not None:
            for edge in p2b_connectivity:
                if edge[0] == -1:
                    continue
                _, bid, w = int(edge[0]), int(edge[1]), float(edge[2])
                if bid == block_id:
                    weight += abs(w)
        return (-weight, -block_map[block_id].area, block_id)

    def _placement_proxy(
        self,
        block_id: int,
        cx: float,
        cy: float,
        positions: Dict[int, Tuple[float, float, float, float]],
        b2b_connectivity: Optional[torch.Tensor],
        p2b_connectivity: Optional[torch.Tensor],
        pins_pos: Optional[torch.Tensor],
    ) -> float:
        total = 0.0
        used = False
        if p2b_connectivity is not None and pins_pos is not None:
            for edge in p2b_connectivity:
                if edge[0] == -1:
                    continue
                pin_idx, bid, weight = int(edge[0]), int(edge[1]), float(edge[2])
                if bid != block_id or pin_idx >= len(pins_pos):
                    continue
                px, py = float(pins_pos[pin_idx][0]), float(pins_pos[pin_idx][1])
                total += weight * (abs(cx - px) + abs(cy - py))
                used = True
        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if edge[0] == -1:
                    continue
                i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
                other = None
                if i == block_id and j in positions:
                    other = positions[j]
                elif j == block_id and i in positions:
                    other = positions[i]
                if other is None:
                    continue
                ox, oy, ow, oh = other
                total += weight * (abs(cx - (ox + 0.5 * ow)) + abs(cy - (oy + 0.5 * oh)))
                used = True
        if used:
            return total
        if positions:
            x0, y0, x1, y1 = self._bbox_xyxy(positions.values())
            return abs(cx - 0.5 * (x0 + x1)) + abs(cy - 0.5 * (y0 + y1))
        return 0.0

    def _axis_targets(
        self,
        block_id: int,
        axis: int,
        positions: Dict[int, Tuple[float, float, float, float]],
        b2b_connectivity: Optional[torch.Tensor],
        p2b_connectivity: Optional[torch.Tensor],
        pins_pos: Optional[torch.Tensor],
    ) -> List[Tuple[float, float]]:
        targets: List[Tuple[float, float]] = []
        if p2b_connectivity is not None and pins_pos is not None:
            for edge in p2b_connectivity:
                if edge[0] == -1:
                    continue
                pin_idx, bid, weight = int(edge[0]), int(edge[1]), float(edge[2])
                if bid == block_id and pin_idx < len(pins_pos):
                    targets.append((float(pins_pos[pin_idx][axis]), abs(weight)))
        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if edge[0] == -1:
                    continue
                i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
                other = None
                if i == block_id and j in positions:
                    other = positions[j]
                elif j == block_id and i in positions:
                    other = positions[i]
                if other is None:
                    continue
                value = other[axis] + 0.5 * other[axis + 2]
                targets.append((value, abs(weight)))
        return targets

    def _solve_region_with_pava(
        self,
        region: FreeRect,
        block_ids: Sequence[int],
        positions: Dict[int, Tuple[float, float, float, float]],
        block_map: Dict[int, BlockSpec],
        b2b_connectivity: Optional[torch.Tensor],
        p2b_connectivity: Optional[torch.Tensor],
        pins_pos: Optional[torch.Tensor],
        precomputed_x_targets: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    ) -> None:
        if not block_ids or region.w <= EPS or region.h <= EPS:
            return

        height = region.h
        widths = {bid: block_map[bid].area / max(height, EPS) for bid in block_ids}
        total_w = sum(widths.values())
        if total_w > region.w + 1e-6:
            # Numerical or assignment fallback: use a top strip sized exactly
            # to the assigned area instead of shrinking any block.
            height = sum(block_map[bid].area for bid in block_ids) / max(region.w, EPS)
            widths = {bid: block_map[bid].area / max(height, EPS) for bid in block_ids}

        target_lists: Dict[int, List[Tuple[float, float]]] = {}
        clipped_values: List[float] = []
        for bid in block_ids:
            targets = (
                list(precomputed_x_targets.get(bid, []))
                if precomputed_x_targets is not None
                else self._axis_targets(
                    bid, 0, positions, b2b_connectivity, p2b_connectivity, pins_pos
                )
            )
            if not targets:
                targets = [(region.x + 0.5 * region.w, 1.0)]
            clipped: List[Tuple[float, float]] = []
            for value, weight in targets:
                value = min(max(value, region.x), region.x + region.w)
                clipped.append((value, max(weight, EPS)))
                clipped_values.append(value)
            target_lists[bid] = clipped

        unique_values = sorted({round(value, 8) for value in clipped_values})
        rank = {value: idx for idx, value in enumerate(unique_values)}

        def order_score(bid: int) -> Tuple[float, float, int]:
            weighted = 0.0
            total = 0.0
            for value, weight in target_lists[bid]:
                weighted += rank[round(value, 8)] * weight
                total += weight
            return (weighted / max(total, EPS), -block_map[bid].area, bid)

        order = sorted(block_ids, key=order_score)
        centers = self._pava_centers(order, widths, target_lists, region.x, region.x + region.w)
        if centers is None:
            centers = {}
            cursor = region.x + max(0.0, region.w - sum(widths[bid] for bid in order)) * 0.5
            for bid in order:
                centers[bid] = cursor + 0.5 * widths[bid]
                cursor += widths[bid]

        for bid in order:
            w = widths[bid]
            cx = centers[bid]
            positions[bid] = (cx - 0.5 * w, region.y, w, height)

    def _pava_centers(
        self,
        order: Sequence[int],
        widths: Dict[int, float],
        target_lists: Dict[int, List[Tuple[float, float]]],
        left: float,
        right: float,
    ) -> Optional[Dict[int, float]]:
        offsets: List[float] = []
        cursor = 0.0
        for idx, bid in enumerate(order):
            if idx == 0:
                offsets.append(0.0)
            else:
                prev = order[idx - 1]
                cursor += 0.5 * (widths[prev] + widths[bid])
                offsets.append(cursor)

        lows: List[float] = []
        highs: List[float] = []
        shifted_targets: List[List[Tuple[float, float]]] = []
        for idx, bid in enumerate(order):
            c = offsets[idx]
            lows.append(left + 0.5 * widths[bid] - c)
            highs.append(right - 0.5 * widths[bid] - c)
            shifted_targets.append([(value - c, weight) for value, weight in target_lists[bid]])

        feasible_y = -float("inf")
        for lo, hi in zip(lows, highs):
            feasible_y = max(feasible_y, lo)
            if feasible_y > hi + 1e-6:
                return None

        stack: List[Dict[str, object]] = []
        for idx in range(len(order)):
            block = {
                "start": idx,
                "end": idx,
                "lo": lows[idx],
                "hi": highs[idx],
                "targets": list(shifted_targets[idx]),
            }
            block["value"] = self._block_l1_value(
                block["targets"], block["lo"], block["hi"]
            )
            stack.append(block)
            while len(stack) >= 2 and float(stack[-2]["value"]) > float(stack[-1]["value"]) + 1e-9:
                right_block = stack.pop()
                left_block = stack.pop()
                merged_targets = list(left_block["targets"]) + list(right_block["targets"])
                merged_lo = max(float(left_block["lo"]), float(right_block["lo"]))
                merged_hi = min(float(left_block["hi"]), float(right_block["hi"]))
                if merged_lo > merged_hi + 1e-7:
                    return None
                merged = {
                    "start": int(left_block["start"]),
                    "end": int(right_block["end"]),
                    "lo": merged_lo,
                    "hi": merged_hi,
                    "targets": merged_targets,
                }
                merged["value"] = self._block_l1_value(merged_targets, merged_lo, merged_hi)
                stack.append(merged)

        centers: Dict[int, float] = {}
        for block in stack:
            value = float(block["value"])
            for idx in range(int(block["start"]), int(block["end"]) + 1):
                centers[order[idx]] = value + offsets[idx]
        return centers

    def _block_l1_value(
        self,
        targets: Sequence[Tuple[float, float]],
        lo: float,
        hi: float,
    ) -> float:
        if not targets:
            return lo
        median = self._weighted_median(targets)
        return min(max(median, lo), hi)

    def _weighted_median(self, targets: Sequence[Tuple[float, float]]) -> float:
        ordered = sorted(targets, key=lambda item: item[0])
        total = sum(max(weight, EPS) for _, weight in ordered)
        acc = 0.0
        for value, weight in ordered:
            acc += max(weight, EPS)
            if acc + EPS >= 0.5 * total:
                return value
        return ordered[-1][0]

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
            # Effective corner dimensions prevent side blocks from
            # overlapping at the corners. Each virtual corner region
            # must be wide/tall enough to separate perpendicular sides.
            eff_w_bl = max(w_bl, L)
            eff_h_bl = max(h_bl, B)
            eff_w_br = max(w_br, R)
            eff_h_br = max(h_br, B)
            eff_w_tl = max(w_tl, L)
            eff_h_tl = max(h_tl, T)
            eff_w_tr = max(w_tr, R)
            eff_h_tr = max(h_tr, T)

            W = core_w + L + R
            H = core_h + T + B
            gap_left = max(H - eff_h_tl - eff_h_bl, 1e-6)
            gap_right = max(H - eff_h_tr - eff_h_br, 1e-6)
            gap_top = max(W - eff_w_tl - eff_w_tr, 1e-6)
            gap_bottom = max(W - eff_w_bl - eff_w_br, 1e-6)

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

        # Final effective corner dimensions for placement bounds
        eff_w_bl = max(w_bl, L)
        eff_h_bl = max(h_bl, B)
        eff_w_br = max(w_br, R)
        eff_h_br = max(h_br, B)
        eff_w_tl = max(w_tl, L)
        eff_h_tl = max(h_tl, T)
        eff_w_tr = max(w_tr, R)
        eff_h_tr = max(h_tr, T)

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

        # Place sides within their allocated rectangular regions,
        # bounded by effective corner dimensions to prevent overlap.
        self._place_vertical_side(
            left_blocks, X0, Y0 + eff_h_bl, Y1 - eff_h_tl,
            L, positions,
        )
        self._place_vertical_side(
            right_blocks, X1, Y0 + eff_h_br, Y1 - eff_h_tr,
            R, positions, right_align=True,
        )
        self._place_horizontal_side(
            top_blocks, X0 + eff_w_tl, Y1, X1 - eff_w_tr,
            T, positions, top_align=True,
        )
        self._place_horizontal_side(
            bottom_blocks, X0 + eff_w_bl, Y0, X1 - eff_w_br,
            B, positions, top_align=False,
        )

        # Safety: resolve any remaining overlaps among boundary blocks
        # that might arise from floating-point or convergence imprecision.
        positions = self._resolve_boundary_overlaps(
            positions, boundary_ids, block_map,
        )

        return positions

    def _choose_corner_dims(
        self,
        corners: Dict[int, Optional[BlockSpec]],
        core_w: float,
        core_h: float,
    ) -> Dict[int, Tuple[float, float]]:
        choices: Dict[int, List[Tuple[float, float]]] = {}
        ratio = max(core_w, 1.0) / max(core_h, 1.0)
        for code, block in corners.items():
            if block is None:
                choices[code] = [(0.0, 0.0)]
                continue
            if block.fixed:
                choices[code] = [(block.width, block.height)]
            else:
                width = math.sqrt(max(block.area * ratio, 1e-6))
                height = block.area / max(width, EPS)
                alt = (height, width)
                primary = (width, height)
                if abs(primary[0] - alt[0]) <= 1e-9 and abs(primary[1] - alt[1]) <= 1e-9:
                    choices[code] = [primary]
                else:
                    choices[code] = [primary, alt]

        codes = [5, 6, 9, 10]
        best_dims: Optional[Dict[int, Tuple[float, float]]] = None
        best_score = (float("inf"), float("inf"))

        def rec(idx: int, current: Dict[int, Tuple[float, float]]) -> None:
            nonlocal best_dims, best_score
            if idx == len(codes):
                w_tl, h_tl = current.get(5, (0.0, 0.0))
                w_tr, h_tr = current.get(6, (0.0, 0.0))
                w_bl, h_bl = current.get(9, (0.0, 0.0))
                w_br, h_br = current.get(10, (0.0, 0.0))
                left = max(w_tl, w_bl)
                right = max(w_tr, w_br)
                top = max(h_tl, h_tr)
                bottom = max(h_bl, h_br)
                final_w = core_w + left + right
                final_h = core_h + top + bottom
                score = (final_w * final_h, abs(final_w - final_h))
                if score < best_score:
                    best_score = score
                    best_dims = dict(current)
                return

            code = codes[idx]
            for dims in choices[code]:
                current[code] = dims
                rec(idx + 1, current)

        rec(0, {})
        return best_dims or {code: choices[code][0] for code in codes}

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
        end_y: float,
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
            if h <= EPS:
                continue
            x = side_x - w if right_align else side_x
            positions[block.block_id] = (x, cursor_y, w, h)
            cursor_y += h

    def _place_horizontal_side(
        self,
        blocks: Sequence[BlockSpec],
        start_x: float,
        side_y: float,
        end_x: float,
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
            if w <= EPS:
                continue
            y = side_y - h if top_align else side_y
            positions[block.block_id] = (cursor_x, y, w, h)
            cursor_x += w

    # ------------------------------------------------------------------
    # Post-placement overlap resolution
    # ------------------------------------------------------------------
    def _resolve_boundary_overlaps(
        self,
        positions: Dict[int, Tuple[float, float, float, float]],
        boundary_ids: Sequence[int],
        block_map: Dict[int, BlockSpec],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        """Resolve overlaps among boundary blocks by pushing overlapping
        blocks outward (away from the core bounding box)."""
        if not boundary_ids or len(boundary_ids) < 2:
            return positions

        boundary_set = set(boundary_ids)
        # Compute core bbox from non-boundary blocks
        core_rects = [
            rect for bid, rect in positions.items() if bid not in boundary_set
        ]
        if core_rects:
            core_x0, core_y0, core_x1, core_y1 = self._bbox_xyxy(core_rects)
        else:
            core_x0 = core_y0 = core_x1 = core_y1 = 0.0

        for _ in range(5):
            fixed = False
            for i_idx in range(len(boundary_ids)):
                for j_idx in range(i_idx + 1, len(boundary_ids)):
                    bi = boundary_ids[i_idx]
                    bj = boundary_ids[j_idx]
                    p1 = positions.get(bi)
                    p2 = positions.get(bj)
                    if p1 is None or p2 is None:
                        continue
                    ox = min(p1[0] + p1[2], p2[0] + p2[2]) - max(p1[0], p2[0])
                    oy = min(p1[1] + p1[3], p2[1] + p2[3]) - max(p1[1], p2[1])
                    if ox <= 1e-6 or oy <= 1e-6:
                        continue
                    # Push the block farther from core bbox outward
                    c1x = p1[0] + 0.5 * p1[2]
                    c1y = p1[1] + 0.5 * p1[3]
                    c2x = p2[0] + 0.5 * p2[2]
                    c2y = p2[1] + 0.5 * p2[3]
                    core_cx = 0.5 * (core_x0 + core_x1)
                    core_cy = 0.5 * (core_y0 + core_y1)
                    d1 = abs(c1x - core_cx) + abs(c1y - core_cy)
                    d2 = abs(c2x - core_cx) + abs(c2y - core_cy)
                    # Move the closer-to-core block inward, the farther outward
                    if d1 < d2:
                        # Push p2 outward
                        dx = 0.0
                        dy = 0.0
                        if c2x >= core_cx:
                            dx = ox
                        else:
                            dx = -ox
                        if c2y >= core_cy:
                            dy = oy
                        else:
                            dy = -oy
                        positions[bj] = (p2[0] + dx, p2[1] + dy, p2[2], p2[3])
                    else:
                        dx = 0.0
                        dy = 0.0
                        if c1x >= core_cx:
                            dx = ox
                        else:
                            dx = -ox
                        if c1y >= core_cy:
                            dy = oy
                        else:
                            dy = -oy
                        positions[bi] = (p1[0] + dx, p1[1] + dy, p1[2], p1[3])
                    fixed = True
            if not fixed:
                break
        return positions

    def _resolve_all_overlaps(
        self,
        positions: List[Tuple[float, float, float, float]],
        blocks: Optional[Sequence[BlockSpec]] = None,
    ) -> List[Tuple[float, float, float, float]]:
        """Final safety net: detect and resolve any remaining overlaps by
        shifting movable blocks apart along the line connecting centers."""
        result = list(positions)
        for _ in range(10):
            fixed = False
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    x1, y1, w1, h1 = result[i]
                    x2, y2, w2, h2 = result[j]
                    ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                    oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                    if ox <= 1e-6 or oy <= 1e-6:
                        continue
                    movable_i = not (blocks is not None and blocks[i].preplaced)
                    movable_j = not (blocks is not None and blocks[j].preplaced)
                    if not movable_i and not movable_j:
                        continue
                    # Shift each block apart by half the overlap
                    cx1, cy1 = x1 + 0.5 * w1, y1 + 0.5 * h1
                    cx2, cy2 = x2 + 0.5 * w2, y2 + 0.5 * h2
                    dx = cx2 - cx1
                    dy = cy2 - cy1
                    dist = math.sqrt(dx * dx + dy * dy) + 1e-9
                    # Normalize and scale
                    shift = 0.5 * (ox + oy)
                    sx = (dx / dist) * shift
                    sy = (dy / dist) * shift
                    if movable_i and movable_j:
                        result[i] = (x1 - sx, y1 - sy, w1, h1)
                        result[j] = (x2 + sx, y2 + sy, w2, h2)
                    elif movable_i:
                        result[i] = (x1 - 2.0 * sx, y1 - 2.0 * sy, w1, h1)
                    else:
                        result[j] = (x2 + 2.0 * sx, y2 + 2.0 * sy, w2, h2)
                    fixed = True
            if not fixed:
                break
        return result

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _local_bbox(
        self,
        rects: Iterable[Tuple[float, float, float, float]],
    ) -> Tuple[float, float]:
        bbox = self._bbox_xyxy(rects)
        if bbox == (0.0, 0.0, 0.0, 0.0):
            return (0.0, 0.0)
        x0, y0, x1, y1 = bbox
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
        iterator = iter(rects)
        try:
            x0, y0, w0, h0 = next(iterator)
        except StopIteration:
            return (0.0, 0.0, 0.0, 0.0)
        min_x = x0
        min_y = y0
        max_x = x0 + w0
        max_y = y0 + h0
        for x, y, w, h in iterator:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            x2 = x + w
            y2 = y + h
            if x2 > max_x:
                max_x = x2
            if y2 > max_y:
                max_y = y2
        return min_x, min_y, max_x, max_y

    def _expand_bbox(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
        rect: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        x, y, w, h = rect
        x2 = x + w
        y2 = y + h
        if bbox is None:
            return (x, y, x2, y2)
        return (
            min(bbox[0], x),
            min(bbox[1], y),
            max(bbox[2], x2),
            max(bbox[3], y2),
        )

    def _score_bbox(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float]:
        if bbox is None:
            return (0.0, 0.0, 0.0)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return (w * h, w + h, max(w, h))

    def _score_bbox_with_rect(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
        rect: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float]:
        return self._score_bbox(self._expand_bbox(bbox, rect))

    def _state_score_occ(
        self,
        fixed_rects: Sequence[Tuple[float, float, float, float]],
        state_rects: Iterable[Tuple[float, float, float, float]],
        extra: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float]:
        bbox = None
        for rect in fixed_rects:
            bbox = self._expand_bbox(bbox, rect)
        for rect in state_rects:
            bbox = self._expand_bbox(bbox, rect)
        if extra is not None:
            bbox = self._expand_bbox(bbox, extra)
        return self._score_bbox(bbox)

    def _overlaps_any(
        self,
        rect: Tuple[float, float, float, float],
        others: Iterable[Tuple[float, float, float, float]],
    ) -> bool:
        x1, y1, w1, h1 = rect
        x1_hi = x1 + w1
        y1_hi = y1 + h1
        for x2, y2, w2, h2 in others:
            overlap_x = min(x1_hi, x2 + w2) - max(x1, x2)
            overlap_y = min(y1_hi, y2 + h2) - max(y1, y2)
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
