#!/usr/bin/env python3
"""
BF3: grouping-as-item + MIB templates + local slicing-tree soft packing.

Main rules for this version:
1. Grouping subproblems are built only from non-preplaced, non-boundary blocks;
   boundary+grouping follows boundary and accepts grouping fragmentation.
2. Preplaced blocks ignore all soft constraints and are never moved.
3. Boundary blocks are placed after the core layout.
4. Blocks in the same MIB group share a template only when doing so keeps
   all hard area/fixed/preplaced constraints legal.
5. Standalone soft blocks are assigned to free regions and solved with a
   bounded local slicing-tree search; 1-D L1 PAVA remains a fallback.
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


class SliceNode:
    def __init__(
        self,
        area: float,
        block_id: Optional[int] = None,
        cut: Optional[str] = None,
        left: Optional["SliceNode"] = None,
        right: Optional["SliceNode"] = None,
    ):
        self.area = area
        self.block_id = block_id
        self.cut = cut
        self.left = left
        self.right = right


class RegionLayoutCandidate:
    def __init__(
        self,
        positions: Dict[int, Tuple[float, float, float, float]],
        score: float,
        source: str,
    ):
        self.positions = positions
        self.score = score
        self.source = source


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.beam_width = 1
        self.state_candidate_limit = 1
        self.local_passes = 1
        self.slice_eval_limit = 60
        self.slice_keep_candidates = 4

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
        self._current_block_map = block_map
        self._current_pins_pos = pins_pos
        self._current_b2b_edges, self._current_p2b_edges = self._build_python_edges(
            b2b_connectivity, p2b_connectivity
        )

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

        b2b_edges = self._current_b2b_edges
        p2b_edges = self._current_p2b_edges

        if not positions:
            total_area = sum(block_map[bid].area for bid in soft_ids)
            height = math.sqrt(max(total_area, 1.0))
            width = total_area / max(height, EPS)
            guides, _ = self._precompute_soft_guides(
                soft_ids, positions, b2b_edges, p2b_edges, pins_pos
            )
            x_targets = self._precompute_axis_targets(
                soft_ids, positions, b2b_edges, p2b_edges, pins_pos, axis=0
            )
            candidates, _ = self._build_region_layout_candidates(
                FreeRect(0.0, 0.0, width, height),
                list(soft_ids),
                positions,
                block_map,
                guides,
                x_targets,
                b2b_edges,
                p2b_edges,
                pins_pos,
                self.slice_eval_limit,
            )
            positions.update(candidates[0].positions)
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
            b2b_edges,
            p2b_edges,
            pins_pos,
        )
        x_targets = self._precompute_axis_targets(
            soft_ids,
            positions,
            b2b_edges,
            p2b_edges,
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

        region_records: List[
            Tuple[
                List[int],
                List[RegionLayoutCandidate],
                List[Tuple[int, int, float]],
                List[Tuple[int, int, float]],
            ]
        ] = []
        eval_budget = self.slice_eval_limit
        region_tasks = [
            (idx, block_ids)
            for idx, block_ids in assignments.items()
            if block_ids
        ]
        region_tasks.sort(
            key=lambda item: (-len(item[1]), -sum(block_map[bid].area for bid in item[1]), item[0])
        )

        for idx, block_ids in region_tasks:
            candidates, used = self._build_region_layout_candidates(
                bins[idx],
                block_ids,
                positions,
                block_map,
                guides,
                x_targets,
                b2b_edges,
                p2b_edges,
                pins_pos,
                eval_budget,
            )
            eval_budget = max(0, eval_budget - used)
            positions.update(candidates[0].positions)
            if len(candidates) > 1:
                region_set = set(block_ids)
                region_b2b = [
                    edge for edge in b2b_edges
                    if edge[0] in region_set or edge[1] in region_set
                ]
                region_p2b = [
                    edge for edge in p2b_edges if edge[1] in region_set
                ]
                region_records.append(
                    (list(block_ids), candidates, region_b2b, region_p2b)
                )

        if unplaced:
            self._pack_remaining_soft_strip(positions, block_map, unplaced)

        if region_records:
            self._refine_region_candidates(
                region_records,
                positions,
                guides,
                pins_pos,
            )
        return positions

    def _build_python_edges(
        self,
        b2b_connectivity: Optional[torch.Tensor],
        p2b_connectivity: Optional[torch.Tensor],
    ) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
        b2b_edges: List[Tuple[int, int, float]] = []
        p2b_edges: List[Tuple[int, int, float]] = []
        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if edge[0] == -1:
                    continue
                b2b_edges.append((int(edge[0]), int(edge[1]), abs(float(edge[2]))))
        if p2b_connectivity is not None:
            for edge in p2b_connectivity:
                if edge[0] == -1:
                    continue
                p2b_edges.append((int(edge[0]), int(edge[1]), abs(float(edge[2]))))
        return b2b_edges, p2b_edges

    def _build_region_layout_candidates(
        self,
        region: FreeRect,
        block_ids: Sequence[int],
        base_positions: Dict[int, Tuple[float, float, float, float]],
        block_map: Dict[int, BlockSpec],
        guides: Dict[int, Tuple[float, float, float]],
        x_targets: Dict[int, List[Tuple[float, float]]],
        b2b_edges: Sequence[Tuple[int, int, float]],
        p2b_edges: Sequence[Tuple[int, int, float]],
        pins_pos: Optional[torch.Tensor],
        eval_budget: int,
    ) -> Tuple[List[RegionLayoutCandidate], int]:
        ids = list(block_ids)
        region_set = set(ids)
        incident_b2b = [
            edge for edge in b2b_edges
            if edge[0] in region_set or edge[1] in region_set
        ]
        incident_p2b = [
            edge for edge in p2b_edges if edge[1] in region_set
        ]
        internal_b2b = [
            edge for edge in incident_b2b
            if edge[0] in region_set and edge[1] in region_set
        ]
        pava_positions = dict(base_positions)
        self._solve_region_with_pava(
            region,
            ids,
            pava_positions,
            block_map,
            None,
            None,
            pins_pos,
            x_targets,
        )
        pava_layout = {bid: pava_positions[bid] for bid in ids}
        pava_score = self._score_region_layout(
            pava_layout, ids, base_positions, guides, incident_b2b, incident_p2b, pins_pos
        )
        pava_candidate = RegionLayoutCandidate(pava_layout, pava_score, "pava")

        # Residual non-template MIB members keep the conservative PAVA path.
        if eval_budget <= 0 or any(
            block_map[bid].mib_id > 0 or block_map[bid].group_id > 0
            for bid in ids
        ):
            return [pava_candidate], 0

        trees = self._build_slice_tree_variants(ids, block_map, guides, internal_b2b)
        ratios = self._slice_root_ratios(region, ids, block_map, guides)
        total_area = sum(block_map[bid].area for bid in ids)
        candidates: List[RegionLayoutCandidate] = [pava_candidate]
        seen_layouts = {self._region_layout_key(pava_layout)}
        used = 0

        for ratio in ratios:
            for tree in trees:
                if used >= eval_budget:
                    break
                root_w = math.sqrt(max(total_area * ratio, EPS))
                root_h = total_area / max(root_w, EPS)
                if root_w > region.w + 1e-7 or root_h > region.h + 1e-7:
                    continue
                root_positions = self._slice_root_positions(
                    region, root_w, root_h, ids, guides
                )
                if len(ids) > 6:
                    root_positions = root_positions[:2]
                for root_x, root_y in root_positions:
                    if used >= eval_budget:
                        break
                    used += 1
                    layout: Dict[int, Tuple[float, float, float, float]] = {}
                    self._materialize_slice_tree(
                        tree, root_x, root_y, root_w, root_h, layout
                    )
                    if not self._validate_region_layout(region, ids, layout, block_map):
                        continue
                    key = self._region_layout_key(layout)
                    if key in seen_layouts:
                        continue
                    seen_layouts.add(key)
                    score = self._score_region_layout(
                        layout,
                        ids,
                        base_positions,
                        guides,
                        incident_b2b,
                        incident_p2b,
                        pins_pos,
                    )
                    candidates.append(RegionLayoutCandidate(layout, score, "slicing"))
            if used >= eval_budget:
                break

        slicing = sorted(
            (candidate for candidate in candidates if candidate.source == "slicing"),
            key=lambda candidate: candidate.score,
        )
        kept = slicing[: max(0, self.slice_keep_candidates - 1)] + [pava_candidate]
        kept.sort(key=lambda candidate: candidate.score)
        return kept, used

    def _build_slice_tree_variants(
        self,
        block_ids: Sequence[int],
        block_map: Dict[int, BlockSpec],
        guides: Dict[int, Tuple[float, float, float]],
        b2b_edges: Sequence[Tuple[int, int, float]],
    ) -> List[SliceNode]:
        ids = list(block_ids)
        if len(ids) == 1:
            bid = ids[0]
            return [SliceNode(area=block_map[bid].area, block_id=bid)]

        configs = [
            ("spread", 0.50),
            ("vertical", 0.50),
            ("horizontal", 0.50),
            ("spread", 1.0 / 3.0),
            ("spread", 2.0 / 3.0),
            ("vertical", 1.0 / 3.0),
            ("horizontal", 2.0 / 3.0),
            ("vertical", 2.0 / 3.0),
        ]
        if len(ids) <= 3:
            configs = configs[:5]
        elif len(ids) > 10:
            configs = configs[:6]

        trees: List[SliceNode] = []
        seen = set()
        for mode, split_fraction in configs:
            tree = self._build_slice_tree(
                ids, block_map, guides, b2b_edges, mode, split_fraction, 0
            )
            key = self._slice_tree_key(tree)
            if key in seen:
                continue
            seen.add(key)
            trees.append(tree)
        return trees

    def _build_slice_tree(
        self,
        block_ids: Sequence[int],
        block_map: Dict[int, BlockSpec],
        guides: Dict[int, Tuple[float, float, float]],
        b2b_edges: Sequence[Tuple[int, int, float]],
        mode: str,
        split_fraction: float,
        depth: int,
    ) -> SliceNode:
        ids = list(block_ids)
        if len(ids) == 1:
            bid = ids[0]
            return SliceNode(area=block_map[bid].area, block_id=bid)

        if mode == "vertical":
            axis = "V" if depth % 2 == 0 else "H"
        elif mode == "horizontal":
            axis = "H" if depth % 2 == 0 else "V"
        else:
            xs = [guides[bid][0] for bid in ids]
            ys = [guides[bid][1] for bid in ids]
            span_x = max(xs) - min(xs) if len(xs) > 1 else 0.0
            span_y = max(ys) - min(ys) if len(ys) > 1 else 0.0
            axis = "V" if span_x >= span_y else "H"

        coord = 0 if axis == "V" else 1
        ordered = sorted(ids, key=lambda bid: (guides[bid][coord], -block_map[bid].area, bid))
        cut_idx = self._choose_slice_cut(
            ordered, block_map, b2b_edges, split_fraction
        )
        left_ids = ordered[:cut_idx]
        right_ids = ordered[cut_idx:]
        left = self._build_slice_tree(
            left_ids, block_map, guides, b2b_edges, mode, split_fraction, depth + 1
        )
        right = self._build_slice_tree(
            right_ids, block_map, guides, b2b_edges, mode, split_fraction, depth + 1
        )
        return SliceNode(
            area=left.area + right.area,
            cut=axis,
            left=left,
            right=right,
        )

    def _choose_slice_cut(
        self,
        ordered: Sequence[int],
        block_map: Dict[int, BlockSpec],
        b2b_edges: Sequence[Tuple[int, int, float]],
        split_fraction: float,
    ) -> int:
        if len(ordered) == 2:
            return 1
        subset = set(ordered)
        internal_edges = [
            (i, j, weight)
            for i, j, weight in b2b_edges
            if i in subset and j in subset
        ]
        total_internal = sum(weight for _, _, weight in internal_edges)
        total_area = sum(block_map[bid].area for bid in ordered)
        prefix_area = 0.0
        best_idx = 1
        best_score = float("inf")
        for idx in range(1, len(ordered)):
            prefix_area += block_map[ordered[idx - 1]].area
            left_set = set(ordered[:idx])
            cross = sum(
                weight
                for i, j, weight in internal_edges
                if (i in left_set) != (j in left_set)
            )
            balance = abs(prefix_area / max(total_area, EPS) - split_fraction)
            score = balance + 0.12 * cross / max(total_internal, 1.0)
            if score + EPS < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _slice_tree_key(self, node: SliceNode) -> Tuple:
        if node.block_id is not None:
            return ("B", node.block_id)
        return (
            node.cut,
            self._slice_tree_key(node.left),
            self._slice_tree_key(node.right),
        )

    def _slice_root_ratios(
        self,
        region: FreeRect,
        block_ids: Sequence[int],
        block_map: Dict[int, BlockSpec],
        guides: Dict[int, Tuple[float, float, float]],
    ) -> List[float]:
        total_area = sum(block_map[bid].area for bid in block_ids)
        rho_min = total_area / max(region.h * region.h, EPS)
        rho_max = region.w * region.w / max(total_area, EPS)
        if rho_min > rho_max + 1e-7:
            return []

        xs = [guides[bid][0] for bid in block_ids]
        ys = [guides[bid][1] for bid in block_ids]
        span_x = max(xs) - min(xs) if len(xs) > 1 else 0.0
        span_y = max(ys) - min(ys) if len(ys) > 1 else 0.0
        if span_x <= EPS and span_y <= EPS:
            target_ratio = region.w / max(region.h, EPS)
        else:
            scale = math.sqrt(max(total_area / max(len(block_ids), 1), EPS))
            target_ratio = (span_x + scale) / max(span_y + scale, EPS)

        raw = [
            target_ratio,
            1.0,
            region.w / max(region.h, EPS),
            0.5,
            2.0,
            rho_min,
            rho_max,
        ]
        values: List[float] = []
        seen = set()
        for value in raw:
            value = min(max(value, rho_min), rho_max)
            key = round(value, 8)
            if key in seen:
                continue
            seen.add(key)
            values.append(value)
        values.sort(key=lambda value: abs(math.log(max(value, EPS) / max(target_ratio, EPS))))
        return values[:6]

    def _slice_root_positions(
        self,
        region: FreeRect,
        root_w: float,
        root_h: float,
        block_ids: Sequence[int],
        guides: Dict[int, Tuple[float, float, float]],
    ) -> List[Tuple[float, float]]:
        total_weight = sum(max(guides[bid][2], 1.0) for bid in block_ids)
        gx = sum(guides[bid][0] * max(guides[bid][2], 1.0) for bid in block_ids) / max(total_weight, EPS)
        gy = sum(guides[bid][1] * max(guides[bid][2], 1.0) for bid in block_ids) / max(total_weight, EPS)
        min_x = region.x
        max_x = region.x + region.w - root_w
        min_y = region.y
        max_y = region.y + region.h - root_h

        target_x = min(max(gx - 0.5 * root_w, min_x), max_x)
        target_y = min(max(gy - 0.5 * root_h, min_y), max_y)
        center_x = region.x + 0.5 * (region.w - root_w)
        center_y = region.y + 0.5 * (region.h - root_h)
        edge_x = min_x if gx <= region.x + 0.5 * region.w else max_x
        edge_y = min_y if gy <= region.y + 0.5 * region.h else max_y

        result: List[Tuple[float, float]] = []
        seen = set()
        for x, y in [(target_x, target_y), (center_x, center_y), (edge_x, edge_y)]:
            key = (round(x, 8), round(y, 8))
            if key in seen:
                continue
            seen.add(key)
            result.append((x, y))
        return result

    def _materialize_slice_tree(
        self,
        node: SliceNode,
        x: float,
        y: float,
        w: float,
        h: float,
        output: Dict[int, Tuple[float, float, float, float]],
    ) -> None:
        if node.block_id is not None:
            output[node.block_id] = (x, y, w, h)
            return
        if node.left is None or node.right is None:
            return
        ratio = node.left.area / max(node.area, EPS)
        if node.cut == "V":
            left_w = w * ratio
            self._materialize_slice_tree(node.left, x, y, left_w, h, output)
            self._materialize_slice_tree(node.right, x + left_w, y, w - left_w, h, output)
        else:
            bottom_h = h * ratio
            self._materialize_slice_tree(node.left, x, y, w, bottom_h, output)
            self._materialize_slice_tree(node.right, x, y + bottom_h, w, h - bottom_h, output)

    def _validate_region_layout(
        self,
        region: FreeRect,
        block_ids: Sequence[int],
        layout: Dict[int, Tuple[float, float, float, float]],
        block_map: Dict[int, BlockSpec],
    ) -> bool:
        if set(layout) != set(block_ids):
            return False
        rects: List[Tuple[float, float, float, float]] = []
        for bid in block_ids:
            x, y, w, h = layout[bid]
            if w <= EPS or h <= EPS:
                return False
            if (
                x < region.x - 1e-6
                or y < region.y - 1e-6
                or x + w > region.x + region.w + 1e-6
                or y + h > region.y + region.h + 1e-6
            ):
                return False
            area_error = abs(w * h - block_map[bid].area) / max(block_map[bid].area, EPS)
            if area_error > 1e-6:
                return False
            if self._overlaps_any((x, y, w, h), rects):
                return False
            rects.append((x, y, w, h))
        return True

    def _score_region_layout(
        self,
        layout: Dict[int, Tuple[float, float, float, float]],
        block_ids: Sequence[int],
        outside_positions: Dict[int, Tuple[float, float, float, float]],
        guides: Dict[int, Tuple[float, float, float]],
        b2b_edges: Sequence[Tuple[int, int, float]],
        p2b_edges: Sequence[Tuple[int, int, float]],
        pins_pos: Optional[torch.Tensor],
    ) -> float:
        region_set = set(block_ids)

        def center(bid: int) -> Tuple[float, float]:
            rect = layout.get(bid)
            if rect is None:
                rect = outside_positions.get(bid)
            if rect is not None:
                return rect[0] + 0.5 * rect[2], rect[1] + 0.5 * rect[3]
            guide = guides.get(bid)
            if guide is not None:
                return guide[0], guide[1]
            return 0.0, 0.0

        hpwl = 0.0
        total_weight = 0.0
        for i, j, weight in b2b_edges:
            if i not in region_set and j not in region_set:
                continue
            ix, iy = center(i)
            jx, jy = center(j)
            hpwl += weight * (abs(ix - jx) + abs(iy - jy))
            total_weight += weight
        if pins_pos is not None:
            for pin_idx, bid, weight in p2b_edges:
                if bid not in region_set or pin_idx >= len(pins_pos):
                    continue
                bx, by = center(bid)
                px, py = float(pins_pos[pin_idx][0]), float(pins_pos[pin_idx][1])
                hpwl += weight * (abs(bx - px) + abs(by - py))
                total_weight += weight

        area_sum = sum(rect[2] * rect[3] for rect in layout.values())
        hpwl_scale = max(total_weight * math.sqrt(max(area_sum, EPS)), 1.0)
        shape_values: List[float] = []
        extreme = 0.0
        for _, _, w, h in layout.values():
            log_ratio = math.log(max(w, EPS) / max(h, EPS))
            shape_values.append(log_ratio * log_ratio)
            ratio = max(w / max(h, EPS), h / max(w, EPS))
            if ratio > 12.0:
                extreme += ((ratio - 12.0) / 12.0) ** 2
        shape_penalty = sum(shape_values) / max(len(shape_values), 1)
        extreme_penalty = extreme / max(len(shape_values), 1)
        return hpwl / hpwl_scale + 0.02 * shape_penalty + 0.05 * extreme_penalty

    def _region_layout_key(
        self,
        layout: Dict[int, Tuple[float, float, float, float]],
    ) -> Tuple:
        return tuple(
            (bid, *(round(value, 7) for value in layout[bid]))
            for bid in sorted(layout)
        )

    def _refine_region_candidates(
        self,
        region_records: Sequence[
            Tuple[
                List[int],
                List[RegionLayoutCandidate],
                List[Tuple[int, int, float]],
                List[Tuple[int, int, float]],
            ]
        ],
        positions: Dict[int, Tuple[float, float, float, float]],
        guides: Dict[int, Tuple[float, float, float]],
        pins_pos: Optional[torch.Tensor],
    ) -> None:
        for _ in range(2):
            improved = False
            for block_ids, candidates, region_b2b, region_p2b in region_records:
                current_layout = {bid: positions[bid] for bid in block_ids}
                best_layout = current_layout
                best_score = self._score_region_layout(
                    current_layout,
                    block_ids,
                    positions,
                    guides,
                    region_b2b,
                    region_p2b,
                    pins_pos,
                )
                for candidate in candidates:
                    score = self._score_region_layout(
                        candidate.positions,
                        block_ids,
                        positions,
                        guides,
                        region_b2b,
                        region_p2b,
                        pins_pos,
                    )
                    if score + 1e-10 < best_score:
                        best_score = score
                        best_layout = candidate.positions
                if self._region_layout_key(best_layout) != self._region_layout_key(current_layout):
                    positions.update(best_layout)
                    improved = True
            if not improved:
                break

    def _precompute_soft_guides(
        self,
        soft_ids: Sequence[int],
        positions: Dict[int, Tuple[float, float, float, float]],
        b2b_edges: Sequence[Tuple[int, int, float]],
        p2b_edges: Sequence[Tuple[int, int, float]],
        pins_pos: Optional[torch.Tensor],
    ) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, float]]:
        soft_set = set(soft_ids)
        sums: Dict[int, List[float]] = {bid: [0.0, 0.0, 0.0] for bid in soft_ids}
        incident: Dict[int, float] = {bid: 0.0 for bid in soft_ids}

        if pins_pos is not None:
            for pin_idx, bid, weight in p2b_edges:
                if bid not in soft_set or pin_idx >= len(pins_pos):
                    continue
                px, py = float(pins_pos[pin_idx][0]), float(pins_pos[pin_idx][1])
                sums[bid][0] += px * weight
                sums[bid][1] += py * weight
                sums[bid][2] += weight
                incident[bid] += weight

        for i, j, weight in b2b_edges:
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
        b2b_edges: Sequence[Tuple[int, int, float]],
        p2b_edges: Sequence[Tuple[int, int, float]],
        pins_pos: Optional[torch.Tensor],
        axis: int,
    ) -> Dict[int, List[Tuple[float, float]]]:
        soft_set = set(soft_ids)
        targets: Dict[int, List[Tuple[float, float]]] = {bid: [] for bid in soft_ids}

        if pins_pos is not None:
            for pin_idx, bid, weight in p2b_edges:
                if bid in soft_set and pin_idx < len(pins_pos):
                    targets[bid].append((float(pins_pos[pin_idx][axis]), max(weight, EPS)))

        for i, j, weight in b2b_edges:
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
        corner_dims, corner_modes, extents = self._choose_boundary_geometry(
            corners,
            core_w,
            core_h,
            left_blocks,
            right_blocks,
            top_blocks,
            bottom_blocks,
        )

        w_tl, h_tl = corner_dims.get(5, (0.0, 0.0))
        w_tr, h_tr = corner_dims.get(6, (0.0, 0.0))
        w_bl, h_bl = corner_dims.get(9, (0.0, 0.0))
        w_br, h_br = corner_dims.get(10, (0.0, 0.0))

        L, R, T, B = extents

        # Give each corner cell to exactly one adjacent side. Horizontal mode
        # consumes top/bottom length and reserves the full horizontal strip
        # from the vertical side; vertical mode is the symmetric case.
        if corner_modes.get(5, "horizontal") == "horizontal":
            eff_w_tl, eff_h_tl = w_tl, T
        else:
            eff_w_tl, eff_h_tl = L, h_tl
        if corner_modes.get(6, "horizontal") == "horizontal":
            eff_w_tr, eff_h_tr = w_tr, T
        else:
            eff_w_tr, eff_h_tr = R, h_tr
        if corner_modes.get(9, "horizontal") == "horizontal":
            eff_w_bl, eff_h_bl = w_bl, B
        else:
            eff_w_bl, eff_h_bl = L, h_bl
        if corner_modes.get(10, "horizontal") == "horizontal":
            eff_w_br, eff_h_br = w_br, B
        else:
            eff_w_br, eff_h_br = R, h_br

        X0 = core_x0 - L
        X1 = core_x1 + R
        Y0 = core_y0 - B
        Y1 = core_y1 + T

        gap_left = max((Y1 - Y0) - eff_h_tl - eff_h_bl, 1e-6)
        gap_right = max((Y1 - Y0) - eff_h_tr - eff_h_br, 1e-6)
        gap_top = max((X1 - X0) - eff_w_tl - eff_w_tr, 1e-6)
        gap_bottom = max((X1 - X0) - eff_w_bl - eff_w_br, 1e-6)
        left_strip_w, _ = self._vertical_side_requirements(left_blocks, gap_left)
        right_strip_w, _ = self._vertical_side_requirements(right_blocks, gap_right)
        top_strip_h, _ = self._horizontal_side_requirements(top_blocks, gap_top)
        bottom_strip_h, _ = self._horizontal_side_requirements(bottom_blocks, gap_bottom)

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
            left_strip_w, positions,
        )
        self._place_vertical_side(
            right_blocks, X1, Y0 + eff_h_br, Y1 - eff_h_tr,
            right_strip_w, positions, right_align=True,
        )
        self._place_horizontal_side(
            top_blocks, X0 + eff_w_tl, Y1, X1 - eff_w_tr,
            top_strip_h, positions, top_align=True,
        )
        self._place_horizontal_side(
            bottom_blocks, X0 + eff_w_bl, Y0, X1 - eff_w_br,
            bottom_strip_h, positions, top_align=False,
        )

        # Safety: resolve any remaining overlaps among boundary blocks
        # that might arise from floating-point or convergence imprecision.
        positions = self._resolve_boundary_overlaps(
            positions, boundary_ids, block_map,
        )

        return positions

    def _choose_boundary_geometry(
        self,
        corners: Dict[int, Optional[BlockSpec]],
        core_w: float,
        core_h: float,
        left_blocks: Sequence[BlockSpec],
        right_blocks: Sequence[BlockSpec],
        top_blocks: Sequence[BlockSpec],
        bottom_blocks: Sequence[BlockSpec],
    ) -> Tuple[
        Dict[int, Tuple[float, float]],
        Dict[int, str],
        Tuple[float, float, float, float],
    ]:
        codes = (5, 6, 9, 10)
        best_dims: Optional[Dict[int, Tuple[float, float]]] = None
        best_modes: Optional[Dict[int, str]] = None
        best_extents: Optional[Tuple[float, float, float, float]] = None
        best_score = (float("inf"), float("inf"), float("inf"))

        # Empty corners also receive a direction: this assigns the empty
        # corner cell to one of its two adjacent side strips. Four binary
        # decisions give only 16 combinations.
        for mask in range(1 << len(codes)):
            modes = {
                code: "vertical" if mask & (1 << idx) else "horizontal"
                for idx, code in enumerate(codes)
            }
            solved = self._solve_boundary_extents(
                corners,
                modes,
                core_w,
                core_h,
                left_blocks,
                right_blocks,
                top_blocks,
                bottom_blocks,
            )
            if solved is None:
                continue
            dims, extents = solved
            L, R, T, B = extents
            final_w = core_w + L + R
            final_h = core_h + T + B
            score = (final_w * final_h, final_w + final_h, max(final_w, final_h))
            if score < best_score:
                best_score = score
                best_dims = dims
                best_modes = modes
                best_extents = extents

        if best_dims is None or best_modes is None or best_extents is None:
            raise RuntimeError("No feasible boundary corner direction assignment")
        return best_dims, best_modes, best_extents

    def _solve_boundary_extents(
        self,
        corners: Dict[int, Optional[BlockSpec]],
        corner_modes: Dict[int, str],
        core_w: float,
        core_h: float,
        left_blocks: Sequence[BlockSpec],
        right_blocks: Sequence[BlockSpec],
        top_blocks: Sequence[BlockSpec],
        bottom_blocks: Sequence[BlockSpec],
    ) -> Optional[
        Tuple[
            Dict[int, Tuple[float, float]],
            Tuple[float, float, float, float],
        ]
    ]:
        mode_tl = corner_modes[5]
        mode_tr = corner_modes[6]
        mode_bl = corner_modes[9]
        mode_br = corner_modes[10]

        left_content = list(left_blocks)
        right_content = list(right_blocks)
        top_content = list(top_blocks)
        bottom_content = list(bottom_blocks)
        for code, block in corners.items():
            if block is None:
                continue
            if corner_modes[code] == "horizontal":
                (top_content if code in (5, 6) else bottom_content).append(block)
            else:
                (left_content if code in (5, 9) else right_content).append(block)

        all_content = left_content + right_content + top_content + bottom_content
        area_scale = math.sqrt(max(sum(block.area for block in all_content), 1.0))
        seed_h = max(core_h, area_scale, 1.0)
        seed_w = max(core_w, area_scale, 1.0)
        L, _ = self._vertical_side_requirements(left_content, seed_h)
        R, _ = self._vertical_side_requirements(right_content, seed_h)
        T, _ = self._horizontal_side_requirements(top_content, seed_w)
        B, _ = self._horizontal_side_requirements(bottom_content, seed_w)

        fixed_h_left = sum(block.height for block in left_content if block.fixed)
        fixed_h_right = sum(block.height for block in right_content if block.fixed)
        fixed_w_top = sum(block.width for block in top_content if block.fixed)
        fixed_w_bottom = sum(block.width for block in bottom_content if block.fixed)

        def grow_pair(
            first: float,
            second: float,
            shortage: float,
            first_allowed: bool,
            second_allowed: bool,
        ) -> Optional[Tuple[float, float]]:
            if shortage <= 1e-8:
                return first, second
            if first_allowed and second_allowed:
                return first + 0.5 * shortage, second + 0.5 * shortage
            if first_allowed:
                return first + shortage, second
            if second_allowed:
                return first, second + shortage
            return None

        def required_extents(
            curL: float,
            curR: float,
            curT: float,
            curB: float,
        ) -> Optional[Tuple[float, float, float, float]]:
            # A horizontal corner owns the corner cell on top/bottom, so that
            # strip thickness is removed from the adjacent vertical edge.
            gap_left = core_h
            gap_left += curT if mode_tl == "vertical" else 0.0
            gap_left += curB if mode_bl == "vertical" else 0.0
            gap_right = core_h
            gap_right += curT if mode_tr == "vertical" else 0.0
            gap_right += curB if mode_br == "vertical" else 0.0
            gap_top = core_w
            gap_top += curL if mode_tl == "horizontal" else 0.0
            gap_top += curR if mode_tr == "horizontal" else 0.0
            gap_bottom = core_w
            gap_bottom += curL if mode_bl == "horizontal" else 0.0
            gap_bottom += curR if mode_br == "horizontal" else 0.0

            newL, _ = self._vertical_side_requirements(left_content, max(gap_left, 1e-6))
            newR, _ = self._vertical_side_requirements(right_content, max(gap_right, 1e-6))
            newT, _ = self._horizontal_side_requirements(top_content, max(gap_top, 1e-6))
            newB, _ = self._horizontal_side_requirements(bottom_content, max(gap_bottom, 1e-6))

            # Fixed blocks need actual edge length; unlike soft blocks they
            # cannot compensate for a shortage by increasing strip thickness.
            shortage = fixed_h_left - (
                core_h
                + (newT if mode_tl == "vertical" else 0.0)
                + (newB if mode_bl == "vertical" else 0.0)
            )
            grown = grow_pair(
                newT, newB, shortage,
                mode_tl == "vertical", mode_bl == "vertical",
            )
            if grown is None:
                return None
            newT, newB = grown

            shortage = fixed_h_right - (
                core_h
                + (newT if mode_tr == "vertical" else 0.0)
                + (newB if mode_br == "vertical" else 0.0)
            )
            grown = grow_pair(
                newT, newB, shortage,
                mode_tr == "vertical", mode_br == "vertical",
            )
            if grown is None:
                return None
            newT, newB = grown

            shortage = fixed_w_top - (
                core_w
                + (newL if mode_tl == "horizontal" else 0.0)
                + (newR if mode_tr == "horizontal" else 0.0)
            )
            grown = grow_pair(
                newL, newR, shortage,
                mode_tl == "horizontal", mode_tr == "horizontal",
            )
            if grown is None:
                return None
            newL, newR = grown

            shortage = fixed_w_bottom - (
                core_w
                + (newL if mode_bl == "horizontal" else 0.0)
                + (newR if mode_br == "horizontal" else 0.0)
            )
            grown = grow_pair(
                newL, newR, shortage,
                mode_bl == "horizontal", mode_br == "horizontal",
            )
            if grown is None:
                return None
            newL, newR = grown
            return newL, newR, newT, newB

        # The horizontal requirements depend only on L/R and the vertical
        # requirements only on T/B. Damping removes the possible two-cycle.
        for _ in range(96):
            required = required_extents(L, R, T, B)
            if required is None:
                return None
            newL, newR, newT, newB = required
            scale = max(1.0, L, R, T, B, newL, newR, newT, newB)
            delta = max(
                abs(newL - L), abs(newR - R),
                abs(newT - T), abs(newB - B),
            )
            if delta <= 1e-8 * scale:
                L, R, T, B = required
                break
            L = 0.5 * (L + newL)
            R = 0.5 * (R + newR)
            T = 0.5 * (T + newT)
            B = 0.5 * (B + newB)
        else:
            required = required_extents(L, R, T, B)
            if required is None:
                return None
            L, R, T, B = required

        if not all(math.isfinite(value) and value >= 0.0 for value in (L, R, T, B)):
            return None

        corner_dims: Dict[int, Tuple[float, float]] = {}
        for code, block in corners.items():
            if block is None:
                corner_dims[code] = (0.0, 0.0)
                continue
            if block.fixed:
                corner_dims[code] = (block.width, block.height)
                continue
            if corner_modes[code] == "horizontal":
                thickness = T if code in (5, 6) else B
                if thickness <= EPS:
                    return None
                corner_dims[code] = (block.area / thickness, thickness)
            else:
                thickness = L if code in (5, 9) else R
                if thickness <= EPS:
                    return None
                corner_dims[code] = (thickness, block.area / thickness)

        total_w = core_w + L + R
        total_h = core_h + T + B
        w_tl, h_tl = corner_dims[5]
        w_tr, h_tr = corner_dims[6]
        w_bl, h_bl = corner_dims[9]
        w_br, h_br = corner_dims[10]
        if max(w_tl + w_tr, w_bl + w_br) > total_w + 1e-5:
            return None
        if max(h_tl + h_bl, h_tr + h_br) > total_h + 1e-5:
            return None
        return corner_dims, (L, R, T, B)

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
        heights: Dict[int, float] = {}
        widths: Dict[int, float] = {}
        for block in blocks:
            if block.fixed:
                w, h = block.width, block.height
            else:
                w = strip_w
                h = block.area / max(w, EPS)
            widths[block.block_id] = w
            heights[block.block_id] = h

        ordered, centers = self._boundary_side_centers(
            blocks, heights, start_y, end_y, 1, positions
        )
        for block in ordered:
            w = widths[block.block_id]
            h = heights[block.block_id]
            x = side_x - w if right_align else side_x
            positions[block.block_id] = (x, centers[block.block_id] - 0.5 * h, w, h)

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
        widths: Dict[int, float] = {}
        heights: Dict[int, float] = {}
        for block in blocks:
            if block.fixed:
                w, h = block.width, block.height
            else:
                h = strip_h
                w = block.area / max(h, EPS)
            widths[block.block_id] = w
            heights[block.block_id] = h

        ordered, centers = self._boundary_side_centers(
            blocks, widths, start_x, end_x, 0, positions
        )
        for block in ordered:
            w = widths[block.block_id]
            h = heights[block.block_id]
            y = side_y - h if top_align else side_y
            positions[block.block_id] = (centers[block.block_id] - 0.5 * w, y, w, h)

    def _boundary_side_centers(
        self,
        blocks: Sequence[BlockSpec],
        lengths: Dict[int, float],
        lower: float,
        upper: float,
        axis: int,
        positions: Dict[int, Tuple[float, float, float, float]],
    ) -> Tuple[List[BlockSpec], Dict[int, float]]:
        if not blocks:
            return [], {}
        block_ids = {block.block_id for block in blocks}
        targets: Dict[int, List[Tuple[float, float]]] = {
            block.block_id: [] for block in blocks
        }

        pins_pos = getattr(self, "_current_pins_pos", None)
        if pins_pos is not None:
            for pin_idx, bid, weight in getattr(self, "_current_p2b_edges", []):
                if bid in block_ids and pin_idx < len(pins_pos):
                    targets[bid].append((float(pins_pos[pin_idx][axis]), max(weight, EPS)))

        for i, j, weight in getattr(self, "_current_b2b_edges", []):
            if i in block_ids and j in positions:
                rect = positions[j]
                targets[i].append((rect[axis] + 0.5 * rect[axis + 2], max(weight, EPS)))
            if j in block_ids and i in positions:
                rect = positions[i]
                targets[j].append((rect[axis] + 0.5 * rect[axis + 2], max(weight, EPS)))

        block_map = getattr(self, "_current_block_map", {})
        group_members: Dict[int, List[int]] = {}
        for bid, block in block_map.items():
            if block.group_id > 0 and bid in positions:
                group_members.setdefault(block.group_id, []).append(bid)
        for block in blocks:
            if block.group_id <= 0:
                continue
            existing_weight = sum(weight for _, weight in targets[block.block_id])
            group_weight = max(existing_weight, 1.0) * 2.0
            for member_id in group_members.get(block.group_id, []):
                rect = positions[member_id]
                targets[block.block_id].append(
                    (rect[axis] + 0.5 * rect[axis + 2], group_weight)
                )

        midpoint = 0.5 * (lower + upper)
        values: List[float] = []
        for block in blocks:
            bid = block.block_id
            if not targets[bid]:
                targets[bid] = [(midpoint, 1.0)]
            clipped = []
            for value, weight in targets[bid]:
                value = min(max(value, lower), upper)
                clipped.append((value, weight))
                values.append(value)
            targets[bid] = clipped

        unique_values = sorted({round(value, 8) for value in values})
        rank = {value: idx for idx, value in enumerate(unique_values)}

        def order_key(block: BlockSpec) -> Tuple[float, float, int]:
            weighted_rank = sum(
                rank[round(value, 8)] * weight
                for value, weight in targets[block.block_id]
            )
            total_weight = sum(weight for _, weight in targets[block.block_id])
            return weighted_rank / max(total_weight, EPS), -block.area, block.block_id

        ordered = sorted(blocks, key=order_key)
        order_ids = [block.block_id for block in ordered]
        centers = self._pava_centers(order_ids, lengths, targets, lower, upper)
        if centers is None:
            centers = {}
            cursor = lower + max(0.0, upper - lower - sum(lengths[bid] for bid in order_ids)) * 0.5
            for bid in order_ids:
                centers[bid] = cursor + 0.5 * lengths[bid]
                cursor += lengths[bid]
        return ordered, centers

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
