#!/usr/bin/env python3
"""
Grouping-as-item + boundary-post-placement heuristic with MIB square templates.

Main rules for this version:
1. Grouping subproblems are built only from non-preplaced, non-boundary blocks.
2. Preplaced blocks ignore all soft constraints.
3. Boundary blocks are placed after the core layout.
4. Blocks in the same MIB group are forced to share one immutable template:
   - if the group contains fixed-shape blocks, use the fixed block shape
     for the whole group
   - otherwise use a square template derived from the common area target
"""

import math
import sys
import importlib.util
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
        self.state_candidate_limit = 3
        self.local_passes = 1
        self.hpwl_weight = 1.0
        self.bbox_weight = 0.40
        self.anchor_weight = 0.08
        self.baseline_optimizer = None
        self.b2b_edges: List[Tuple[int, int, float]] = []
        self.p2b_edges: List[Tuple[int, int, float]] = []
        self.incident_b2b: List[List[int]] = []
        self.incident_p2b: List[List[Tuple[int, float]]] = []
        self.block_wire_weight: List[float] = []
        self.block_pin_weight: List[float] = []
        self.target_centers: Dict[int, Tuple[float, float]] = {}
        self.item_target_xy: Dict[int, Tuple[float, float]] = {}
        self.item_wire_weight: Dict[int, float] = {}
        self.total_area_norm = 1.0
        self.sqrt_area_norm = 1.0
        self.hpwl_norm = 1.0
        self.pins_pos = None
        self.has_p2b_anchor = False

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
                if block.mib_id > 0:
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
        self._prepare_wire_model(block_count, b2b_connectivity, p2b_connectivity, pins_pos)
        self._estimate_target_centers(blocks, pins_pos)
        self._prepare_item_targets(core_items)

        best_positions: Optional[List[Tuple[float, float, float, float]]] = None
        best_score: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

        if not core_items and not fixed_positions and soft_single_ids:
            soft_blocks = [block_map[idx] for idx in soft_single_ids]
            core_positions = self._solve_soft_only(soft_blocks)
            core_rects = {block.block_id: core_positions[block.block_id] for block in soft_blocks}
            candidate = self._finalize_layout(blocks, core_rects, boundary_ids)
            best_positions = candidate
            best_score = self._layout_score(candidate)
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

                core_positions = self._fill_soft_singles(block_map, core_positions, soft_single_ids)
                candidate = self._finalize_layout(blocks, core_positions, boundary_ids)
                score = self._layout_score(candidate)
                if score < best_score:
                    best_score = score
                    best_positions = candidate

        if best_positions is None:
            fallback_positions = dict(fixed_positions)
            fallback_positions = self._fill_soft_singles(block_map, fallback_positions, soft_single_ids)
            best_positions = self._finalize_layout(blocks, fallback_positions, boundary_ids)

        if not has_preplaced and not self.has_p2b_anchor:
            best_positions = self._shift_to_origin(best_positions)

        # Final safety net: resolve any remaining overlaps
        best_positions = self._resolve_all_overlaps(best_positions)
        best_positions = self._adjacent_hpwl_swaps(best_positions, blocks)
        best_positions = self._choose_against_baseline(
            best_positions,
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
            blocks,
        )

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
            # Hard constraints take priority: if a MIB group contains any
            # fixed-shape block, the whole group follows that fixed template.
            fixed_members = [block for block in members if block.fixed]
            if fixed_members:
                ref = fixed_members[0]
                template_w = ref.width
                template_h = ref.height
            else:
                area = members[0].area
                side = math.sqrt(max(area, 1e-6))
                template_w = side
                template_h = side

            for block in members:
                if block.preplaced or block.fixed:
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
    # Wirelength model and target centers
    # ------------------------------------------------------------------
    def _prepare_wire_model(
        self,
        block_count: int,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> None:
        self.pins_pos = pins_pos
        self.b2b_edges = []
        self.p2b_edges: List[Tuple[int, int, float]] = []
        self.incident_b2b = [[] for _ in range(block_count)]
        self.incident_p2b = [[] for _ in range(block_count)]
        self.block_wire_weight = [0.0 for _ in range(block_count)]
        self.block_pin_weight = [0.0 for _ in range(block_count)]
        self.has_p2b_anchor = False

        total_weight = 0.0
        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                block_i = int(edge[0])
                if block_i == -1:
                    continue
                block_j = int(edge[1])
                if not (0 <= block_i < block_count and 0 <= block_j < block_count):
                    continue
                weight = max(0.0, float(edge[2]))
                edge_idx = len(self.b2b_edges)
                self.b2b_edges.append((block_i, block_j, weight))
                self.incident_b2b[block_i].append(edge_idx)
                if block_j != block_i:
                    self.incident_b2b[block_j].append(edge_idx)
                self.block_wire_weight[block_i] += weight
                self.block_wire_weight[block_j] += weight
                total_weight += weight

        pin_count = len(pins_pos) if pins_pos is not None else 0
        if p2b_connectivity is not None:
            for edge in p2b_connectivity:
                pin_idx = int(edge[0])
                if pin_idx == -1:
                    continue
                block_idx = int(edge[1])
                if not (0 <= block_idx < block_count and 0 <= pin_idx < pin_count):
                    continue
                weight = max(0.0, float(edge[2]))
                self.p2b_edges.append((pin_idx, block_idx, weight))
                self.incident_p2b[block_idx].append((pin_idx, weight))
                self.block_wire_weight[block_idx] += weight
                self.block_pin_weight[block_idx] += weight
                self.has_p2b_anchor = True
                total_weight += weight

        self.hpwl_norm = max(1.0, total_weight)

    def _estimate_target_centers(
        self,
        blocks: Sequence[BlockSpec],
        pins_pos: torch.Tensor,
    ) -> None:
        total_area = 0.0
        preplaced_centers: List[Tuple[float, float]] = []
        for block in blocks:
            if block.width is not None and block.height is not None:
                total_area += max(block.width * block.height, EPS)
            else:
                total_area += max(block.area, EPS)
            if block.preplaced:
                preplaced_centers.append(
                    (block.x + 0.5 * block.width, block.y + 0.5 * block.height)
                )

        self.total_area_norm = max(total_area, 1.0)
        self.sqrt_area_norm = math.sqrt(self.total_area_norm)
        self.hpwl_norm = max(self.hpwl_norm * self.sqrt_area_norm, 1.0)

        pin_centers: List[Tuple[float, float]] = []
        if pins_pos is not None and self.p2b_edges:
            used_pins = {pin_idx for pin_idx, _, _ in self.p2b_edges}
            for pin_idx in used_pins:
                if 0 <= pin_idx < len(pins_pos):
                    pin_centers.append((float(pins_pos[pin_idx][0]), float(pins_pos[pin_idx][1])))

        if preplaced_centers:
            anchor_x = sum(x for x, _ in preplaced_centers) / len(preplaced_centers)
            anchor_y = sum(y for _, y in preplaced_centers) / len(preplaced_centers)
        elif pin_centers:
            anchor_x = sum(x for x, _ in pin_centers) / len(pin_centers)
            anchor_y = sum(y for _, y in pin_centers) / len(pin_centers)
        else:
            anchor_x = 0.0
            anchor_y = 0.0

        centers: Dict[int, Tuple[float, float]] = {}
        for block in blocks:
            if block.preplaced:
                centers[block.block_id] = (
                    block.x + 0.5 * block.width,
                    block.y + 0.5 * block.height,
                )
            else:
                centers[block.block_id] = (anchor_x, anchor_y)

        for _ in range(7):
            prev = centers
            centers = dict(prev)
            for block in blocks:
                block_id = block.block_id
                if block.preplaced:
                    continue
                xs: List[Tuple[float, float]] = []
                ys: List[Tuple[float, float]] = []
                for edge_idx in self.incident_b2b[block_id]:
                    a, b, weight = self.b2b_edges[edge_idx]
                    other = b if a == block_id else a
                    ox, oy = prev.get(other, (anchor_x, anchor_y))
                    xs.append((ox, weight))
                    ys.append((oy, weight))
                if pins_pos is not None:
                    for pin_idx, weight in self.incident_p2b[block_id]:
                        if 0 <= pin_idx < len(pins_pos):
                            xs.append((float(pins_pos[pin_idx][0]), weight))
                            ys.append((float(pins_pos[pin_idx][1]), weight))
                if xs:
                    centers[block_id] = (
                        self._weighted_median(xs),
                        self._weighted_median(ys),
                    )

        self.target_centers = centers

    def _prepare_item_targets(self, items: Sequence[LayoutItem]) -> None:
        self.item_target_xy = {}
        self.item_wire_weight = {}
        for item in items:
            sum_w = 0.0
            sum_x = 0.0
            sum_y = 0.0
            for block_id, (lx, ly, w, h) in item.local_rects.items():
                target_x, target_y = self.target_centers.get(block_id, (0.0, 0.0))
                weight = max(self.block_wire_weight[block_id], 1.0)
                sum_w += weight
                sum_x += weight * (target_x - lx - 0.5 * w)
                sum_y += weight * (target_y - ly - 0.5 * h)
            if sum_w <= EPS:
                self.item_target_xy[item.item_id] = (0.0, 0.0)
                self.item_wire_weight[item.item_id] = 0.0
            else:
                self.item_target_xy[item.item_id] = (sum_x / sum_w, sum_y / sum_w)
                self.item_wire_weight[item.item_id] = sum_w

    def _weighted_median(self, values: Sequence[Tuple[float, float]]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values, key=lambda item: item[0])
        total = sum(max(weight, 0.0) for _, weight in ordered)
        if total <= EPS:
            return ordered[len(ordered) // 2][0]
        acc = 0.0
        half = 0.5 * total
        for value, weight in ordered:
            acc += max(weight, 0.0)
            if acc + EPS >= half:
                return value
        return ordered[-1][0]

    def _layout_score(
        self,
        positions: Sequence[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float]:
        hpwl = self._positions_hpwl(positions)
        area = calculate_bbox_area(list(positions))
        mixed = self.hpwl_weight * (hpwl / self.hpwl_norm) + self.bbox_weight * (area / self.total_area_norm)
        return (mixed, area, hpwl)

    def _positions_hpwl(
        self,
        positions: Sequence[Tuple[float, float, float, float]],
    ) -> float:
        total = 0.0
        for block_i, block_j, weight in self.b2b_edges:
            if block_i >= len(positions) or block_j >= len(positions):
                continue
            total += self._rect_center_distance(positions[block_i], positions[block_j], weight)
        if self.pins_pos is not None:
            for pin_idx, block_id, weight in self.p2b_edges:
                if block_id >= len(positions) or pin_idx >= len(self.pins_pos):
                    continue
                rect = positions[block_id]
                bx = rect[0] + 0.5 * rect[2]
                by = rect[1] + 0.5 * rect[3]
                px = float(self.pins_pos[pin_idx][0])
                py = float(self.pins_pos[pin_idx][1])
                total += weight * (abs(bx - px) + abs(by - py))
        return total

    def _rect_center_distance(
        self,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
        weight: float,
    ) -> float:
        ax = a[0] + 0.5 * a[2]
        ay = a[1] + 0.5 * a[3]
        bx = b[0] + 0.5 * b[2]
        by = b[1] + 0.5 * b[3]
        return weight * (abs(ax - bx) + abs(ay - by))

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
                        placed_layout=state,
                        item_map=item_map,
                        fixed_positions=fixed_positions,
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
                improved = self._local_refine_items(
                    fixed_rects, fixed_bbox, dict(state), item_map, order, fixed_positions
                )
                best_states.append((self._score_item_state(fixed_positions, improved, item_map), improved))

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
            sorted(items, key=lambda item: (-self.item_wire_weight.get(item.item_id, 0.0), -item.area, item.item_id)),
            sorted(items, key=lambda item: (center_metric(item), -item.area, item.item_id)),
            sorted(items, key=lambda item: (self.item_target_xy.get(item.item_id, (0.0, 0.0))[0], item.item_id)),
            sorted(items, key=lambda item: (self.item_target_xy.get(item.item_id, (0.0, 0.0))[1], item.item_id)),
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
        placed_layout: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        item_map: Optional[Dict[int, LayoutItem]] = None,
        fixed_positions: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    ) -> List[Tuple[Tuple[float, float, float, float, float], float, float]]:
        w, h = item.width, item.height
        target_x, target_y = self.item_target_xy.get(item.item_id, (0.0, 0.0))

        if occupied:
            x0, y0, x1, y1 = occupied_bbox if occupied_bbox is not None else self._bbox_xyxy(occupied)
            xs = {x0, x1, x0 - w, x1 - w, target_x}
            ys = {y0, y1, y0 - h, y1 - h, target_y}
            for rx, ry, rw, rh in occupied:
                xs.update([rx - w, rx, rx + rw - w, rx + rw])
                ys.update([ry - h, ry, ry + rh - h, ry + rh])
                xs.update([rx + 0.5 * rw - 0.5 * w, rx + rw - w, rx])
                ys.update([ry + 0.5 * rh - 0.5 * h, ry + rh - h, ry])
        else:
            xs = {0.0, target_x}
            ys = {0.0, target_y}
            occupied_bbox = None

        xs = self._limited_candidate_coords(xs, target_x, 14)
        ys = self._limited_candidate_coords(ys, target_y, 14)

        candidates = []
        seen = set()
        overlaps_any = self._overlaps_any
        placed_blocks = self._placed_block_positions(
            placed_layout or {},
            item_map or {},
            fixed_positions or {},
        )
        for x in xs:
            for y in ys:
                key = (round(x, 6), round(y, 6))
                if key in seen:
                    continue
                seen.add(key)
                rect = (x, y, w, h)
                if overlaps_any(rect, occupied):
                    continue
                score = self._score_item_candidate(
                    occupied_bbox,
                    item,
                    rect,
                    placed_blocks,
                )
                candidates.append((score, x, y))

        if allow_origin and not seen:
            rect = (0.0, 0.0, w, h)
            score = self._score_item_candidate(occupied_bbox, item, rect, placed_blocks)
            candidates.append((score, 0.0, 0.0))

        candidates.sort(key=lambda item_state: item_state[0])
        return candidates[: self.state_candidate_limit]

    def _limited_candidate_coords(
        self,
        coords: Iterable[float],
        target: float,
        limit: int,
    ) -> List[float]:
        unique = sorted({round(float(coord), 7) for coord in coords})
        if len(unique) <= limit:
            return unique
        by_target = sorted(unique, key=lambda coord: (abs(coord - target), coord))[: max(4, limit - 4)]
        extremes = [unique[0], unique[-1]]
        middle = [unique[len(unique) // 4], unique[len(unique) // 2], unique[(3 * len(unique)) // 4]]
        selected = []
        seen = set()
        for coord in by_target + extremes + middle:
            if coord in seen:
                continue
            seen.add(coord)
            selected.append(coord)
            if len(selected) >= limit:
                break
        return selected

    def _placed_block_positions(
        self,
        placed_layout: Dict[int, Tuple[float, float, float, float]],
        item_map: Dict[int, LayoutItem],
        fixed_positions: Dict[int, Tuple[float, float, float, float]],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        positions = dict(fixed_positions)
        for item_id, rect in placed_layout.items():
            item = item_map.get(item_id)
            if item is None:
                continue
            ix, iy, _, _ = rect
            for block_id, (lx, ly, w, h) in item.local_rects.items():
                positions[block_id] = (ix + lx, iy + ly, w, h)
        return positions

    def _score_item_candidate(
        self,
        occupied_bbox: Optional[Tuple[float, float, float, float]],
        item: LayoutItem,
        rect: Tuple[float, float, float, float],
        placed_blocks: Dict[int, Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float, float, float]:
        bbox = self._expand_bbox(occupied_bbox, rect)
        bbox_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0.0)
        bbox_perimeter = max((bbox[2] - bbox[0]) + (bbox[3] - bbox[1]), 0.0)
        wire = self._item_candidate_wire(item, rect, placed_blocks)
        anchor = self._item_anchor_distance(item, rect)
        mixed = (
            self.hpwl_weight * (wire / self.hpwl_norm)
            + self.bbox_weight * (bbox_area / self.total_area_norm)
            + self.anchor_weight * (anchor / self.sqrt_area_norm)
        )
        return (mixed, bbox_area, wire, anchor, bbox_perimeter)

    def _item_candidate_wire(
        self,
        item: LayoutItem,
        item_rect: Tuple[float, float, float, float],
        placed_blocks: Dict[int, Tuple[float, float, float, float]],
    ) -> float:
        ix, iy, _, _ = item_rect
        member_rects: Dict[int, Tuple[float, float, float, float]] = {}
        for block_id, (lx, ly, w, h) in item.local_rects.items():
            member_rects[block_id] = (ix + lx, iy + ly, w, h)

        total = 0.0
        member_set = set(member_rects)
        for block_id, rect in member_rects.items():
            cx = rect[0] + 0.5 * rect[2]
            cy = rect[1] + 0.5 * rect[3]
            for edge_idx in self.incident_b2b[block_id]:
                a, b, weight = self.b2b_edges[edge_idx]
                other = b if a == block_id else a
                if other in member_set:
                    continue
                if other in placed_blocks:
                    other_rect = placed_blocks[other]
                    ox = other_rect[0] + 0.5 * other_rect[2]
                    oy = other_rect[1] + 0.5 * other_rect[3]
                else:
                    ox, oy = self.target_centers.get(other, (cx, cy))
                total += weight * (abs(cx - ox) + abs(cy - oy))
            if self.pins_pos is not None:
                for pin_idx, weight in self.incident_p2b[block_id]:
                    if 0 <= pin_idx < len(self.pins_pos):
                        px = float(self.pins_pos[pin_idx][0])
                        py = float(self.pins_pos[pin_idx][1])
                        total += weight * (abs(cx - px) + abs(cy - py))
        return total

    def _item_anchor_distance(
        self,
        item: LayoutItem,
        item_rect: Tuple[float, float, float, float],
    ) -> float:
        target_x, target_y = self.item_target_xy.get(item.item_id, (0.0, 0.0))
        return abs(item_rect[0] - target_x) + abs(item_rect[1] - target_y)

    def _score_item_state(
        self,
        fixed_positions: Dict[int, Tuple[float, float, float, float]],
        state: Dict[int, Tuple[float, float, float, float]],
        item_map: Dict[int, LayoutItem],
    ) -> Tuple[float, float, float, float, float]:
        block_positions = self._placed_block_positions(state, item_map, fixed_positions)
        rects = list(block_positions.values())
        bbox_area = 0.0
        bbox_perimeter = 0.0
        if rects:
            x0, y0, x1, y1 = self._bbox_xyxy(rects)
            bbox_area = max((x1 - x0) * (y1 - y0), 0.0)
            bbox_perimeter = max((x1 - x0) + (y1 - y0), 0.0)

        wire = 0.0
        for a, b, weight in self.b2b_edges:
            a_pos = block_positions.get(a)
            b_pos = block_positions.get(b)
            if a_pos is not None:
                ax = a_pos[0] + 0.5 * a_pos[2]
                ay = a_pos[1] + 0.5 * a_pos[3]
            else:
                ax, ay = self.target_centers.get(a, (0.0, 0.0))
            if b_pos is not None:
                bx = b_pos[0] + 0.5 * b_pos[2]
                by = b_pos[1] + 0.5 * b_pos[3]
            else:
                bx, by = self.target_centers.get(b, (ax, ay))
            wire += weight * (abs(ax - bx) + abs(ay - by))

        if self.pins_pos is not None:
            for pin_idx, block_id, weight in self.p2b_edges:
                rect = block_positions.get(block_id)
                if rect is None or pin_idx >= len(self.pins_pos):
                    continue
                bx = rect[0] + 0.5 * rect[2]
                by = rect[1] + 0.5 * rect[3]
                px = float(self.pins_pos[pin_idx][0])
                py = float(self.pins_pos[pin_idx][1])
                wire += weight * (abs(bx - px) + abs(by - py))

        anchor = 0.0
        for item_id, rect in state.items():
            item = item_map.get(item_id)
            if item is not None:
                anchor += self._item_anchor_distance(item, rect)

        mixed = (
            self.hpwl_weight * (wire / self.hpwl_norm)
            + self.bbox_weight * (bbox_area / self.total_area_norm)
            + self.anchor_weight * (anchor / max(self.sqrt_area_norm, 1.0))
        )
        return (mixed, bbox_area, wire, anchor, bbox_perimeter)

    def _local_refine_items(
        self,
        fixed_rects: Sequence[Tuple[float, float, float, float]],
        fixed_bbox: Optional[Tuple[float, float, float, float]],
        state: Dict[int, Tuple[float, float, float, float]],
        item_map: Dict[int, LayoutItem],
        order: Sequence[int],
        fixed_positions: Dict[int, Tuple[float, float, float, float]],
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
                    placed_blocks = self._placed_block_positions(state, item_map, fixed_positions)
                    best_score = self._score_item_candidate(
                        occupied_bbox,
                        item,
                        current,
                        placed_blocks,
                    )
                    candidates = [(best_score, current[0], current[1])]
                candidates.extend(
                    self._rank_item_candidates(
                        occupied,
                        occupied_bbox,
                        item,
                        allow_origin=(not occupied),
                        placed_layout=state,
                        item_map=item_map,
                        fixed_positions=fixed_positions,
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
    ) -> Dict[int, Tuple[float, float, float, float]]:
        if not soft_ids:
            return positions

        if not positions:
            total_area = sum(block_map[bid].area for bid in soft_ids)
            height = math.sqrt(max(total_area, 1.0))
            ordered_ids = self._ordered_soft_ids(block_map, soft_ids)
            widths = {block_id: block_map[block_id].area / height for block_id in ordered_ids}
            total_w = sum(widths.values())
            if self.has_p2b_anchor:
                avg_x = sum(self.target_centers.get(bid, (0.0, 0.0))[0] for bid in ordered_ids) / max(len(ordered_ids), 1)
                avg_y = sum(self.target_centers.get(bid, (0.0, 0.0))[1] for bid in ordered_ids) / max(len(ordered_ids), 1)
                cursor_x = avg_x - 0.5 * total_w
                y = avg_y - 0.5 * height
            else:
                cursor_x = 0.0
                y = 0.0
            for block_id in ordered_ids:
                area = block_map[block_id].area
                width = widths[block_id]
                positions[block_id] = (cursor_x, y, width, height)
                cursor_x += width
            return positions

        free_rects = self._extract_free_rectangles(positions.values())
        bins = [FreeRect(*rect) for rect in free_rects]
        unplaced: List[int] = []
        for block_id in self._ordered_soft_ids(block_map, soft_ids):
            choice = self._choose_soft_placement(bins, positions, block_map, block_id)
            if choice is None:
                unplaced.append(block_id)
                continue
            chosen_idx, placed, leftover = choice
            positions[block_id] = placed
            if leftover is None or leftover.w <= EPS or leftover.h <= EPS:
                bins.pop(chosen_idx)
            else:
                bins[chosen_idx] = leftover

        if unplaced:
            self._pack_remaining_soft_strip(positions, block_map, unplaced)
        return positions

    def _ordered_soft_ids(
        self,
        block_map: Dict[int, BlockSpec],
        soft_ids: Sequence[int],
    ) -> List[int]:
        return sorted(
            soft_ids,
            key=lambda bid: (
                -self.block_wire_weight[bid] if bid < len(self.block_wire_weight) else 0.0,
                -self.block_pin_weight[bid] if bid < len(self.block_pin_weight) else 0.0,
                -block_map[bid].area,
                self.target_centers.get(bid, (0.0, 0.0))[1],
                self.target_centers.get(bid, (0.0, 0.0))[0],
                bid,
            ),
        )

    def _choose_soft_placement(
        self,
        bins: Sequence[FreeRect],
        positions: Dict[int, Tuple[float, float, float, float]],
        block_map: Dict[int, BlockSpec],
        block_id: int,
    ) -> Optional[Tuple[int, Tuple[float, float, float, float], Optional[FreeRect]]]:
        area = block_map[block_id].area
        occupied = tuple(positions.values())
        best = None
        best_score = (float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))
        for idx, rect in enumerate(bins):
            if rect.area + EPS < area:
                continue
            for placed, leftover in self._slice_rect_variants(rect, area):
                if self._overlaps_any(placed, occupied):
                    continue
                score = self._score_soft_candidate(positions, block_id, placed)
                if score < best_score:
                    best_score = score
                    best = (idx, placed, leftover)
        return best

    def _slice_rect_variants(
        self,
        rect: FreeRect,
        area: float,
    ) -> List[Tuple[Tuple[float, float, float, float], Optional[FreeRect]]]:
        variants: List[Tuple[Tuple[float, float, float, float], Optional[FreeRect]]] = []
        if rect.w > EPS:
            height = area / rect.w
            if height <= rect.h + EPS:
                height = min(height, rect.h)
                rem_h = rect.h - height
                placed_bottom = (rect.x, rect.y, rect.w, height)
                leftover_top = None if rem_h <= EPS else FreeRect(rect.x, rect.y + height, rect.w, rem_h)
                variants.append((placed_bottom, leftover_top))
                placed_top = (rect.x, rect.y + rem_h, rect.w, height)
                leftover_bottom = None if rem_h <= EPS else FreeRect(rect.x, rect.y, rect.w, rem_h)
                variants.append((placed_top, leftover_bottom))
        if rect.h > EPS:
            width = area / rect.h
            if width <= rect.w + EPS:
                width = min(width, rect.w)
                rem_w = rect.w - width
                placed_left = (rect.x, rect.y, width, rect.h)
                leftover_right = None if rem_w <= EPS else FreeRect(rect.x + width, rect.y, rem_w, rect.h)
                variants.append((placed_left, leftover_right))
                placed_right = (rect.x + rem_w, rect.y, width, rect.h)
                leftover_left = None if rem_w <= EPS else FreeRect(rect.x, rect.y, rem_w, rect.h)
                variants.append((placed_right, leftover_left))
        return variants

    def _score_soft_candidate(
        self,
        positions: Dict[int, Tuple[float, float, float, float]],
        block_id: int,
        rect: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float, float]:
        occupied_bbox = self._bbox_xyxy(positions.values()) if positions else None
        bbox = self._expand_bbox(occupied_bbox, rect)
        bbox_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0.0)
        bbox_perimeter = max((bbox[2] - bbox[0]) + (bbox[3] - bbox[1]), 0.0)
        wire = self._soft_candidate_wire(positions, block_id, rect)
        cx = rect[0] + 0.5 * rect[2]
        cy = rect[1] + 0.5 * rect[3]
        tx, ty = self.target_centers.get(block_id, (cx, cy))
        anchor = abs(cx - tx) + abs(cy - ty)
        mixed = (
            self.hpwl_weight * (wire / self.hpwl_norm)
            + self.bbox_weight * (bbox_area / self.total_area_norm)
            + self.anchor_weight * (anchor / self.sqrt_area_norm)
        )
        return (mixed, bbox_area, wire, anchor, bbox_perimeter)

    def _soft_candidate_wire(
        self,
        positions: Dict[int, Tuple[float, float, float, float]],
        block_id: int,
        rect: Tuple[float, float, float, float],
    ) -> float:
        cx = rect[0] + 0.5 * rect[2]
        cy = rect[1] + 0.5 * rect[3]
        total = 0.0
        for edge_idx in self.incident_b2b[block_id]:
            a, b, weight = self.b2b_edges[edge_idx]
            other = b if a == block_id else a
            other_rect = positions.get(other)
            if other_rect is not None:
                ox = other_rect[0] + 0.5 * other_rect[2]
                oy = other_rect[1] + 0.5 * other_rect[3]
            else:
                ox, oy = self.target_centers.get(other, (cx, cy))
            total += weight * (abs(cx - ox) + abs(cy - oy))
        if self.pins_pos is not None:
            for pin_idx, weight in self.incident_p2b[block_id]:
                if 0 <= pin_idx < len(self.pins_pos):
                    px = float(self.pins_pos[pin_idx][0])
                    py = float(self.pins_pos[pin_idx][1])
                    total += weight * (abs(cx - px) + abs(cy - py))
        return total

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
            ordered = sorted(
                remaining_ids,
                key=lambda bid: (self.target_centers.get(bid, (0.0, 0.0))[0], -self.block_wire_weight[bid], bid),
            )
            for block_id in ordered:
                area = block_map[block_id].area
                rect_w = area / strip_h
                positions[block_id] = (cursor_x, y1, rect_w, strip_h)
                cursor_x += rect_w
        else:
            strip_w = remain_area / max(height, EPS)
            cursor_y = y0
            ordered = sorted(
                remaining_ids,
                key=lambda bid: (self.target_centers.get(bid, (0.0, 0.0))[1], -self.block_wire_weight[bid], bid),
            )
            for block_id in ordered:
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
        end_y: float,
        strip_w: float,
        positions: Dict[int, Tuple[float, float, float, float]],
        right_align: bool = False,
    ) -> None:
        cursor_y = start_y
        ordered = sorted(
            blocks,
            key=lambda block: (
                self.target_centers.get(block.block_id, (0.0, 0.0))[1],
                0 if block.fixed else 1,
                -self.block_wire_weight[block.block_id] if block.block_id < len(self.block_wire_weight) else 0.0,
                block.block_id,
            ),
        )
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
        ordered = sorted(
            blocks,
            key=lambda block: (
                self.target_centers.get(block.block_id, (0.0, 0.0))[0],
                0 if block.fixed else 1,
                -self.block_wire_weight[block.block_id] if block.block_id < len(self.block_wire_weight) else 0.0,
                block.block_id,
            ),
        )
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
    ) -> List[Tuple[float, float, float, float]]:
        """Final safety net: detect and resolve any remaining overlaps by
        shifting blocks apart along the line connecting their centers."""
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
                    result[i] = (x1 - sx, y1 - sy, w1, h1)
                    result[j] = (x2 + sx, y2 + sy, w2, h2)
                    fixed = True
            if not fixed:
                break
        return result

    # ------------------------------------------------------------------
    # Conservative HPWL post-pass
    # ------------------------------------------------------------------
    def _adjacent_hpwl_swaps(
        self,
        positions: List[Tuple[float, float, float, float]],
        blocks: Sequence[BlockSpec],
    ) -> List[Tuple[float, float, float, float]]:
        current = [tuple(float(v) for v in rect) for rect in positions]
        if len(current) < 2:
            return current

        for _ in range(1):
            improved = False
            for i in range(len(current)):
                if not self._swap_allowed(blocks[i]):
                    continue
                for j in range(i + 1, len(current)):
                    if not self._swap_allowed(blocks[j]):
                        continue
                    if not self._are_adjacent(current[i], current[j]):
                        continue
                    swapped = self._adjacent_order_swap_rects(current[i], current[j])
                    if swapped is None:
                        continue
                    rect_i, rect_j = swapped
                    if self._rects_overlap(rect_i, rect_j):
                        continue
                    if self._overlaps_except(rect_i, current, {i, j}):
                        continue
                    if self._overlaps_except(rect_j, current, {i, j}):
                        continue
                    delta = self._two_block_hpwl_delta(current, i, j, rect_i, rect_j)
                    if delta < -1e-7:
                        current[i] = rect_i
                        current[j] = rect_j
                        improved = True
            if not improved:
                break
        return current

    def _area_recovery_moves(
        self,
        positions: List[Tuple[float, float, float, float]],
        blocks: Sequence[BlockSpec],
    ) -> List[Tuple[float, float, float, float]]:
        current = [tuple(float(v) for v in rect) for rect in positions]
        if len(current) < 2:
            return current

        for _ in range(1):
            improved = False
            bbox = self._bbox_xyxy(current)
            boundary_touchers = self._bbox_touching_blocks(current, bbox)
            for block_id in sorted(boundary_touchers, key=lambda bid: -self._block_area(current[bid])):
                if not self._area_move_allowed(blocks[block_id]):
                    continue
                candidate = self._best_area_reinsert(current, block_id)
                if candidate is None:
                    continue
                current[block_id] = candidate
                improved = True
            if not improved:
                break
        return current

    def _bbox_touching_blocks(
        self,
        positions: Sequence[Tuple[float, float, float, float]],
        bbox: Tuple[float, float, float, float],
    ) -> List[int]:
        x0, y0, x1, y1 = bbox
        result = []
        for idx, (x, y, w, h) in enumerate(positions):
            if (
                abs(x - x0) <= 1e-6
                or abs(y - y0) <= 1e-6
                or abs(x + w - x1) <= 1e-6
                or abs(y + h - y1) <= 1e-6
            ):
                result.append(idx)
        return result

    def _area_move_allowed(self, block: BlockSpec) -> bool:
        if block.preplaced:
            return False
        if block.boundary_code != 0:
            return False
        if block.group_id > 0:
            return False
        return True

    def _best_area_reinsert(
        self,
        positions: List[Tuple[float, float, float, float]],
        block_id: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        current_rect = positions[block_id]
        other_rects = [rect for idx, rect in enumerate(positions) if idx != block_id]
        if not other_rects:
            return None
        other_bbox = self._bbox_xyxy(other_rects)
        curr_area = calculate_bbox_area(positions)
        curr_hpwl = self._positions_hpwl(positions)
        curr_score = curr_hpwl / self.hpwl_norm + 1.25 * curr_area / self.total_area_norm

        x, y, w, h = current_rect
        ox0, oy0, ox1, oy1 = other_bbox
        xs = {x, ox0, ox1 - w, 0.5 * (ox0 + ox1 - w)}
        ys = {y, oy0, oy1 - h, 0.5 * (oy0 + oy1 - h)}
        for rx, ry, rw, rh in other_rects:
            xs.update([rx - w, rx + rw, rx, rx + rw - w])
            ys.update([ry - h, ry + rh, ry, ry + rh - h])

        xs = self._limited_candidate_coords(xs, 0.5 * (ox0 + ox1 - w), 10)
        ys = self._limited_candidate_coords(ys, 0.5 * (oy0 + oy1 - h), 10)

        best_rect = None
        best_score = curr_score
        for nx in xs:
            for ny in ys:
                rect = (nx, ny, w, h)
                if abs(nx - x) <= 1e-7 and abs(ny - y) <= 1e-7:
                    continue
                if self._overlaps_any(rect, other_rects):
                    continue
                trial = list(positions)
                trial[block_id] = rect
                area = calculate_bbox_area(trial)
                if area >= curr_area - 1e-7:
                    continue
                hpwl = self._positions_hpwl(trial)
                score = hpwl / self.hpwl_norm + 1.25 * area / self.total_area_norm
                if score + 1e-9 < best_score:
                    best_score = score
                    best_rect = rect
        return best_rect

    def _block_area(self, rect: Tuple[float, float, float, float]) -> float:
        return rect[2] * rect[3]

    def _choose_against_baseline(
        self,
        hpwl_positions: List[Tuple[float, float, float, float]],
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        blocks: Sequence[BlockSpec],
    ) -> List[Tuple[float, float, float, float]]:
        baseline = self._run_bf1_baseline(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )
        if baseline is None:
            return hpwl_positions

        hpwl_ok = self._hard_ok(hpwl_positions, blocks)
        base_ok = self._hard_ok(baseline, blocks)
        if hpwl_ok and not base_ok:
            return hpwl_positions
        if base_ok and not hpwl_ok:
            return baseline
        if not hpwl_ok and not base_ok:
            return hpwl_positions

        hpwl_area = calculate_bbox_area(hpwl_positions)
        base_area = calculate_bbox_area(baseline)
        hpwl_wire = self._positions_hpwl(hpwl_positions)
        base_wire = self._positions_hpwl(baseline)
        hpwl_vrel = self._soft_violation_relative(hpwl_positions, constraints)
        base_vrel = self._soft_violation_relative(baseline, constraints)

        wire_gain = (base_wire - hpwl_wire) / max(base_wire, 1.0)
        area_loss = (hpwl_area - base_area) / max(base_area, 1.0)
        if area_loss > 0.12 and wire_gain < 0.18:
            return baseline
        if hpwl_vrel > base_vrel + 0.015 and area_loss > 0.03:
            return baseline

        hpwl_score = self._candidate_choice_score(hpwl_positions, constraints)
        base_score = self._candidate_choice_score(baseline, constraints)
        return hpwl_positions if hpwl_score + 1e-9 < base_score else baseline

    def _run_bf1_baseline(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        optimizer = self._load_bf1_optimizer()
        if optimizer is None:
            return None
        try:
            return optimizer.solve(
                block_count,
                area_targets,
                b2b_connectivity,
                p2b_connectivity,
                pins_pos,
                constraints,
                target_positions,
            )
        except Exception:
            return None

    def _load_bf1_optimizer(self):
        if self.baseline_optimizer is not None:
            return self.baseline_optimizer
        bf1_path = Path(__file__).resolve().parents[1] / "BF1" / "my_optimizer.py"
        if not bf1_path.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location("_bf1_optimizer_module", bf1_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.baseline_optimizer = module.MyOptimizer(verbose=False)
            return self.baseline_optimizer
        except Exception:
            return None

    def _candidate_choice_score(
        self,
        positions: Sequence[Tuple[float, float, float, float]],
        constraints: torch.Tensor,
    ) -> float:
        hpwl = self._positions_hpwl(positions) / self.hpwl_norm
        area = calculate_bbox_area(list(positions)) / self.total_area_norm
        vrel = self._soft_violation_relative(positions, constraints)
        return hpwl + 0.95 * area + 1.50 * vrel

    def _hard_ok(
        self,
        positions: Sequence[Tuple[float, float, float, float]],
        blocks: Sequence[BlockSpec],
    ) -> bool:
        if len(positions) != len(blocks):
            return False
        for i in range(len(positions)):
            if self._overlaps_except(positions[i], positions, {i}):
                return False
        for block in blocks:
            x, y, w, h = positions[block.block_id]
            if block.preplaced:
                if (
                    abs(x - block.x) > 1e-4
                    or abs(y - block.y) > 1e-4
                    or abs(w - block.width) > 1e-4
                    or abs(h - block.height) > 1e-4
                ):
                    return False
            elif block.fixed:
                if abs(w - block.width) > 1e-4 or abs(h - block.height) > 1e-4:
                    return False
            else:
                if block.area > EPS and abs(w * h - block.area) / block.area > 0.01:
                    return False
        return True

    def _soft_violation_relative(
        self,
        positions: Sequence[Tuple[float, float, float, float]],
        constraints: torch.Tensor,
    ) -> float:
        if constraints is None or constraints.ndim != 2 or not positions:
            return 0.0
        block_count = min(len(positions), constraints.shape[0])
        ncols = int(constraints.shape[1])
        boundary_count = 0
        nsoft = 0
        violations = 0

        if ncols > 4:
            boundary_count = int((constraints[:block_count, 4] != 0).sum().item())
            nsoft += boundary_count
            if boundary_count:
                x0, y0, x1, y1 = self._bbox_xyxy(positions)
                for i in range(block_count):
                    code = int(constraints[i, 4].item())
                    if code == 0:
                        continue
                    x, y, w, h = positions[i]
                    touches = {
                        1: abs(x - x0) < 1e-6,
                        2: abs(x + w - x1) < 1e-6,
                        4: abs(y + h - y1) < 1e-6,
                        8: abs(y - y0) < 1e-6,
                    }
                    if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                        violations += 1

        if ncols > 2:
            for _, members in self._constraint_groups_from_tensor(constraints, 2, block_count).items():
                nsoft += max(0, len(members) - 1)
                shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in members}
                violations += max(0, len(shapes) - 1)

        if ncols > 3:
            for _, members in self._constraint_groups_from_tensor(constraints, 3, block_count).items():
                nsoft += max(0, len(members) - 1)
                violations += max(0, self._component_count_for_positions(members, positions) - 1)

        return violations / max(nsoft, 1)

    def _constraint_groups_from_tensor(
        self,
        constraints: torch.Tensor,
        column: int,
        block_count: int,
    ) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        if constraints is None or constraints.ndim != 2 or constraints.shape[1] <= column:
            return groups
        for i in range(min(block_count, constraints.shape[0])):
            group_id = int(constraints[i, column].item())
            if group_id > 0:
                groups.setdefault(group_id, []).append(i)
        return groups

    def _component_count_for_positions(
        self,
        members: Sequence[int],
        positions: Sequence[Tuple[float, float, float, float]],
    ) -> int:
        if not members:
            return 0
        remaining = set(members)
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                i = stack.pop()
                for j in list(remaining):
                    if self._share_edge(positions[i], positions[j]):
                        remaining.remove(j)
                        stack.append(j)
        return count

    def _share_edge(
        self,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
        tol: float = 1e-6,
    ) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax1 = ax + aw
        ay1 = ay + ah
        bx1 = bx + bw
        by1 = by + bh
        x_overlap = min(ax1, bx1) - max(ax, bx)
        y_overlap = min(ay1, by1) - max(ay, by)
        return (
            ((abs(ax1 - bx) <= tol or abs(bx1 - ax) <= tol) and y_overlap > tol)
            or ((abs(ay1 - by) <= tol or abs(by1 - ay) <= tol) and x_overlap > tol)
        )

    def _swap_allowed(self, block: BlockSpec) -> bool:
        if block.preplaced:
            return False
        if block.boundary_code != 0:
            return False
        if block.group_id > 0:
            return False
        return True

    def _two_block_hpwl_delta(
        self,
        positions: Sequence[Tuple[float, float, float, float]],
        block_i: int,
        block_j: int,
        rect_i_after: Tuple[float, float, float, float],
        rect_j_after: Tuple[float, float, float, float],
    ) -> float:
        before = 0.0
        after = 0.0
        edge_indices = set(self.incident_b2b[block_i])
        edge_indices.update(self.incident_b2b[block_j])
        for edge_idx in edge_indices:
            a, b, weight = self.b2b_edges[edge_idx]
            before += self._edge_hpwl(positions[a], positions[b], weight)
            rect_a = rect_i_after if a == block_i else rect_j_after if a == block_j else positions[a]
            rect_b = rect_i_after if b == block_i else rect_j_after if b == block_j else positions[b]
            after += self._edge_hpwl(rect_a, rect_b, weight)

        if self.pins_pos is not None:
            for block_id, rect_after in ((block_i, rect_i_after), (block_j, rect_j_after)):
                for pin_idx, weight in self.incident_p2b[block_id]:
                    if pin_idx >= len(self.pins_pos):
                        continue
                    before += self._pin_hpwl(positions[block_id], pin_idx, weight)
                    after += self._pin_hpwl(rect_after, pin_idx, weight)
        return after - before

    def _edge_hpwl(
        self,
        rect_a: Tuple[float, float, float, float],
        rect_b: Tuple[float, float, float, float],
        weight: float,
    ) -> float:
        ax = rect_a[0] + 0.5 * rect_a[2]
        ay = rect_a[1] + 0.5 * rect_a[3]
        bx = rect_b[0] + 0.5 * rect_b[2]
        by = rect_b[1] + 0.5 * rect_b[3]
        return weight * (abs(ax - bx) + abs(ay - by))

    def _pin_hpwl(
        self,
        rect: Tuple[float, float, float, float],
        pin_idx: int,
        weight: float,
    ) -> float:
        bx = rect[0] + 0.5 * rect[2]
        by = rect[1] + 0.5 * rect[3]
        px = float(self.pins_pos[pin_idx][0])
        py = float(self.pins_pos[pin_idx][1])
        return weight * (abs(bx - px) + abs(by - py))

    def _are_adjacent(
        self,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
        tol: float = 1e-5,
    ) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax1 = ax + aw
        ay1 = ay + ah
        bx1 = bx + bw
        by1 = by + bh
        x_overlap = min(ax1, bx1) - max(ax, bx)
        y_overlap = min(ay1, by1) - max(ay, by)
        vertical_touch = (abs(ax1 - bx) <= tol or abs(bx1 - ax) <= tol) and y_overlap > tol
        horizontal_touch = (abs(ay1 - by) <= tol or abs(by1 - ay) <= tol) and x_overlap > tol
        return vertical_touch or horizontal_touch

    def _adjacent_order_swap_rects(
        self,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
        tol: float = 1e-5,
    ) -> Optional[Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]]:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax1 = ax + aw
        ay1 = ay + ah
        bx1 = bx + bw
        by1 = by + bh

        x_overlap = min(ax1, bx1) - max(ax, bx)
        y_overlap = min(ay1, by1) - max(ay, by)
        if x_overlap > tol and abs(ay1 - by) <= tol:
            bottom = ay
            return (ax, bottom + bh, aw, ah), (bx, bottom, bw, bh)
        if x_overlap > tol and abs(by1 - ay) <= tol:
            bottom = by
            return (ax, bottom, aw, ah), (bx, bottom + ah, bw, bh)
        if y_overlap > tol and abs(ax1 - bx) <= tol:
            left = ax
            return (left + bw, ay, aw, ah), (left, by, bw, bh)
        if y_overlap > tol and abs(bx1 - ax) <= tol:
            left = bx
            return (left, ay, aw, ah), (left + aw, by, bw, bh)
        return None

    def _overlaps_except(
        self,
        rect: Tuple[float, float, float, float],
        positions: Sequence[Tuple[float, float, float, float]],
        skip: set,
    ) -> bool:
        for idx, other in enumerate(positions):
            if idx in skip:
                continue
            if self._rects_overlap(rect, other):
                return True
        return False

    def _rects_overlap(
        self,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ox = min(ax + aw, bx + bw) - max(ax, bx)
        oy = min(ay + ah, by + bh) - max(ay, by)
        return ox > 1e-6 and oy > 1e-6

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
