#!/usr/bin/env python3
"""
PeF_ev1: simplified PeF-style optimizer for the reduced objective
HPWLgap + Areagap_bbox. Soft constraints are intentionally ignored; fixed
shape and preplaced constraints remain hard constraints.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

CONTEST_ROOT = Path(__file__).resolve().parents[2]
if str(CONTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEST_ROOT))

from iccad2026_evaluate import FloorplanOptimizer


EPS = 1e-9


Placement = Tuple[float, float, float, float]


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
        shape_locked: bool = False,
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
        self.shape_locked = shape_locked


class LayoutItem:
    def __init__(
        self,
        item_id: int,
        member_ids: List[int],
        width: float,
        height: float,
        area: float,
        local_rects: Dict[int, Placement],
        fixed_position: bool = False,
        fixed_x: float = 0.0,
        fixed_y: float = 0.0,
        resizable: bool = False,
    ):
        self.item_id = item_id
        self.member_ids = member_ids
        self.width = width
        self.height = height
        self.area = area
        self.local_rects = local_rects
        self.fixed_position = fixed_position
        self.fixed_x = fixed_x
        self.fixed_y = fixed_y
        self.resizable = resizable


def rect_bounds(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def bbox_xyxy(rects: Iterable[Placement]) -> Tuple[float, float, float, float]:
    rect_list = list(rects)
    if not rect_list:
        return 0.0, 0.0, 0.0, 0.0
    x0 = min(r[0] for r in rect_list)
    y0 = min(r[1] for r in rect_list)
    x1 = max(r[0] + r[2] for r in rect_list)
    y1 = max(r[1] + r[3] for r in rect_list)
    return x0, y0, x1, y1


def local_bbox(rects: Iterable[Placement]) -> Tuple[float, float]:
    rect_list = list(rects)
    if not rect_list:
        return 0.0, 0.0
    x0 = min(r[0] for r in rect_list)
    y0 = min(r[1] for r in rect_list)
    x1 = max(r[0] + r[2] for r in rect_list)
    y1 = max(r[1] + r[3] for r in rect_list)
    return x1 - x0, y1 - y0


def overlaps(a: Placement, b: Placement) -> bool:
    return not (
        a[0] + a[2] <= b[0] + EPS
        or b[0] + b[2] <= a[0] + EPS
        or a[1] + a[3] <= b[1] + EPS
        or b[1] + b[3] <= a[1] + EPS
    )


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
    ) -> List[Placement]:
        if block_count <= 0:
            return []

        blocks = self._build_blocks(block_count, area_targets, constraints, target_positions)
        core_items = self._build_core_items(blocks)

        if core_items:
            item_edges, pin_targets = self._aggregate_item_connectivity(
                core_items, b2b_connectivity, p2b_connectivity, pins_pos
            )
            outline_w, outline_h = self._estimate_outline(core_items, pin_targets, pins_pos)
            init_x, init_y = self._seed_item_positions(core_items, outline_w, outline_h, item_edges, pin_targets)
            x, y, w, h = self._run_pef_core(
                core_items,
                init_x,
                init_y,
                outline_w,
                outline_h,
                item_edges,
                pin_targets,
            )
            fixed_mask = np.array([item.fixed_position for item in core_items], dtype=bool)
            fixed_x = np.array([item.fixed_x for item in core_items], dtype=float)
            fixed_y = np.array([item.fixed_y for item in core_items], dtype=float)
            x, y, outline_w, outline_h = self._legalize_item_arrays(
                x,
                y,
                w,
                h,
                outline_w,
                outline_h,
                fixed_mask,
                fixed_x,
                fixed_y,
            )
            x, y = enforce_pairwise_separation_iteratively(
                x,
                y,
                w,
                h,
                outline_w,
                outline_h,
                fixed_mask,
                fixed_x,
                fixed_y,
                passes=10,
            )
            self._sync_item_shapes(core_items, w, h)
            core_positions = self._expand_items(core_items, x, y)
        else:
            core_positions = {}

        result = [core_positions[i] for i in range(block_count)]
        result = self._legalize_final_blocks(blocks, result)
        result = self._resolve_block_overlaps(blocks, result)
        result = self._compact_layout(blocks, result, rounds=4)
        result = self._resolve_block_overlaps(blocks, result)
        result = self._compact_layout(blocks, result, rounds=2)

        if not any(block.preplaced for block in blocks):
            result = self._shift_to_origin(result)

        return result

    # ------------------------------------------------------------------
    # Input parsing. Soft constraints are parsed but not optimized in ev1.
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

    def _apply_mib_templates(self, blocks: Sequence[BlockSpec]) -> None:
        groups: Dict[int, List[BlockSpec]] = {}
        for block in blocks:
            if block.mib_id > 0:
                groups.setdefault(block.mib_id, []).append(block)

        for members in groups.values():
            if len(members) < 2:
                continue

            hard_members = [b for b in members if b.preplaced or b.fixed]
            if hard_members:
                ref = hard_members[0]
                if ref.width is None or ref.height is None:
                    continue
                consistent = all(
                    other.width is not None
                    and other.height is not None
                    and abs(other.width - ref.width) <= 1e-4
                    and abs(other.height - ref.height) <= 1e-4
                    for other in hard_members
                )
                if not consistent:
                    continue
                template_area = ref.width * ref.height
                for block in members:
                    if block.preplaced:
                        continue
                    if abs(block.area - template_area) / max(block.area, 1.0) <= 0.01:
                        block.width = ref.width
                        block.height = ref.height
                        block.shape_locked = True
                continue

            areas = [block.area for block in members]
            if max(areas) - min(areas) > 0.01 * max(max(areas), 1.0):
                continue
            area = sum(areas) / len(areas)
            side = math.sqrt(max(area, 1.0))
            for block in members:
                block.width = side
                block.height = side
                block.shape_locked = True

    def _build_group_items(
        self,
        blocks: Sequence[BlockSpec],
    ) -> Tuple[List[LayoutItem], set[int]]:
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
        grouped_ids: set[int] = set()
        next_item_id = len(blocks) + 1

        for members in groups.values():
            if len(members) < 2:
                continue
            item = self._make_group_item(next_item_id, members)
            next_item_id += 1
            items.append(item)
            grouped_ids.update(block.block_id for block in members)

        return items, grouped_ids

    def _make_group_item(self, item_id: int, members: Sequence[BlockSpec]) -> LayoutItem:
        total_area = sum(block.area for block in members)
        rigid = lambda block: block.fixed or block.shape_locked
        orders = [
            sorted(members, key=lambda b: (-b.area, b.block_id)),
            sorted(members, key=lambda b: (0 if rigid(b) else 1, -b.area, b.block_id)),
        ]

        candidates: List[Tuple[float, float, float, Dict[int, Placement]]] = []
        max_rigid_h = max((block.height for block in members if rigid(block) and block.height is not None), default=0.0)
        max_rigid_w = max((block.width for block in members if rigid(block) and block.width is not None), default=0.0)

        for order in orders:
            row_h = max(max_rigid_h, math.sqrt(max(total_area, 1.0)))
            x = 0.0
            local_h: Dict[int, Placement] = {}
            max_h = 0.0
            for block in order:
                if rigid(block) and block.width is not None and block.height is not None:
                    bw, bh = block.width, block.height
                else:
                    bh = row_h
                    bw = block.area / bh
                local_h[block.block_id] = (x, 0.0, bw, bh)
                x += bw
                max_h = max(max_h, bh)
            candidates.append((x * max_h, abs(x - max_h), 0.0, local_h))

            col_w = max(max_rigid_w, math.sqrt(max(total_area, 1.0)))
            y = 0.0
            local_v: Dict[int, Placement] = {}
            max_w = 0.0
            for block in order:
                if rigid(block) and block.width is not None and block.height is not None:
                    bw, bh = block.width, block.height
                else:
                    bw = col_w
                    bh = block.area / bw
                local_v[block.block_id] = (0.0, y, bw, bh)
                y += bh
                max_w = max(max_w, bw)
            candidates.append((max_w * y, abs(max_w - y), 1.0, local_v))

        best_local = min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]
        width, height = local_bbox(best_local.values())
        return LayoutItem(
            item_id=item_id,
            member_ids=[block.block_id for block in members],
            width=width,
            height=height,
            area=width * height,
            local_rects=best_local,
            fixed_position=False,
            resizable=False,
        )

    def _build_core_items(
        self,
        blocks: Sequence[BlockSpec],
    ) -> List[LayoutItem]:
        items: List[LayoutItem] = []

        for block in blocks:
            if block.preplaced:
                width = block.width if block.width is not None else math.sqrt(block.area)
                height = block.height if block.height is not None else block.area / max(width, EPS)
                items.append(
                    LayoutItem(
                        item_id=block.block_id,
                        member_ids=[block.block_id],
                        width=width,
                        height=height,
                        area=width * height,
                        local_rects={block.block_id: (0.0, 0.0, width, height)},
                        fixed_position=True,
                        fixed_x=float(block.x) + width / 2.0,
                        fixed_y=float(block.y) + height / 2.0,
                        resizable=False,
                    )
                )
                continue

            width, height = self._default_block_shape(block)
            items.append(
                LayoutItem(
                    item_id=block.block_id,
                    member_ids=[block.block_id],
                    width=width,
                    height=height,
                    area=block.area if not block.fixed else width * height,
                    local_rects={block.block_id: (0.0, 0.0, width, height)},
                    fixed_position=False,
                    resizable=not block.fixed and not block.shape_locked,
                )
            )

        return items

    def _default_block_shape(self, block: BlockSpec) -> Tuple[float, float]:
        if block.width is not None and block.height is not None:
            return block.width, block.height
        side = math.sqrt(max(block.area, 1.0))
        return side, side

    # ------------------------------------------------------------------
    # Item-level connectivity
    # ------------------------------------------------------------------
    def _aggregate_item_connectivity(
        self,
        items: Sequence[LayoutItem],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> Tuple[List[Tuple[int, int, float]], Dict[int, Tuple[float, float, float]]]:
        item_of_block: Dict[int, int] = {}
        for idx, item in enumerate(items):
            for block_id in item.member_ids:
                item_of_block[block_id] = idx

        pair_weights: Dict[Tuple[int, int], float] = {}
        if b2b_connectivity is not None:
            for edge in b2b_connectivity:
                if int(edge[0].item()) == -1:
                    continue
                b1 = int(edge[0].item())
                b2 = int(edge[1].item())
                w = float(edge[2].item())
                i = item_of_block.get(b1)
                j = item_of_block.get(b2)
                if i is None or j is None or i == j or w <= 0.0:
                    continue
                key = (i, j) if i < j else (j, i)
                pair_weights[key] = pair_weights.get(key, 0.0) + w

        pin_sum_x: Dict[int, float] = {}
        pin_sum_y: Dict[int, float] = {}
        pin_sum_w: Dict[int, float] = {}
        if p2b_connectivity is not None and pins_pos is not None:
            for edge in p2b_connectivity:
                if int(edge[0].item()) == -1:
                    continue
                pin_idx = int(edge[0].item())
                block_idx = int(edge[1].item())
                w = float(edge[2].item())
                if pin_idx < 0 or pin_idx >= len(pins_pos) or w <= 0.0:
                    continue
                item_idx = item_of_block.get(block_idx)
                if item_idx is None:
                    continue
                px = float(pins_pos[pin_idx, 0].item())
                py = float(pins_pos[pin_idx, 1].item())
                pin_sum_x[item_idx] = pin_sum_x.get(item_idx, 0.0) + w * px
                pin_sum_y[item_idx] = pin_sum_y.get(item_idx, 0.0) + w * py
                pin_sum_w[item_idx] = pin_sum_w.get(item_idx, 0.0) + w

        pin_targets: Dict[int, Tuple[float, float, float]] = {}
        for item_idx, total_w in pin_sum_w.items():
            if total_w > 0.0:
                pin_targets[item_idx] = (
                    pin_sum_x[item_idx] / total_w,
                    pin_sum_y[item_idx] / total_w,
                    total_w,
                )

        edges = [(i, j, w) for (i, j), w in pair_weights.items()]
        return edges, pin_targets

    # ------------------------------------------------------------------
    # PeF-style core placement
    # ------------------------------------------------------------------
    def _estimate_outline(
        self,
        items: Sequence[LayoutItem],
        pin_targets: Dict[int, Tuple[float, float, float]],
        pins_pos: torch.Tensor,
    ) -> Tuple[float, float]:
        total_area = sum(max(item.area, 1.0) for item in items)
        base_side = math.sqrt(max(total_area, 1.0))

        fixed_rects = [
            (
                item.fixed_x - item.width / 2.0,
                item.fixed_y - item.height / 2.0,
                item.width,
                item.height,
            )
            for item in items
            if item.fixed_position
        ]
        ax0, ay0, ax1, ay1 = bbox_xyxy(fixed_rects)
        anchor_w = max(ax1 - ax0, 0.0)
        anchor_h = max(ay1 - ay0, 0.0)

        aspect = 1.0
        valid_pins = []
        if pins_pos is not None:
            for pin in pins_pos:
                px = float(pin[0].item())
                py = float(pin[1].item())
                if px == -1.0 and py == -1.0:
                    continue
                valid_pins.append((px, py))
        if len(valid_pins) >= 2:
            xs = [p[0] for p in valid_pins]
            ys = [p[1] for p in valid_pins]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            if span_x > EPS and span_y > EPS:
                aspect = min(1.8, max(0.55, span_x / span_y))
        elif anchor_w > EPS and anchor_h > EPS:
            aspect = min(1.8, max(0.55, anchor_w / anchor_h))

        outline_area = total_area / 0.82
        outline_w = math.sqrt(outline_area * aspect)
        outline_h = outline_area / max(outline_w, 1e-6)
        pad = 0.18 * base_side

        if fixed_rects:
            outline_w = max(outline_w, ax1 + pad, anchor_w + 2.0 * pad)
            outline_h = max(outline_h, ay1 + pad, anchor_h + 2.0 * pad)

        outline_w = max(outline_w, 1.12 * base_side)
        outline_h = max(outline_h, 1.12 * base_side)
        return outline_w, outline_h

    def _seed_item_positions(
        self,
        items: Sequence[LayoutItem],
        outline_w: float,
        outline_h: float,
        item_edges: Sequence[Tuple[int, int, float]],
        pin_targets: Dict[int, Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(items)
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)

        degree = [0.0] * n
        neighbor_fixed_sum_x = [0.0] * n
        neighbor_fixed_sum_y = [0.0] * n
        neighbor_fixed_w = [0.0] * n
        for i, j, wgt in item_edges:
            degree[i] += wgt
            degree[j] += wgt
            if items[i].fixed_position and not items[j].fixed_position:
                neighbor_fixed_sum_x[j] += wgt * items[i].fixed_x
                neighbor_fixed_sum_y[j] += wgt * items[i].fixed_y
                neighbor_fixed_w[j] += wgt
            if items[j].fixed_position and not items[i].fixed_position:
                neighbor_fixed_sum_x[i] += wgt * items[j].fixed_x
                neighbor_fixed_sum_y[i] += wgt * items[j].fixed_y
                neighbor_fixed_w[i] += wgt

        movable: List[int] = []
        anchors: Dict[int, Tuple[float, float]] = {}
        for idx, item in enumerate(items):
            if item.fixed_position:
                x[idx] = item.fixed_x
                y[idx] = item.fixed_y
                continue
            movable.append(idx)
            sx = outline_w * 0.5
            sy = outline_h * 0.5
            sw = 1.0
            pin_target = pin_targets.get(idx)
            if pin_target is not None:
                px, py, pw = pin_target
                sx += px * pw
                sy += py * pw
                sw += pw
            if neighbor_fixed_w[idx] > 0.0:
                sx += neighbor_fixed_sum_x[idx]
                sy += neighbor_fixed_sum_y[idx]
                sw += neighbor_fixed_w[idx]
            anchors[idx] = (sx / sw, sy / sw)

        if movable:
            aspect = outline_w / max(outline_h, 1e-6)
            cols = max(1, int(math.ceil(math.sqrt(len(movable) * aspect))))
            rows = max(1, int(math.ceil(len(movable) / cols)))
            cell_w = outline_w / cols
            cell_h = outline_h / rows
            cells = []
            for row in range(rows):
                for col in range(cols):
                    cells.append(((col + 0.5) * cell_w, (row + 0.5) * cell_h))

            ordered = sorted(movable, key=lambda idx: (-degree[idx], -items[idx].area, idx))
            used = [False] * len(cells)
            for idx in ordered:
                ax, ay = anchors[idx]
                best_cell = None
                best_score = None
                for cell_idx, (cx, cy) in enumerate(cells):
                    if used[cell_idx]:
                        continue
                    score = (cx - ax) ** 2 + (cy - ay) ** 2
                    if best_score is None or score < best_score:
                        best_score = score
                        best_cell = cell_idx
                if best_cell is None:
                    best_cell = 0
                used[best_cell] = True
                x[idx], y[idx] = cells[best_cell]

        self._clamp_item_positions(items, x, y, outline_w, outline_h)
        return x, y

    def _run_pef_core(
        self,
        items: Sequence[LayoutItem],
        init_x: np.ndarray,
        init_y: np.ndarray,
        outline_w: float,
        outline_h: float,
        item_edges: Sequence[Tuple[int, int, float]],
        pin_targets: Dict[int, Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = init_x.copy()
        y = init_y.copy()
        w = np.array([item.width for item in items], dtype=float)
        h = np.array([item.height for item in items], dtype=float)
        area = np.array([item.area for item in items], dtype=float)
        fixed_mask = np.array([item.fixed_position for item in items], dtype=bool)
        resizable_mask = np.array([item.resizable for item in items], dtype=bool)
        fixed_x = np.array([item.fixed_x for item in items], dtype=float)
        fixed_y = np.array([item.fixed_y for item in items], dtype=float)
        movable = ~fixed_mask

        grid_size = 36 if len(items) <= 64 else 44
        iterations = min(84, max(42, 28 + len(items)))
        step_base = min(outline_w, outline_h) / max(grid_size, 1)
        scale_conn = max(min(outline_w, outline_h) / 6.0, 1.0)

        rng = np.random.default_rng(17)

        for it in range(iterations):
            density = rasterize_density(x, y, w, h, outline_w, outline_h, grid_size)
            potential = solve_poisson_fft(density)
            ex_grid, ey_grid = electric_field_from_potential(potential, outline_w, outline_h)
            ex = bilinear_sample(ex_grid, x, y, outline_w, outline_h)
            ey = bilinear_sample(ey_grid, x, y, outline_w, outline_h)
            grad_x, grad_y = connectivity_gradients(x, y, item_edges, pin_targets, scale_conn)

            grad_x[fixed_mask] = 0.0
            grad_y[fixed_mask] = 0.0
            ex[fixed_mask] = 0.0
            ey[fixed_mask] = 0.0

            conn_scale = max(
                np.max(np.abs(grad_x[movable])) if np.any(movable) else 0.0,
                np.max(np.abs(grad_y[movable])) if np.any(movable) else 0.0,
                1e-9,
            )
            field_scale = max(
                np.max(np.abs(ex[movable])) if np.any(movable) else 0.0,
                np.max(np.abs(ey[movable])) if np.any(movable) else 0.0,
                1e-9,
            )

            frac = it / max(iterations - 1, 1)
            conn_step = step_base * (0.42 - 0.18 * frac)
            dens_step = step_base * (0.92 + 0.25 * frac)
            x[movable] -= conn_step * grad_x[movable] / conn_scale
            y[movable] -= conn_step * grad_y[movable] / conn_scale
            x[movable] += dens_step * ex[movable] / field_scale
            y[movable] += dens_step * ey[movable] / field_scale

            if np.any(resizable_mask):
                width_grad = np.zeros_like(w)
                for idx in np.where(resizable_mask)[0]:
                    nominal = math.sqrt(max(area[idx], 1.0))
                    width_min = nominal / 4.0
                    width_max = nominal * 4.0
                    eps = max(0.04 * w[idx], 0.15)
                    w_plus = min(width_max, w[idx] + eps)
                    w_minus = max(width_min, w[idx] - eps)
                    if abs(w_plus - w_minus) < 1e-9:
                        continue
                    e_plus = module_potential_energy(
                        x[idx], y[idx], w_plus, area[idx], potential, outline_w, outline_h
                    )
                    e_minus = module_potential_energy(
                        x[idx], y[idx], w_minus, area[idx], potential, outline_w, outline_h
                    )
                    width_grad[idx] = (e_plus - e_minus) / (w_plus - w_minus)
                width_scale = max(np.max(np.abs(width_grad[resizable_mask])), 1e-9)
                width_step = 0.14 * step_base
                w[resizable_mask] -= width_step * width_grad[resizable_mask] / width_scale
                for idx in np.where(resizable_mask)[0]:
                    nominal = math.sqrt(max(area[idx], 1.0))
                    width_min = nominal / 4.0
                    width_max = nominal * 4.0
                    w[idx] = min(width_max, max(width_min, w[idx]))
                    h[idx] = area[idx] / max(w[idx], EPS)

            self._clamp_item_arrays(w, h, x, y, outline_w, outline_h, fixed_mask, fixed_x, fixed_y)

            if np.any(movable):
                jitter = 0.020 * step_base * (1.0 - frac)
                x[movable] += rng.normal(0.0, jitter, size=np.count_nonzero(movable))
                y[movable] += rng.normal(0.0, jitter, size=np.count_nonzero(movable))
                self._clamp_item_arrays(w, h, x, y, outline_w, outline_h, fixed_mask, fixed_x, fixed_y)

        return x, y, w, h

    def _sync_item_shapes(
        self,
        items: Sequence[LayoutItem],
        widths: np.ndarray,
        heights: np.ndarray,
    ) -> None:
        for idx, item in enumerate(items):
            if len(item.member_ids) != 1:
                continue
            block_id = item.member_ids[0]
            width = float(widths[idx])
            height = float(heights[idx])
            item.width = width
            item.height = height
            item.area = width * height
            item.local_rects = {block_id: (0.0, 0.0, width, height)}

    def _legalize_item_arrays(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        h: np.ndarray,
        outline_w: float,
        outline_h: float,
        fixed_mask: np.ndarray,
        fixed_x: np.ndarray,
        fixed_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        last_error: Optional[Exception] = None
        for factor in (1.0, 1.12, 1.30, 1.55, 1.90, 2.35):
            trial_w = outline_w * factor
            trial_h = outline_h * factor
            try:
                lx, ly, _, _ = legalize_with_obstacle_aware_graphs(
                    x,
                    y,
                    w,
                    h,
                    trial_w,
                    trial_h,
                    fixed_mask,
                    fixed_x,
                    fixed_y,
                )
                return lx, ly, trial_w, trial_h
            except RuntimeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("Obstacle-aware packing failed")

    def _clamp_item_positions(
        self,
        items: Sequence[LayoutItem],
        x: np.ndarray,
        y: np.ndarray,
        outline_w: float,
        outline_h: float,
    ) -> None:
        for idx, item in enumerate(items):
            x[idx] = min(max(x[idx], item.width / 2.0), outline_w - item.width / 2.0)
            y[idx] = min(max(y[idx], item.height / 2.0), outline_h - item.height / 2.0)
            if item.fixed_position:
                x[idx] = item.fixed_x
                y[idx] = item.fixed_y

    def _clamp_item_arrays(
        self,
        w: np.ndarray,
        h: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        outline_w: float,
        outline_h: float,
        fixed_mask: np.ndarray,
        fixed_x: np.ndarray,
        fixed_y: np.ndarray,
    ) -> None:
        x[:] = np.clip(x, w / 2.0, outline_w - w / 2.0)
        y[:] = np.clip(y, h / 2.0, outline_h - h / 2.0)
        x[fixed_mask] = fixed_x[fixed_mask]
        y[fixed_mask] = fixed_y[fixed_mask]

    def _expand_items(
        self,
        items: Sequence[LayoutItem],
        x: np.ndarray,
        y: np.ndarray,
    ) -> Dict[int, Placement]:
        placements: Dict[int, Placement] = {}
        for idx, item in enumerate(items):
            left = x[idx] - item.width / 2.0
            bottom = y[idx] - item.height / 2.0
            for block_id, (lx, ly, w, h) in item.local_rects.items():
                placements[block_id] = (left + lx, bottom + ly, w, h)
        return placements

    # ------------------------------------------------------------------
    # Boundary post-placement
    # ------------------------------------------------------------------
    def _choose_boundary_shape(self, block: BlockSpec, primary: float, vertical: bool) -> Tuple[float, float]:
        if block.width is not None and block.height is not None:
            return block.width, block.height
        if vertical:
            w = max(primary, math.sqrt(max(block.area, 1.0)) / 2.0)
            h = block.area / max(w, EPS)
            return w, h
        h = max(primary, math.sqrt(max(block.area, 1.0)) / 2.0)
        w = block.area / max(h, EPS)
        return w, h

    def _side_stack_height(self, blocks: Sequence[BlockSpec], width: float) -> float:
        total = 0.0
        for block in blocks:
            bw, bh = self._choose_boundary_shape(block, width, vertical=True)
            total += bh
        return total

    def _side_stack_width(self, blocks: Sequence[BlockSpec], height: float) -> float:
        total = 0.0
        for block in blocks:
            bw, bh = self._choose_boundary_shape(block, height, vertical=False)
            total += bw
        return total

    def _estimate_vertical_width(self, blocks: Sequence[BlockSpec], available_h: float, min_width: float) -> float:
        if not blocks:
            return min_width
        fixed_h = 0.0
        fixed_w = min_width
        soft_area = 0.0
        for block in blocks:
            if block.width is not None and block.height is not None:
                fixed_h += block.height
                fixed_w = max(fixed_w, block.width)
            else:
                soft_area += block.area
        remaining_h = max(available_h - fixed_h, 1e-6)
        return max(fixed_w, soft_area / remaining_h)

    def _estimate_horizontal_height(self, blocks: Sequence[BlockSpec], available_w: float, min_height: float) -> float:
        if not blocks:
            return min_height
        fixed_w = 0.0
        fixed_h = min_height
        soft_area = 0.0
        for block in blocks:
            if block.width is not None and block.height is not None:
                fixed_w += block.width
                fixed_h = max(fixed_h, block.height)
            else:
                soft_area += block.area
        remaining_w = max(available_w - fixed_w, 1e-6)
        return max(fixed_h, soft_area / remaining_w)

    def _place_boundary_blocks(
        self,
        blocks: Sequence[BlockSpec],
        core_positions: Dict[int, Placement],
        boundary_blocks: Sequence[BlockSpec],
    ) -> Dict[int, Placement]:
        positions = dict(core_positions)
        if not boundary_blocks:
            return positions

        if positions:
            core_x0, core_y0, core_x1, core_y1 = bbox_xyxy(positions.values())
        else:
            pseudo = math.sqrt(max(sum(block.area for block in boundary_blocks), 1.0))
            core_x0 = 0.0
            core_y0 = 0.0
            core_x1 = pseudo
            core_y1 = pseudo

        corner_buckets: Dict[int, List[BlockSpec]] = {5: [], 6: [], 9: [], 10: []}
        left_blocks: List[BlockSpec] = []
        right_blocks: List[BlockSpec] = []
        top_blocks: List[BlockSpec] = []
        bottom_blocks: List[BlockSpec] = []

        for block in boundary_blocks:
            code = block.boundary_code
            if code in corner_buckets:
                corner_buckets[code].append(block)
            elif code == 1:
                left_blocks.append(block)
            elif code == 2:
                right_blocks.append(block)
            elif code == 4:
                top_blocks.append(block)
            elif code == 8:
                bottom_blocks.append(block)
            elif code & 1:
                left_blocks.append(block)
            elif code & 2:
                right_blocks.append(block)
            elif code & 4:
                top_blocks.append(block)
            elif code & 8:
                bottom_blocks.append(block)

        corners: Dict[int, Optional[BlockSpec]] = {}
        for code, members in corner_buckets.items():
            if not members:
                corners[code] = None
                continue
            ordered = sorted(members, key=lambda b: (0 if (b.width is not None and b.height is not None) else 1, -b.area, b.block_id))
            corners[code] = ordered[0]
            extra = ordered[1:]
            if code in (5, 9):
                left_blocks.extend(extra)
            else:
                right_blocks.extend(extra)

        core_w = core_x1 - core_x0
        core_h = core_y1 - core_y0

        corner_dims: Dict[int, Tuple[float, float]] = {}
        for code, block in corners.items():
            if block is None:
                corner_dims[code] = (0.0, 0.0)
                continue
            if block.width is not None and block.height is not None:
                corner_dims[code] = (block.width, block.height)
            else:
                side = math.sqrt(max(block.area, 1.0))
                corner_dims[code] = (side, side)

        w_tl, h_tl = corner_dims[5]
        w_tr, h_tr = corner_dims[6]
        w_bl, h_bl = corner_dims[9]
        w_br, h_br = corner_dims[10]

        L = max(w_tl, w_bl)
        R = max(w_tr, w_br)
        T = max(h_tl, h_tr)
        B = max(h_bl, h_br)

        for _ in range(8):
            final_h = core_h + T + B
            left_available = max(final_h - h_tl - h_bl, 1e-6)
            right_available = max(final_h - h_tr - h_br, 1e-6)
            L = max(L, self._estimate_vertical_width(left_blocks, left_available, L))
            R = max(R, self._estimate_vertical_width(right_blocks, right_available, R))

            final_w = core_w + L + R
            top_available = max(final_w - w_tl - w_tr, 1e-6)
            bottom_available = max(final_w - w_bl - w_br, 1e-6)
            T = max(T, self._estimate_horizontal_height(top_blocks, top_available, T))
            B = max(B, self._estimate_horizontal_height(bottom_blocks, bottom_available, B))

            need_h = max(
                core_h + T + B,
                h_tl + h_bl + self._side_stack_height(left_blocks, max(L, 1e-6)),
                h_tr + h_br + self._side_stack_height(right_blocks, max(R, 1e-6)),
            )
            cur_h = core_h + T + B
            if need_h > cur_h + 1e-6:
                extra = need_h - cur_h
                T += 0.5 * extra
                B += 0.5 * extra

            need_w = max(
                core_w + L + R,
                w_tl + w_tr + self._side_stack_width(top_blocks, max(T, 1e-6)),
                w_bl + w_br + self._side_stack_width(bottom_blocks, max(B, 1e-6)),
            )
            cur_w = core_w + L + R
            if need_w > cur_w + 1e-6:
                extra = need_w - cur_w
                L += 0.5 * extra
                R += 0.5 * extra

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
            else:
                positions[block.block_id] = (X1 - w, Y0, w, h)

        left_cursor = Y0 + h_bl
        for block in sorted(left_blocks, key=lambda b: (0 if (b.width is not None and b.height is not None) else 1, -b.area, b.block_id)):
            bw, bh = self._choose_boundary_shape(block, L, vertical=True)
            positions[block.block_id] = (X0, left_cursor, bw, bh)
            left_cursor += bh

        right_cursor = Y0 + h_br
        for block in sorted(right_blocks, key=lambda b: (0 if (b.width is not None and b.height is not None) else 1, -b.area, b.block_id)):
            bw, bh = self._choose_boundary_shape(block, R, vertical=True)
            positions[block.block_id] = (X1 - bw, right_cursor, bw, bh)
            right_cursor += bh

        top_cursor = X0 + w_tl
        for block in sorted(top_blocks, key=lambda b: (0 if (b.width is not None and b.height is not None) else 1, -b.area, b.block_id)):
            bw, bh = self._choose_boundary_shape(block, T, vertical=False)
            positions[block.block_id] = (top_cursor, Y1 - bh, bw, bh)
            top_cursor += bw

        bottom_cursor = X0 + w_bl
        for block in sorted(bottom_blocks, key=lambda b: (0 if (b.width is not None and b.height is not None) else 1, -b.area, b.block_id)):
            bw, bh = self._choose_boundary_shape(block, B, vertical=False)
            positions[block.block_id] = (bottom_cursor, Y0, bw, bh)
            bottom_cursor += bw

        return positions

    def _shift_to_origin(self, positions: Sequence[Placement]) -> List[Placement]:
        if not positions:
            return []
        x0 = min(p[0] for p in positions)
        y0 = min(p[1] for p in positions)
        return [(x - x0, y - y0, w, h) for x, y, w, h in positions]

    def _resolve_block_overlaps(
        self,
        blocks: Sequence[BlockSpec],
        positions: Sequence[Placement],
    ) -> List[Placement]:
        rects = [list(p) for p in positions]
        preplaced_ids = {block.block_id for block in blocks if block.preplaced}

        for _ in range(14):
            moved = False
            for i in range(len(rects)):
                xi, yi, wi, hi = rects[i]
                for j in range(i + 1, len(rects)):
                    xj, yj, wj, hj = rects[j]
                    ox = min(xi + wi, xj + wj) - max(xi, xj)
                    oy = min(yi + hi, yj + hj) - max(yi, yj)
                    if ox <= 1e-6 or oy <= 1e-6:
                        continue

                    sep_x = ox + 1e-3
                    sep_y = oy + 1e-3
                    if i in preplaced_ids and j in preplaced_ids:
                        continue

                    if i in preplaced_ids or j in preplaced_ids:
                        fixed = i if i in preplaced_ids else j
                        mov = j if fixed == i else i
                        if sep_x <= sep_y:
                            direction = 1.0 if rects[mov][0] >= rects[fixed][0] else -1.0
                            rects[mov][0] += direction * sep_x
                        else:
                            direction = 1.0 if rects[mov][1] >= rects[fixed][1] else -1.0
                            rects[mov][1] += direction * sep_y
                        moved = True
                        continue

                    if sep_x <= sep_y:
                        direction = 1.0 if xj >= xi else -1.0
                        rects[i][0] -= direction * sep_x / 2.0
                        rects[j][0] += direction * sep_x / 2.0
                    else:
                        direction = 1.0 if yj >= yi else -1.0
                        rects[i][1] -= direction * sep_y / 2.0
                        rects[j][1] += direction * sep_y / 2.0
                    moved = True

            for block in blocks:
                if not block.preplaced:
                    continue
                bid = block.block_id
                rects[bid][0] = float(block.x)
                rects[bid][1] = float(block.y)

            if not moved:
                break

        return [tuple(r) for r in rects]

    def _compact_layout(
        self,
        blocks: Sequence[BlockSpec],
        positions: Sequence[Placement],
        rounds: int = 3,
    ) -> List[Placement]:
        rects = [list(p) for p in positions]
        preplaced_ids = {block.block_id for block in blocks if block.preplaced}

        def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
            return min(a1, b1) > max(a0, b0) + 1e-9

        for _ in range(rounds):
            changed = False

            order_x = sorted(
                [i for i in range(len(rects)) if i not in preplaced_ids],
                key=lambda i: (rects[i][0], rects[i][1], i),
            )
            for i in order_x:
                x, y, w, h = rects[i]
                min_x = 0.0 if not preplaced_ids else min(r[0] for r in rects)
                for j, (ox, oy, ow, oh) in enumerate(rects):
                    if i == j:
                        continue
                    if interval_overlap(y, y + h, oy, oy + oh) and ox + ow <= x + 1e-9:
                        min_x = max(min_x, ox + ow)
                if min_x < x - 1e-6:
                    rects[i][0] = min_x
                    changed = True

            order_y = sorted(
                [i for i in range(len(rects)) if i not in preplaced_ids],
                key=lambda i: (rects[i][1], rects[i][0], i),
            )
            for i in order_y:
                x, y, w, h = rects[i]
                min_y = 0.0 if not preplaced_ids else min(r[1] for r in rects)
                for j, (ox, oy, ow, oh) in enumerate(rects):
                    if i == j:
                        continue
                    if interval_overlap(x, x + w, ox, ox + ow) and oy + oh <= y + 1e-9:
                        min_y = max(min_y, oy + oh)
                if min_y < y - 1e-6:
                    rects[i][1] = min_y
                    changed = True

            for block in blocks:
                if not block.preplaced:
                    continue
                bid = block.block_id
                rects[bid][0] = float(block.x)
                rects[bid][1] = float(block.y)

            if not changed:
                break

        return [tuple(r) for r in rects]

    def _legalize_final_blocks(
        self,
        blocks: Sequence[BlockSpec],
        positions: Sequence[Placement],
    ) -> List[Placement]:
        if not positions:
            return []

        rects = [tuple(p) for p in positions]
        x0, y0, x1, y1 = bbox_xyxy(rects)
        pad = 0.25
        shift_x = -min(0.0, x0) + pad
        shift_y = -min(0.0, y0) + pad

        widths = np.array([p[2] for p in rects], dtype=float)
        heights = np.array([p[3] for p in rects], dtype=float)
        centers_x = np.array([p[0] + p[2] / 2.0 + shift_x for p in rects], dtype=float)
        centers_y = np.array([p[1] + p[3] / 2.0 + shift_y for p in rects], dtype=float)

        fixed_mask = np.array([block.preplaced for block in blocks], dtype=bool)
        fixed_x = centers_x.copy()
        fixed_y = centers_y.copy()
        for idx, block in enumerate(blocks):
            if not block.preplaced:
                continue
            fixed_x[idx] = float(block.x) + widths[idx] / 2.0 + shift_x
            fixed_y[idx] = float(block.y) + heights[idx] / 2.0 + shift_y
            centers_x[idx] = fixed_x[idx]
            centers_y[idx] = fixed_y[idx]

        outline_w = max(np.max(centers_x + widths / 2.0) + pad, (x1 - x0) + 2.0 * pad)
        outline_h = max(np.max(centers_y + heights / 2.0) + pad, (y1 - y0) + 2.0 * pad)

        centers_x, centers_y, outline_w, outline_h = self._legalize_item_arrays(
            centers_x,
            centers_y,
            widths,
            heights,
            outline_w,
            outline_h,
            fixed_mask,
            fixed_x,
            fixed_y,
        )
        centers_x, centers_y = enforce_pairwise_separation_iteratively(
            centers_x,
            centers_y,
            widths,
            heights,
            outline_w,
            outline_h,
            fixed_mask,
            fixed_x,
            fixed_y,
            passes=8,
        )

        legalized: List[Placement] = []
        for idx in range(len(rects)):
            left = centers_x[idx] - widths[idx] / 2.0 - shift_x
            bottom = centers_y[idx] - heights[idx] / 2.0 - shift_y
            if blocks[idx].preplaced:
                left = float(blocks[idx].x)
                bottom = float(blocks[idx].y)
            legalized.append((left, bottom, widths[idx], heights[idx]))
        return legalized


# ----------------------------------------------------------------------
# PeF helper functions
# ----------------------------------------------------------------------
def rasterize_density(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    outline_w: float,
    outline_h: float,
    grid_size: int,
) -> np.ndarray:
    density = np.zeros((grid_size, grid_size), dtype=float)
    bin_w = outline_w / grid_size
    bin_h = outline_h / grid_size
    bin_area = bin_w * bin_h

    for i in range(len(x)):
        left, bottom, right, top = rect_bounds(x[i], y[i], w[i], h[i])
        ix0 = max(int(np.floor(left / bin_w)), 0)
        ix1 = min(int(np.ceil(right / bin_w)), grid_size)
        iy0 = max(int(np.floor(bottom / bin_h)), 0)
        iy1 = min(int(np.ceil(top / bin_h)), grid_size)

        for ix in range(ix0, ix1):
            bin_left = ix * bin_w
            bin_right = bin_left + bin_w
            overlap_x = max(0.0, min(right, bin_right) - max(left, bin_left))
            if overlap_x <= 0.0:
                continue
            for iy in range(iy0, iy1):
                bin_bottom = iy * bin_h
                bin_top = bin_bottom + bin_h
                overlap_y = max(0.0, min(top, bin_top) - max(bottom, bin_bottom))
                if overlap_y <= 0.0:
                    continue
                density[iy, ix] += (overlap_x * overlap_y) / bin_area

    return density


def solve_poisson_fft(density: np.ndarray) -> np.ndarray:
    rho = density - density.mean()
    rho_hat = np.fft.fft2(rho)
    nrows, ncols = density.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(nrows)
    kx = 2.0 * np.pi * np.fft.fftfreq(ncols)
    lap = (2.0 * np.cos(ky)[:, None] - 2.0) + (2.0 * np.cos(kx)[None, :] - 2.0)

    psi_hat = np.zeros_like(rho_hat, dtype=complex)
    mask = lap != 0.0
    psi_hat[mask] = -rho_hat[mask] / lap[mask]
    psi = np.fft.ifft2(psi_hat).real
    psi -= psi.mean()
    return psi


def electric_field_from_potential(
    potential: np.ndarray,
    outline_w: float,
    outline_h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    grid_size = potential.shape[0]
    bin_w = outline_w / grid_size
    bin_h = outline_h / grid_size
    dpsi_dy, dpsi_dx = np.gradient(potential, bin_h, bin_w)
    return -dpsi_dx, -dpsi_dy


def bilinear_sample(
    grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    outline_w: float,
    outline_h: float,
) -> np.ndarray:
    grid_h, grid_w = grid.shape
    px = np.clip(xs / outline_w * (grid_w - 1), 0.0, grid_w - 1.0)
    py = np.clip(ys / outline_h * (grid_h - 1), 0.0, grid_h - 1.0)

    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    x1 = np.clip(x0 + 1, 0, grid_w - 1)
    y1 = np.clip(y0 + 1, 0, grid_h - 1)

    tx = px - x0
    ty = py - y0

    v00 = grid[y0, x0]
    v01 = grid[y0, x1]
    v10 = grid[y1, x0]
    v11 = grid[y1, x1]
    top = (1.0 - tx) * v00 + tx * v01
    bottom = (1.0 - tx) * v10 + tx * v11
    return (1.0 - ty) * top + ty * bottom


def module_potential_energy(
    cx: float,
    cy: float,
    width: float,
    area: float,
    potential: np.ndarray,
    outline_w: float,
    outline_h: float,
) -> float:
    height = area / max(width, EPS)
    grid_size = potential.shape[0]
    bin_w = outline_w / grid_size
    bin_h = outline_h / grid_size
    left, bottom, right, top = rect_bounds(cx, cy, width, height)

    ix0 = max(int(np.floor(left / bin_w)), 0)
    ix1 = min(int(np.ceil(right / bin_w)), grid_size)
    iy0 = max(int(np.floor(bottom / bin_h)), 0)
    iy1 = min(int(np.ceil(top / bin_h)), grid_size)

    energy = 0.0
    for ix in range(ix0, ix1):
        bin_left = ix * bin_w
        bin_right = bin_left + bin_w
        overlap_x = max(0.0, min(right, bin_right) - max(left, bin_left))
        if overlap_x <= 0.0:
            continue
        for iy in range(iy0, iy1):
            bin_bottom = iy * bin_h
            bin_top = bin_bottom + bin_h
            overlap_y = max(0.0, min(top, bin_top) - max(bottom, bin_bottom))
            if overlap_y <= 0.0:
                continue
            energy += overlap_x * overlap_y * potential[iy, ix]
    return float(energy)


def connectivity_gradients(
    x: np.ndarray,
    y: np.ndarray,
    item_edges: Sequence[Tuple[int, int, float]],
    pin_targets: Dict[int, Tuple[float, float, float]],
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    grad_x = np.zeros_like(x)
    grad_y = np.zeros_like(y)

    for i, j, weight in item_edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        grad_x[i] += weight * math.tanh(dx / scale)
        grad_x[j] -= weight * math.tanh(dx / scale)
        grad_y[i] += weight * math.tanh(dy / scale)
        grad_y[j] -= weight * math.tanh(dy / scale)

    for idx, (px, py, weight) in pin_targets.items():
        grad_x[idx] += weight * math.tanh((x[idx] - px) / scale)
        grad_y[idx] += weight * math.tanh((y[idx] - py) / scale)

    return grad_x, grad_y


# ----------------------------------------------------------------------
# Obstacle-aware legalization
# ----------------------------------------------------------------------
def add_edge(graph: List[set[int]], src: int, dst: int) -> None:
    if src == dst:
        return
    graph[src].add(dst)


def build_obstacle_aware_constraint_graphs(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    fixed_mask: np.ndarray,
) -> Tuple[List[set[int]], List[set[int]]]:
    n = len(x)
    hcg = [set() for _ in range(n)]
    vcg = [set() for _ in range(n)]

    for i in range(n):
        li, bi, ri, ti = rect_bounds(x[i], y[i], w[i], h[i])
        for j in range(i + 1, n):
            lj, bj, rj, tj = rect_bounds(x[j], y[j], w[j], h[j])
            overlap_x = min(ri, rj) - max(li, lj)
            overlap_y = min(ti, tj) - max(bi, bj)
            center_dx = x[j] - x[i]
            center_dy = y[j] - y[i]

            if overlap_x <= 0.0 and overlap_y <= 0.0:
                gap_x = max(lj - ri, li - rj, 0.0)
                gap_y = max(bj - ti, bi - tj, 0.0)
                if gap_x >= gap_y:
                    add_edge(hcg, i if center_dx >= 0 else j, j if center_dx >= 0 else i)
                else:
                    add_edge(vcg, i if center_dy >= 0 else j, j if center_dy >= 0 else i)
                continue

            if overlap_y > 0.0 and overlap_x <= 0.0:
                add_edge(hcg, i if center_dx >= 0 else j, j if center_dx >= 0 else i)
                continue

            if overlap_x > 0.0 and overlap_y <= 0.0:
                add_edge(vcg, i if center_dy >= 0 else j, j if center_dy >= 0 else i)
                continue

            if fixed_mask[i] and fixed_mask[j]:
                if abs(center_dx) >= abs(center_dy):
                    add_edge(hcg, i if center_dx >= 0 else j, j if center_dx >= 0 else i)
                else:
                    add_edge(vcg, i if center_dy >= 0 else j, j if center_dy >= 0 else i)
                continue

            if fixed_mask[i] ^ fixed_mask[j]:
                fixed = i if fixed_mask[i] else j
                mov = j if fixed == i else i
                if overlap_x >= overlap_y:
                    add_edge(hcg, fixed if x[mov] >= x[fixed] else mov, mov if x[mov] >= x[fixed] else fixed)
                else:
                    add_edge(vcg, fixed if y[mov] >= y[fixed] else mov, mov if y[mov] >= y[fixed] else fixed)
                continue

            if overlap_x >= overlap_y:
                add_edge(hcg, i if center_dx >= 0 else j, j if center_dx >= 0 else i)
            else:
                add_edge(vcg, i if center_dy >= 0 else j, j if center_dy >= 0 else i)

    return hcg, vcg


def topological_order(graph: List[set[int]]) -> List[int]:
    indeg = [0] * len(graph)
    for nbrs in graph:
        for dst in nbrs:
            indeg[dst] += 1

    queue = [i for i, deg in enumerate(indeg) if deg == 0]
    order: List[int] = []
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        order.append(node)
        for dst in graph[node]:
            indeg[dst] -= 1
            if indeg[dst] == 0:
                queue.append(dst)

    if len(order) != len(graph):
        return list(range(len(graph)))
    return order


def solve_positions_from_constraint_graph(
    graph: List[set[int]],
    sizes: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    n = len(graph)
    pos = lower_bounds.copy()
    pos[fixed_mask] = fixed_values[fixed_mask]
    order = topological_order(graph)
    preds = [set() for _ in range(n)]
    for src in range(n):
        for dst in graph[src]:
            preds[dst].add(src)

    for _ in range(2):
        for node in order:
            if fixed_mask[node]:
                pos[node] = fixed_values[node]
                continue
            need = lower_bounds[node]
            for pred in preds[node]:
                need = max(need, pos[pred] + (sizes[pred] + sizes[node]) / 2.0)
            pos[node] = min(max(need, lower_bounds[node]), upper_bounds[node])

        for node in reversed(order):
            if fixed_mask[node]:
                pos[node] = fixed_values[node]
                continue
            allowed = upper_bounds[node]
            for succ in graph[node]:
                allowed = min(allowed, pos[succ] - (sizes[node] + sizes[succ]) / 2.0)
            if allowed >= lower_bounds[node]:
                pos[node] = max(lower_bounds[node], min(pos[node], allowed))

    pos[fixed_mask] = fixed_values[fixed_mask]
    return pos


def constrained_bottom_left_pack(
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    outline_w: float,
    outline_h: float,
    fixed_mask: np.ndarray,
    fixed_x: np.ndarray,
    fixed_y: np.ndarray,
    hcg: List[set[int]],
    vcg: List[set[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(anchor_x)
    lefts = np.full(n, np.nan)
    bottoms = np.full(n, np.nan)
    placed = np.zeros(n, dtype=bool)

    for idx in np.where(fixed_mask)[0]:
        lefts[idx] = fixed_x[idx] - widths[idx] / 2.0
        bottoms[idx] = fixed_y[idx] - heights[idx] / 2.0
        placed[idx] = True

    order_h = topological_order(hcg)
    order_v = topological_order(vcg)
    rank_h = np.zeros(n, dtype=int)
    rank_v = np.zeros(n, dtype=int)
    for pos, idx in enumerate(order_h):
        rank_h[idx] = pos
    for pos, idx in enumerate(order_v):
        rank_v[idx] = pos

    movable = [idx for idx in range(n) if not fixed_mask[idx]]
    movable.sort(key=lambda idx: (rank_h[idx] + rank_v[idx], rank_v[idx], rank_h[idx], anchor_y[idx], anchor_x[idx], idx))

    pred_h = [set() for _ in range(n)]
    pred_v = [set() for _ in range(n)]
    for src in range(n):
        for dst in hcg[src]:
            pred_h[dst].add(src)
        for dst in vcg[src]:
            pred_v[dst].add(src)

    def overlaps_placed(idx: int, left: float, bottom: float) -> bool:
        right = left + widths[idx]
        top = bottom + heights[idx]
        for j in np.where(placed)[0]:
            other_left = lefts[j]
            other_bottom = bottoms[j]
            other_right = other_left + widths[j]
            other_top = other_bottom + heights[j]
            if right <= other_left + 1e-9 or other_right <= left + 1e-9:
                continue
            if top <= other_bottom + 1e-9 or other_top <= bottom + 1e-9:
                continue
            return True
        return False

    def fallback_candidates(limit: int, max_value: float) -> List[float]:
        if max_value <= 0.0:
            return [0.0]
        return list(np.linspace(0.0, max_value, limit))

    for idx in movable:
        anchor_left = np.clip(anchor_x[idx] - widths[idx] / 2.0, 0.0, outline_w - widths[idx])
        anchor_bottom = np.clip(anchor_y[idx] - heights[idx] / 2.0, 0.0, outline_h - heights[idx])

        min_left = 0.0
        min_bottom = 0.0
        max_left = outline_w - widths[idx]
        max_bottom = outline_h - heights[idx]

        for j in pred_h[idx]:
            if placed[j]:
                min_left = max(min_left, lefts[j] + widths[j])
        for j in pred_v[idx]:
            if placed[j]:
                min_bottom = max(min_bottom, bottoms[j] + heights[j])
        for j in hcg[idx]:
            if placed[j]:
                max_left = min(max_left, lefts[j] - widths[idx])
        for j in vcg[idx]:
            if placed[j]:
                max_bottom = min(max_bottom, bottoms[j] - heights[idx])

        if max_left < min_left:
            max_left = outline_w - widths[idx]
        if max_bottom < min_bottom:
            max_bottom = outline_h - heights[idx]

        x_candidates = {0.0, anchor_left, min_left, max(0.0, max_left)}
        y_candidates = {0.0, anchor_bottom, min_bottom, max(0.0, max_bottom)}
        for j in np.where(placed)[0]:
            x_candidates.add(lefts[j] + widths[j])
            x_candidates.add(max(0.0, lefts[j] - widths[idx]))
            y_candidates.add(bottoms[j] + heights[j])
            y_candidates.add(max(0.0, bottoms[j] - heights[idx]))

        best: Optional[Tuple[float, float]] = None
        best_score = None

        def count_graph_violations(left: float, bottom: float) -> int:
            right = left + widths[idx]
            top = bottom + heights[idx]
            violations = 0
            for j in np.where(placed)[0]:
                other_left = lefts[j]
                other_bottom = bottoms[j]
                other_right = other_left + widths[j]
                other_top = other_bottom + heights[j]
                if idx in hcg[j] and left < other_right - 1e-9:
                    violations += 1
                if j in hcg[idx] and right > other_left + 1e-9:
                    violations += 1
                if idx in vcg[j] and bottom < other_top - 1e-9:
                    violations += 1
                if j in vcg[idx] and top > other_bottom + 1e-9:
                    violations += 1
            return violations

        def search(xs: List[float], ys: List[float], strict: bool) -> None:
            nonlocal best, best_score
            for bottom in ys:
                if strict and (bottom < min_bottom - 1e-9 or bottom > max_bottom + 1e-9):
                    continue
                if bottom + heights[idx] > outline_h + 1e-9:
                    continue
                for left in xs:
                    if strict and (left < min_left - 1e-9 or left > max_left + 1e-9):
                        continue
                    if left + widths[idx] > outline_w + 1e-9:
                        continue
                    if overlaps_placed(idx, left, bottom):
                        continue
                    center_x = left + widths[idx] / 2.0
                    center_y = bottom + heights[idx] / 2.0
                    score = (
                        count_graph_violations(left, bottom),
                        (center_x - anchor_x[idx]) ** 2 + (center_y - anchor_y[idx]) ** 2,
                        abs(bottom - anchor_bottom),
                        abs(left - anchor_left),
                        bottom,
                        left,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (left, bottom)

        search(sorted(x_candidates), sorted(y_candidates), strict=True)
        if best is None:
            search(fallback_candidates(30, outline_w - widths[idx]), fallback_candidates(30, outline_h - heights[idx]), strict=True)
        if best is None:
            search(sorted(x_candidates), sorted(y_candidates), strict=False)
        if best is None:
            search(fallback_candidates(36, outline_w - widths[idx]), fallback_candidates(36, outline_h - heights[idx]), strict=False)
        if best is None:
            raise RuntimeError("Obstacle-aware packing failed")

        lefts[idx], bottoms[idx] = best
        placed[idx] = True

    return lefts + widths / 2.0, bottoms + heights / 2.0


def enforce_pairwise_separation_iteratively(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    outline_w: float,
    outline_h: float,
    fixed_mask: np.ndarray,
    fixed_x: np.ndarray,
    fixed_y: np.ndarray,
    passes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    x = x.copy()
    y = y.copy()
    movable = ~fixed_mask

    for _ in range(passes):
        moved = False
        for i in range(len(x)):
            li, bi, ri, ti = rect_bounds(x[i], y[i], w[i], h[i])
            for j in range(i + 1, len(x)):
                lj, bj, rj, tj = rect_bounds(x[j], y[j], w[j], h[j])
                overlap_x = min(ri, rj) - max(li, lj)
                overlap_y = min(ti, tj) - max(bi, bj)
                if overlap_x <= 1e-9 or overlap_y <= 1e-9:
                    continue

                sep_h = overlap_x + 1e-3
                sep_v = overlap_y + 1e-3

                if fixed_mask[i] and fixed_mask[j]:
                    continue
                if fixed_mask[i] ^ fixed_mask[j]:
                    fixed = i if fixed_mask[i] else j
                    mov = j if fixed == i else i
                    if sep_h <= sep_v:
                        direction = 1.0 if x[mov] >= x[fixed] else -1.0
                        x[mov] += direction * sep_h
                    else:
                        direction = 1.0 if y[mov] >= y[fixed] else -1.0
                        y[mov] += direction * sep_v
                    moved = True
                    continue

                if movable[i] and movable[j]:
                    if sep_h <= sep_v:
                        direction = 1.0 if x[j] >= x[i] else -1.0
                        x[i] -= direction * sep_h / 2.0
                        x[j] += direction * sep_h / 2.0
                    else:
                        direction = 1.0 if y[j] >= y[i] else -1.0
                        y[i] -= direction * sep_v / 2.0
                        y[j] += direction * sep_v / 2.0
                    moved = True

        x[:] = np.clip(x, w / 2.0, outline_w - w / 2.0)
        y[:] = np.clip(y, h / 2.0, outline_h - h / 2.0)
        x[fixed_mask] = fixed_x[fixed_mask]
        y[fixed_mask] = fixed_y[fixed_mask]
        if not moved:
            break

    return x, y


def legalize_with_obstacle_aware_graphs(
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    outline_w: float,
    outline_h: float,
    fixed_mask: np.ndarray,
    fixed_x: np.ndarray,
    fixed_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[set[int]], List[set[int]]]:
    hcg, vcg = build_obstacle_aware_constraint_graphs(anchor_x, anchor_y, widths, heights, fixed_mask)

    lower_x = widths / 2.0
    upper_x = outline_w - widths / 2.0
    lower_y = heights / 2.0
    upper_y = outline_h - heights / 2.0

    legal_x = solve_positions_from_constraint_graph(hcg, widths, fixed_mask, fixed_x, lower_x, upper_x)
    legal_y = solve_positions_from_constraint_graph(vcg, heights, fixed_mask, fixed_y, lower_y, upper_y)
    legal_x, legal_y = constrained_bottom_left_pack(
        legal_x,
        legal_y,
        widths,
        heights,
        outline_w,
        outline_h,
        fixed_mask,
        fixed_x,
        fixed_y,
        hcg,
        vcg,
    )
    return legal_x, legal_y, hcg, vcg
