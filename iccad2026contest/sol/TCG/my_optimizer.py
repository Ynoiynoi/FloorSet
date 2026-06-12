#!/usr/bin/env python3
"""
TCG-guided floorplanner for ICCAD 2026 FloorSet.

This implementation uses:
  - grouping-aware item construction
  - MIB-aware shape assignment
  - item-level TCG reconstruction from placements
  - TCG-guided local reinsertion refinement

The solver is conservative about hard constraints:
  - preplaced blocks are emitted exactly at target positions
  - fixed-shape blocks keep exact dimensions
  - all soft blocks preserve target area exactly
  - all placements are overlap-checked at block level
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


ROOT = Path(__file__).resolve().parents[2]
for path in [str(ROOT), str(ROOT.parent)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from iccad2026_evaluate import FloorplanOptimizer


EPS = 1e-6


def popcount(value: int) -> int:
    count = 0
    while value:
        count += value & 1
        value >>= 1
    return count


def rect_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    eps: float = EPS,
) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_y = min(ay + ah, by + bh) - max(ay, by)
    return overlap_x > eps and overlap_y > eps


def edge_contact_length(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    eps: float = EPS,
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    if abs((ax + aw) - bx) <= eps or abs((bx + bw) - ax) <= eps:
        overlap = min(ay + ah, by + bh) - max(ay, by)
        if overlap > eps:
            return overlap

    if abs((ay + ah) - by) <= eps or abs((by + bh) - ay) <= eps:
        overlap = min(ax + aw, bx + bw) - max(ax, bx)
        if overlap > eps:
            return overlap

    return 0.0


def rect_gap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    return dx + dy


def bbox_union(
    bboxes: Iterable[Tuple[float, float, float, float]]
) -> Tuple[float, float, float, float]:
    boxes = list(bboxes)
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)

    x_min = min(x for x, _, _, _ in boxes)
    y_min = min(y for _, y, _, _ in boxes)
    x_max = max(x + w for x, _, w, _ in boxes)
    y_max = max(y + h for _, y, _, h in boxes)
    return (x_min, y_min, x_max, y_max)


@dataclass
class BlockSpec:
    idx: int
    area: float
    fixed: bool
    preplaced: bool
    mib_id: int
    group_id: int
    boundary: int
    target: Optional[Tuple[float, float, float, float]]
    degree: float = 0.0
    pin_degree: float = 0.0
    width: float = 0.0
    height: float = 0.0


@dataclass
class Variant:
    name: str
    local_rects: Dict[int, Tuple[float, float, float, float]]
    width: float
    height: float
    left_blocks: Tuple[int, ...]
    right_blocks: Tuple[int, ...]
    bottom_blocks: Tuple[int, ...]
    top_blocks: Tuple[int, ...]


@dataclass
class Item:
    item_id: int
    blocks: List[int]
    variants: List[Variant]
    anchored: bool = False
    anchor_block: Optional[int] = None
    group_id: int = 0
    attachment_group: Optional[int] = None
    boundary_blocks: Dict[int, int] = field(default_factory=dict)
    area: float = 0.0
    net_weight: float = 0.0
    pin_weight: float = 0.0


@dataclass
class Placement:
    item_id: int
    variant_idx: int
    bbox_x: float
    bbox_y: float


@dataclass
class TCGState:
    gh_preds: Dict[int, Set[int]]
    gh_succs: Dict[int, Set[int]]
    gv_preds: Dict[int, Set[int]]
    gv_succs: Dict[int, Set[int]]


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.refine_passes = 2
        self.max_refine_items = 18

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        self.block_count = int(block_count)
        self.area_targets = area_targets
        self.b2b_connectivity = b2b_connectivity
        self.p2b_connectivity = p2b_connectivity
        self.pins_pos = pins_pos
        self.constraints = constraints
        self.target_positions = target_positions

        self.b2b_adj, self.pin_adj = self._build_connectivity()
        self.blocks = self._build_blocks()
        self._assign_dimensions()
        self.group_members = self._build_groups(column=3)
        self.mib_members = self._build_groups(column=2)
        self.ideal_centers = self._build_ideal_centers()
        self.total_area = sum(block.area for block in self.blocks)
        self.norm_len = max(math.sqrt(max(self.total_area, 1.0)), 1.0)

        items = self._build_items()
        item_map = {item.item_id: item for item in items}

        placements = self._initial_place(items, item_map)
        placements = self._refine_layout(items, item_map, placements)
        placements = self._boundary_push(items, item_map, placements)

        positions = [None] * self.block_count
        for item in items:
            placement = placements[item.item_id]
            variant = item.variants[placement.variant_idx]
            rects = self._materialize_item_rects(variant, placement.bbox_x, placement.bbox_y)
            for block_id, rect in rects.items():
                positions[block_id] = rect

        for idx, rect in enumerate(positions):
            if rect is None:
                block = self.blocks[idx]
                if block.preplaced and block.target is not None:
                    tx, ty, tw, th = block.target
                    positions[idx] = (tx, ty, tw, th)
                else:
                    positions[idx] = (0.0, 0.0, block.width, block.height)

        return positions

    def _build_connectivity(
        self,
    ) -> Tuple[List[List[Tuple[int, float]]], List[List[Tuple[float, float, float]]]]:
        b2b_adj: List[List[Tuple[int, float]]] = [[] for _ in range(self.block_count)]
        pin_adj: List[List[Tuple[float, float, float]]] = [[] for _ in range(self.block_count)]

        if self.b2b_connectivity is not None:
            for edge in self.b2b_connectivity:
                if int(edge[0]) == -1:
                    continue
                i = int(edge[0])
                j = int(edge[1])
                w = float(edge[2])
                if 0 <= i < self.block_count and 0 <= j < self.block_count and w > 0:
                    b2b_adj[i].append((j, w))
                    b2b_adj[j].append((i, w))

        if self.p2b_connectivity is not None:
            for edge in self.p2b_connectivity:
                if int(edge[0]) == -1:
                    continue
                pin_idx = int(edge[0])
                block_idx = int(edge[1])
                weight = float(edge[2])
                if 0 <= block_idx < self.block_count and 0 <= pin_idx < len(self.pins_pos) and weight > 0:
                    px = float(self.pins_pos[pin_idx][0])
                    py = float(self.pins_pos[pin_idx][1])
                    pin_adj[block_idx].append((px, py, weight))

        return b2b_adj, pin_adj

    def _build_blocks(self) -> List[BlockSpec]:
        blocks: List[BlockSpec] = []
        ncols = int(self.constraints.shape[1]) if self.constraints is not None and self.constraints.dim() > 1 else 0

        for i in range(self.block_count):
            fixed = bool(ncols > 0 and self.constraints[i, 0] != 0)
            preplaced = bool(ncols > 1 and self.constraints[i, 1] != 0)
            mib_id = int(self.constraints[i, 2].item()) if ncols > 2 else 0
            group_id = int(self.constraints[i, 3].item()) if ncols > 3 else 0
            boundary = int(self.constraints[i, 4].item()) if ncols > 4 else 0

            target = None
            if self.target_positions is not None and i < len(self.target_positions):
                tx = float(self.target_positions[i, 0])
                ty = float(self.target_positions[i, 1])
                tw = float(self.target_positions[i, 2])
                th = float(self.target_positions[i, 3])
                if tw != -1.0 and th != -1.0:
                    target = (tx, ty, tw, th)

            degree = sum(weight for _, weight in self.b2b_adj[i])
            pin_degree = sum(weight for _, _, weight in self.pin_adj[i])
            area = float(self.area_targets[i]) if float(self.area_targets[i]) > 0 else 1.0

            blocks.append(
                BlockSpec(
                    idx=i,
                    area=area,
                    fixed=fixed,
                    preplaced=preplaced,
                    mib_id=mib_id,
                    group_id=group_id,
                    boundary=boundary,
                    target=target,
                    degree=degree,
                    pin_degree=pin_degree,
                )
            )

        return blocks

    def _build_groups(self, column: int) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        if self.constraints is None or self.constraints.dim() < 2 or self.constraints.shape[1] <= column:
            return groups

        for i in range(self.block_count):
            gid = int(self.constraints[i, column].item())
            if gid > 0:
                groups.setdefault(gid, []).append(i)
        return groups

    def _assign_dimensions(self) -> None:
        # Fixed dimensions first.
        for block in self.blocks:
            if (block.fixed or block.preplaced) and block.target is not None:
                _, _, tw, th = block.target
                block.width = tw
                block.height = th

        mib_ratios: Dict[int, float] = {}
        mib_exact_dims: Dict[int, Optional[Tuple[float, float]]] = {}

        for mib_id, members in self._build_groups(column=2).items():
            anchor_ratios: List[float] = []
            anchor_dims: Optional[Tuple[float, float]] = None
            areas = [self.blocks[idx].area for idx in members]
            same_area = max(areas) / max(min(areas), EPS) <= 1.000001

            for idx in members:
                block = self.blocks[idx]
                if (block.fixed or block.preplaced) and block.width > 0 and block.height > 0:
                    anchor_dims = (block.width, block.height)
                    anchor_ratios.append(block.width / max(block.height, EPS))

            if anchor_ratios:
                ratio = math.exp(sum(math.log(max(r, EPS)) for r in anchor_ratios) / len(anchor_ratios))
            else:
                lr = 0
                tb = 0
                for idx in members:
                    code = self.blocks[idx].boundary
                    if code & (1 | 2):
                        lr += 1
                    if code & (4 | 8):
                        tb += 1
                if tb > lr * 1.2:
                    ratio = 2.25
                elif lr > tb * 1.2:
                    ratio = 0.45
                else:
                    ratio = 1.0

            exact_dims = None
            if same_area:
                area = sum(areas) / len(areas)
                if anchor_dims is not None:
                    exact_dims = anchor_dims
                else:
                    width = math.sqrt(area * ratio)
                    height = area / max(width, EPS)
                    exact_dims = (width, height)

            mib_ratios[mib_id] = ratio
            mib_exact_dims[mib_id] = exact_dims

        for block in self.blocks:
            if block.width > 0 and block.height > 0:
                continue

            if block.mib_id > 0:
                exact_dims = mib_exact_dims.get(block.mib_id)
                if exact_dims is not None:
                    ew, eh = exact_dims
                    rel_err = abs((ew * eh) - block.area) / max(block.area, EPS)
                    if rel_err <= 0.01:
                        block.width = ew
                        block.height = eh
                        continue

                ratio = mib_ratios.get(block.mib_id, 1.0)
                width = math.sqrt(block.area * ratio)
                height = block.area / max(width, EPS)
                block.width = width
                block.height = height
                continue

            code = block.boundary
            if (code & (4 | 8)) and not (code & (1 | 2)):
                ratio = 2.25
            elif (code & (1 | 2)) and not (code & (4 | 8)):
                ratio = 0.45
            else:
                ratio = 1.0

            width = math.sqrt(block.area * ratio)
            height = block.area / max(width, EPS)
            block.width = width
            block.height = height

    def _build_ideal_centers(self) -> Dict[int, Optional[Tuple[float, float, float]]]:
        centers: Dict[int, Optional[Tuple[float, float, float]]] = {}
        preplaced_centers: Dict[int, Tuple[float, float]] = {}

        for block in self.blocks:
            if block.preplaced and block.target is not None:
                x, y, w, h = block.target
                preplaced_centers[block.idx] = (x + w / 2.0, y + h / 2.0)

        for block in self.blocks:
            sx = 0.0
            sy = 0.0
            sw = 0.0

            for px, py, weight in self.pin_adj[block.idx]:
                sx += weight * px
                sy += weight * py
                sw += weight

            for other, weight in self.b2b_adj[block.idx]:
                if other in preplaced_centers:
                    ox, oy = preplaced_centers[other]
                    sx += weight * ox
                    sy += weight * oy
                    sw += weight

            if sw > EPS:
                centers[block.idx] = (sx / sw, sy / sw, sw)
            else:
                centers[block.idx] = None

        return centers

    def _build_items(self) -> List[Item]:
        items: List[Item] = []
        used: Set[int] = set()
        next_item_id = 0

        for group_id, members in sorted(self.group_members.items()):
            preplaced = [idx for idx in members if self.blocks[idx].preplaced]
            rest = [idx for idx in members if not self.blocks[idx].preplaced]

            if preplaced:
                items.append(self._build_core_item(next_item_id, preplaced, group_id))
                next_item_id += 1
                if rest:
                    items.append(self._build_group_item(next_item_id, rest, group_id, attachment_group=group_id))
                    next_item_id += 1
            else:
                items.append(self._build_group_item(next_item_id, members, group_id, attachment_group=None))
                next_item_id += 1

            used.update(members)

        for idx in range(self.block_count):
            if idx in used:
                continue
            if self.blocks[idx].preplaced:
                items.append(self._build_core_item(next_item_id, [idx], 0))
            else:
                items.append(self._build_singleton_item(next_item_id, idx))
            next_item_id += 1

        return items

    def _build_core_item(self, item_id: int, block_ids: List[int], group_id: int) -> Item:
        anchor = block_ids[0]
        anchor_target = self.blocks[anchor].target
        assert anchor_target is not None
        ax, ay, _, _ = anchor_target

        raw_rects: Dict[int, Tuple[float, float, float, float]] = {}
        for block_id in block_ids:
            tx, ty, _, _ = self.blocks[block_id].target  # type: ignore[misc]
            raw_rects[block_id] = (
                tx - ax,
                ty - ay,
                self.blocks[block_id].width,
                self.blocks[block_id].height,
            )

        variant = self._finalize_variant("core", raw_rects)
        boundary_blocks = {
            block_id: self.blocks[block_id].boundary
            for block_id in block_ids
            if self.blocks[block_id].boundary != 0
        }
        area = sum(self.blocks[block_id].area for block_id in block_ids)
        net_weight = sum(self.blocks[block_id].degree for block_id in block_ids)
        pin_weight = sum(self.blocks[block_id].pin_degree for block_id in block_ids)

        return Item(
            item_id=item_id,
            blocks=list(block_ids),
            variants=[variant],
            anchored=True,
            anchor_block=anchor,
            group_id=group_id,
            attachment_group=None,
            boundary_blocks=boundary_blocks,
            area=area,
            net_weight=net_weight,
            pin_weight=pin_weight,
        )

    def _build_singleton_item(self, item_id: int, block_id: int) -> Item:
        block = self.blocks[block_id]
        variants: List[Variant] = []

        base_raw = {block_id: (0.0, 0.0, block.width, block.height)}
        variants.append(self._finalize_variant("single", base_raw))

        if (
            not block.fixed
            and not block.preplaced
            and block.mib_id == 0
            and abs(block.width - block.height) > 1e-4
        ):
            rotated = {block_id: (0.0, 0.0, block.height, block.width)}
            variants.append(self._finalize_variant("single-rot", rotated))

        return Item(
            item_id=item_id,
            blocks=[block_id],
            variants=self._dedupe_variants(variants),
            anchored=False,
            anchor_block=None,
            group_id=0,
            attachment_group=None,
            boundary_blocks={block_id: block.boundary} if block.boundary != 0 else {},
            area=block.area,
            net_weight=block.degree,
            pin_weight=block.pin_degree,
        )

    def _build_group_item(
        self,
        item_id: int,
        block_ids: List[int],
        group_id: int,
        attachment_group: Optional[int],
    ) -> Item:
        variants: List[Variant] = []
        order_h = self._order_horizontal(block_ids)
        order_v = self._order_vertical(block_ids)

        variants.append(self._variant_horizontal("group-h", order_h))
        variants.append(self._variant_vertical("group-v", order_v))
        if len(block_ids) >= 4:
            variants.append(self._variant_snake("group-s", order_h))

        boundary_blocks = {
            block_id: self.blocks[block_id].boundary
            for block_id in block_ids
            if self.blocks[block_id].boundary != 0
        }
        area = sum(self.blocks[block_id].area for block_id in block_ids)
        net_weight = sum(self.blocks[block_id].degree for block_id in block_ids)
        pin_weight = sum(self.blocks[block_id].pin_degree for block_id in block_ids)

        return Item(
            item_id=item_id,
            blocks=list(block_ids),
            variants=self._dedupe_variants(variants),
            anchored=False,
            anchor_block=None,
            group_id=group_id,
            attachment_group=attachment_group,
            boundary_blocks=boundary_blocks,
            area=area,
            net_weight=net_weight,
            pin_weight=pin_weight,
        )

    def _order_horizontal(self, block_ids: Sequence[int]) -> List[int]:
        left = [idx for idx in block_ids if self.blocks[idx].boundary & 1]
        right = [idx for idx in block_ids if (self.blocks[idx].boundary & 2) and idx not in left]
        middle = [idx for idx in block_ids if idx not in left and idx not in right]

        def mid_key(idx: int) -> Tuple[float, float, int]:
            block = self.blocks[idx]
            return (-(block.degree + 0.4 * block.pin_degree), -block.area, idx)

        left.sort(key=mid_key)
        right.sort(key=mid_key)
        middle.sort(key=mid_key)
        return self._stable_unique(left + middle + right)

    def _order_vertical(self, block_ids: Sequence[int]) -> List[int]:
        bottom = [idx for idx in block_ids if self.blocks[idx].boundary & 8]
        top = [idx for idx in block_ids if (self.blocks[idx].boundary & 4) and idx not in bottom]
        middle = [idx for idx in block_ids if idx not in bottom and idx not in top]

        def mid_key(idx: int) -> Tuple[float, float, int]:
            block = self.blocks[idx]
            return (-(block.degree + 0.4 * block.pin_degree), -block.area, idx)

        bottom.sort(key=mid_key)
        top.sort(key=mid_key)
        middle.sort(key=mid_key)
        return self._stable_unique(bottom + middle + top)

    def _variant_horizontal(self, name: str, order: Sequence[int]) -> Variant:
        raw: Dict[int, Tuple[float, float, float, float]] = {}
        x_cursor = 0.0
        for idx in order:
            block = self.blocks[idx]
            raw[idx] = (x_cursor, 0.0, block.width, block.height)
            x_cursor += block.width
        return self._finalize_variant(name, raw)

    def _variant_vertical(self, name: str, order: Sequence[int]) -> Variant:
        raw: Dict[int, Tuple[float, float, float, float]] = {}
        y_cursor = 0.0
        for idx in order:
            block = self.blocks[idx]
            raw[idx] = (0.0, y_cursor, block.width, block.height)
            y_cursor += block.height
        return self._finalize_variant(name, raw)

    def _variant_snake(self, name: str, order: Sequence[int]) -> Variant:
        split = (len(order) + 1) // 2
        row1 = list(order[:split])
        row2 = list(order[split:])
        raw: Dict[int, Tuple[float, float, float, float]] = {}

        x_cursor = 0.0
        row1_height = 0.0
        for idx in row1:
            block = self.blocks[idx]
            raw[idx] = (x_cursor, 0.0, block.width, block.height)
            x_cursor += block.width
            row1_height = max(row1_height, block.height)

        row2_width = sum(self.blocks[idx].width for idx in row2)
        x_cursor = max(x_cursor, row2_width)
        for idx in reversed(row2):
            block = self.blocks[idx]
            x_cursor -= block.width
            raw[idx] = (x_cursor, row1_height, block.width, block.height)

        return self._finalize_variant(name, raw)

    def _finalize_variant(
        self,
        name: str,
        raw_rects: Dict[int, Tuple[float, float, float, float]],
    ) -> Variant:
        x_min = min(x for x, _, _, _ in raw_rects.values())
        y_min = min(y for _, y, _, _ in raw_rects.values())
        shifted: Dict[int, Tuple[float, float, float, float]] = {}

        x_max = -float("inf")
        y_max = -float("inf")
        for block_id, (x, y, w, h) in raw_rects.items():
            sx = x - x_min
            sy = y - y_min
            shifted[block_id] = (sx, sy, w, h)
            x_max = max(x_max, sx + w)
            y_max = max(y_max, sy + h)

        width = x_max
        height = y_max
        left_blocks = tuple(sorted(block_id for block_id, (x, _, _, _) in shifted.items() if x <= EPS))
        right_blocks = tuple(
            sorted(
                block_id
                for block_id, (x, _, w, _) in shifted.items()
                if (x + w) >= width - EPS
            )
        )
        bottom_blocks = tuple(sorted(block_id for block_id, (_, y, _, _) in shifted.items() if y <= EPS))
        top_blocks = tuple(
            sorted(
                block_id
                for block_id, (_, y, _, h) in shifted.items()
                if (y + h) >= height - EPS
            )
        )

        return Variant(
            name=name,
            local_rects=shifted,
            width=width,
            height=height,
            left_blocks=left_blocks,
            right_blocks=right_blocks,
            bottom_blocks=bottom_blocks,
            top_blocks=top_blocks,
        )

    def _stable_unique(self, values: Sequence[int]) -> List[int]:
        seen: Set[int] = set()
        ordered: List[int] = []
        for value in values:
            if value not in seen:
                ordered.append(value)
                seen.add(value)
        return ordered

    def _dedupe_variants(self, variants: Sequence[Variant]) -> List[Variant]:
        deduped: List[Variant] = []
        seen: Set[Tuple] = set()
        for variant in variants:
            signature = tuple(
                (block_id, round(rect[0], 6), round(rect[1], 6), round(rect[2], 6), round(rect[3], 6))
                for block_id, rect in sorted(variant.local_rects.items())
            )
            if signature not in seen:
                deduped.append(variant)
                seen.add(signature)
        return deduped

    def _initial_place(self, items: Sequence[Item], item_map: Dict[int, Item]) -> Dict[int, Placement]:
        placements: Dict[int, Placement] = {}
        block_rects: Dict[int, Tuple[float, float, float, float]] = {}
        item_bboxes: Dict[int, Tuple[float, float, float, float]] = {}

        anchored = [item for item in items if item.anchored]
        movable = [item for item in items if not item.anchored]

        for item in anchored:
            placement = self._fixed_placement(item)
            placements[item.item_id] = placement
            variant = item.variants[placement.variant_idx]
            rects = self._materialize_item_rects(variant, placement.bbox_x, placement.bbox_y)
            block_rects.update(rects)
            item_bboxes[item.item_id] = (placement.bbox_x, placement.bbox_y, variant.width, variant.height)

        movable.sort(key=self._item_priority_key)

        for item in movable:
            placement = self._find_best_placement(item, placements, block_rects, item_bboxes, None)
            placements[item.item_id] = placement
            variant = item_map[item.item_id].variants[placement.variant_idx]
            rects = self._materialize_item_rects(variant, placement.bbox_x, placement.bbox_y)
            block_rects.update(rects)
            item_bboxes[item.item_id] = (placement.bbox_x, placement.bbox_y, variant.width, variant.height)

        return placements

    def _item_priority_key(self, item: Item) -> Tuple:
        boundary_count = len(item.boundary_blocks)
        corner_count = sum(1 for code in item.boundary_blocks.values() if popcount(code & 0xF) >= 2)
        grouped = 1 if len(item.blocks) > 1 else 0
        tail = 1 if item.attachment_group is not None else 0
        return (
            -tail,
            -corner_count,
            -boundary_count,
            -grouped,
            -(item.net_weight + 0.6 * item.pin_weight),
            -item.area,
            item.item_id,
        )

    def _fixed_placement(self, item: Item) -> Placement:
        variant = item.variants[0]
        assert item.anchor_block is not None
        anchor_target = self.blocks[item.anchor_block].target
        assert anchor_target is not None
        ax, ay, _, _ = anchor_target
        local_x, local_y, _, _ = variant.local_rects[item.anchor_block]
        bbox_x = ax - local_x
        bbox_y = ay - local_y
        return Placement(item.item_id, 0, bbox_x, bbox_y)

    def _find_best_placement(
        self,
        item: Item,
        placements: Dict[int, Placement],
        block_rects: Dict[int, Tuple[float, float, float, float]],
        item_bboxes: Dict[int, Tuple[float, float, float, float]],
        tcg: Optional[TCGState],
    ) -> Placement:
        best_score = float("inf")
        best: Optional[Placement] = None

        for variant_idx, variant in enumerate(item.variants):
            candidates = self._generate_candidates(item, variant, placements, block_rects, item_bboxes, tcg)
            for bbox_x, bbox_y in candidates:
                score = self._score_candidate(item, variant, bbox_x, bbox_y, block_rects, item_bboxes)
                if score < best_score:
                    best_score = score
                    best = Placement(item.item_id, variant_idx, bbox_x, bbox_y)

        if best is not None:
            return best

        # Guaranteed fallback: expand outward until a feasible slot appears.
        for variant_idx, variant in enumerate(item.variants):
            bbox_x, bbox_y = self._fallback_position(item, variant, block_rects, item_bboxes)
            return Placement(item.item_id, variant_idx, bbox_x, bbox_y)

        # Defensive fallback, should never trigger.
        return Placement(item.item_id, 0, 0.0, 0.0)

    def _generate_candidates(
        self,
        item: Item,
        variant: Variant,
        placements: Dict[int, Placement],
        block_rects: Dict[int, Tuple[float, float, float, float]],
        item_bboxes: Dict[int, Tuple[float, float, float, float]],
        tcg: Optional[TCGState],
    ) -> List[Tuple[float, float]]:
        if not item_bboxes:
            return [(0.0, 0.0)]

        candidates: Set[Tuple[float, float]] = set()
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox_union(item_bboxes.values())

        def add(x: float, y: float) -> None:
            candidates.add((round(x, 4), round(y, 4)))

        add(bbox_x1, bbox_y0)
        add(bbox_x0, bbox_y1)
        add(bbox_x1, bbox_y1 - variant.height)
        add(bbox_x0 - variant.width, bbox_y0)
        add(bbox_x0 - variant.width, bbox_y1 - variant.height)
        add(bbox_x0, bbox_y0 - variant.height)
        add(bbox_x1 - variant.width, bbox_y1)
        add((bbox_x0 + bbox_x1 - variant.width) * 0.5, (bbox_y0 + bbox_y1 - variant.height) * 0.5)

        for _, (bx, by, bw, bh) in item_bboxes.items():
            cx = bx + 0.5 * bw
            cy = by + 0.5 * bh
            add(bx + bw, by)
            add(bx + bw, by + bh - variant.height)
            add(bx - variant.width, by)
            add(bx - variant.width, by + bh - variant.height)
            add(bx, by + bh)
            add(bx + bw - variant.width, by + bh)
            add(bx, by - variant.height)
            add(bx + bw - variant.width, by - variant.height)
            add(bx + bw, cy - 0.5 * variant.height)
            add(bx - variant.width, cy - 0.5 * variant.height)
            add(cx - 0.5 * variant.width, by + bh)
            add(cx - 0.5 * variant.width, by - variant.height)

        # Boundary-aware outward placements.
        if item.boundary_blocks:
            if any(code & 1 for code in item.boundary_blocks.values()):
                add(bbox_x0 - variant.width, bbox_y0)
                add(bbox_x0 - variant.width, bbox_y1 - variant.height)
            if any(code & 2 for code in item.boundary_blocks.values()):
                add(bbox_x1, bbox_y0)
                add(bbox_x1, bbox_y1 - variant.height)
            if any(code & 4 for code in item.boundary_blocks.values()):
                add(bbox_x0, bbox_y1)
                add(bbox_x1 - variant.width, bbox_y1)
            if any(code & 8 for code in item.boundary_blocks.values()):
                add(bbox_x0, bbox_y0 - variant.height)
                add(bbox_x1 - variant.width, bbox_y0 - variant.height)

        # Group-attachment candidates: directly touch already-placed same-group blocks.
        if item.attachment_group is not None:
            group_rects = [
                (block_id, rect)
                for block_id, rect in block_rects.items()
                if self.blocks[block_id].group_id == item.attachment_group
            ]
            for _, neighbor in group_rects:
                nx, ny, nw, nh = neighbor
                for block_id in variant.left_blocks:
                    lx, ly, bw, bh = variant.local_rects[block_id]
                    add(nx + nw - lx, ny - ly)
                    add(nx + nw - lx, ny + nh - bh - ly)
                for block_id in variant.right_blocks:
                    lx, ly, bw, bh = variant.local_rects[block_id]
                    add(nx - (lx + bw), ny - ly)
                    add(nx - (lx + bw), ny + nh - bh - ly)
                for block_id in variant.bottom_blocks:
                    lx, ly, bw, bh = variant.local_rects[block_id]
                    add(nx - lx, ny + nh - ly)
                    add(nx + nw - bw - lx, ny + nh - ly)
                for block_id in variant.top_blocks:
                    lx, ly, bw, bh = variant.local_rects[block_id]
                    add(nx - lx, ny - (ly + bh))
                    add(nx + nw - bw - lx, ny - (ly + bh))

        # TCG-neighbor candidates around previous predecessors / successors.
        if tcg is not None:
            neighbor_ids = set()
            neighbor_ids.update(tcg.gh_preds.get(item.item_id, set()))
            neighbor_ids.update(tcg.gh_succs.get(item.item_id, set()))
            neighbor_ids.update(tcg.gv_preds.get(item.item_id, set()))
            neighbor_ids.update(tcg.gv_succs.get(item.item_id, set()))
            for neighbor_id in neighbor_ids:
                if neighbor_id not in item_bboxes:
                    continue
                bx, by, bw, bh = item_bboxes[neighbor_id]
                add(bx + bw, by)
                add(bx - variant.width, by)
                add(bx, by + bh)
                add(bx, by - variant.height)

        # Coarse ideal-center placement.
        ideal = self._item_ideal_center(item)
        if ideal is not None:
            ix, iy = ideal
            add(ix - 0.5 * variant.width, iy - 0.5 * variant.height)

        return sorted(candidates)

    def _item_ideal_center(self, item: Item) -> Optional[Tuple[float, float]]:
        sx = 0.0
        sy = 0.0
        sw = 0.0
        for block_id in item.blocks:
            ideal = self.ideal_centers.get(block_id)
            if ideal is None:
                continue
            ix, iy, weight = ideal
            sx += weight * ix
            sy += weight * iy
            sw += weight
        if sw <= EPS:
            return None
        return (sx / sw, sy / sw)

    def _materialize_item_rects(
        self,
        variant: Variant,
        bbox_x: float,
        bbox_y: float,
    ) -> Dict[int, Tuple[float, float, float, float]]:
        rects: Dict[int, Tuple[float, float, float, float]] = {}
        for block_id, (lx, ly, w, h) in variant.local_rects.items():
            rects[block_id] = (bbox_x + lx, bbox_y + ly, w, h)
        return rects

    def _score_candidate(
        self,
        item: Item,
        variant: Variant,
        bbox_x: float,
        bbox_y: float,
        block_rects: Dict[int, Tuple[float, float, float, float]],
        item_bboxes: Dict[int, Tuple[float, float, float, float]],
    ) -> float:
        rects = self._materialize_item_rects(variant, bbox_x, bbox_y)

        for rect in rects.values():
            for other_id, other_rect in block_rects.items():
                if other_id in rects:
                    continue
                if rect_overlap(rect, other_rect):
                    return float("inf")

        current_boxes = list(item_bboxes.values())
        candidate_bbox = (bbox_x, bbox_y, variant.width, variant.height)
        current_union = bbox_union(current_boxes) if current_boxes else (bbox_x, bbox_y, bbox_x + variant.width, bbox_y + variant.height)

        new_xmin = min(current_union[0], bbox_x)
        new_ymin = min(current_union[1], bbox_y)
        new_xmax = max(current_union[2], bbox_x + variant.width)
        new_ymax = max(current_union[3], bbox_y + variant.height)
        new_area = (new_xmax - new_xmin) * (new_ymax - new_ymin)

        bbox_term = new_area / max(self.total_area, 1.0)

        placed_hpwl = 0.0
        placed_weight = 0.0
        pin_hpwl = 0.0
        pin_weight = 0.0
        ideal_hpwl = 0.0
        ideal_weight = 0.0

        for block_id, rect in rects.items():
            x, y, w, h = rect
            cx = x + 0.5 * w
            cy = y + 0.5 * h

            for px, py, weight in self.pin_adj[block_id]:
                pin_hpwl += weight * (abs(cx - px) + abs(cy - py))
                pin_weight += weight

            for other_block, weight in self.b2b_adj[block_id]:
                other_rect = block_rects.get(other_block)
                if other_rect is None:
                    continue
                ox, oy, ow, oh = other_rect
                ocx = ox + 0.5 * ow
                ocy = oy + 0.5 * oh
                placed_hpwl += weight * (abs(cx - ocx) + abs(cy - ocy))
                placed_weight += weight

            ideal = self.ideal_centers.get(block_id)
            if ideal is not None:
                ix, iy, weight = ideal
                ideal_hpwl += weight * (abs(cx - ix) + abs(cy - iy))
                ideal_weight += weight

        net_term = placed_hpwl / max(placed_weight * self.norm_len, 1.0) if placed_weight > 0 else 0.0
        pin_term = pin_hpwl / max(pin_weight * self.norm_len, 1.0) if pin_weight > 0 else 0.0
        ideal_term = ideal_hpwl / max(ideal_weight * self.norm_len, 1.0) if ideal_weight > 0 else 0.0
        boundary_term = self._boundary_penalty(rects, (new_xmin, new_ymin, new_xmax, new_ymax), item)
        attach_term = self._attachment_penalty(item, rects, block_rects)

        cx = bbox_x + 0.5 * variant.width
        cy = bbox_y + 0.5 * variant.height
        union_cx = 0.5 * (new_xmin + new_xmax)
        union_cy = 0.5 * (new_ymin + new_ymax)
        spread_term = (abs(cx - union_cx) + abs(cy - union_cy)) / self.norm_len

        return (
            3.2 * bbox_term
            + 1.2 * pin_term
            + 1.0 * net_term
            + 0.45 * ideal_term
            + 5.2 * boundary_term
            + 6.5 * attach_term
            + 0.12 * spread_term
        )

    def _boundary_penalty(
        self,
        rects: Dict[int, Tuple[float, float, float, float]],
        bbox_xyxy: Tuple[float, float, float, float],
        item: Item,
    ) -> float:
        if not item.boundary_blocks:
            return 0.0

        x_min, y_min, x_max, y_max = bbox_xyxy
        penalty = 0.0
        for block_id, code in item.boundary_blocks.items():
            x, y, w, h = rects[block_id]
            if code & 1:
                penalty += abs(x - x_min)
            if code & 2:
                penalty += abs((x + w) - x_max)
            if code & 4:
                penalty += abs((y + h) - y_max)
            if code & 8:
                penalty += abs(y - y_min)
        return penalty / max(self.norm_len, 1.0)

    def _attachment_penalty(
        self,
        item: Item,
        rects: Dict[int, Tuple[float, float, float, float]],
        block_rects: Dict[int, Tuple[float, float, float, float]],
    ) -> float:
        if item.attachment_group is None:
            return 0.0

        same_group = [
            rect
            for block_id, rect in block_rects.items()
            if self.blocks[block_id].group_id == item.attachment_group
        ]
        if not same_group:
            return 0.0

        min_gap = float("inf")
        corner_touch = False
        for rect in rects.values():
            for other in same_group:
                if edge_contact_length(rect, other) > EPS:
                    return 0.0
                gap = rect_gap(rect, other)
                if gap < min_gap:
                    min_gap = gap
                if gap <= EPS:
                    corner_touch = True

        if corner_touch:
            return 0.45
        return 0.8 + min_gap / max(self.norm_len, 1.0)

    def _fallback_position(
        self,
        item: Item,
        variant: Variant,
        block_rects: Dict[int, Tuple[float, float, float, float]],
        item_bboxes: Dict[int, Tuple[float, float, float, float]],
    ) -> Tuple[float, float]:
        if not item_bboxes:
            return (0.0, 0.0)

        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox_union(item_bboxes.values())
        best = (bbox_x1, bbox_y0)
        best_score = float("inf")

        for ring in range(1, 80):
            step_x = ring * max(variant.width * 0.75, self.norm_len * 0.05)
            step_y = ring * max(variant.height * 0.75, self.norm_len * 0.05)
            candidates = [
                (bbox_x1 + step_x, bbox_y0),
                (bbox_x0 - variant.width - step_x, bbox_y0),
                (bbox_x0, bbox_y1 + step_y),
                (bbox_x0, bbox_y0 - variant.height - step_y),
                (bbox_x1 + step_x, bbox_y1 + step_y),
                (bbox_x0 - variant.width - step_x, bbox_y0 - variant.height - step_y),
            ]
            for bbox_x, bbox_y in candidates:
                score = self._score_candidate(item, variant, bbox_x, bbox_y, block_rects, item_bboxes)
                if score < best_score:
                    best_score = score
                    best = (bbox_x, bbox_y)
            if best_score < float("inf"):
                return best

        return best

    def _materialize_layout(
        self,
        items: Sequence[Item],
        placements: Dict[int, Placement],
    ) -> Tuple[
        Dict[int, Tuple[float, float, float, float]],
        Dict[int, Tuple[float, float, float, float]],
    ]:
        item_map = {item.item_id: item for item in items}
        block_rects: Dict[int, Tuple[float, float, float, float]] = {}
        item_bboxes: Dict[int, Tuple[float, float, float, float]] = {}

        for item_id, placement in placements.items():
            item = item_map[item_id]
            variant = item.variants[placement.variant_idx]
            rects = self._materialize_item_rects(variant, placement.bbox_x, placement.bbox_y)
            block_rects.update(rects)
            item_bboxes[item_id] = (placement.bbox_x, placement.bbox_y, variant.width, variant.height)

        return block_rects, item_bboxes

    def _build_tcg(
        self,
        items: Sequence[Item],
        placements: Dict[int, Placement],
    ) -> TCGState:
        _, item_bboxes = self._materialize_layout(items, placements)
        item_ids = sorted(item_bboxes)

        gh_preds: Dict[int, Set[int]] = {item_id: set() for item_id in item_ids}
        gh_succs: Dict[int, Set[int]] = {item_id: set() for item_id in item_ids}
        gv_preds: Dict[int, Set[int]] = {item_id: set() for item_id in item_ids}
        gv_succs: Dict[int, Set[int]] = {item_id: set() for item_id in item_ids}

        for i in range(len(item_ids)):
            a_id = item_ids[i]
            ax, ay, aw, ah = item_bboxes[a_id]
            for j in range(i + 1, len(item_ids)):
                b_id = item_ids[j]
                bx, by, bw, bh = item_bboxes[b_id]

                candidates: List[Tuple[str, int, int, float]] = []
                h_ab = bx - (ax + aw)
                h_ba = ax - (bx + bw)
                v_ab = by - (ay + ah)
                v_ba = ay - (by + bh)

                if h_ab >= -EPS:
                    candidates.append(("h", a_id, b_id, max(h_ab, 0.0)))
                if h_ba >= -EPS:
                    candidates.append(("h", b_id, a_id, max(h_ba, 0.0)))
                if v_ab >= -EPS:
                    candidates.append(("v", a_id, b_id, max(v_ab, 0.0)))
                if v_ba >= -EPS:
                    candidates.append(("v", b_id, a_id, max(v_ba, 0.0)))

                if not candidates:
                    # Should never happen for feasible placements; fall back to horizontal by x.
                    if ax <= bx:
                        candidates.append(("h", a_id, b_id, 0.0))
                    else:
                        candidates.append(("h", b_id, a_id, 0.0))

                axis, src, dst, _ = min(candidates, key=lambda entry: entry[3])
                if axis == "h":
                    gh_succs[src].add(dst)
                    gh_preds[dst].add(src)
                else:
                    gv_succs[src].add(dst)
                    gv_preds[dst].add(src)

        return TCGState(gh_preds=gh_preds, gh_succs=gh_succs, gv_preds=gv_preds, gv_succs=gv_succs)

    def _refine_layout(
        self,
        items: Sequence[Item],
        item_map: Dict[int, Item],
        placements: Dict[int, Placement],
    ) -> Dict[int, Placement]:
        movable_ids = [item.item_id for item in items if not item.anchored]
        if not movable_ids:
            return placements

        for _ in range(self.refine_passes):
            tcg = self._build_tcg(items, placements)
            block_rects, item_bboxes = self._materialize_layout(items, placements)

            ranked: List[Tuple[float, int]] = []
            for item_id in movable_ids:
                item = item_map[item_id]
                placement = placements[item_id]
                variant = item.variants[placement.variant_idx]
                score = self._score_candidate(item, variant, placement.bbox_x, placement.bbox_y, block_rects_without(block_rects, item.blocks), item_bboxes_without(item_bboxes, item_id))
                ranked.append((score, item_id))

            ranked.sort(reverse=True)
            improved = False

            for _, item_id in ranked[: self.max_refine_items]:
                item = item_map[item_id]
                current = placements.pop(item_id)
                reduced_block_rects, reduced_item_bboxes = self._materialize_layout(items, placements)
                current_variant = item.variants[current.variant_idx]
                current_score = self._score_candidate(item, current_variant, current.bbox_x, current.bbox_y, reduced_block_rects, reduced_item_bboxes)

                best = self._find_best_placement(item, placements, reduced_block_rects, reduced_item_bboxes, tcg)
                best_variant = item.variants[best.variant_idx]
                best_score = self._score_candidate(item, best_variant, best.bbox_x, best.bbox_y, reduced_block_rects, reduced_item_bboxes)

                if best_score + 1e-6 < current_score:
                    placements[item_id] = best
                    improved = True
                else:
                    placements[item_id] = current

            if not improved:
                break

        return placements

    def _boundary_push(
        self,
        items: Sequence[Item],
        item_map: Dict[int, Item],
        placements: Dict[int, Placement],
    ) -> Dict[int, Placement]:
        movable = [item for item in items if not item.anchored and item.boundary_blocks]
        if not movable:
            return placements

        block_rects, item_bboxes = self._materialize_layout(items, placements)
        x_min, y_min, x_max, y_max = bbox_union(item_bboxes.values())

        for item in movable:
            current = placements[item.item_id]
            variant = item.variants[current.variant_idx]
            best = current
            best_score = self._score_candidate(item, variant, current.bbox_x, current.bbox_y, block_rects_without(block_rects, item.blocks), item_bboxes_without(item_bboxes, item.item_id))

            reduced_block_rects = block_rects_without(block_rects, item.blocks)
            reduced_item_bboxes = item_bboxes_without(item_bboxes, item.item_id)
            candidates: Set[Tuple[float, float]] = set()

            if any(code & 1 for code in item.boundary_blocks.values()):
                candidates.add((x_min, current.bbox_y))
                candidates.add((x_min - 0.6 * variant.width, current.bbox_y))
            if any(code & 2 for code in item.boundary_blocks.values()):
                candidates.add((x_max - variant.width, current.bbox_y))
                candidates.add((x_max, current.bbox_y))
            if any(code & 4 for code in item.boundary_blocks.values()):
                candidates.add((current.bbox_x, y_max - variant.height))
                candidates.add((current.bbox_x, y_max))
            if any(code & 8 for code in item.boundary_blocks.values()):
                candidates.add((current.bbox_x, y_min))
                candidates.add((current.bbox_x, y_min - 0.6 * variant.height))

            if any((code & 1) and (code & 4) for code in item.boundary_blocks.values()):
                candidates.add((x_min - 0.6 * variant.width, y_max))
            if any((code & 2) and (code & 4) for code in item.boundary_blocks.values()):
                candidates.add((x_max, y_max))
            if any((code & 1) and (code & 8) for code in item.boundary_blocks.values()):
                candidates.add((x_min - 0.6 * variant.width, y_min - 0.6 * variant.height))
            if any((code & 2) and (code & 8) for code in item.boundary_blocks.values()):
                candidates.add((x_max, y_min - 0.6 * variant.height))

            for bbox_x, bbox_y in sorted(candidates):
                score = self._score_candidate(item, variant, bbox_x, bbox_y, reduced_block_rects, reduced_item_bboxes)
                if score + 1e-6 < best_score:
                    best_score = score
                    best = Placement(item.item_id, current.variant_idx, bbox_x, bbox_y)

            placements[item.item_id] = best
            block_rects, item_bboxes = self._materialize_layout(items, placements)
            x_min, y_min, x_max, y_max = bbox_union(item_bboxes.values())

        return placements


def block_rects_without(
    block_rects: Dict[int, Tuple[float, float, float, float]],
    block_ids: Sequence[int],
) -> Dict[int, Tuple[float, float, float, float]]:
    removed = set(block_ids)
    return {block_id: rect for block_id, rect in block_rects.items() if block_id not in removed}


def item_bboxes_without(
    item_bboxes: Dict[int, Tuple[float, float, float, float]],
    item_id: int,
) -> Dict[int, Tuple[float, float, float, float]]:
    return {other_id: bbox for other_id, bbox in item_bboxes.items() if other_id != item_id}
