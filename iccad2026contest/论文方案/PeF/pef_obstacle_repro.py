from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


@dataclass
class FloorplanInstanceObstacle:
    outline_w: float
    outline_h: float
    areas: np.ndarray
    soft_mask: np.ndarray
    fixed_mask: np.ndarray
    ar_low: np.ndarray
    ar_high: np.ndarray
    widths: np.ndarray
    heights: np.ndarray
    x: np.ndarray
    y: np.ndarray
    fixed_x: np.ndarray
    fixed_y: np.ndarray
    nets: list[list[int]]


def rect_bounds(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def clamp_positions(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    outline_w: float,
    outline_h: float,
    fixed_mask: np.ndarray | None = None,
    fixed_x: np.ndarray | None = None,
    fixed_y: np.ndarray | None = None,
) -> None:
    x[:] = np.clip(x, w / 2.0, outline_w - w / 2.0)
    y[:] = np.clip(y, h / 2.0, outline_h - h / 2.0)
    if fixed_mask is not None and fixed_x is not None and fixed_y is not None:
        x[fixed_mask] = fixed_x[fixed_mask]
        y[fixed_mask] = fixed_y[fixed_mask]


def total_overlap_area(x: np.ndarray, y: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    total = 0.0
    n = len(x)
    for i in range(n):
        left_i, bottom_i, right_i, top_i = rect_bounds(x[i], y[i], w[i], h[i])
        for j in range(i + 1, n):
            left_j, bottom_j, right_j, top_j = rect_bounds(x[j], y[j], w[j], h[j])
            overlap_x = max(0.0, min(right_i, right_j) - max(left_i, left_j))
            overlap_y = max(0.0, min(top_i, top_j) - max(bottom_i, bottom_j))
            total += overlap_x * overlap_y
    return float(total)


def hpwl(x: np.ndarray, y: np.ndarray, nets: list[list[int]]) -> float:
    total = 0.0
    for net in nets:
        xs = x[net]
        ys = y[net]
        total += (xs.max() - xs.min()) + (ys.max() - ys.min())
    return float(total)


def lse_wirelength_and_grad(x: np.ndarray, y: np.ndarray, nets: list[list[int]], alpha: float) -> tuple[float, np.ndarray, np.ndarray]:
    grad_x = np.zeros_like(x)
    grad_y = np.zeros_like(y)
    total = 0.0

    for net in nets:
        xs = x[net]
        ys = y[net]

        exp_px = np.exp((xs - xs.max()) / alpha)
        exp_nx = np.exp((-xs - (-xs).max()) / alpha)
        exp_py = np.exp((ys - ys.max()) / alpha)
        exp_ny = np.exp((-ys - (-ys).max()) / alpha)

        sum_px = exp_px.sum()
        sum_nx = exp_nx.sum()
        sum_py = exp_py.sum()
        sum_ny = exp_ny.sum()

        total += (
            xs.max()
            + alpha * np.log(sum_px)
            + (-xs).max()
            + alpha * np.log(sum_nx)
            + ys.max()
            + alpha * np.log(sum_py)
            + (-ys).max()
            + alpha * np.log(sum_ny)
        )

        grad_x[net] += exp_px / sum_px - exp_nx / sum_nx
        grad_y[net] += exp_py / sum_py - exp_ny / sum_ny

    return float(total), grad_x, grad_y


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


def electric_field_from_potential(potential: np.ndarray, outline_w: float, outline_h: float) -> tuple[np.ndarray, np.ndarray]:
    grid_size = potential.shape[0]
    bin_w = outline_w / grid_size
    bin_h = outline_h / grid_size
    dpsi_dy, dpsi_dx = np.gradient(potential, bin_h, bin_w)
    ex = -dpsi_dx
    ey = -dpsi_dy
    return ex, ey


def bilinear_sample(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray, outline_w: float, outline_h: float) -> np.ndarray:
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
    height = area / width
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


def project_soft_widths(
    widths: np.ndarray,
    heights: np.ndarray,
    areas: np.ndarray,
    soft_mask: np.ndarray,
    fixed_mask: np.ndarray,
    ar_low: np.ndarray,
    ar_high: np.ndarray,
) -> None:
    movable_soft = soft_mask & ~fixed_mask
    for i in range(len(widths)):
        if not movable_soft[i]:
            heights[i] = areas[i] / widths[i]
            continue
        width_min = np.sqrt(areas[i] / ar_high[i])
        width_max = np.sqrt(areas[i] / ar_low[i])
        widths[i] = np.clip(widths[i], width_min, width_max)
        heights[i] = areas[i] / widths[i]


def generate_random_instance_with_obstacles(
    num_modules: int,
    num_nets: int,
    soft_ratio: float,
    fixed_ratio: float,
    whitespace: float,
    outline_aspect: float,
    seed: int,
) -> FloorplanInstanceObstacle:
    rng = np.random.default_rng(seed)

    areas = rng.uniform(80.0, 260.0, size=num_modules)
    soft_mask = rng.random(num_modules) < soft_ratio
    fixed_mask = np.zeros(num_modules, dtype=bool)

    fixed_count = max(1, int(round(num_modules * fixed_ratio)))
    fixed_indices = rng.choice(num_modules, size=fixed_count, replace=False)
    fixed_mask[fixed_indices] = True

    ar_low = np.full(num_modules, 0.5)
    ar_high = np.full(num_modules, 2.0)
    widths = np.zeros(num_modules)
    heights = np.zeros(num_modules)

    for i in range(num_modules):
        if fixed_mask[i]:
            aspect = rng.uniform(0.8, 1.7)
            widths[i] = np.sqrt(areas[i] / aspect)
            heights[i] = areas[i] / widths[i]
            soft_mask[i] = False
            ar_value = heights[i] / widths[i]
            ar_low[i] = ar_value
            ar_high[i] = ar_value
        elif soft_mask[i]:
            width_min = np.sqrt(areas[i] / ar_high[i])
            width_max = np.sqrt(areas[i] / ar_low[i])
            widths[i] = rng.uniform(width_min, width_max)
            heights[i] = areas[i] / widths[i]
        else:
            aspect = rng.uniform(0.7, 1.8)
            widths[i] = np.sqrt(areas[i] / aspect)
            heights[i] = areas[i] / widths[i]
            ar_value = heights[i] / widths[i]
            ar_low[i] = ar_value
            ar_high[i] = ar_value

    total_area = float(np.sum(areas))
    outline_area = total_area / max(1.0 - whitespace, 0.12)
    outline_w = np.sqrt(outline_area * outline_aspect)
    outline_h = outline_area / outline_w

    x = np.zeros(num_modules)
    y = np.zeros(num_modules)

    fixed_x = np.zeros(num_modules)
    fixed_y = np.zeros(num_modules)

    fixed_order = list(np.where(fixed_mask)[0])
    movable_order = list(np.where(~fixed_mask)[0])

    fixed_cols = max(1, int(np.ceil(np.sqrt(len(fixed_order) * outline_aspect))))
    fixed_rows = max(1, int(np.ceil(len(fixed_order) / fixed_cols)))
    margin_w = outline_w / max(2 * fixed_cols + 1, 3)
    margin_h = outline_h / max(2 * fixed_rows + 1, 3)

    for n, idx in enumerate(fixed_order):
        row = n // fixed_cols
        col = n % fixed_cols
        fx = (2 * col + 1.5) * margin_w + rng.uniform(-0.12, 0.12) * margin_w
        fy = (2 * row + 1.5) * margin_h + rng.uniform(-0.12, 0.12) * margin_h
        x[idx] = np.clip(fx, widths[idx] / 2.0, outline_w - widths[idx] / 2.0)
        y[idx] = np.clip(fy, heights[idx] / 2.0, outline_h - heights[idx] / 2.0)
        fixed_x[idx] = x[idx]
        fixed_y[idx] = y[idx]

    grid_cols = int(np.ceil(np.sqrt(max(len(movable_order), 1) * outline_aspect)))
    grid_rows = int(np.ceil(max(len(movable_order), 1) / max(grid_cols, 1)))
    cell_w = outline_w / max(grid_cols, 1)
    cell_h = outline_h / max(grid_rows, 1)
    move_perm = rng.permutation(movable_order)
    for n, idx in enumerate(move_perm):
        row = n // grid_cols
        col = n % grid_cols
        x[idx] = (col + 0.5) * cell_w + rng.uniform(-0.18, 0.18) * cell_w
        y[idx] = (row + 0.5) * cell_h + rng.uniform(-0.18, 0.18) * cell_h

    clamp_positions(x, y, widths, heights, outline_w, outline_h, fixed_mask, fixed_x, fixed_y)

    nets: list[list[int]] = []
    degree = np.zeros(num_modules, dtype=int)
    for _ in range(num_nets):
        net_size = int(rng.integers(2, min(5, num_modules) + 1))
        pins = rng.choice(num_modules, size=net_size, replace=False)
        pins_list = sorted(int(v) for v in pins)
        nets.append(pins_list)
        degree[pins_list] += 1

    lonely = np.where(degree == 0)[0]
    for idx in lonely:
        mate = int(rng.integers(0, num_modules - 1))
        if mate >= idx:
            mate += 1
        nets.append(sorted([int(idx), int(mate)]))

    return FloorplanInstanceObstacle(
        outline_w=outline_w,
        outline_h=outline_h,
        areas=areas,
        soft_mask=soft_mask,
        fixed_mask=fixed_mask,
        ar_low=ar_low,
        ar_high=ar_high,
        widths=widths,
        heights=heights,
        x=x,
        y=y,
        fixed_x=fixed_x,
        fixed_y=fixed_y,
        nets=nets,
    )


def global_floorplan_with_obstacles(
    inst: FloorplanInstanceObstacle,
    grid_size: int,
    iterations: int,
    density_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 3000)
    x = inst.x.copy()
    y = inst.y.copy()
    w = inst.widths.copy()
    h = inst.heights.copy()

    alpha = 0.15 * min(inst.outline_w, inst.outline_h)
    step_base = min(inst.outline_w, inst.outline_h) / max(grid_size, 1)
    movable = ~inst.fixed_mask
    movable_soft = inst.soft_mask & ~inst.fixed_mask

    for it in range(iterations):
        density = rasterize_density(x, y, w, h, inst.outline_w, inst.outline_h, grid_size)
        potential = solve_poisson_fft(density)
        ex_grid, ey_grid = electric_field_from_potential(potential, inst.outline_w, inst.outline_h)
        _, grad_wl_x, grad_wl_y = lse_wirelength_and_grad(x, y, inst.nets, alpha=alpha)

        ex = bilinear_sample(ex_grid, x, y, inst.outline_w, inst.outline_h)
        ey = bilinear_sample(ey_grid, x, y, inst.outline_w, inst.outline_h)

        grad_wl_x[inst.fixed_mask] = 0.0
        grad_wl_y[inst.fixed_mask] = 0.0
        ex[inst.fixed_mask] = 0.0
        ey[inst.fixed_mask] = 0.0

        wl_scale = max(np.max(np.abs(grad_wl_x[movable])) if np.any(movable) else 0.0, np.max(np.abs(grad_wl_y[movable])) if np.any(movable) else 0.0, 1e-9)
        field_scale = max(np.max(np.abs(ex[movable])) if np.any(movable) else 0.0, np.max(np.abs(ey[movable])) if np.any(movable) else 0.0, 1e-9)

        wl_step = step_base * (0.38 - 0.16 * (it / max(iterations - 1, 1)))
        density_step = step_base * density_weight * (0.8 + 0.2 * (it / max(iterations - 1, 1)))

        x[movable] -= wl_step * grad_wl_x[movable] / wl_scale
        y[movable] -= wl_step * grad_wl_y[movable] / wl_scale
        x[movable] += density_step * ex[movable] / field_scale
        y[movable] += density_step * ey[movable] / field_scale

        if np.any(movable_soft):
            width_grad = np.zeros_like(w)
            for i in np.where(movable_soft)[0]:
                width_min = np.sqrt(inst.areas[i] / inst.ar_high[i])
                width_max = np.sqrt(inst.areas[i] / inst.ar_low[i])
                eps = max(0.03 * w[i], 0.15)
                w_plus = np.clip(w[i] + eps, width_min, width_max)
                w_minus = np.clip(w[i] - eps, width_min, width_max)
                if abs(w_plus - w_minus) < 1e-9:
                    continue
                e_plus = module_potential_energy(x[i], y[i], w_plus, inst.areas[i], potential, inst.outline_w, inst.outline_h)
                e_minus = module_potential_energy(x[i], y[i], w_minus, inst.areas[i], potential, inst.outline_w, inst.outline_h)
                width_grad[i] = (e_plus - e_minus) / (w_plus - w_minus)

            width_scale = max(np.max(np.abs(width_grad[movable_soft])), 1e-9)
            width_step = 0.16 * step_base
            w[movable_soft] -= width_step * width_grad[movable_soft] / width_scale

        project_soft_widths(w, h, inst.areas, inst.soft_mask, inst.fixed_mask, inst.ar_low, inst.ar_high)
        clamp_positions(x, y, w, h, inst.outline_w, inst.outline_h, inst.fixed_mask, inst.fixed_x, inst.fixed_y)

        if np.any(movable):
            jitter = 0.025 * step_base * (1.0 - it / max(iterations, 1))
            x[movable] += rng.normal(0.0, jitter, size=np.count_nonzero(movable))
            y[movable] += rng.normal(0.0, jitter, size=np.count_nonzero(movable))

        clamp_positions(x, y, w, h, inst.outline_w, inst.outline_h, inst.fixed_mask, inst.fixed_x, inst.fixed_y)

    final_density = rasterize_density(x, y, w, h, inst.outline_w, inst.outline_h, grid_size)
    final_potential = solve_poisson_fft(final_density)
    return x, y, w, h, final_potential


def add_edge(graph: list[set[int]], src: int, dst: int) -> None:
    if src == dst:
        return
    graph[src].add(dst)


def build_obstacle_aware_constraint_graphs(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    fixed_mask: np.ndarray,
) -> tuple[list[set[int]], list[set[int]]]:
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
                    if x[mov] >= x[fixed]:
                        add_edge(hcg, fixed, mov)
                    else:
                        add_edge(hcg, mov, fixed)
                else:
                    if y[mov] >= y[fixed]:
                        add_edge(vcg, fixed, mov)
                    else:
                        add_edge(vcg, mov, fixed)
                continue

            if overlap_x >= overlap_y:
                add_edge(hcg, i if center_dx >= 0 else j, j if center_dx >= 0 else i)
            else:
                add_edge(vcg, i if center_dy >= 0 else j, j if center_dy >= 0 else i)

    return hcg, vcg


def topological_order(graph: list[set[int]]) -> list[int]:
    indeg = [0] * len(graph)
    for src, nbrs in enumerate(graph):
        for dst in nbrs:
            indeg[dst] += 1

    queue = [i for i, deg in enumerate(indeg) if deg == 0]
    order: list[int] = []
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
        order = list(range(len(graph)))
    return order


def solve_positions_from_constraint_graph(
    graph: list[set[int]],
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

    for _ in range(2):
        for node in order:
            if fixed_mask[node]:
                pos[node] = fixed_values[node]
                continue
            need = lower_bounds[node]
            for pred in range(n):
                if node in graph[pred]:
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
                pos[node] = min(pos[node], allowed)
                pos[node] = max(pos[node], lower_bounds[node])

    pos[fixed_mask] = fixed_values[fixed_mask]
    return pos


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
    passes: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
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

                if fixed_mask[i] and fixed_mask[j]:
                    continue

                sep_h = overlap_x + 1e-3
                sep_v = overlap_y + 1e-3

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

        clamp_positions(x, y, w, h, outline_w, outline_h, fixed_mask, fixed_x, fixed_y)
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
) -> tuple[np.ndarray, np.ndarray, list[set[int]], list[set[int]]]:
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
    hcg: list[set[int]],
    vcg: list[set[int]],
) -> tuple[np.ndarray, np.ndarray]:
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

    def fallback_candidates(limit: int, max_value: float) -> list[float]:
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

        x_candidates = {0.0, anchor_left}
        y_candidates = {0.0, anchor_bottom}
        for j in np.where(placed)[0]:
            x_candidates.add(lefts[j] + widths[j])
            x_candidates.add(max(0.0, lefts[j] - widths[idx]))
            y_candidates.add(bottoms[j] + heights[j])
            y_candidates.add(max(0.0, bottoms[j] - heights[idx]))

        x_candidates.add(min_left)
        x_candidates.add(max(0.0, max_left))
        y_candidates.add(min_bottom)
        y_candidates.add(max(0.0, max_bottom))

        best = None
        best_score = None

        def count_graph_violations(idx_local: int, left: float, bottom: float) -> int:
            right = left + widths[idx_local]
            top = bottom + heights[idx_local]
            violations = 0
            for j in np.where(placed)[0]:
                other_left = lefts[j]
                other_bottom = bottoms[j]
                other_right = other_left + widths[j]
                other_top = other_bottom + heights[j]

                if idx_local in hcg[j] and left < other_right - 1e-9:
                    violations += 1
                if j in hcg[idx_local] and right > other_left + 1e-9:
                    violations += 1
                if idx_local in vcg[j] and bottom < other_top - 1e-9:
                    violations += 1
                if j in vcg[idx_local] and top > other_bottom + 1e-9:
                    violations += 1
            return violations

        def search_candidates(xs: list[float], ys: list[float], strict_bounds: bool) -> tuple[float, float] | None:
            nonlocal best, best_score
            for bottom in ys:
                if strict_bounds and (bottom < min_bottom - 1e-9 or bottom > max_bottom + 1e-9):
                    continue
                if bottom + heights[idx] > outline_h + 1e-9:
                    continue
                for left in xs:
                    if strict_bounds and (left < min_left - 1e-9 or left > max_left + 1e-9):
                        continue
                    if left + widths[idx] > outline_w + 1e-9:
                        continue
                    if overlaps_placed(idx, left, bottom):
                        continue

                    center_x = left + widths[idx] / 2.0
                    center_y = bottom + heights[idx] / 2.0
                    violations = count_graph_violations(idx, left, bottom)
                    score = (
                        violations,
                        (center_x - anchor_x[idx]) ** 2 + (center_y - anchor_y[idx]) ** 2,
                        abs(bottom - anchor_bottom),
                        abs(left - anchor_left),
                        bottom,
                        left,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (left, bottom)
            return best

        search_candidates(sorted(x_candidates), sorted(y_candidates), strict_bounds=True)

        if best is None:
            dense_x = fallback_candidates(30, outline_w - widths[idx])
            dense_y = fallback_candidates(30, outline_h - heights[idx])
            search_candidates(dense_x, dense_y, strict_bounds=True)

        if best is None:
            search_candidates(sorted(x_candidates), sorted(y_candidates), strict_bounds=False)

        if best is None:
            dense_x = fallback_candidates(36, outline_w - widths[idx])
            dense_y = fallback_candidates(36, outline_h - heights[idx])
            search_candidates(dense_x, dense_y, strict_bounds=False)

        if best is None:
            raise RuntimeError("Obstacle-aware packing failed. Increase whitespace or reduce fixed ratio.")

        lefts[idx], bottoms[idx] = best
        placed[idx] = True

    legal_x = lefts + widths / 2.0
    legal_y = bottoms + heights / 2.0
    return legal_x, legal_y


def draw_fixed_marker(ax: plt.Axes, left: float, bottom: float, width: float, height: float) -> None:
    ax.plot([left, left + width], [bottom, bottom + height], color="#3b0a45", linewidth=1.0, alpha=0.8)
    ax.plot([left + width, left], [bottom, bottom + height], color="#3b0a45", linewidth=1.0, alpha=0.8)


def draw_floorplan(
    ax: plt.Axes,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    soft_mask: np.ndarray,
    fixed_mask: np.ndarray,
    outline_w: float,
    outline_h: float,
    show_ids: bool = True,
) -> None:
    ax.set_title(title)
    ax.set_xlim(0.0, outline_w)
    ax.set_ylim(0.0, outline_h)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f6f8")

    outline = Rectangle((0.0, 0.0), outline_w, outline_h, fill=False, edgecolor="black", linewidth=1.8)
    ax.add_patch(outline)

    for i in range(len(x)):
        left, bottom, _, _ = rect_bounds(x[i], y[i], w[i], h[i])
        if fixed_mask[i]:
            face = "#c51b7d"
            edge = "#4d004b"
            hatch = "xx"
            alpha = 0.55
        elif soft_mask[i]:
            face = "#6aaed6"
            edge = "#1f1f1f"
            hatch = None
            alpha = 0.78
        else:
            face = "#f4a259"
            edge = "#1f1f1f"
            hatch = None
            alpha = 0.78

        patch = Rectangle((left, bottom), w[i], h[i], facecolor=face, edgecolor=edge, linewidth=0.9, alpha=alpha, hatch=hatch)
        ax.add_patch(patch)
        if fixed_mask[i]:
            draw_fixed_marker(ax, left, bottom, w[i], h[i])
        if show_ids:
            label = f"F{i}" if fixed_mask[i] else str(i)
            ax.text(x[i], y[i], label, ha="center", va="center", fontsize=7, color="#111111", weight="bold" if fixed_mask[i] else "normal")

    ax.set_xticks([])
    ax.set_yticks([])


def draw_density(ax: plt.Axes, density: np.ndarray, title: str, outline_w: float, outline_h: float) -> None:
    ax.set_title(title)
    ax.imshow(
        density,
        origin="lower",
        extent=(0.0, outline_w, 0.0, outline_h),
        cmap="magma",
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def count_edges(graph: Iterable[set[int]]) -> int:
    return int(sum(len(v) for v in graph))


def save_visualization(
    out_path: Path,
    inst: FloorplanInstanceObstacle,
    init_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    global_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    legal_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    density: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    draw_floorplan(axes[0, 0], "Initial Placement With Fixed Obstacles", init_state[0], init_state[1], init_state[2], init_state[3], inst.soft_mask, inst.fixed_mask, inst.outline_w, inst.outline_h)
    draw_floorplan(axes[0, 1], "After Global Floorplanning", global_state[0], global_state[1], global_state[2], global_state[3], inst.soft_mask, inst.fixed_mask, inst.outline_w, inst.outline_h)
    draw_floorplan(axes[1, 0], "After Obstacle-Aware Legalization", legal_state[0], legal_state[1], legal_state[2], legal_state[3], inst.soft_mask, inst.fixed_mask, inst.outline_w, inst.outline_h)
    draw_density(axes[1, 1], density, "Density Map After Global Floorplanning", inst.outline_w, inst.outline_h)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#6aaed6", edgecolor="#1f1f1f", alpha=0.78, label="Movable Soft"),
        Rectangle((0, 0), 1, 1, facecolor="#f4a259", edgecolor="#1f1f1f", alpha=0.78, label="Movable Hard"),
        Rectangle((0, 0), 1, 1, facecolor="#c51b7d", edgecolor="#4d004b", hatch="xx", alpha=0.55, label="Fixed Obstacle"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("PeF Prototype With Fixed Obstacles", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inst = generate_random_instance_with_obstacles(
        num_modules=args.modules,
        num_nets=args.nets,
        soft_ratio=args.soft_ratio,
        fixed_ratio=args.fixed_ratio,
        whitespace=args.whitespace,
        outline_aspect=args.aspect,
        seed=args.seed,
    )

    init_x = inst.x.copy()
    init_y = inst.y.copy()
    init_w = inst.widths.copy()
    init_h = inst.heights.copy()

    global_x, global_y, global_w, global_h, _ = global_floorplan_with_obstacles(
        inst,
        grid_size=args.grid,
        iterations=args.iters,
        density_weight=args.density_weight,
        seed=args.seed,
    )

    density = rasterize_density(global_x, global_y, global_w, global_h, inst.outline_w, inst.outline_h, args.grid)
    legal_x, legal_y, hcg, vcg = legalize_with_obstacle_aware_graphs(
        global_x,
        global_y,
        global_w,
        global_h,
        inst.outline_w,
        inst.outline_h,
        inst.fixed_mask,
        inst.fixed_x,
        inst.fixed_y,
    )

    image_path = outdir / "pef_obstacle_floorplan.png"
    save_visualization(
        image_path,
        inst,
        (init_x, init_y, init_w, init_h),
        (global_x, global_y, global_w, global_h),
        (legal_x, legal_y, global_w, global_h),
        density,
    )

    final_only = outdir / "pef_obstacle_floorplan_final.png"
    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    draw_floorplan(ax, "Final Legal Floorplan With Fixed Obstacles", legal_x, legal_y, global_w, global_h, inst.soft_mask, inst.fixed_mask, inst.outline_w, inst.outline_h, show_ids=args.show_ids)
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor="#6aaed6", edgecolor="#1f1f1f", alpha=0.78, label="Movable Soft"),
        Rectangle((0, 0), 1, 1, facecolor="#f4a259", edgecolor="#1f1f1f", alpha=0.78, label="Movable Hard"),
        Rectangle((0, 0), 1, 1, facecolor="#c51b7d", edgecolor="#4d004b", hatch="xx", alpha=0.55, label="Fixed Obstacle"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.savefig(final_only, dpi=200)
    plt.close(fig)

    metrics = {
        "seed": int(args.seed),
        "modules": int(args.modules),
        "nets": int(len(inst.nets)),
        "fixed_modules": int(np.count_nonzero(inst.fixed_mask)),
        "movable_modules": int(np.count_nonzero(~inst.fixed_mask)),
        "outline_w": float(inst.outline_w),
        "outline_h": float(inst.outline_h),
        "hpwl_initial": hpwl(init_x, init_y, inst.nets),
        "hpwl_global": hpwl(global_x, global_y, inst.nets),
        "hpwl_final": hpwl(legal_x, legal_y, inst.nets),
        "overlap_initial": total_overlap_area(init_x, init_y, init_w, init_h),
        "overlap_global": total_overlap_area(global_x, global_y, global_w, global_h),
        "overlap_final": total_overlap_area(legal_x, legal_y, global_w, global_h),
        "hcg_edges": count_edges(hcg),
        "vcg_edges": count_edges(vcg),
        "visualization": str(image_path),
        "final_image": str(final_only),
    }

    metrics_path = outdir / "pef_obstacle_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simplified PeF prototype with fixed obstacles and obstacle-aware HCG/VCG legalization.")
    parser.add_argument("--modules", type=int, default=34, help="Number of total modules.")
    parser.add_argument("--nets", type=int, default=78, help="Number of random nets.")
    parser.add_argument("--soft-ratio", type=float, default=0.60, help="Fraction of soft modules among non-fixed modules.")
    parser.add_argument("--fixed-ratio", type=float, default=0.18, help="Fraction of fixed modules.")
    parser.add_argument("--whitespace", type=float, default=0.34, help="Whitespace fraction.")
    parser.add_argument("--aspect", type=float, default=1.0, help="Outline aspect ratio W / H.")
    parser.add_argument("--grid", type=int, default=48, help="Grid size for Poisson solve.")
    parser.add_argument("--iters", type=int, default=180, help="Global floorplanning iterations.")
    parser.add_argument("--density-weight", type=float, default=1.20, help="Density spreading strength.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument("--outdir", type=str, default="output_obstacle", help="Output directory.")
    parser.add_argument("--show-ids", action="store_true", help="Show module ids on final image.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2))
