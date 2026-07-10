from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


@dataclass
class FloorplanInstance:
    outline_w: float
    outline_h: float
    areas: np.ndarray
    soft_mask: np.ndarray
    ar_low: np.ndarray
    ar_high: np.ndarray
    widths: np.ndarray
    heights: np.ndarray
    x: np.ndarray
    y: np.ndarray
    nets: list[list[int]]


def generate_random_instance(
    num_modules: int,
    num_nets: int,
    soft_ratio: float,
    whitespace: float,
    outline_aspect: float,
    seed: int,
) -> FloorplanInstance:
    rng = np.random.default_rng(seed)

    areas = rng.uniform(80.0, 260.0, size=num_modules)
    soft_mask = rng.random(num_modules) < soft_ratio
    ar_low = np.full(num_modules, 0.5)
    ar_high = np.full(num_modules, 2.0)

    widths = np.zeros(num_modules)
    heights = np.zeros(num_modules)

    for i in range(num_modules):
        if soft_mask[i]:
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
    outline_area = total_area / max(1.0 - whitespace, 0.1)
    outline_w = np.sqrt(outline_area * outline_aspect)
    outline_h = outline_area / outline_w

    grid_cols = int(np.ceil(np.sqrt(num_modules * outline_aspect)))
    grid_rows = int(np.ceil(num_modules / max(grid_cols, 1)))
    cell_w = outline_w / max(grid_cols, 1)
    cell_h = outline_h / max(grid_rows, 1)

    x = np.zeros(num_modules)
    y = np.zeros(num_modules)
    order = rng.permutation(num_modules)
    for n, idx in enumerate(order):
        row = n // grid_cols
        col = n % grid_cols
        x[idx] = (col + 0.5) * cell_w + rng.uniform(-0.15, 0.15) * cell_w
        y[idx] = (row + 0.5) * cell_h + rng.uniform(-0.15, 0.15) * cell_h

    x = np.clip(x, widths / 2.0, outline_w - widths / 2.0)
    y = np.clip(y, heights / 2.0, outline_h - heights / 2.0)

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

    return FloorplanInstance(
        outline_w=outline_w,
        outline_h=outline_h,
        areas=areas,
        soft_mask=soft_mask,
        ar_low=ar_low,
        ar_high=ar_high,
        widths=widths,
        heights=heights,
        x=x,
        y=y,
        nets=nets,
    )


def clamp_positions(x: np.ndarray, y: np.ndarray, w: np.ndarray, h: np.ndarray, outline_w: float, outline_h: float) -> None:
    x[:] = np.clip(x, w / 2.0, outline_w - w / 2.0)
    y[:] = np.clip(y, h / 2.0, outline_h - h / 2.0)


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

        lse_x_pos = xs.max() + alpha * np.log(sum_px)
        lse_x_neg = (-xs).max() + alpha * np.log(sum_nx)
        lse_y_pos = ys.max() + alpha * np.log(sum_py)
        lse_y_neg = (-ys).max() + alpha * np.log(sum_ny)
        total += lse_x_pos + lse_x_neg + lse_y_pos + lse_y_neg

        grad_x_net = exp_px / sum_px - exp_nx / sum_nx
        grad_y_net = exp_py / sum_py - exp_ny / sum_ny

        grad_x[net] += grad_x_net
        grad_y[net] += grad_y_net

    return float(total), grad_x, grad_y


def rect_bounds(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


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
    ar_low: np.ndarray,
    ar_high: np.ndarray,
) -> None:
    for i in range(len(widths)):
        if not soft_mask[i]:
            heights[i] = areas[i] / widths[i]
            continue
        width_min = np.sqrt(areas[i] / ar_high[i])
        width_max = np.sqrt(areas[i] / ar_low[i])
        widths[i] = np.clip(widths[i], width_min, width_max)
        heights[i] = areas[i] / widths[i]


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


def global_floorplan(
    inst: FloorplanInstance,
    grid_size: int,
    iterations: int,
    density_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1000)
    x = inst.x.copy()
    y = inst.y.copy()
    w = inst.widths.copy()
    h = inst.heights.copy()

    alpha = 0.15 * min(inst.outline_w, inst.outline_h)
    step_base = min(inst.outline_w, inst.outline_h) / max(grid_size, 1)

    for it in range(iterations):
        density = rasterize_density(x, y, w, h, inst.outline_w, inst.outline_h, grid_size)
        potential = solve_poisson_fft(density)
        ex_grid, ey_grid = electric_field_from_potential(potential, inst.outline_w, inst.outline_h)
        _, grad_wl_x, grad_wl_y = lse_wirelength_and_grad(x, y, inst.nets, alpha=alpha)

        ex = bilinear_sample(ex_grid, x, y, inst.outline_w, inst.outline_h)
        ey = bilinear_sample(ey_grid, x, y, inst.outline_w, inst.outline_h)

        wl_scale = max(np.max(np.abs(grad_wl_x)), np.max(np.abs(grad_wl_y)), 1e-9)
        field_scale = max(np.max(np.abs(ex)), np.max(np.abs(ey)), 1e-9)

        wl_step = step_base * (0.40 - 0.18 * (it / max(iterations - 1, 1)))
        density_step = step_base * density_weight * (0.85 + 0.15 * (it / max(iterations - 1, 1)))

        x -= wl_step * grad_wl_x / wl_scale
        y -= wl_step * grad_wl_y / wl_scale
        x += density_step * ex / field_scale
        y += density_step * ey / field_scale

        if np.any(inst.soft_mask):
            width_grad = np.zeros_like(w)
            for i in np.where(inst.soft_mask)[0]:
                width_min = np.sqrt(inst.areas[i] / inst.ar_high[i])
                width_max = np.sqrt(inst.areas[i] / inst.ar_low[i])
                eps = max(0.03 * w[i], 0.15)

                w_plus = np.clip(w[i] + eps, width_min, width_max)
                w_minus = np.clip(w[i] - eps, width_min, width_max)
                if abs(w_plus - w_minus) < 1e-9:
                    continue

                e_plus = module_potential_energy(
                    x[i],
                    y[i],
                    w_plus,
                    inst.areas[i],
                    potential,
                    inst.outline_w,
                    inst.outline_h,
                )
                e_minus = module_potential_energy(
                    x[i],
                    y[i],
                    w_minus,
                    inst.areas[i],
                    potential,
                    inst.outline_w,
                    inst.outline_h,
                )
                width_grad[i] = (e_plus - e_minus) / (w_plus - w_minus)

            width_scale = max(np.max(np.abs(width_grad[inst.soft_mask])), 1e-9)
            width_step = 0.18 * step_base
            w[inst.soft_mask] -= width_step * width_grad[inst.soft_mask] / width_scale

        project_soft_widths(w, h, inst.areas, inst.soft_mask, inst.ar_low, inst.ar_high)
        clamp_positions(x, y, w, h, inst.outline_w, inst.outline_h)

        jitter = 0.03 * step_base * (1.0 - it / max(iterations, 1))
        x += rng.normal(0.0, jitter, size=len(x))
        y += rng.normal(0.0, jitter, size=len(y))
        clamp_positions(x, y, w, h, inst.outline_w, inst.outline_h)

    final_density = rasterize_density(x, y, w, h, inst.outline_w, inst.outline_h, grid_size)
    final_potential = solve_poisson_fft(final_density)
    return x, y, w, h, final_potential


def bottom_left_legalize(
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    outline_w: float,
    outline_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(anchor_x)
    lefts = np.zeros(n)
    bottoms = np.zeros(n)
    placed: list[int] = []

    order = list(np.lexsort((anchor_x, anchor_y)))

    def overlaps(idx: int, left: float, bottom: float) -> bool:
        right = left + widths[idx]
        top = bottom + heights[idx]
        for j in placed:
            other_left = lefts[j]
            other_bottom = bottoms[j]
            other_right = other_left + widths[j]
            other_top = other_bottom + heights[j]
            if right <= other_left or other_right <= left:
                continue
            if top <= other_bottom or other_top <= bottom:
                continue
            return True
        return False

    for idx in order:
        x_candidates = {0.0}
        y_candidates = {0.0}
        for j in placed:
            x_candidates.add(lefts[j] + widths[j])
            y_candidates.add(bottoms[j] + heights[j])

        best = None
        best_key = None
        for bottom in sorted(y_candidates):
            if bottom + heights[idx] > outline_h + 1e-9:
                continue
            for left in sorted(x_candidates):
                if left + widths[idx] > outline_w + 1e-9:
                    continue
                if overlaps(idx, left, bottom):
                    continue

                center_x = left + widths[idx] / 2.0
                center_y = bottom + heights[idx] / 2.0
                distance_cost = (center_x - anchor_x[idx]) ** 2 + (center_y - anchor_y[idx]) ** 2
                key = (round(bottom, 6), round(left, 6), round(distance_cost, 6))
                if best_key is None or key < best_key:
                    best_key = key
                    best = (left, bottom)

        if best is None:
            raise RuntimeError("Bottom-left legalization failed. Increase whitespace or reduce module count.")

        lefts[idx], bottoms[idx] = best
        placed.append(idx)

    legal_x = lefts + widths / 2.0
    legal_y = bottoms + heights / 2.0
    return legal_x, legal_y


def draw_floorplan(
    ax: plt.Axes,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    soft_mask: np.ndarray,
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
        face = "#6aaed6" if soft_mask[i] else "#f4a259"
        patch = Rectangle((left, bottom), w[i], h[i], facecolor=face, edgecolor="#1f1f1f", linewidth=0.8, alpha=0.75)
        ax.add_patch(patch)
        if show_ids:
            ax.text(x[i], y[i], str(i), ha="center", va="center", fontsize=7, color="#111111")

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


def save_visualization(
    out_path: Path,
    inst: FloorplanInstance,
    init_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    global_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    legal_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    density: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    draw_floorplan(
        axes[0, 0],
        "Initial Placement",
        init_state[0],
        init_state[1],
        init_state[2],
        init_state[3],
        inst.soft_mask,
        inst.outline_w,
        inst.outline_h,
    )
    draw_floorplan(
        axes[0, 1],
        "After Global Floorplanning",
        global_state[0],
        global_state[1],
        global_state[2],
        global_state[3],
        inst.soft_mask,
        inst.outline_w,
        inst.outline_h,
    )
    draw_floorplan(
        axes[1, 0],
        "After Legalization",
        legal_state[0],
        legal_state[1],
        legal_state[2],
        legal_state[3],
        inst.soft_mask,
        inst.outline_w,
        inst.outline_h,
    )
    draw_density(axes[1, 1], density, "Density Map After Global Floorplanning", inst.outline_w, inst.outline_h)

    fig.suptitle("Simplified PeF Reproduction", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inst = generate_random_instance(
        num_modules=args.modules,
        num_nets=args.nets,
        soft_ratio=args.soft_ratio,
        whitespace=args.whitespace,
        outline_aspect=args.aspect,
        seed=args.seed,
    )

    init_x = inst.x.copy()
    init_y = inst.y.copy()
    init_w = inst.widths.copy()
    init_h = inst.heights.copy()

    global_x, global_y, global_w, global_h, _ = global_floorplan(
        inst,
        grid_size=args.grid,
        iterations=args.iters,
        density_weight=args.density_weight,
        seed=args.seed,
    )

    density = rasterize_density(global_x, global_y, global_w, global_h, inst.outline_w, inst.outline_h, args.grid)
    legal_x, legal_y = bottom_left_legalize(global_x, global_y, global_w, global_h, inst.outline_w, inst.outline_h)

    image_path = outdir / "pef_random_floorplan.png"
    save_visualization(
        image_path,
        inst,
        (init_x, init_y, init_w, init_h),
        (global_x, global_y, global_w, global_h),
        (legal_x, legal_y, global_w, global_h),
        density,
    )

    final_only = outdir / "pef_random_floorplan_final.png"
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    draw_floorplan(
        ax,
        "Final Legal Floorplan",
        legal_x,
        legal_y,
        global_w,
        global_h,
        inst.soft_mask,
        inst.outline_w,
        inst.outline_h,
        show_ids=args.show_ids,
    )
    fig.savefig(final_only, dpi=200)
    plt.close(fig)

    metrics = {
        "seed": int(args.seed),
        "modules": int(args.modules),
        "nets": int(len(inst.nets)),
        "outline_w": float(inst.outline_w),
        "outline_h": float(inst.outline_h),
        "hpwl_initial": hpwl(init_x, init_y, inst.nets),
        "hpwl_global": hpwl(global_x, global_y, inst.nets),
        "hpwl_final": hpwl(legal_x, legal_y, inst.nets),
        "overlap_initial": total_overlap_area(init_x, init_y, init_w, init_h),
        "overlap_global": total_overlap_area(global_x, global_y, global_w, global_h),
        "overlap_final": total_overlap_area(legal_x, legal_y, global_w, global_h),
        "visualization": str(image_path),
        "final_image": str(final_only),
    }

    metrics_path = outdir / "pef_random_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simplified Python reproduction of the PeF floorplanning paper.")
    parser.add_argument("--modules", type=int, default=36, help="Number of modules to generate.")
    parser.add_argument("--nets", type=int, default=72, help="Number of random nets to generate.")
    parser.add_argument("--soft-ratio", type=float, default=0.65, help="Fraction of soft modules.")
    parser.add_argument("--whitespace", type=float, default=0.30, help="Whitespace fraction for the outline.")
    parser.add_argument("--aspect", type=float, default=1.0, help="Outline aspect ratio W / H.")
    parser.add_argument("--grid", type=int, default=48, help="Poisson solver grid size.")
    parser.add_argument("--iters", type=int, default=180, help="Global floorplanning iterations.")
    parser.add_argument("--density-weight", type=float, default=1.25, help="Strength of density spreading.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--outdir", type=str, default="output", help="Directory for generated artifacts.")
    parser.add_argument("--show-ids", action="store_true", help="Show module ids on the final-only plot.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2))
