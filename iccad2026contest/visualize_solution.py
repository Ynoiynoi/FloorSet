#!/usr/bin/env python3
"""
Visualize one ICCAD 2026 FloorSet layout solution for a given validation case.

Supported sources:
  1. Evaluation result JSON (`*_results.json`)
  2. Saved solutions JSON (`*_solutions.json`)
  3. An optimizer python file (`my_optimizer.py`)

Examples:
  python visualize_solution.py --results-json my_optimizer_results.json --test-id 0
  python visualize_solution.py --solutions-json my_optimizer_solutions.json --test-id 0
  python visualize_solution.py --optimizer my_optimizer.py --test-id 0 --draw-b2b --draw-p2b
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from iccad2026_evaluate import ContestEvaluator, FloorplanDatasetLiteTest, evaluate_solution


Position = Tuple[float, float, float, float]


def extract_positions_from_polygons(polygons: Sequence[torch.Tensor], block_count: int) -> List[Position]:
    positions: List[Position] = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            positions.append(
                (
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                )
            )
        else:
            positions.append((0.0, 0.0, 1.0, 1.0))
    return positions


def build_optimizer_target_positions(
    constraints: torch.Tensor,
    target_positions: Sequence[Position],
    block_count: int,
) -> torch.Tensor:
    opt_target_pos = torch.full((block_count, 4), -1.0)
    if constraints is None:
        return opt_target_pos

    ncols = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = ncols > 0 and constraints[i, 0] != 0
        is_preplaced = ncols > 1 and constraints[i, 1] != 0
        if is_preplaced:
            tx, ty, tw, th = target_positions[i]
            opt_target_pos[i] = torch.tensor([tx, ty, tw, th], dtype=torch.float32)
        elif is_fixed:
            _, _, tw, th = target_positions[i]
            opt_target_pos[i, 2] = tw
            opt_target_pos[i, 3] = th
    return opt_target_pos


def load_test_case(test_id: int, data_path: str) -> Dict:
    dataset = FloorplanDatasetLiteTest(data_path)
    sample = dataset[test_id]
    inputs, labels = sample["input"], sample["label"]
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    polygons, raw_metrics = labels
    block_count = int((area_target != -1).sum().item())

    evaluator = ContestEvaluator(data_path=data_path, verbose=False)
    baseline, target_positions = evaluator._extract_baseline(
        test_id, labels, b2b_conn, p2b_conn, pins_pos, block_count
    )

    return {
        "area_target": area_target,
        "b2b_conn": b2b_conn,
        "p2b_conn": p2b_conn,
        "pins_pos": pins_pos,
        "constraints": constraints,
        "polygons": polygons,
        "raw_metrics": raw_metrics,
        "block_count": block_count,
        "baseline": baseline,
        "target_positions": target_positions,
        "opt_target_positions": build_optimizer_target_positions(
            constraints, target_positions, block_count
        ),
    }


def normalize_positions(positions: Sequence[Sequence[float]], block_count: int) -> List[Position]:
    normalized = [tuple(map(float, p)) for p in positions]
    if len(normalized) != block_count:
        raise ValueError(f"Expected {block_count} blocks, got {len(normalized)}")
    return normalized


def load_positions_from_json(json_path: str, test_id: int, block_count: int) -> Tuple[List[Position], Dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    def match_record(records: Sequence[Dict]) -> Dict:
        for record in records:
            if int(record.get("test_id", -1)) == test_id:
                return record
        raise KeyError(f"test_id={test_id} not found in {json_path}")

    record: Optional[Dict] = None
    if isinstance(data, dict) and "test_results" in data:
        record = match_record(data["test_results"])
    elif isinstance(data, dict) and "solutions" in data:
        record = match_record(data["solutions"])
    elif isinstance(data, list):
        record = match_record(data)
    elif isinstance(data, dict) and "positions" in data:
        record = data
    else:
        raise ValueError(f"Unsupported JSON format: {json_path}")

    if "positions" not in record or record["positions"] is None:
        raise ValueError(f"Record for test_id={test_id} has no positions")

    return normalize_positions(record["positions"], block_count), record


def solve_with_optimizer(optimizer_path: str, case: Dict, data_path: str) -> Tuple[List[Position], Dict]:
    evaluator = ContestEvaluator(data_path=data_path, verbose=False)
    optimizer = evaluator._load_optimizer(optimizer_path)

    start = time.time()
    positions = optimizer.solve(
        case["block_count"],
        case["area_target"],
        case["b2b_conn"],
        case["p2b_conn"],
        case["pins_pos"],
        case["constraints"],
        case["opt_target_positions"],
    )
    runtime = time.time() - start

    return normalize_positions(positions, case["block_count"]), {
        "source": str(Path(optimizer_path).resolve()),
        "runtime_seconds": runtime,
    }


def evaluate_positions(case: Dict, positions: List[Position], runtime_seconds: Optional[float]) -> Dict:
    runtime_for_eval = runtime_seconds if runtime_seconds and runtime_seconds > 0 else 1.0
    metrics = evaluate_solution(
        {"positions": positions, "runtime": runtime_for_eval},
        case["baseline"],
        case["constraints"],
        case["b2b_conn"],
        case["p2b_conn"],
        case["pins_pos"],
        case["area_target"],
        case["target_positions"],
        median_runtime=runtime_for_eval,
    )
    return {
        "is_feasible": metrics.is_feasible,
        "cost": metrics.cost,
        "hpwl_gap": metrics.hpwl_gap,
        "area_gap": metrics.area_gap,
        "violations_relative": metrics.violations_relative,
        "overlap_violations": metrics.overlap_violations,
        "area_violations": metrics.area_violations,
        "dimension_violations": metrics.dimension_violations,
        "boundary_violations": metrics.boundary_violations,
        "grouping_violations": metrics.grouping_violations,
        "mib_violations": metrics.mib_violations,
        "hpwl_total": metrics.hpwl_total,
        "bbox_area": metrics.bbox_area,
    }


def _constraint_style(constraint_row: torch.Tensor) -> Tuple[str, float, Optional[str]]:
    is_fixed = bool(constraint_row[0].item()) if len(constraint_row) > 0 else False
    is_preplaced = bool(constraint_row[1].item()) if len(constraint_row) > 1 else False
    has_mib = bool(constraint_row[2].item()) if len(constraint_row) > 2 else False
    has_cluster = bool(constraint_row[3].item()) if len(constraint_row) > 3 else False
    has_boundary = bool(constraint_row[4].item()) if len(constraint_row) > 4 else False

    if is_preplaced:
        return "black", 2.5, "xx"
    if is_fixed:
        return "darkviolet", 2.2, "//"
    if has_boundary:
        return "olive", 2.0, None
    if has_cluster:
        return "firebrick", 1.8, None
    if has_mib:
        return "darkgreen", 1.8, None
    return "black", 1.0, None


def _block_fill_colors(constraints: torch.Tensor, block_count: int):
    import matplotlib.pyplot as plt

    palette = list(plt.cm.tab20.colors)
    colors = [palette[i % len(palette)] for i in range(block_count)]
    group_to_color: Dict[int, Tuple[float, float, float]] = {}
    next_group_color = 0

    ncols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0
    for i in range(block_count):
        group_id = int(constraints[i, 3].item()) if ncols > 3 else 0
        if group_id > 0:
            if group_id not in group_to_color:
                group_to_color[group_id] = palette[next_group_color % len(palette)]
                next_group_color += 1
            colors[i] = group_to_color[group_id]

    return colors


def _draw_nets(ax, positions: List[Position], b2b_conn: torch.Tensor, p2b_conn: torch.Tensor, pins_pos: torch.Tensor,
               draw_b2b: bool, draw_p2b: bool) -> None:
    centers = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in positions]

    if draw_b2b:
        for edge in b2b_conn:
            if int(edge[0]) == -1:
                continue
            i, j = int(edge[0]), int(edge[1])
            if i < len(centers) and j < len(centers):
                (x1, y1), (x2, y2) = centers[i], centers[j]
                ax.plot((x1, x2), (y1, y2), color="tab:red", linewidth=0.4, alpha=0.25, zorder=1)

    if draw_p2b:
        for edge in p2b_conn:
            if int(edge[0]) == -1:
                continue
            pin_idx, block_idx = int(edge[0]), int(edge[1])
            if pin_idx < len(pins_pos) and block_idx < len(centers):
                px, py = float(pins_pos[pin_idx][0]), float(pins_pos[pin_idx][1])
                bx, by = centers[block_idx]
                ax.plot((px, bx), (py, by), color="tab:blue", linewidth=0.35, alpha=0.2, zorder=1)


def draw_layout(
    ax,
    positions: List[Position],
    constraints: torch.Tensor,
    pins_pos: torch.Tensor,
    title: str,
    metrics: Optional[Dict],
    draw_b2b: bool,
    draw_p2b: bool,
    b2b_conn: torch.Tensor,
    p2b_conn: torch.Tensor,
) -> None:
    import matplotlib.patches as mpatches

    block_count = len(positions)
    colors = _block_fill_colors(constraints, block_count)

    xmin = min(x for x, _, _, _ in positions)
    ymin = min(y for _, y, _, _ in positions)
    xmax = max(x + w for x, _, w, _ in positions)
    ymax = max(y + h for _, y, _, h in positions)

    _draw_nets(ax, positions, b2b_conn, p2b_conn, pins_pos, draw_b2b, draw_p2b)

    for i, (x, y, w, h) in enumerate(positions):
        edgecolor, linewidth, hatch = _constraint_style(constraints[i])
        rect = mpatches.Rectangle(
            (x, y),
            w,
            h,
            facecolor=colors[i],
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
            alpha=0.72,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2.0,
            y + h / 2.0,
            str(i),
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            zorder=3,
        )

    ax.add_patch(
        mpatches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=1.2,
            zorder=4,
        )
    )

    if pins_pos is not None and len(pins_pos) > 0:
        px = [float(pin[0]) for pin in pins_pos]
        py = [float(pin[1]) for pin in pins_pos]
        ax.scatter(px, py, marker="x", s=24, color="green", linewidths=1.0, zorder=5)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    margin_x = max((xmax - xmin) * 0.05, 1.0)
    margin_y = max((ymax - ymin) * 0.05, 1.0)
    ax.set_xlim(xmin - margin_x, xmax + margin_x)
    ax.set_ylim(ymin - margin_y, ymax + margin_y)

    if metrics is not None:
        summary = (
            f"feasible={metrics['is_feasible']}  local_cost={metrics['cost']:.4f}\n"
            f"hpwl_gap={metrics['hpwl_gap']:.4f}  area_gap={metrics['area_gap']:.4f}\n"
            f"v_rel={metrics['violations_relative']:.4f}  "
            f"overlap={metrics['overlap_violations']}  area={metrics['area_violations']}  dim={metrics['dimension_violations']}"
        )
        ax.text(
            0.01,
            0.99,
            summary,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )


def save_figure(
    case: Dict,
    positions: List[Position],
    metrics: Dict,
    output_path: str,
    source_label: str,
    test_id: int,
    with_reference: bool,
    draw_b2b: bool,
    draw_p2b: bool,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    gt_positions = extract_positions_from_polygons(case["polygons"], case["block_count"])
    fig, axes = plt.subplots(1, 2 if with_reference else 1, figsize=(16 if with_reference else 9, 8))
    if not isinstance(axes, (list, tuple)):
        axes = [axes] if not hasattr(axes, "__len__") else axes

    draw_layout(
        axes[0],
        positions,
        case["constraints"],
        case["pins_pos"],
        f"Solution | test_id={test_id} | blocks={case['block_count']}",
        metrics,
        draw_b2b,
        draw_p2b,
        case["b2b_conn"],
        case["p2b_conn"],
    )

    if with_reference:
        draw_layout(
            axes[1],
            gt_positions,
            case["constraints"],
            case["pins_pos"],
            "Ground Truth / Baseline",
            None,
            False,
            False,
            case["b2b_conn"],
            case["p2b_conn"],
        )

    fig.suptitle(source_label, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved image to: {Path(output_path).resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one FloorSet solution for a given validation case."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-json", help="Path to *_results.json produced by --evaluate")
    source.add_argument("--solutions-json", help="Path to *_solutions.json produced by --save-solutions")
    source.add_argument("--optimizer", help="Path to optimizer python file")

    parser.add_argument("--test-id", type=int, required=True, help="Validation case id (0-99)")
    parser.add_argument("--data-path", default="..", help="Path to FloorSet root from iccad2026contest/")
    parser.add_argument("--output", "-o", default=None, help="Output PNG path")
    parser.add_argument("--with-reference", action="store_true", help="Draw ground truth beside the solution")
    parser.add_argument("--draw-b2b", action="store_true", help="Overlay block-to-block nets")
    parser.add_argument("--draw-p2b", action="store_true", help="Overlay pin-to-block nets")
    parser.add_argument("--show", action="store_true", help="Show the figure window after saving")
    return parser.parse_args()


def choose_output_path(args: argparse.Namespace) -> str:
    if args.output:
        return args.output

    if args.optimizer:
        stem = Path(args.optimizer).stem
    elif args.results_json:
        stem = Path(args.results_json).stem
    else:
        stem = Path(args.solutions_json).stem
    return f"{stem}_test_{args.test_id}.png"


def main() -> None:
    args = parse_args()
    case = load_test_case(args.test_id, args.data_path)

    if args.optimizer:
        positions, record = solve_with_optimizer(args.optimizer, case, args.data_path)
        source_label = f"optimizer={Path(args.optimizer).resolve()}"
        runtime_seconds = record["runtime_seconds"]
    else:
        json_path = args.results_json or args.solutions_json
        positions, record = load_positions_from_json(json_path, args.test_id, case["block_count"])
        source_label = f"source={Path(json_path).resolve()}"
        runtime_seconds = record.get("runtime_seconds", record.get("runtime"))

    metrics = evaluate_positions(case, positions, runtime_seconds)
    output_path = choose_output_path(args)

    print(f"test_id={args.test_id}, block_count={case['block_count']}")
    print(
        "metrics: "
        f"feasible={metrics['is_feasible']}, "
        f"local_cost={metrics['cost']:.4f}, "
        f"hpwl_gap={metrics['hpwl_gap']:.4f}, "
        f"area_gap={metrics['area_gap']:.4f}, "
        f"v_rel={metrics['violations_relative']:.4f}"
    )

    if "cost" in record:
        print(f"recorded_cost={float(record['cost']):.4f}")
    if runtime_seconds is not None:
        print(f"runtime_seconds={float(runtime_seconds):.6f}")

    save_figure(
        case,
        positions,
        metrics,
        output_path,
        source_label,
        args.test_id,
        args.with_reference,
        args.draw_b2b,
        args.draw_p2b,
        args.show,
    )


if __name__ == "__main__":
    main()
