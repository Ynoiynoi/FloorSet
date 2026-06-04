#!/usr/bin/env python3
"""
Analyze average metric contributions to contest cost on validation cases.

This utility is meant for comparing different floorplanning methods using the
same validation set. It reports the averages of:

- HPWL_gap
- Area_gap (bbox)
- Violations_relative
- Vfixed, Vpreplaced, Vgrouping, Vboundary, Vmib

It also reports a few derived terms from the no-runtime contest formula:

    Cost_no_runtime = (1 + 0.5 * (max(0, HPWL_gap) + max(0, Area_gap)))
                      * exp(2 * Violations_relative)

Why no runtime by default:
- runtime is submission-relative in the official ranking
- the requested analysis is about quality and constraint terms
- comparing methods is much cleaner without runtime mixed in

Usage examples:
    python analyze_cost_contributions.py --evaluate my_optimizer.py
    python analyze_cost_contributions.py --score my_optimizer_solutions.json
    python analyze_cost_contributions.py --evaluate my_optimizer.py --output summary.json
    python analyze_cost_contributions.py --evaluate my_optimizer.py --test-id 0
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import ALPHA, BETA, FloorplanOptimizer
from iccad2026_evaluate import calculate_bbox_area, calculate_hpwl_b2b, calculate_hpwl_p2b
from iccad2026_evaluate import compute_total_score
from iccad2026_evaluate_no_runtime import evaluate_solution_no_runtime
from litetestLoader import FloorplanDatasetLiteTest


def load_optimizer(optimizer_path: str, verbose: bool) -> FloorplanOptimizer:
    path = Path(optimizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Optimizer file not found: {optimizer_path}")

    spec = importlib.util.spec_from_file_location("optimizer_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import optimizer from {optimizer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ["MyOptimizer", "Optimizer", "ContestOptimizer"]:
        if hasattr(module, name):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, FloorplanOptimizer)
                and obj.__name__ != "FloorplanOptimizer"
                and getattr(obj, "__module__", None) == module.__name__
            ):
                return obj(verbose=verbose)

    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, FloorplanOptimizer)
            and obj.__name__ != "FloorplanOptimizer"
            and getattr(obj, "__module__", None) == module.__name__
        ):
            return obj(verbose=verbose)

    raise ValueError(f"No optimizer class found in {optimizer_path}")


def extract_baseline_and_targets(
    labels: Tuple[torch.Tensor, Optional[torch.Tensor]],
    b2b_conn: torch.Tensor,
    p2b_conn: torch.Tensor,
    pins_pos: torch.Tensor,
    block_count: int,
) -> Tuple[Dict[str, float], List[Tuple[float, float, float, float]]]:
    polygons, metrics = labels

    target_positions: List[Tuple[float, float, float, float]] = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            target_positions.append(
                (
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                )
            )
        else:
            target_positions.append((0.0, 0.0, 1.0, 1.0))

    hpwl_b2b = calculate_hpwl_b2b(target_positions, b2b_conn)
    hpwl_p2b = calculate_hpwl_p2b(target_positions, p2b_conn, pins_pos)
    area = calculate_bbox_area(target_positions)

    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0:
            area = float(metrics[0])
        if metrics[-2] > 0:
            hpwl_b2b = float(metrics[-2])
        if metrics[-1] >= 0:
            hpwl_p2b = float(metrics[-1])

    baseline = {
        "hpwl_baseline": hpwl_b2b + hpwl_p2b,
        "area_baseline": area,
    }
    return baseline, target_positions


def build_optimizer_target_positions(
    constraints: torch.Tensor,
    target_positions: List[Tuple[float, float, float, float]],
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
            opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = target_positions[i]
            opt_target_pos[i, 2] = tw
            opt_target_pos[i, 3] = th
    return opt_target_pos


def load_solutions(solutions_path: str) -> Dict[int, Dict[str, Any]]:
    path = Path(solutions_path)
    if not path.exists():
        raise FileNotFoundError(f"Solutions file not found: {solutions_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "solutions" in data:
        records = data["solutions"]
    elif isinstance(data, dict) and "test_results" in data:
        records = data["test_results"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            "Unsupported solutions JSON format. Expected keys 'solutions' or 'test_results', or a raw list."
        )

    indexed: Dict[int, Dict[str, Any]] = {}
    for record in records:
        test_id = int(record["test_id"])
        indexed[test_id] = record
    return indexed


def aggregate_case_metrics(case_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not case_metrics:
        return {
            "num_cases": 0,
            "num_feasible": 0,
            "avg_block_count": 0.0,
        }

    n = len(case_metrics)

    def avg(key: str) -> float:
        return sum(float(case[key]) for case in case_metrics) / n

    costs = [float(case["cost_no_runtime"]) for case in case_metrics]
    blocks = [int(case["block_count"]) for case in case_metrics]

    summary = {
        "num_cases": n,
        "num_feasible": sum(1 for case in case_metrics if case["is_feasible"]),
        "avg_block_count": avg("block_count"),
        "avg_runtime_seconds": avg("runtime_seconds"),
        "avg_hpwl_gap": avg("hpwl_gap"),
        "avg_area_gap_bbox": avg("area_gap_bbox"),
        "avg_violations_relative": avg("violations_relative"),
        "avg_vfixed": avg("vfixed"),
        "avg_vpreplaced": avg("vpreplaced"),
        "avg_vgrouping": avg("vgrouping"),
        "avg_vboundary": avg("vboundary"),
        "avg_vmib": avg("vmib"),
        "avg_total_soft_violations": avg("total_soft_violations"),
        "avg_max_possible_violations": avg("max_possible_violations"),
        "avg_overlap_violations": avg("overlap_violations"),
        "avg_area_violations": avg("area_violations"),
        "avg_dimension_violations": avg("dimension_violations"),
        "avg_hpwl_term": avg("hpwl_term"),
        "avg_area_term": avg("area_term"),
        "avg_quality_factor": avg("quality_factor"),
        "avg_violation_factor": avg("violation_factor"),
        "avg_cost_no_runtime": avg("cost_no_runtime"),
        "weighted_total_score_no_runtime": compute_total_score(costs, blocks),
    }
    return summary


def format_summary_text(summary: Dict[str, Any]) -> str:
    lines = [
        "ICCAD 2026 cost contribution summary (no runtime factor)",
        f"timestamp: {summary['timestamp']}",
        f"source_type: {summary['source_type']}",
        f"source: {summary['source']}",
        f"num_cases: {summary['aggregate']['num_cases']}",
        f"num_feasible: {summary['aggregate']['num_feasible']}",
        f"avg_block_count: {summary['aggregate']['avg_block_count']:.4f}",
        f"avg_runtime_seconds: {summary['aggregate']['avg_runtime_seconds']:.4f}",
        "",
        "Primary metrics",
        f"avg_HPWLgap: {summary['aggregate']['avg_hpwl_gap']:.6f}",
        f"avg_Areagap_bbox: {summary['aggregate']['avg_area_gap_bbox']:.6f}",
        f"avg_Violationsrelative: {summary['aggregate']['avg_violations_relative']:.6f}",
        "",
        "Violation components",
        f"avg_Vfixed: {summary['aggregate']['avg_vfixed']:.6f}",
        f"avg_Vpreplaced: {summary['aggregate']['avg_vpreplaced']:.6f}",
        f"avg_Vgrouping: {summary['aggregate']['avg_vgrouping']:.6f}",
        f"avg_Vboundary: {summary['aggregate']['avg_vboundary']:.6f}",
        f"avg_Vmib: {summary['aggregate']['avg_vmib']:.6f}",
        f"avg_total_soft_violations: {summary['aggregate']['avg_total_soft_violations']:.6f}",
        f"avg_max_possible_violations: {summary['aggregate']['avg_max_possible_violations']:.6f}",
        "",
        "Derived no-runtime cost terms",
        f"avg_hpwl_term: {summary['aggregate']['avg_hpwl_term']:.6f}",
        f"avg_area_term: {summary['aggregate']['avg_area_term']:.6f}",
        f"avg_quality_factor: {summary['aggregate']['avg_quality_factor']:.6f}",
        f"avg_violation_factor: {summary['aggregate']['avg_violation_factor']:.6f}",
        f"avg_cost_no_runtime: {summary['aggregate']['avg_cost_no_runtime']:.6f}",
        f"weighted_total_score_no_runtime: {summary['aggregate']['weighted_total_score_no_runtime']:.6f}",
        "",
        "Notes",
        "Vfixed and Vpreplaced are reported for debugging/information.",
        "The current official soft penalty uses boundary, grouping, and MIB only.",
    ]
    return "\n".join(lines) + "\n"


def write_output(output_path: str, summary: Dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".txt":
        path.write_text(format_summary_text(summary), encoding="utf-8")
    else:
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze average cost contributions on validation cases."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evaluate", "-e", metavar="OPTIMIZER", help="Run an optimizer on validation set")
    mode.add_argument("--score", "-s", metavar="SOLUTIONS_JSON", help="Analyze a saved solutions/results JSON")

    parser.add_argument("--data-path", "-d", default="../", help="Path to FloorSet data (default: ../)")
    parser.add_argument("--test-id", "-t", type=int, default=None, help="Analyze one validation case only")
    parser.add_argument("--output", "-o", default=None, help="Optional output file (.json or .txt)")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bar and per-run chatter")

    args = parser.parse_args()

    verbose = not args.quiet
    dataset = FloorplanDatasetLiteTest(args.data_path)
    test_ids = [args.test_id] if args.test_id is not None else list(range(len(dataset)))

    optimizer: Optional[FloorplanOptimizer] = None
    solutions_by_test_id: Optional[Dict[int, Dict[str, Any]]] = None
    source_type: str
    source: str

    if args.evaluate:
        optimizer = load_optimizer(args.evaluate, verbose=verbose)
        source_type = "optimizer"
        source = str(Path(args.evaluate).resolve())
    else:
        solutions_by_test_id = load_solutions(args.score)
        source_type = "solutions_json"
        source = str(Path(args.score).resolve())

    iterator: Iterable[int] = tqdm(test_ids, desc="Analyzing") if verbose else test_ids
    case_metrics: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for test_id in iterator:
        try:
            sample = dataset[test_id]
            inputs, labels = sample["input"], sample["label"]
            area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
            block_count = int((area_target != -1).sum().item())
            baseline, target_positions = extract_baseline_and_targets(
                labels, b2b_conn, p2b_conn, pins_pos, block_count
            )

            runtime = 0.0
            if optimizer is not None:
                opt_target_pos = build_optimizer_target_positions(constraints, target_positions, block_count)
                start = time.time()
                positions = optimizer.solve(
                    block_count,
                    area_target,
                    b2b_conn,
                    p2b_conn,
                    pins_pos,
                    constraints,
                    opt_target_pos,
                )
                runtime = time.time() - start
            else:
                assert solutions_by_test_id is not None
                if test_id not in solutions_by_test_id:
                    raise KeyError(f"Missing solution for test_id={test_id}")
                record = solutions_by_test_id[test_id]
                if "positions" not in record or record["positions"] is None:
                    raise ValueError(f"Solution record for test_id={test_id} has no positions")
                positions = [tuple(p) for p in record["positions"]]
                runtime = float(record.get("runtime", record.get("runtime_seconds", 0.0)))

            metrics = evaluate_solution_no_runtime(
                {"positions": positions, "runtime": runtime},
                baseline,
                constraints,
                b2b_conn,
                p2b_conn,
                pins_pos,
                area_target,
                target_positions,
            )

            hpwl_term = ALPHA * max(0.0, metrics.hpwl_gap)
            area_term = ALPHA * max(0.0, metrics.area_gap)
            quality_factor = 1.0 + hpwl_term + area_term
            violation_factor = math.exp(BETA * metrics.violations_relative)

            case_metrics.append(
                {
                    "test_id": test_id,
                    "block_count": block_count,
                    "is_feasible": metrics.is_feasible,
                    "runtime_seconds": runtime,
                    "hpwl_gap": metrics.hpwl_gap,
                    "area_gap_bbox": metrics.area_gap,
                    "violations_relative": metrics.violations_relative,
                    "vfixed": metrics.fixed_violations,
                    "vpreplaced": metrics.preplaced_violations,
                    "vgrouping": metrics.grouping_violations,
                    "vboundary": metrics.boundary_violations,
                    "vmib": metrics.mib_violations,
                    "total_soft_violations": metrics.total_soft_violations,
                    "max_possible_violations": metrics.max_possible_violations,
                    "overlap_violations": metrics.overlap_violations,
                    "area_violations": metrics.area_violations,
                    "dimension_violations": metrics.dimension_violations,
                    "hpwl_term": hpwl_term,
                    "area_term": area_term,
                    "quality_factor": quality_factor,
                    "violation_factor": violation_factor,
                    "cost_no_runtime": metrics.cost,
                }
            )
        except Exception as exc:
            errors.append({"test_id": test_id, "error": str(exc)})

    summary = {
        "timestamp": datetime.now().isoformat(),
        "source_type": source_type,
        "source": source,
        "aggregate": aggregate_case_metrics(case_metrics),
        "errors": errors,
        "cases": case_metrics,
    }

    text = format_summary_text(summary)
    print(text, end="")
    if errors:
        print(f"errors: {len(errors)}")

    if args.output:
        write_output(args.output, summary)
        if verbose:
            print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
