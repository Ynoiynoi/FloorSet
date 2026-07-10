#!/usr/bin/env python3
"""
Compute average P2B/B2B net-count ratios on all LiteTensorDataTest cases.

For each test case:

    p2b_ratio = num_p2b_edges / (num_p2b_edges + num_b2b_edges)
    b2b_ratio = num_b2b_edges / (num_p2b_edges + num_b2b_edges)

The script reports both:

1. Arithmetic mean over cases: average of each per-case ratio.
2. Global weighted ratio: sum of each edge count divided by total edge count.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
CONTEST_ROOT = SCRIPT_DIR.parent
DATA_ROOT = CONTEST_ROOT.parent

for path in [str(CONTEST_ROOT), str(DATA_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from litetestLoader import FloorplanDatasetLiteTest  # noqa: E402


@dataclass
class CaseNetStats:
    test_id: int
    block_count: int
    b2b_edges: int
    p2b_edges: int
    total_edges: int
    b2b_ratio: float
    p2b_ratio: float


def infer_block_count(area_target: torch.Tensor) -> int:
    return int((area_target != -1).sum().item())


def count_valid_edges(connectivity: torch.Tensor) -> int:
    if connectivity is None or connectivity.numel() == 0:
        return 0
    return int((connectivity[:, 0] != -1).sum().item())


def compute_case_stats(test_id: int, sample: Dict[str, object]) -> CaseNetStats:
    area_target, b2b_conn, p2b_conn, _, _ = sample["input"]

    block_count = infer_block_count(area_target)
    b2b_edges = count_valid_edges(b2b_conn)
    p2b_edges = count_valid_edges(p2b_conn)
    total_edges = b2b_edges + p2b_edges

    if total_edges == 0:
        b2b_ratio = 0.0
        p2b_ratio = 0.0
    else:
        b2b_ratio = b2b_edges / total_edges
        p2b_ratio = p2b_edges / total_edges

    return CaseNetStats(
        test_id=test_id,
        block_count=block_count,
        b2b_edges=b2b_edges,
        p2b_edges=p2b_edges,
        total_edges=total_edges,
        b2b_ratio=b2b_ratio,
        p2b_ratio=p2b_ratio,
    )


def analyze_dataset(data_path: str) -> Dict[str, object]:
    dataset = FloorplanDatasetLiteTest(data_path)

    cases: List[CaseNetStats] = []
    zero_net_cases = 0
    total_b2b_edges = 0
    total_p2b_edges = 0
    total_edges = 0

    for test_id in range(len(dataset)):
        case = compute_case_stats(test_id, dataset[test_id])
        cases.append(case)

        if case.total_edges == 0:
            zero_net_cases += 1

        total_b2b_edges += case.b2b_edges
        total_p2b_edges += case.p2b_edges
        total_edges += case.total_edges

    case_count = len(cases)
    avg_b2b_ratio = sum(case.b2b_ratio for case in cases) / case_count if case_count else 0.0
    avg_p2b_ratio = sum(case.p2b_ratio for case in cases) / case_count if case_count else 0.0

    weighted_b2b_ratio = total_b2b_edges / total_edges if total_edges else 0.0
    weighted_p2b_ratio = total_p2b_edges / total_edges if total_edges else 0.0

    return {
        "num_cases": case_count,
        "zero_net_cases": zero_net_cases,
        "average_case_ratios": {
            "p2b": avg_p2b_ratio,
            "b2b": avg_b2b_ratio,
        },
        "global_weighted_ratios": {
            "p2b": weighted_p2b_ratio,
            "b2b": weighted_b2b_ratio,
        },
        "totals": {
            "p2b_edges": total_p2b_edges,
            "b2b_edges": total_b2b_edges,
            "all_edges": total_edges,
        },
        "cases": [asdict(case) for case in cases],
    }


def format_report(result: Dict[str, object]) -> str:
    avg = result["average_case_ratios"]
    weighted = result["global_weighted_ratios"]
    totals = result["totals"]

    lines = [
        "Average P2B/B2B net-count ratios",
        f"num_cases: {result['num_cases']}",
        f"zero_net_cases: {result['zero_net_cases']}",
        "",
        "Arithmetic mean over cases:",
        f"  p2b_edges / (p2b_edges + b2b_edges) = {avg['p2b']:.6f}",
        f"  b2b_edges / (p2b_edges + b2b_edges) = {avg['b2b']:.6f}",
        "",
        "Global weighted ratios (sum edges / sum all edges):",
        f"  p2b                                  = {weighted['p2b']:.6f}",
        f"  b2b                                  = {weighted['b2b']:.6f}",
        "",
        "Totals:",
        f"  p2b_edges                            = {totals['p2b_edges']}",
        f"  b2b_edges                            = {totals['b2b_edges']}",
        f"  all_edges                            = {totals['all_edges']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze average P2B and B2B net-count ratios on validation data."
    )
    parser.add_argument(
        "--data-path",
        default="../",
        help="Dataset root path containing LiteTensorDataTest/ (default: ../)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file. Use .json to save raw structured results.",
    )
    args = parser.parse_args()

    result = analyze_dataset(args.data_path)
    report = format_report(result)
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            output_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
