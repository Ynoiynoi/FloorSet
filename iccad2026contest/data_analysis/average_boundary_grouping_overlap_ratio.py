#!/usr/bin/env python3
"""
Compute average overlap ratios for blocks that simultaneously have
boundary and grouping constraints on the validation set.

Per test case:

    overlap_block_ratio = (# blocks with boundary != 0 and grouping != 0) / block_count
    overlap_nsoft_ratio = (# blocks with boundary != 0 and grouping != 0) / N_soft

where:

    N_soft = |B_boundary| + sum_p (|G_p| - 1) + sum_q (|M_q| - 1)

The script reports arithmetic means across all cases. It also reports the
mean of exp(beta * overlap_nsoft_ratio), where the default beta matches the
contest evaluator (beta = 2.0). Global weighted ratios are also reported:

    total_overlap_blocks / total_blocks
    total_overlap_blocks / total_n_soft
"""

from __future__ import annotations

import argparse
import json
import math
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


DEFAULT_BETA = 2.0


@dataclass
class CaseOverlapStats:
    test_id: int
    block_count: int
    boundary_count: int
    grouping_count: int
    boundary_units: int
    grouping_units: int
    mib_units: int
    n_soft: int
    overlap_count: int
    overlap_block_ratio: float
    overlap_nsoft_ratio: float


def infer_block_count(area_target: torch.Tensor) -> int:
    return int((area_target != -1).sum().item())


def count_group_units(group_ids: torch.Tensor) -> int:
    if group_ids.numel() == 0:
        return 0

    group_ids = group_ids.to(torch.int64)
    positive_ids = torch.unique(group_ids[group_ids > 0])

    total = 0
    for group_id in positive_ids.tolist():
        group_size = int((group_ids == group_id).sum().item())
        total += max(0, group_size - 1)
    return total


def compute_case_stats(test_id: int, sample: Dict[str, object]) -> CaseOverlapStats:
    area_target, _, _, _, constraints = sample["input"]
    block_count = infer_block_count(area_target)
    constraints = constraints[:block_count]

    if constraints.shape[1] > 4:
        boundary_mask = constraints[:, 4] != 0
    else:
        boundary_mask = torch.zeros(block_count, dtype=torch.bool)

    if constraints.shape[1] > 3:
        grouping_mask = constraints[:, 3] != 0
    else:
        grouping_mask = torch.zeros(block_count, dtype=torch.bool)

    boundary_units = int(boundary_mask.sum().item())
    grouping_units = count_group_units(constraints[:, 3]) if constraints.shape[1] > 3 else 0
    mib_units = count_group_units(constraints[:, 2]) if constraints.shape[1] > 2 else 0
    n_soft = boundary_units + grouping_units + mib_units

    overlap_mask = boundary_mask & grouping_mask

    boundary_count = boundary_units
    grouping_count = int(grouping_mask.sum().item())
    overlap_count = int(overlap_mask.sum().item())
    overlap_block_ratio = overlap_count / block_count if block_count > 0 else 0.0
    overlap_nsoft_ratio = overlap_count / n_soft if n_soft > 0 else 0.0

    return CaseOverlapStats(
        test_id=test_id,
        block_count=block_count,
        boundary_count=boundary_count,
        grouping_count=grouping_count,
        boundary_units=boundary_units,
        grouping_units=grouping_units,
        mib_units=mib_units,
        n_soft=n_soft,
        overlap_count=overlap_count,
        overlap_block_ratio=overlap_block_ratio,
        overlap_nsoft_ratio=overlap_nsoft_ratio,
    )


def analyze_dataset(data_path: str, beta: float) -> Dict[str, object]:
    dataset = FloorplanDatasetLiteTest(data_path)

    cases: List[CaseOverlapStats] = []
    total_blocks = 0
    total_boundary_blocks = 0
    total_grouping_blocks = 0
    total_overlap_blocks = 0
    total_n_soft = 0

    for test_id in range(len(dataset)):
        case = compute_case_stats(test_id, dataset[test_id])
        cases.append(case)

        total_blocks += case.block_count
        total_boundary_blocks += case.boundary_count
        total_grouping_blocks += case.grouping_count
        total_overlap_blocks += case.overlap_count
        total_n_soft += case.n_soft

    case_count = len(cases)
    avg_overlap_block_ratio = (
        sum(case.overlap_block_ratio for case in cases) / case_count if case_count else 0.0
    )
    avg_overlap_nsoft_ratio = (
        sum(case.overlap_nsoft_ratio for case in cases) / case_count if case_count else 0.0
    )
    avg_exp_beta_overlap_nsoft = (
        sum(math.exp(beta * case.overlap_nsoft_ratio) for case in cases) / case_count
        if case_count else 0.0
    )
    weighted_overlap_block_ratio = total_overlap_blocks / total_blocks if total_blocks else 0.0
    weighted_overlap_nsoft_ratio = total_overlap_blocks / total_n_soft if total_n_soft else 0.0
    exp_beta_weighted_overlap_nsoft = math.exp(beta * weighted_overlap_nsoft_ratio)

    return {
        "beta": beta,
        "num_cases": case_count,
        "average_case_overlap_block_ratio": avg_overlap_block_ratio,
        "average_case_overlap_nsoft_ratio": avg_overlap_nsoft_ratio,
        "average_case_exp_beta_overlap_nsoft": avg_exp_beta_overlap_nsoft,
        "global_weighted_overlap_block_ratio": weighted_overlap_block_ratio,
        "global_weighted_overlap_nsoft_ratio": weighted_overlap_nsoft_ratio,
        "exp_beta_global_weighted_overlap_nsoft": exp_beta_weighted_overlap_nsoft,
        "totals": {
            "blocks": total_blocks,
            "boundary_blocks": total_boundary_blocks,
            "grouping_blocks": total_grouping_blocks,
            "overlap_blocks": total_overlap_blocks,
            "n_soft": total_n_soft,
        },
        "cases": [asdict(case) for case in cases],
    }


def format_report(result: Dict[str, object]) -> str:
    totals = result["totals"]
    lines = [
        "Average ratio of blocks with both boundary and grouping constraints",
        f"beta: {result['beta']:.6f}",
        f"num_cases: {result['num_cases']}",
        "",
        "Arithmetic mean over cases:",
        f"  overlap_blocks / block_count        = {result['average_case_overlap_block_ratio']:.6f}",
        f"  overlap_blocks / Nsoft             = {result['average_case_overlap_nsoft_ratio']:.6f}",
        f"  exp(beta * overlap_blocks / Nsoft) = {result['average_case_exp_beta_overlap_nsoft']:.6f}",
        "",
        "Global weighted ratio:",
        f"  total_overlap_blocks / total_blocks = {result['global_weighted_overlap_block_ratio']:.6f}",
        f"  total_overlap_blocks / total_Nsoft = {result['global_weighted_overlap_nsoft_ratio']:.6f}",
        f"  exp(beta * total_overlap_blocks / total_Nsoft) = {result['exp_beta_global_weighted_overlap_nsoft']:.6f}",
        "",
        "Totals:",
        f"  total_blocks                        = {totals['blocks']}",
        f"  total_boundary_blocks               = {totals['boundary_blocks']}",
        f"  total_grouping_blocks               = {totals['grouping_blocks']}",
        f"  total_overlap_blocks                = {totals['overlap_blocks']}",
        f"  total_n_soft                        = {totals['n_soft']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the average ratio of blocks with both boundary and grouping constraints."
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
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Beta used in exp(beta * V). Default: 2.0",
    )
    args = parser.parse_args()

    result = analyze_dataset(args.data_path, args.beta)
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
