#!/usr/bin/env python3
"""
Compute average soft-constraint contribution ratios on all validation cases.

For each test case, under the "fully violated" assumption:

    boundary_ratio = |B_boundary| / N_soft
    grouping_ratio = sum_p (|G_p| - 1) / N_soft
    mib_ratio      = sum_q (|M_q| - 1) / N_soft

where:

    N_soft = |B_boundary| + sum_p (|G_p| - 1) + sum_q (|M_q| - 1)

The script reports the arithmetic mean of the three ratios across all cases.
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
class CaseContribution:
    test_id: int
    block_count: int
    boundary_units: int
    grouping_units: int
    mib_units: int
    n_soft: int
    boundary_ratio: float
    grouping_ratio: float
    mib_ratio: float


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


def compute_case_contribution(test_id: int, sample: Dict[str, object]) -> CaseContribution:
    area_target, _, _, _, constraints = sample["input"]
    block_count = infer_block_count(area_target)
    constraints = constraints[:block_count]

    boundary_units = 0
    grouping_units = 0
    mib_units = 0

    if constraints.shape[1] > 4:
        boundary_units = int((constraints[:, 4] != 0).sum().item())
    if constraints.shape[1] > 3:
        grouping_units = count_group_units(constraints[:, 3])
    if constraints.shape[1] > 2:
        mib_units = count_group_units(constraints[:, 2])

    n_soft = boundary_units + grouping_units + mib_units

    if n_soft == 0:
        boundary_ratio = 0.0
        grouping_ratio = 0.0
        mib_ratio = 0.0
    else:
        boundary_ratio = boundary_units / n_soft
        grouping_ratio = grouping_units / n_soft
        mib_ratio = mib_units / n_soft

    return CaseContribution(
        test_id=test_id,
        block_count=block_count,
        boundary_units=boundary_units,
        grouping_units=grouping_units,
        mib_units=mib_units,
        n_soft=n_soft,
        boundary_ratio=boundary_ratio,
        grouping_ratio=grouping_ratio,
        mib_ratio=mib_ratio,
    )


def analyze_dataset(data_path: str) -> Dict[str, object]:
    dataset = FloorplanDatasetLiteTest(data_path)

    cases: List[CaseContribution] = []
    zero_soft_cases = 0
    total_boundary_units = 0
    total_grouping_units = 0
    total_mib_units = 0
    total_n_soft = 0

    for test_id in range(len(dataset)):
        case = compute_case_contribution(test_id, dataset[test_id])
        cases.append(case)

        if case.n_soft == 0:
            zero_soft_cases += 1

        total_boundary_units += case.boundary_units
        total_grouping_units += case.grouping_units
        total_mib_units += case.mib_units
        total_n_soft += case.n_soft

    case_count = len(cases)
    avg_boundary_ratio = sum(case.boundary_ratio for case in cases) / case_count if case_count else 0.0
    avg_grouping_ratio = sum(case.grouping_ratio for case in cases) / case_count if case_count else 0.0
    avg_mib_ratio = sum(case.mib_ratio for case in cases) / case_count if case_count else 0.0

    weighted_boundary_ratio = total_boundary_units / total_n_soft if total_n_soft else 0.0
    weighted_grouping_ratio = total_grouping_units / total_n_soft if total_n_soft else 0.0
    weighted_mib_ratio = total_mib_units / total_n_soft if total_n_soft else 0.0

    return {
        "num_cases": case_count,
        "zero_soft_cases": zero_soft_cases,
        "average_case_ratios": {
            "boundary": avg_boundary_ratio,
            "grouping": avg_grouping_ratio,
            "mib": avg_mib_ratio,
        },
        "global_weighted_ratios": {
            "boundary": weighted_boundary_ratio,
            "grouping": weighted_grouping_ratio,
            "mib": weighted_mib_ratio,
        },
        "totals": {
            "boundary_units": total_boundary_units,
            "grouping_units": total_grouping_units,
            "mib_units": total_mib_units,
            "n_soft": total_n_soft,
        },
        "cases": [asdict(case) for case in cases],
    }


def format_report(result: Dict[str, object]) -> str:
    avg = result["average_case_ratios"]
    weighted = result["global_weighted_ratios"]
    totals = result["totals"]

    lines = [
        "Average soft-constraint contribution ratios under full violation",
        f"num_cases: {result['num_cases']}",
        f"zero_soft_cases: {result['zero_soft_cases']}",
        "",
        "Arithmetic mean over cases:",
        f"  |Bboundary| / Nsoft               = {avg['boundary']:.6f}",
        f"  sum_p (|Gp| - 1) / Nsoft         = {avg['grouping']:.6f}",
        f"  sum_q (|Mq| - 1) / Nsoft         = {avg['mib']:.6f}",
        "",
        "Global weighted ratios (sum units / sum Nsoft):",
        f"  boundary                         = {weighted['boundary']:.6f}",
        f"  grouping                         = {weighted['grouping']:.6f}",
        f"  mib                              = {weighted['mib']:.6f}",
        "",
        "Totals:",
        f"  boundary_units                   = {totals['boundary_units']}",
        f"  grouping_units                   = {totals['grouping_units']}",
        f"  mib_units                        = {totals['mib_units']}",
        f"  n_soft                           = {totals['n_soft']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze average soft-constraint contribution ratios on validation data."
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
