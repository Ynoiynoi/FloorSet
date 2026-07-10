#!/usr/bin/env python3
"""
Report per-case contest scores and weighted contributions.

Why this exists:
- The official Total Score is not a plain mean of case costs.
- It is an exponentially weighted average by block count:
    weight_i ∝ exp(n_i)
  implemented numerically as exp(n_i - max_n).
- So even if many small cases score around 2.0~2.5, a few large cases with
  worse costs can pull the final Total Score much higher.

Usage:
    python sol\\eazyver2\\report_case_scores.py --optimizer sol\\eazyver2\\my_optimizer.py
    python sol\\eazyver2\\report_case_scores.py --optimizer sol\\eazyver2\\my_optimizer_mib_square.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List

CONTEST_ROOT = Path(__file__).resolve().parents[2]
if str(CONTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEST_ROOT))

from iccad2026_evaluate import ContestEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report per-test-case costs, weights, and weighted score contributions."
    )
    parser.add_argument(
        "--optimizer",
        "-e",
        required=True,
        help="Path to optimizer .py file",
    )
    parser.add_argument(
        "--data-path",
        "-d",
        default="../",
        help="Path to FloorSet data directory (default: ../)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output text file",
    )
    args = parser.parse_args()

    evaluator = ContestEvaluator(args.data_path, verbose=False)
    result = evaluator.evaluate(args.optimizer)

    rows = result.test_results
    costs: List[float] = [row.cost for row in rows]
    blocks: List[int] = [row.block_count for row in rows]
    max_n = max(blocks) if blocks else 0
    raw_weights = [math.exp(n - max_n) for n in blocks]
    total_weight = sum(raw_weights) if raw_weights else 1.0
    norm_weights = [w / total_weight for w in raw_weights]
    contributions = [c * w for c, w in zip(costs, norm_weights)]

    mean_cost = sum(costs) / len(costs) if costs else 0.0
    total_score = sum(contributions)

    lines: List[str] = []
    lines.append(f"optimizer: {Path(args.optimizer).resolve()}")
    lines.append(f"tests: {len(rows)}")
    lines.append(f"feasible: {sum(1 for row in rows if row.is_feasible)}")
    lines.append(f"plain_avg_cost: {mean_cost:.6f}")
    lines.append(f"contest_total_score: {total_score:.6f}")
    lines.append(f"delta(total - mean): {total_score - mean_cost:.6f}")
    lines.append("")
    lines.append("Per-case:")
    lines.append(
        "test_id blocks feasible cost norm_weight weighted_contrib "
        "hpwl_gap area_gap v_rel"
    )

    sorted_idx = sorted(
        range(len(rows)),
        key=lambda i: contributions[i],
        reverse=True,
    )
    for i in sorted_idx:
        row = rows[i]
        lines.append(
            f"{row.test_id:3d} "
            f"{row.block_count:3d} "
            f"{int(row.is_feasible):1d} "
            f"{row.cost:10.6f} "
            f"{norm_weights[i]:12.8f} "
            f"{contributions[i]:14.8f} "
            f"{row.hpwl_gap:9.6f} "
            f"{row.area_gap:9.6f} "
            f"{row.violations_relative:9.6f}"
        )

    lines.append("")
    lines.append("Top 10 weighted contributors:")
    for rank, i in enumerate(sorted_idx[:10], start=1):
        row = rows[i]
        lines.append(
            f"{rank:2d}. test_id={row.test_id:3d}, blocks={row.block_count:3d}, "
            f"cost={row.cost:.6f}, weight={norm_weights[i]:.8f}, "
            f"contrib={contributions[i]:.8f}"
        )

    text = "\n".join(lines)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
