#!/usr/bin/env python3
"""
Verify MIB Constraint Violations in my_optimizer_grouping.py

Checks whether the optimizer produces any MIB (Multi-Instance Block) violations.
MIB constraint requires all blocks in the same group to have identical dimensions.
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import get_validation_dataloader
from my_optimizer_grouping import MyOptimizer


def parse_mib_groups(constraints: torch.Tensor, block_count: int) -> Dict[int, List[int]]:
    """
    Parse MIB groups from constraint tensor.

    Returns:
        Dictionary mapping group_id to list of block indices
    """
    mib_groups = defaultdict(list)

    if constraints is None or block_count == 0:
        return mib_groups

    for i in range(block_count):
        mib_id = int(constraints[i, 2])  # Column 2 is MIB
        if mib_id > 0:
            mib_groups[mib_id].append(i)

    return mib_groups


def check_mib_violations(solution: List[Tuple[float, float, float, float]],
                        mib_groups: Dict[int, List[int]],
                        tolerance: float = 1e-3) -> Tuple[int, Dict]:
    """
    Check MIB violations in the solution.

    Returns:
        (total_violations, detailed_info)
    """
    total_violations = 0
    detailed_info = {}

    for group_id, blocks in mib_groups.items():
        if len(blocks) <= 1:
            continue

        # Collect all dimensions (rounded to avoid floating point issues)
        dimensions = []
        for block in blocks:
            w = round(solution[block][2], 3)
            h = round(solution[block][3], 3)
            dimensions.append((w, h))

        # Count unique dimensions
        unique_dims = set(dimensions)
        violations = len(unique_dims) - 1
        total_violations += violations

        detailed_info[group_id] = {
            'blocks': blocks,
            'dimensions': dimensions,
            'unique_dims': unique_dims,
            'violations': violations
        }

    return total_violations, detailed_info


def main():
    """Verify MIB violations for validation cases."""

    print("="*80)
    print("MIB VIOLATION VERIFICATION")
    print("="*80)
    print("\nVerifying my_optimizer_grouping.py for MIB constraint violations...")
    print()

    # Load optimizer
    optimizer = MyOptimizer(verbose=False)

    # Load validation data
    val_loader = get_validation_dataloader()

    # Statistics
    total_cases = 0
    total_mib_groups = 0
    total_violations = 0
    cases_with_violations = 0

    print(f"{'Test ID':<10} {'Blocks':<10} {'MIB Groups':<12} {'Violations':<12} {'Status':<10}")
    print("-"*80)

    for batch in val_loader:
        # Extract data
        data_list = batch[0]

        if not isinstance(data_list, list) or len(data_list) < 5:
            continue

        # Extract fields
        area_target = data_list[0]
        b2b_connectivity = data_list[1]
        p2b_connectivity = data_list[2]
        pins_pos = data_list[3]
        placement_constraints = data_list[4]

        # Get block count
        block_count = area_target.shape[1]
        test_id = total_cases

        # Remove batch dimension from constraints
        if placement_constraints.dim() == 3:
            placement_constraints = placement_constraints[0]

        # Parse MIB groups
        mib_groups = parse_mib_groups(placement_constraints, block_count)

        if not mib_groups:
            # No MIB constraints in this case
            continue

        # Generate solution
        solution = optimizer.solve(
            block_count=block_count,
            area_targets=area_target[0],
            b2b_connectivity=b2b_connectivity[0],
            p2b_connectivity=p2b_connectivity[0],
            pins_pos=pins_pos[0],
            constraints=placement_constraints,
            target_positions=None  # Will be extracted if needed
        )

        # Check MIB violations
        violations, detailed_info = check_mib_violations(solution, mib_groups)

        # Update statistics
        total_cases += 1
        total_mib_groups += len(mib_groups)
        total_violations += violations
        if violations > 0:
            cases_with_violations += 1

        # Print case results
        status = "OK" if violations == 0 else "FAIL"
        print(f"{test_id:<10} {block_count:<10} {len(mib_groups):<12} {violations:<12} {status:<10}")

        # Print detailed info for cases with violations
        if violations > 0:
            print(f"\n  Details for Test {test_id}:")
            for group_id, info in detailed_info.items():
                if info['violations'] > 0:
                    print(f"    Group {group_id}: blocks {info['blocks']}")
                    for i, block in enumerate(info['blocks']):
                        w, h = info['dimensions'][i]
                        print(f"      Block {block}: w={w:.3f}, h={h:.3f}")
                    print(f"      Unique dimensions: {len(info['unique_dims'])}")
                    print(f"      Violations: {info['violations']}")
            print()

        # Show first 20 cases
        if total_cases >= 20:
            print("... (showing first 20 cases with MIB constraints)")
            break

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nCases analyzed: {total_cases}")
    print(f"Total MIB groups: {total_mib_groups}")
    print(f"Total violations: {total_violations}")
    print(f"Cases with violations: {cases_with_violations}")

    if total_violations == 0:
        print("\n[OK] No MIB violations detected!")
        print("The optimizer correctly maintains MIB constraints.")
    else:
        print(f"\n[FAIL] Found {total_violations} MIB violations in {cases_with_violations} cases!")
        print("The optimizer is NOT correctly maintaining MIB constraints.")
        print("\nPossible causes:")
        print("  1. _repair_overlaps method modifies dimensions")
        print("  2. _postprocess_boundary_grouped method modifies dimensions")
        print("  3. Some other post-processing step changes dimensions")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
