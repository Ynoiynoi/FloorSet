#!/usr/bin/env python3
"""
Calculate Exact Constraint Contributions

Calculates the exact contribution of each soft constraint type to Violations_relative
if that constraint type is completely violated.

Based on the formula from the problem statement:
  Violations_relative = (V_fixed + V_preplaced + V_grouping + V_boundary + V_mib) / N_soft
  N_soft = |B_fixed| + |B_preplaced| + |B_boundary| + Σ(|Gp| - 1) + Σ(|Mq| - 1)
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import get_validation_dataloader


def parse_constraints(constraints: torch.Tensor, block_count: int) -> Dict:
    """
    Parse constraint tensor and extract constraint information.

    Returns:
        Dictionary with constraint counts and groups
    """
    result = {
        'fixed_blocks': [],
        'preplaced_blocks': [],
        'boundary_blocks': {},
        'grouping_groups': defaultdict(list),
        'mib_groups': defaultdict(list),
    }

    if constraints is None or block_count == 0:
        return result

    for i in range(block_count):
        # Fixed shape (column 0)
        if constraints[i, 0] == 1:
            result['fixed_blocks'].append(i)

        # Preplaced (column 1)
        if constraints[i, 1] == 1:
            result['preplaced_blocks'].append(i)

        # MIB (column 2)
        mib_id = int(constraints[i, 2])
        if mib_id > 0:
            result['mib_groups'][mib_id].append(i)

        # Grouping (column 3)
        group_id = int(constraints[i, 3])
        if group_id > 0:
            result['grouping_groups'][group_id].append(i)

        # Boundary (column 4)
        boundary_type = int(constraints[i, 4])
        if boundary_type > 0:
            result['boundary_blocks'][i] = boundary_type

    return result


def calculate_N_soft(parsed_constraints: Dict) -> int:
    """
    Calculate N_soft according to the formula:
    N_soft = |B_fixed| + |B_preplaced| + |B_boundary| + Σ(|Gp| - 1) + Σ(|Mq| - 1)
    """
    N_soft = 0

    # Single-block constraints
    N_soft += len(parsed_constraints['fixed_blocks'])
    N_soft += len(parsed_constraints['preplaced_blocks'])
    N_soft += len(parsed_constraints['boundary_blocks'])

    # Grouping constraints
    for group_id, blocks in parsed_constraints['grouping_groups'].items():
        N_soft += len(blocks) - 1

    # MIB constraints
    for group_id, blocks in parsed_constraints['mib_groups'].items():
        N_soft += len(blocks) - 1

    return N_soft


def calculate_max_contributions(parsed_constraints: Dict, N_soft: int) -> Dict[str, float]:
    """
    Calculate the contribution of each constraint type if completely violated.

    Returns:
        Dictionary with percentage contributions (0-100)
    """
    if N_soft == 0:
        return {
            'fixed': 0.0,
            'preplaced': 0.0,
            'boundary': 0.0,
            'grouping': 0.0,
            'mib': 0.0,
        }

    # Maximum violations for each type
    V_fixed_max = len(parsed_constraints['fixed_blocks'])
    V_preplaced_max = len(parsed_constraints['preplaced_blocks'])
    V_boundary_max = len(parsed_constraints['boundary_blocks'])

    V_grouping_max = sum(
        len(blocks) - 1
        for blocks in parsed_constraints['grouping_groups'].values()
    )

    V_mib_max = sum(
        len(blocks) - 1
        for blocks in parsed_constraints['mib_groups'].values()
    )

    # Calculate contributions as percentages
    contributions = {
        'fixed': (V_fixed_max / N_soft) * 100,
        'preplaced': (V_preplaced_max / N_soft) * 100,
        'boundary': (V_boundary_max / N_soft) * 100,
        'grouping': (V_grouping_max / N_soft) * 100,
        'mib': (V_mib_max / N_soft) * 100,
    }

    return contributions


def main():
    """Analyze constraint contributions for all validation cases."""

    print("="*80)
    print("EXACT CONSTRAINT CONTRIBUTION ANALYSIS")
    print("="*80)
    print("\nCalculating contribution of each constraint type if completely violated...")
    print("Based on formula: V_rel = (V_fixed + V_preplaced + V_grouping + V_boundary + V_mib) / N_soft")
    print()

    # Load validation data
    val_loader = get_validation_dataloader()

    # Statistics
    all_contributions = []
    case_count = 0

    print(f"{'Test ID':<10} {'Blocks':<10} {'N_soft':<10} {'Fixed%':<10} {'Prepl%':<10} {'Group%':<10} {'MIB%':<10} {'Bound%':<10}")
    print("-"*90)

    for batch in val_loader:
        # Extract data from batch
        # batch is (data_list, labels)
        data_list = batch[0]

        if not isinstance(data_list, list) or len(data_list) < 5:
            continue

        # Extract fields
        # [0]: area_target (1, n)
        # [1]: b2b_connectivity (1, edges, 3)
        # [2]: p2b_connectivity (1, edges, 3)
        # [3]: pins_pos (1, pins, 2)
        # [4]: placement_constraints (1, n, 5)
        area_target = data_list[0]
        placement_constraints = data_list[4]

        # Get block count from area_target shape
        block_count = area_target.shape[1]
        test_id = case_count

        # Remove batch dimension
        if placement_constraints.dim() == 3:
            placement_constraints = placement_constraints[0]  # (n, 5)

        # Parse constraints
        parsed = parse_constraints(placement_constraints, block_count)

        # Calculate N_soft
        N_soft = calculate_N_soft(parsed)

        if N_soft == 0:
            continue

        # Calculate max contributions
        contributions = calculate_max_contributions(parsed, N_soft)

        # Store for statistics
        all_contributions.append(contributions)

        # Print case results
        print(f"{test_id:<10} {block_count:<10} {N_soft:<10} "
              f"{contributions['fixed']:<10.1f} {contributions['preplaced']:<10.1f} "
              f"{contributions['grouping']:<10.1f} {contributions['mib']:<10.1f} "
              f"{contributions['boundary']:<10.1f}")

        case_count += 1

        # Show first 20 cases
        if case_count >= 20:
            print("... (showing first 20 cases)")
            break

    # Calculate averages across all cases
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (First 20 Cases)")
    print("="*80)

    if all_contributions:
        avg_contributions = {
            key: sum(c[key] for c in all_contributions) / len(all_contributions)
            for key in ['fixed', 'preplaced', 'grouping', 'mib', 'boundary']
        }

        print(f"\nAverage contribution if completely violated:")
        print(f"  Fixed-shape:  {avg_contributions['fixed']:>6.2f}%")
        print(f"  Preplaced:    {avg_contributions['preplaced']:>6.2f}%")
        print(f"  Grouping:     {avg_contributions['grouping']:>6.2f}%")
        print(f"  MIB:          {avg_contributions['mib']:>6.2f}%")
        print(f"  Boundary:     {avg_contributions['boundary']:>6.2f}%")
        print(f"  Total:        {sum(avg_contributions.values()):>6.2f}%")

        # Verify total is 100%
        total = sum(avg_contributions.values())
        if abs(total - 100.0) > 0.1:
            print(f"\n  WARNING: Total is {total:.2f}%, expected 100%")
        else:
            print(f"\n  [OK] Total verified: {total:.2f}% (correct)")

        # Show distribution
        print(f"\nConstraint importance ranking (by max contribution):")
        sorted_constraints = sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)
        for i, (name, value) in enumerate(sorted_constraints, 1):
            bar_length = int(value / 2)  # Scale to fit
            bar = "█" * bar_length
            print(f"  {i}. {name.capitalize():<12} {value:>6.2f}% {bar}")

        # Analysis
        print(f"\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print("\nThese percentages show the MAXIMUM possible contribution of each")
        print("constraint type to V_rel if that type is COMPLETELY violated.")
        print("\nKey insights:")

        max_constraint = max(avg_contributions.items(), key=lambda x: x[1])
        print(f"  - {max_constraint[0].capitalize()} has the highest weight ({max_constraint[1]:.1f}%)")
        print(f"    -> Optimizing this constraint will have the biggest impact on V_rel")

        min_constraint = min(avg_contributions.items(), key=lambda x: x[1])
        print(f"  - {min_constraint[0].capitalize()} has the lowest weight ({min_constraint[1]:.1f}%)")
        print(f"    -> Even if completely violated, impact is limited")

        print(f"\nTo reduce V_rel, prioritize optimizing constraints in this order:")
        for i, (name, value) in enumerate(sorted_constraints, 1):
            print(f"  {i}. {name.capitalize()} (max impact: {value:.1f}%)")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
