#!/usr/bin/env python3
"""
Soft Constraint Violation Analysis Tool

Analyzes the contribution of each soft constraint type to the total violation score.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import get_validation_dataloader


def analyze_soft_constraints(solution: List[Tuple[float, float, float, float]],
                             constraints: torch.Tensor,
                             target_positions: torch.Tensor) -> Dict[str, float]:
    """
    Analyze soft constraint violations.

    Returns:
        Dictionary with violation counts for each constraint type
    """
    block_count = len(solution)
    violations = {
        'fixed': 0,
        'preplaced': 0,
        'grouping': 0,
        'mib': 0,
        'boundary': 0,
        'total': 0,
        'N_soft': 0
    }

    if constraints is None or block_count == 0:
        return violations

    # Parse constraints
    fixed_blocks = []
    preplaced_blocks = []
    grouping_groups = defaultdict(list)
    mib_groups = defaultdict(list)
    boundary_blocks = {}

    for i in range(block_count):
        # Fixed shape (column 0)
        if constraints[i, 0] == 1:
            fixed_blocks.append(i)

        # Preplaced (column 1)
        if constraints[i, 1] == 1:
            preplaced_blocks.append(i)

        # MIB (column 2)
        mib_id = int(constraints[i, 2])
        if mib_id > 0:
            mib_groups[mib_id].append(i)

        # Grouping (column 3)
        group_id = int(constraints[i, 3])
        if group_id > 0:
            grouping_groups[group_id].append(i)

        # Boundary (column 4)
        boundary_type = int(constraints[i, 4])
        if boundary_type > 0:
            boundary_blocks[i] = boundary_type

    # Calculate N_soft (normalization factor)
    N_soft = len(fixed_blocks) + len(preplaced_blocks) + len(boundary_blocks)
    for group_id, blocks in grouping_groups.items():
        N_soft += len(blocks) - 1
    for group_id, blocks in mib_groups.items():
        N_soft += len(blocks) - 1

    violations['N_soft'] = N_soft

    if N_soft == 0:
        return violations

    # Check fixed-shape violations
    V_fixed = 0
    for i in fixed_blocks:
        if target_positions is not None:
            target_w = float(target_positions[i, 2])
            target_h = float(target_positions[i, 3])
            actual_w = solution[i][2]
            actual_h = solution[i][3]

            # Check if dimensions match (with small tolerance)
            if abs(actual_w - target_w) > 1e-3 or abs(actual_h - target_h) > 1e-3:
                V_fixed += 1

    violations['fixed'] = V_fixed

    # Check preplaced violations
    V_preplaced = 0
    for i in preplaced_blocks:
        if target_positions is not None:
            target_x = float(target_positions[i, 0])
            target_y = float(target_positions[i, 1])
            target_w = float(target_positions[i, 2])
            target_h = float(target_positions[i, 3])

            actual_x = solution[i][0]
            actual_y = solution[i][1]
            actual_w = solution[i][2]
            actual_h = solution[i][3]

            # Check if position and dimensions match
            if (abs(actual_x - target_x) > 1e-3 or abs(actual_y - target_y) > 1e-3 or
                abs(actual_w - target_w) > 1e-3 or abs(actual_h - target_h) > 1e-3):
                V_preplaced += 1

    violations['preplaced'] = V_preplaced

    # Check grouping violations
    V_grouping = 0
    for group_id, blocks in grouping_groups.items():
        if len(blocks) <= 1:
            continue

        # Count connected components
        # Two blocks are connected if they share an edge
        adjacency = defaultdict(set)
        for i, b1 in enumerate(blocks):
            for b2 in blocks[i+1:]:
                if are_adjacent(solution[b1], solution[b2]):
                    adjacency[b1].add(b2)
                    adjacency[b2].add(b1)

        # Count connected components using DFS
        visited = set()
        components = 0

        for block in blocks:
            if block not in visited:
                components += 1
                # DFS
                stack = [block]
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)

        # Violation = components - 1
        V_grouping += components - 1

    violations['grouping'] = V_grouping

    # Check MIB violations
    V_mib = 0
    for group_id, blocks in mib_groups.items():
        if len(blocks) <= 1:
            continue

        # Count unique dimensions
        dimensions = set()
        for b in blocks:
            w = round(solution[b][2], 3)
            h = round(solution[b][3], 3)
            dimensions.add((w, h))

        # Violation = unique dimensions - 1
        V_mib += len(dimensions) - 1

    violations['mib'] = V_mib

    # Check boundary violations
    V_boundary = 0
    if boundary_blocks:
        # Calculate bounding box
        min_x = min(p[0] for p in solution)
        max_x = max(p[0] + p[2] for p in solution)
        min_y = min(p[1] for p in solution)
        max_y = max(p[1] + p[3] for p in solution)

        tolerance = 1e-3

        for block, boundary_type in boundary_blocks.items():
            x, y, w, h = solution[block]

            touches_left = abs(x - min_x) < tolerance
            touches_right = abs(x + w - max_x) < tolerance
            touches_top = abs(y + h - max_y) < tolerance
            touches_bottom = abs(y - min_y) < tolerance

            satisfied = False
            if boundary_type == 1 and touches_left:
                satisfied = True
            elif boundary_type == 2 and touches_right:
                satisfied = True
            elif boundary_type == 4 and touches_top:
                satisfied = True
            elif boundary_type == 8 and touches_bottom:
                satisfied = True
            elif boundary_type == 5 and touches_left and touches_top:
                satisfied = True
            elif boundary_type == 6 and touches_right and touches_top:
                satisfied = True
            elif boundary_type == 9 and touches_left and touches_bottom:
                satisfied = True
            elif boundary_type == 10 and touches_right and touches_bottom:
                satisfied = True

            if not satisfied:
                V_boundary += 1

    violations['boundary'] = V_boundary

    # Calculate total
    violations['total'] = V_fixed + V_preplaced + V_grouping + V_mib + V_boundary

    return violations


def are_adjacent(pos1: Tuple[float, float, float, float],
                pos2: Tuple[float, float, float, float],
                tolerance: float = 1e-3) -> bool:
    """Check if two blocks share an edge."""
    x1, y1, w1, h1 = pos1
    x2, y2, w2, h2 = pos2

    # Check if they share a vertical edge
    if abs(x1 + w1 - x2) < tolerance or abs(x2 + w2 - x1) < tolerance:
        # Check y overlap
        y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
        if y_overlap > tolerance:
            return True

    # Check if they share a horizontal edge
    if abs(y1 + h1 - y2) < tolerance or abs(y2 + h2 - y1) < tolerance:
        # Check x overlap
        x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
        if x_overlap > tolerance:
            return True

    return False


def main():
    """Analyze soft constraint violations for all validation cases."""

    print("="*80)
    print("SOFT CONSTRAINT VIOLATION ANALYSIS")
    print("="*80)

    # Load optimizer
    from my_optimizer import MyOptimizer
    optimizer = MyOptimizer(verbose=False)

    # Load validation data
    val_loader = get_validation_dataloader()

    # Statistics
    total_stats = defaultdict(float)
    case_count = 0

    print("\nAnalyzing validation cases...")
    print(f"{'Test ID':<10} {'Fixed':<10} {'Preplaced':<12} {'Grouping':<12} {'MIB':<10} {'Boundary':<12} {'Total':<10} {'V_rel':<10}")
    print("-"*100)

    for batch in val_loader:
        # batch is a tuple: (data_list, labels)
        data_list = batch[0]

        # data_list is a list of tensors/values
        # Extract data based on the structure
        if isinstance(data_list, list) and len(data_list) >= 5:
            # Unpack the data
            test_id_tensor = data_list[0]
            block_count_tensor = data_list[1]
            area_target = data_list[2]
            b2b_connectivity = data_list[3]
            p2b_connectivity = data_list[4]
            pins_pos = data_list[5] if len(data_list) > 5 else torch.tensor([])
            placement_constraints = data_list[6] if len(data_list) > 6 else None
            target_positions = data_list[7] if len(data_list) > 7 else None

            test_id = int(test_id_tensor[0]) if test_id_tensor.numel() > 0 else case_count
            block_count = int(block_count_tensor[0]) if block_count_tensor.numel() > 0 else len(area_target)
        else:
            print(f"Skipping batch with unexpected format")
            continue

        # Solve
        solution = optimizer.solve(
            block_count=block_count,
            area_targets=area_target,
            b2b_connectivity=b2b_connectivity,
            p2b_connectivity=p2b_connectivity,
            pins_pos=pins_pos,
            constraints=placement_constraints,
            target_positions=target_positions
        )

        # Analyze violations
        violations = analyze_soft_constraints(
            solution,
            placement_constraints,
            target_positions
        )

        # Calculate V_rel
        N_soft = violations['N_soft']
        if N_soft > 0:
            V_rel = violations['total'] / N_soft
        else:
            V_rel = 0.0

        # Print case results
        print(f"{test_id:<10} {violations['fixed']:<10} {violations['preplaced']:<12} "
              f"{violations['grouping']:<12} {violations['mib']:<10} {violations['boundary']:<12} "
              f"{violations['total']:<10} {V_rel:<10.4f}")

        # Accumulate statistics
        for key in ['fixed', 'preplaced', 'grouping', 'mib', 'boundary', 'total', 'N_soft']:
            total_stats[key] += violations[key]
        total_stats['V_rel'] += V_rel
        case_count += 1

        # Only show first 20 cases
        if case_count >= 20:
            print("... (showing first 20 cases)")
            break

    # Calculate averages
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    avg_stats = {key: value / case_count for key, value in total_stats.items()}

    print(f"\nAverage violations per case (over {case_count} cases):")
    print(f"  Fixed:      {avg_stats['fixed']:.2f}")
    print(f"  Preplaced:  {avg_stats['preplaced']:.2f}")
    print(f"  Grouping:   {avg_stats['grouping']:.2f}")
    print(f"  MIB:        {avg_stats['mib']:.2f}")
    print(f"  Boundary:   {avg_stats['boundary']:.2f}")
    print(f"  Total:      {avg_stats['total']:.2f}")
    print(f"  N_soft:     {avg_stats['N_soft']:.2f}")
    print(f"  V_rel:      {avg_stats['V_rel']:.4f}")

    # Calculate contribution percentages
    print(f"\nContribution to V_rel (%):")
    if avg_stats['total'] > 0:
        print(f"  Fixed:      {avg_stats['fixed'] / avg_stats['total'] * 100:.1f}%")
        print(f"  Preplaced:  {avg_stats['preplaced'] / avg_stats['total'] * 100:.1f}%")
        print(f"  Grouping:   {avg_stats['grouping'] / avg_stats['total'] * 100:.1f}%")
        print(f"  MIB:        {avg_stats['mib'] / avg_stats['total'] * 100:.1f}%")
        print(f"  Boundary:   {avg_stats['boundary'] / avg_stats['total'] * 100:.1f}%")
    else:
        print("  No violations!")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
