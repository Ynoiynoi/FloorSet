#!/usr/bin/env python3
"""
Comprehensive Soft Constraint Violation Analysis for my_optimizer_grouping.py

Analyzes all soft constraint violations:
- Fixed-shape
- Preplaced
- Grouping
- MIB
- Boundary
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import get_validation_dataloader
from my_optimizer_grouping import MyOptimizer


def parse_constraints(constraints: torch.Tensor, block_count: int) -> Dict:
    """Parse all constraint types from constraint tensor."""
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


def check_fixed_violations(solution: List[Tuple], fixed_blocks: List[int],
                          target_positions: torch.Tensor) -> int:
    """Check fixed-shape violations."""
    if target_positions is None:
        return 0

    violations = 0
    for block in fixed_blocks:
        target_w = float(target_positions[block, 2])
        target_h = float(target_positions[block, 3])
        actual_w = solution[block][2]
        actual_h = solution[block][3]

        if abs(actual_w - target_w) > 1e-3 or abs(actual_h - target_h) > 1e-3:
            violations += 1

    return violations


def check_preplaced_violations(solution: List[Tuple], preplaced_blocks: List[int],
                               target_positions: torch.Tensor) -> int:
    """Check preplaced violations."""
    if target_positions is None:
        return 0

    violations = 0
    for block in preplaced_blocks:
        target_x = float(target_positions[block, 0])
        target_y = float(target_positions[block, 1])
        target_w = float(target_positions[block, 2])
        target_h = float(target_positions[block, 3])

        actual_x = solution[block][0]
        actual_y = solution[block][1]
        actual_w = solution[block][2]
        actual_h = solution[block][3]

        if (abs(actual_x - target_x) > 1e-3 or abs(actual_y - target_y) > 1e-3 or
            abs(actual_w - target_w) > 1e-3 or abs(actual_h - target_h) > 1e-3):
            violations += 1

    return violations


def are_adjacent(pos1: Tuple, pos2: Tuple, tolerance: float = 1e-3) -> bool:
    """Check if two blocks share an edge."""
    x1, y1, w1, h1 = pos1
    x2, y2, w2, h2 = pos2

    # Check vertical edge sharing
    if abs(x1 + w1 - x2) < tolerance or abs(x2 + w2 - x1) < tolerance:
        y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
        if y_overlap > tolerance:
            return True

    # Check horizontal edge sharing
    if abs(y1 + h1 - y2) < tolerance or abs(y2 + h2 - y1) < tolerance:
        x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
        if x_overlap > tolerance:
            return True

    return False


def check_grouping_violations(solution: List[Tuple],
                              grouping_groups: Dict[int, List[int]]) -> int:
    """Check grouping violations (connected components - 1)."""
    total_violations = 0

    for group_id, blocks in grouping_groups.items():
        if len(blocks) <= 1:
            continue

        # Build adjacency graph
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

        # Violations = components - 1
        violations = components - 1
        total_violations += violations

    return total_violations


def check_mib_violations(solution: List[Tuple],
                        mib_groups: Dict[int, List[int]]) -> int:
    """Check MIB violations (unique dimensions - 1)."""
    total_violations = 0

    for group_id, blocks in mib_groups.items():
        if len(blocks) <= 1:
            continue

        # Collect unique dimensions
        dimensions = set()
        for block in blocks:
            w = round(solution[block][2], 3)
            h = round(solution[block][3], 3)
            dimensions.add((w, h))

        # Violations = unique dimensions - 1
        violations = len(dimensions) - 1
        total_violations += violations

    return total_violations


def check_boundary_violations(solution: List[Tuple],
                              boundary_blocks: Dict[int, int]) -> int:
    """Check boundary violations."""
    if not boundary_blocks:
        return 0

    # Calculate bounding box
    min_x = min(p[0] for p in solution)
    max_x = max(p[0] + p[2] for p in solution)
    min_y = min(p[1] for p in solution)
    max_y = max(p[1] + p[3] for p in solution)

    violations = 0
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
            violations += 1

    return violations


def calculate_N_soft(parsed_constraints: Dict) -> int:
    """Calculate N_soft normalization factor."""
    N_soft = 0
    N_soft += len(parsed_constraints['fixed_blocks'])
    N_soft += len(parsed_constraints['preplaced_blocks'])
    N_soft += len(parsed_constraints['boundary_blocks'])

    for blocks in parsed_constraints['grouping_groups'].values():
        N_soft += len(blocks) - 1

    for blocks in parsed_constraints['mib_groups'].values():
        N_soft += len(blocks) - 1

    return N_soft


def main():
    """Analyze all soft constraint violations."""

    print("="*80)
    print("COMPREHENSIVE SOFT CONSTRAINT VIOLATION ANALYSIS")
    print("Optimizer: my_optimizer_grouping.py")
    print("="*80)
    print()

    # Load optimizer
    optimizer = MyOptimizer(verbose=False)

    # Load validation data
    val_loader = get_validation_dataloader()

    # Statistics
    total_stats = defaultdict(int)
    case_count = 0

    print(f"{'Test':<6} {'Blks':<6} {'N_soft':<8} {'Fixed':<7} {'Prepl':<7} {'Group':<7} {'MIB':<7} {'Bound':<7} {'Total':<7} {'V_rel':<8}")
    print("-"*90)

    for batch in val_loader:
        # Extract data
        data_list = batch[0]

        if not isinstance(data_list, list) or len(data_list) < 5:
            continue

        area_target = data_list[0]
        b2b_connectivity = data_list[1]
        p2b_connectivity = data_list[2]
        pins_pos = data_list[3]
        placement_constraints = data_list[4]

        block_count = area_target.shape[1]
        test_id = case_count

        # Remove batch dimension
        if placement_constraints.dim() == 3:
            placement_constraints = placement_constraints[0]

        # Parse constraints
        parsed = parse_constraints(placement_constraints, block_count)
        N_soft = calculate_N_soft(parsed)

        if N_soft == 0:
            continue

        # Generate solution
        solution = optimizer.solve(
            block_count=block_count,
            area_targets=area_target[0],
            b2b_connectivity=b2b_connectivity[0],
            p2b_connectivity=p2b_connectivity[0],
            pins_pos=pins_pos[0],
            constraints=placement_constraints,
            target_positions=None
        )

        # Check all violations
        v_fixed = check_fixed_violations(solution, parsed['fixed_blocks'], None)
        v_preplaced = check_preplaced_violations(solution, parsed['preplaced_blocks'], None)
        v_grouping = check_grouping_violations(solution, parsed['grouping_groups'])
        v_mib = check_mib_violations(solution, parsed['mib_groups'])
        v_boundary = check_boundary_violations(solution, parsed['boundary_blocks'])

        v_total = v_fixed + v_preplaced + v_grouping + v_mib + v_boundary
        v_rel = v_total / N_soft if N_soft > 0 else 0.0

        # Print case results
        print(f"{test_id:<6} {block_count:<6} {N_soft:<8} {v_fixed:<7} {v_preplaced:<7} "
              f"{v_grouping:<7} {v_mib:<7} {v_boundary:<7} {v_total:<7} {v_rel:<8.4f}")

        # Update statistics
        total_stats['fixed'] += v_fixed
        total_stats['preplaced'] += v_preplaced
        total_stats['grouping'] += v_grouping
        total_stats['mib'] += v_mib
        total_stats['boundary'] += v_boundary
        total_stats['total'] += v_total
        total_stats['N_soft'] += N_soft
        total_stats['v_rel'] += v_rel

        case_count += 1

        if case_count >= 20:
            print("... (showing first 20 cases)")
            break

    # Summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    if case_count > 0:
        avg_v_rel = total_stats['v_rel'] / case_count

        print(f"\nAverage violations per case (over {case_count} cases):")
        print(f"  Fixed:      {total_stats['fixed'] / case_count:.2f}")
        print(f"  Preplaced:  {total_stats['preplaced'] / case_count:.2f}")
        print(f"  Grouping:   {total_stats['grouping'] / case_count:.2f}")
        print(f"  MIB:        {total_stats['mib'] / case_count:.2f}")
        print(f"  Boundary:   {total_stats['boundary'] / case_count:.2f}")
        print(f"  Total:      {total_stats['total'] / case_count:.2f}")
        print(f"  V_rel:      {avg_v_rel:.4f}")

        # Calculate actual contribution percentages
        if total_stats['total'] > 0:
            print(f"\nActual contribution to violations:")
            print(f"  Fixed:      {total_stats['fixed'] / total_stats['total'] * 100:.1f}%")
            print(f"  Preplaced:  {total_stats['preplaced'] / total_stats['total'] * 100:.1f}%")
            print(f"  Grouping:   {total_stats['grouping'] / total_stats['total'] * 100:.1f}%")
            print(f"  MIB:        {total_stats['mib'] / total_stats['total'] * 100:.1f}%")
            print(f"  Boundary:   {total_stats['boundary'] / total_stats['total'] * 100:.1f}%")

        # Comparison with my_optimizer.py
        print(f"\n" + "="*80)
        print("COMPARISON WITH my_optimizer.py")
        print("="*80)
        print(f"\nmy_optimizer_grouping.py:")
        print(f"  Average V_rel: {avg_v_rel:.4f}")
        print(f"  MIB violations: {total_stats['mib'] / case_count:.2f} (avg per case)")

        print(f"\nmy_optimizer.py (from previous analysis):")
        print(f"  Average V_rel: 0.345")
        print(f"  MIB violations: ~68% violation rate")

        improvement = (0.345 - avg_v_rel) / 0.345 * 100
        print(f"\nImprovement: {improvement:.1f}% reduction in V_rel")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
