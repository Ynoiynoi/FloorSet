#!/usr/bin/env python3
"""
Compare constraint violations between optimizers.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import get_validation_dataloader
from my_optimizer import MyOptimizer as BaseOptimizer
from my_optimizer_hierarchical import MyOptimizer as HierarchicalOptimizer


def analyze_violations(solution, constraints, block_count):
    """Analyze constraint violations."""
    from collections import defaultdict

    # Parse grouping
    grouping_groups = defaultdict(list)
    for i in range(block_count):
        group_id = int(constraints[i, 3])
        if group_id > 0:
            grouping_groups[group_id].append(i)

    # Check grouping violations
    grouping_violations = 0
    for group_id, blocks in grouping_groups.items():
        if len(blocks) <= 1:
            continue

        # Count connected components (simplified: check if all adjacent)
        # For now, just check if they're close together
        positions = [solution[b] for b in blocks]

        # Check if all blocks are within a bounding box
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] + p[2] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] + p[3] for p in positions)

        bbox_area = (max_x - min_x) * (max_y - min_y)
        total_area = sum(p[2] * p[3] for p in positions)

        # If bbox is much larger than total area, blocks are scattered
        if bbox_area > total_area * 1.5:
            grouping_violations += 1

    # Check boundary violations
    boundary_violations = 0
    if block_count > 0:
        min_x = min(p[0] for p in solution)
        max_x = max(p[0] + p[2] for p in solution)
        min_y = min(p[1] for p in solution)
        max_y = max(p[1] + p[3] for p in solution)

        for i in range(block_count):
            boundary_type = int(constraints[i, 4])
            if boundary_type == 0:
                continue

            x, y, w, h = solution[i]
            tolerance = 1e-3

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
            elif boundary_type in [5, 6, 9, 10]:
                # Corner constraints
                if boundary_type == 5 and touches_left and touches_top:
                    satisfied = True
                elif boundary_type == 6 and touches_right and touches_top:
                    satisfied = True
                elif boundary_type == 9 and touches_left and touches_bottom:
                    satisfied = True
                elif boundary_type == 10 and touches_right and touches_bottom:
                    satisfied = True

            if not satisfied:
                boundary_violations += 1

    return grouping_violations, boundary_violations


def main():
    print("="*80)
    print("CONSTRAINT VIOLATION COMPARISON")
    print("="*80)

    # Load data
    val_loader = get_validation_dataloader()

    # Create optimizers
    base_opt = BaseOptimizer(verbose=False)
    hier_opt = HierarchicalOptimizer(verbose=False)

    print("\nTesting on first 5 cases...")
    print(f"{'Test':<6} {'Method':<15} {'Group_viol':<12} {'Bound_viol':<12} {'Total':<10}")
    print("-"*60)

    case_count = 0
    for batch in val_loader:
        if case_count >= 5:
            break

        data_list = batch[0]
        area_target = data_list[0]
        b2b_connectivity = data_list[1]
        p2b_connectivity = data_list[2]
        pins_pos = data_list[3]
        placement_constraints = data_list[4]

        block_count = area_target.shape[1]
        constraints = placement_constraints[0]

        # Test base optimizer
        base_solution = base_opt.solve(
            block_count=block_count,
            area_targets=area_target[0],
            b2b_connectivity=b2b_connectivity[0],
            p2b_connectivity=p2b_connectivity[0],
            pins_pos=pins_pos[0],
            constraints=constraints,
            target_positions=None
        )

        base_group_viol, base_bound_viol = analyze_violations(base_solution, constraints, block_count)

        # Test hierarchical optimizer
        hier_solution = hier_opt.solve(
            block_count=block_count,
            area_targets=area_target[0],
            b2b_connectivity=b2b_connectivity[0],
            p2b_connectivity=p2b_connectivity[0],
            pins_pos=pins_pos[0],
            constraints=constraints,
            target_positions=None
        )

        hier_group_viol, hier_bound_viol = analyze_violations(hier_solution, constraints, block_count)

        # Print results
        print(f"{case_count:<6} {'Base':<15} {base_group_viol:<12} {base_bound_viol:<12} {base_group_viol + base_bound_viol:<10}")
        print(f"{'':<6} {'Hierarchical':<15} {hier_group_viol:<12} {hier_bound_viol:<12} {hier_group_viol + hier_bound_viol:<10}")
        print()

        case_count += 1

    print("="*80)


if __name__ == '__main__':
    main()
