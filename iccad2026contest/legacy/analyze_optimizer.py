#!/usr/bin/env python3
"""
Analyze optimizer performance and breakdown cost components.

Usage:
    python analyze_optimizer.py --optimizer my_optimizer.py [--test-ids 0 1 2]
    python analyze_optimizer.py --optimizer my_optimizer.py --all
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import get_validation_dataloader


def calculate_actual_violations(solution, constraints, block_count):
    """Calculate actual violations for each constraint type."""

    # Parse constraints
    fixed_blocks = []
    preplaced_blocks = []
    grouping_groups = defaultdict(list)
    mib_groups = defaultdict(list)
    boundary_blocks = {}

    for i in range(block_count):
        if constraints[i, 0] == 1:
            fixed_blocks.append(i)
        if constraints[i, 1] == 1:
            preplaced_blocks.append(i)

        mib_id = int(constraints[i, 2])
        if mib_id > 0:
            mib_groups[mib_id].append(i)

        group_id = int(constraints[i, 3])
        if group_id > 0:
            grouping_groups[group_id].append(i)

        boundary_type = int(constraints[i, 4])
        if boundary_type > 0:
            boundary_blocks[i] = boundary_type

    # Calculate N_soft
    N_soft = (
        len(fixed_blocks) +
        len(preplaced_blocks) +
        len(boundary_blocks) +
        sum(len(g) - 1 for g in grouping_groups.values()) +
        sum(len(g) - 1 for g in mib_groups.values())
    )

    # Check violations
    V_fixed = 0
    V_preplaced = 0

    # Check Grouping violations
    V_grouping = 0
    for group_id, blocks in grouping_groups.items():
        if len(blocks) <= 1:
            continue
        positions = [solution[b] for b in blocks]
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] + p[2] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] + p[3] for p in positions)
        bbox_area = (max_x - min_x) * (max_y - min_y)
        total_area = sum(p[2] * p[3] for p in positions)
        if bbox_area > total_area * 2.0:
            V_grouping += len(blocks) - 1

    # Check MIB violations
    V_mib = 0
    for group_id, blocks in mib_groups.items():
        if len(blocks) <= 1:
            continue
        dimensions = set()
        for b in blocks:
            w = round(solution[b][2], 3)
            h = round(solution[b][3], 3)
            dimensions.add((w, h))
        V_mib += len(dimensions) - 1

    # Check Boundary violations
    V_boundary = 0
    if block_count > 0:
        min_x = min(p[0] for p in solution)
        max_x = max(p[0] + p[2] for p in solution)
        min_y = min(p[1] for p in solution)
        max_y = max(p[1] + p[3] for p in solution)

        for i, boundary_type in boundary_blocks.items():
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
                if boundary_type == 5 and touches_left and touches_top:
                    satisfied = True
                elif boundary_type == 6 and touches_right and touches_top:
                    satisfied = True
                elif boundary_type == 9 and touches_left and touches_bottom:
                    satisfied = True
                elif boundary_type == 10 and touches_right and touches_bottom:
                    satisfied = True

            if not satisfied:
                V_boundary += 1

    V_total = V_fixed + V_preplaced + V_grouping + V_boundary + V_mib
    V_rel = V_total / N_soft if N_soft > 0 else 0

    return {
        'V_fixed': V_fixed,
        'V_preplaced': V_preplaced,
        'V_grouping': V_grouping,
        'V_mib': V_mib,
        'V_boundary': V_boundary,
        'V_total': V_total,
        'N_soft': N_soft,
        'V_rel': V_rel
    }


def analyze_optimizer(optimizer_path, test_ids=None, use_all=False):
    """Analyze optimizer performance."""

    # Load optimizer
    import importlib.util
    spec = importlib.util.spec_from_file_location("optimizer_module", optimizer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    optimizer_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and name == 'MyOptimizer':
            optimizer_class = obj
            break

    if optimizer_class is None:
        raise ValueError(f"No MyOptimizer class found in {optimizer_path}")

    optimizer = optimizer_class(verbose=False)

    # Load validation data
    val_loader = get_validation_dataloader()

    # Collect results
    all_results = []
    case_count = 0

    print("="*80)
    print(f"OPTIMIZER PERFORMANCE ANALYSIS: {Path(optimizer_path).name}")
    print("="*80)
    print()

    for batch in val_loader:
        if not use_all and test_ids is not None and case_count not in test_ids:
            case_count += 1
            continue

        if not use_all and test_ids is None and case_count >= 20:
            break

        data_list = batch[0]
        area_target = data_list[0]
        b2b_connectivity = data_list[1]
        p2b_connectivity = data_list[2]
        pins_pos = data_list[3]
        placement_constraints = data_list[4]

        block_count = area_target.shape[1]
        constraints = placement_constraints[0]

        # Generate solution
        solution = optimizer.solve(
            block_count=block_count,
            area_targets=area_target[0],
            b2b_connectivity=b2b_connectivity[0],
            p2b_connectivity=p2b_connectivity[0],
            pins_pos=pins_pos[0],
            constraints=constraints,
            target_positions=None
        )

        # Calculate violations
        violations = calculate_actual_violations(solution, constraints, block_count)

        all_results.append({
            'test_id': case_count,
            'block_count': block_count,
            **violations
        })

        case_count += 1

        if use_all and case_count % 20 == 0:
            print(f"Processed {case_count} cases...")

    # Load results from JSON if available
    results_file = Path(optimizer_path).stem + "_results.json"
    hpwl_data = {}
    area_data = {}

    if Path(results_file).exists():
        with open(results_file, 'r') as f:
            results_json = json.load(f)
            for test in results_json['test_results']:
                test_id = test['test_id']
                hpwl_data[test_id] = test.get('hpwl_gap', 0.0)
                area_data[test_id] = test.get('area_gap', 0.0)

    # Print summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nAnalyzed {len(all_results)} test cases")
    print()

    # Calculate averages
    avg_v_fixed = sum(r['V_fixed'] for r in all_results) / len(all_results)
    avg_v_preplaced = sum(r['V_preplaced'] for r in all_results) / len(all_results)
    avg_v_grouping = sum(r['V_grouping'] for r in all_results) / len(all_results)
    avg_v_mib = sum(r['V_mib'] for r in all_results) / len(all_results)
    avg_v_boundary = sum(r['V_boundary'] for r in all_results) / len(all_results)
    avg_v_total = sum(r['V_total'] for r in all_results) / len(all_results)
    avg_v_rel = sum(r['V_rel'] for r in all_results) / len(all_results)

    print("Constraint Violations (Average):")
    print(f"  Fixed:     {avg_v_fixed:.2f}")
    print(f"  Preplaced: {avg_v_preplaced:.2f}")
    print(f"  Grouping:  {avg_v_grouping:.2f}")
    print(f"  MIB:       {avg_v_mib:.2f}")
    print(f"  Boundary:  {avg_v_boundary:.2f}")
    print(f"  Total:     {avg_v_total:.2f}")
    print(f"  V_rel:     {avg_v_rel:.4f}")
    print()

    if avg_v_total > 0:
        print("Violation Contribution:")
        print(f"  Fixed:     {avg_v_fixed / avg_v_total * 100:.1f}%")
        print(f"  Preplaced: {avg_v_preplaced / avg_v_total * 100:.1f}%")
        print(f"  Grouping:  {avg_v_grouping / avg_v_total * 100:.1f}%")
        print(f"  MIB:       {avg_v_mib / avg_v_total * 100:.1f}%")
        print(f"  Boundary:  {avg_v_boundary / avg_v_total * 100:.1f}%")
        print()

    # HPWL and Area analysis
    if hpwl_data and area_data:
        test_ids_analyzed = [r['test_id'] for r in all_results]
        avg_hpwl = sum(hpwl_data.get(tid, 0) for tid in test_ids_analyzed) / len(test_ids_analyzed)
        avg_area = sum(area_data.get(tid, 0) for tid in test_ids_analyzed) / len(test_ids_analyzed)

        print("="*80)
        print("COST BREAKDOWN")
        print("="*80)
        print()
        print(f"  HPWL_gap:  {avg_hpwl:.4f}")
        print(f"  Area_gap:  {avg_area:.4f}")
        print(f"  V_rel:     {avg_v_rel:.4f}")
        print()

        total_components = avg_hpwl + avg_area + avg_v_rel
        print("Cost Component Weights:")
        print(f"  HPWL_gap:  {avg_hpwl / total_components * 100:.1f}%")
        print(f"  Area_gap:  {avg_area / total_components * 100:.1f}%")
        print(f"  V_rel:     {avg_v_rel / total_components * 100:.1f}%")
        print()

    print("="*80)
    print("OPTIMIZATION PRIORITIES")
    print("="*80)
    print()

    if hpwl_data and area_data:
        priorities = [
            ('HPWL', avg_hpwl / total_components * 100),
            ('Area', avg_area / total_components * 100),
            ('Boundary (V_rel)', avg_v_rel / total_components * 100)
        ]
        priorities.sort(key=lambda x: x[1], reverse=True)

        for i, (name, weight) in enumerate(priorities, 1):
            stars = '*' * int(weight / 10)
            print(f"  {i}. {name:<20} {weight:>5.1f}% {stars}")

    print()
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze optimizer performance')
    parser.add_argument('--optimizer', required=True, help='Path to optimizer file')
    parser.add_argument('--test-ids', type=int, nargs='+', help='Specific test IDs to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all 100 test cases')

    args = parser.parse_args()

    analyze_optimizer(args.optimizer, args.test_ids, args.all)


if __name__ == '__main__':
    main()
