#!/usr/bin/env python3
"""
Compare different optimization methods on the same test cases.
"""

import json
import time
from pathlib import Path

from iccad2026_evaluate import get_validation_dataloader
from my_optimizer import MyOptimizer as BStarOptimizer
from my_optimizer_v2 import MyOptimizer as SequencePairOptimizer


def test_optimizer(optimizer, name, test_ids):
    """Test an optimizer on given test cases."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    val_loader = get_validation_dataloader()
    results = []

    for batch in val_loader:
        # batch is a list with one element (the data dict)
        if isinstance(batch, list):
            data = batch[0]
        else:
            data = batch

        # Extract test_id - it might be in different formats
        if 'test_id' in data:
            test_id = int(data['test_id'])
        else:
            # Skip if we can't find test_id
            continue

        if test_id not in test_ids:
            continue

        block_count = int(data['block_count'])
        print(f"\nTest {test_id} ({block_count} blocks)...", end=' ')

        start = time.time()
        try:
            solution = optimizer.solve(
                block_count=block_count,
                area_targets=data['area_target'],
                b2b_connectivity=data['b2b_connectivity'],
                p2b_connectivity=data['p2b_connectivity'],
                pins_pos=data['pins_pos'],
                constraints=data['placement_constraints'],
                target_positions=data.get('target_positions')
            )
            runtime = time.time() - start

            # Quick feasibility check
            from iccad2026_evaluate import check_overlap
            has_overlap = check_overlap(solution)

            status = "FAIL" if has_overlap else "PASS"
            print(f"{status} ({runtime:.2f}s)")

            results.append({
                'test_id': test_id,
                'block_count': block_count,
                'runtime': runtime,
                'feasible': not has_overlap
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'test_id': test_id,
                'block_count': block_count,
                'runtime': 0,
                'feasible': False,
                'error': str(e)
            })

    return results


def main():
    # Test cases to compare
    test_ids = [0, 5, 10, 15, 20]

    print("Comparing optimization methods on test cases:", test_ids)

    # Test B*-tree + SA
    bstar_opt = BStarOptimizer(verbose=False)
    bstar_results = test_optimizer(bstar_opt, "B*-tree + Simulated Annealing", test_ids)

    # Test Sequence Pair + Tabu Search
    sp_opt = SequencePairOptimizer(verbose=False)
    sp_results = test_optimizer(sp_opt, "Sequence Pair + Tabu Search", test_ids)

    # Print comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Test ID':<10} {'Blocks':<10} {'B*-tree Time':<15} {'SeqPair Time':<15} {'Winner':<10}")
    print(f"{'-'*80}")

    for i, test_id in enumerate(test_ids):
        if i < len(bstar_results) and i < len(sp_results):
            b_result = bstar_results[i]
            s_result = sp_results[i]

            b_time = b_result['runtime']
            s_time = s_result['runtime']

            if b_time < s_time:
                winner = "B*-tree"
            elif s_time < b_time:
                winner = "SeqPair"
            else:
                winner = "Tie"

            print(f"{test_id:<10} {b_result['block_count']:<10} "
                  f"{b_time:<15.2f} {s_time:<15.2f} {winner:<10}")

    # Calculate averages
    b_avg = sum(r['runtime'] for r in bstar_results) / len(bstar_results) if bstar_results else 0
    s_avg = sum(r['runtime'] for r in sp_results) / len(sp_results) if sp_results else 0

    print(f"{'-'*80}")
    print(f"{'Average':<10} {'':<10} {b_avg:<15.2f} {s_avg:<15.2f}")

    # Feasibility check
    b_feasible = sum(1 for r in bstar_results if r['feasible'])
    s_feasible = sum(1 for r in sp_results if r['feasible'])

    print(f"\nFeasibility: B*-tree {b_feasible}/{len(bstar_results)}, "
          f"SeqPair {s_feasible}/{len(sp_results)}")


if __name__ == '__main__':
    main()
