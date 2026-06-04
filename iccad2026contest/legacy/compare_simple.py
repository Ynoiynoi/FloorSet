#!/usr/bin/env python3
"""
Simple comparison: run both optimizers and compare results from JSON files.
"""

import subprocess
import json
import time

def run_evaluation(optimizer_file, test_ids):
    """Run evaluation and return results."""
    print(f"\nEvaluating {optimizer_file}...")

    # Build command
    cmd = [
        'python', 'iccad2026_evaluate.py',
        '--evaluate', optimizer_file
    ]
    for tid in test_ids:
        cmd.extend(['--test-id', str(tid)])

    # Run evaluation
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    # Read results from JSON
    result_file = optimizer_file.replace('.py', '_results.json')
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to read results: {e}")
        return None


def main():
    test_ids = [0, 3, 5, 10, 15]

    print("="*80)
    print("COMPARING OPTIMIZATION METHODS")
    print("="*80)
    print(f"Test cases: {test_ids}")

    # Test B*-tree
    print("\n" + "="*80)
    print("METHOD 1: B*-tree + Simulated Annealing")
    print("="*80)
    bstar_results = run_evaluation('my_optimizer.py', test_ids)

    # Test Sequence Pair
    print("\n" + "="*80)
    print("METHOD 2: Sequence Pair + Tabu Search")
    print("="*80)
    sp_results = run_evaluation('my_optimizer_v2.py', test_ids)

    # Compare
    if bstar_results and sp_results:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        print(f"\nB*-tree + SA:")
        print(f"  Total Score: {bstar_results['total_score']:.4f}")
        print(f"  Avg Cost: {bstar_results.get('avg_cost', 'N/A')}")
        print(f"  Avg Runtime: {bstar_results.get('avg_runtime_seconds', 'N/A')}")
        print(f"  Feasible: {bstar_results.get('feasible_count', 'N/A')}/{bstar_results.get('test_count', 'N/A')}")

        print(f"\nSequence Pair + Tabu:")
        print(f"  Total Score: {sp_results['total_score']:.4f}")
        print(f"  Avg Cost: {sp_results.get('avg_cost', 'N/A')}")
        print(f"  Avg Runtime: {sp_results.get('avg_runtime_seconds', 'N/A')}")
        print(f"  Feasible: {sp_results.get('feasible_count', 'N/A')}/{sp_results.get('test_count', 'N/A')}")

        # Determine winner
        b_score = bstar_results['total_score']
        s_score = sp_results['total_score']

        print(f"\n{'='*80}")
        if b_score < s_score:
            print(f"WINNER: B*-tree + SA (score {b_score:.4f} < {s_score:.4f})")
        elif s_score < b_score:
            print(f"WINNER: Sequence Pair + Tabu (score {s_score:.4f} < {b_score:.4f})")
        else:
            print(f"TIE: Both methods scored {b_score:.4f}")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
