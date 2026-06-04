#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Training Data Test Script (120 blocks only)

测试脚本：使用前100个规模为120的训练数据样本进行测试
忽略运行时间因素，专注于解的质量评估

Usage:
    python test_training_120blocks.py --evaluate my_optimizer.py
    python test_training_120blocks.py --evaluate my_optimizer.py --num-samples 50
    python test_training_120blocks.py --evaluate my_optimizer.py --test-id 0 --verbose
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026contest.iccad2026_evaluate import (
    FloorplanOptimizer,
    SolutionMetrics,
    TestResult,
    EvaluationResult,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
    check_area_tolerance,
    check_dimension_hard_constraints,
    compute_total_score,
    ALPHA,
    BETA,
    M_PENALTY,
)

from liteLoader import FloorplanDatasetLite
from lite_dataset import floorplan_collate

try:
    from shapely.geometry import box
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("WARNING: shapely not installed. Some constraint checks disabled.")


def compute_cost_no_runtime(
    hpwl_gap: float,
    area_gap: float,
    violations_relative: float,
    is_feasible: bool
) -> float:
    """
    计算代价（不考虑运行时间因素）

    Cost = (1 + α·(HPWL_gap + Area_gap)) × exp(β·V_rel)
         = M (10.0) if infeasible
    """
    if not is_feasible:
        return M_PENALTY

    quality_factor = 1 + ALPHA * (max(0, hpwl_gap) + max(0, area_gap))
    violation_factor = math.exp(BETA * violations_relative)

    return quality_factor * violation_factor


def evaluate_solution_no_runtime(
    solution: Dict,
    baseline_metrics: Dict,
    target_constraints: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    target_areas: torch.Tensor,
    target_positions: Optional[List] = None
) -> SolutionMetrics:
    """评估解（不考虑运行时间）"""
    positions = solution['positions']
    runtime = solution.get('runtime', 0.0)
    block_count = len(positions)

    # Calculate HPWL
    hpwl_b2b = calculate_hpwl_b2b(positions, b2b_connectivity)
    hpwl_p2b = calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)
    hpwl_total = hpwl_b2b + hpwl_p2b

    hpwl_baseline = baseline_metrics.get('hpwl_baseline', hpwl_total)
    hpwl_gap = (hpwl_total - hpwl_baseline) / max(hpwl_baseline, 1e-6)

    # Calculate area
    bbox_area = calculate_bbox_area(positions)
    area_baseline = baseline_metrics.get('area_baseline', bbox_area)
    area_gap = (bbox_area - area_baseline) / max(area_baseline, 1e-6)

    # Check hard constraints
    overlap_violations = check_overlap(positions)

    # Build set of blocks to skip for area tolerance check
    # Skip: fixed, preplaced, AND MIB group members
    fixed_or_preplaced = set()
    mib_members = set()
    if target_constraints is not None and len(target_constraints) >= block_count:
        ncols_hc = target_constraints.shape[1]
        for i in range(block_count):
            if (ncols_hc > 0 and target_constraints[i, 0] != 0) or \
               (ncols_hc > 1 and target_constraints[i, 1] != 0):
                fixed_or_preplaced.add(i)
            # Also skip MIB group members from area tolerance check
            if ncols_hc > 2 and target_constraints[i, 2] != 0:
                mib_members.add(i)

    # Combine all blocks to skip
    skip_area_check = fixed_or_preplaced | mib_members

    area_violations = check_area_tolerance(
        positions, target_areas, skip_indices=skip_area_check)
    dimension_violations = check_dimension_hard_constraints(
        positions, target_positions, target_constraints, block_count)
    is_feasible = (overlap_violations == 0 and area_violations == 0
                   and dimension_violations == 0)


    # Soft constraint violations
    fixed_violations = 0
    preplaced_violations = 0
    boundary_violations = 0
    grouping_violations = 0
    mib_violations = 0
    n_soft = 0

    if target_constraints is not None and len(target_constraints) >= block_count:
        constraints_block = target_constraints[:block_count]
        ncols = constraints_block.shape[1]

        fixed_const = constraints_block[:, 0] if ncols > 0 else torch.zeros(block_count)
        preplaced_const = constraints_block[:, 1] if ncols > 1 else torch.zeros(block_count)
        mib_const = constraints_block[:, 2] if ncols > 2 else torch.zeros(block_count)
        clust_const = constraints_block[:, 3] if ncols > 3 else torch.zeros(block_count)
        bound_const = constraints_block[:, 4] if ncols > 4 else torch.zeros(block_count)

        n_boundary = int((bound_const != 0).sum().item())
        n_soft = n_boundary

        n_mib_groups = int(mib_const.max().item()) if mib_const.numel() > 0 else 0
        for g in range(1, n_mib_groups + 1):
            group_size = int((mib_const == g).sum().item())
            n_soft += max(0, group_size - 1)

        n_clust_groups = int(clust_const.max().item()) if clust_const.numel() > 0 else 0
        for g in range(1, n_clust_groups + 1):
            group_size = int((clust_const == g).sum().item())
            n_soft += max(0, group_size - 1)

        if SHAPELY_AVAILABLE:
            pred_polys = [box(x, y, x + w, y + h) for x, y, w, h in positions]

            for g in range(1, n_clust_groups + 1):
                group_indices = torch.where(clust_const == g)[0].tolist()
                group_polys = [pred_polys[i] for i in group_indices]
                union_result = unary_union(group_polys)
                if union_result.geom_type == 'MultiPolygon':
                    grouping_violations += len(union_result.geoms) - 1

        for g in range(1, n_mib_groups + 1):
            group_indices = torch.where(mib_const == g)[0].tolist()
            distinct_shapes = set()
            for i in group_indices:
                bw, bh = round(positions[i][2], 4), round(positions[i][3], 4)
                distinct_shapes.add((bw, bh))
            mib_violations += len(distinct_shapes) - 1

        if n_boundary > 0:
            x_min_bb = min(p[0] for p in positions)
            y_min_bb = min(p[1] for p in positions)
            x_max_bb = max(p[0] + p[2] for p in positions)
            y_max_bb = max(p[1] + p[3] for p in positions)
            eps = 1e-6

            for i in range(block_count):
                code = int(bound_const[i].item())
                if code == 0:
                    continue
                bx, by, bw, bh = positions[i]
                touches = {
                    1: abs(bx - x_min_bb) < eps,
                    2: abs(bx + bw - x_max_bb) < eps,
                    4: abs(by + bh - y_max_bb) < eps,
                    8: abs(by - y_min_bb) < eps,
                }
                if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                    boundary_violations += 1

    total_soft_violations = (boundary_violations + grouping_violations + mib_violations)
    violations_relative = total_soft_violations / max(n_soft, 1)

    # Compute cost WITHOUT runtime factor
    cost = compute_cost_no_runtime(hpwl_gap, area_gap, violations_relative, is_feasible)

    return SolutionMetrics(
        is_feasible=is_feasible,
        overlap_violations=overlap_violations,
        area_violations=area_violations,
        dimension_violations=dimension_violations,
        hpwl_b2b=hpwl_b2b,
        hpwl_p2b=hpwl_p2b,
        hpwl_total=hpwl_total,
        hpwl_baseline=hpwl_baseline,
        hpwl_gap=hpwl_gap,
        bbox_area=bbox_area,
        bbox_area_baseline=area_baseline,
        area_gap=area_gap,
        fixed_violations=fixed_violations,
        preplaced_violations=preplaced_violations,
        boundary_violations=boundary_violations,
        grouping_violations=grouping_violations,
        mib_violations=mib_violations,
        total_soft_violations=total_soft_violations,
        max_possible_violations=n_soft,
        violations_relative=violations_relative,
        runtime_seconds=runtime,
        cost=cost
    )


class Training120BlocksEvaluator:
    """测试评估器：使用前100个规模为120的训练数据"""

    def __init__(self, data_path: str = "../", verbose: bool = True):
        self.data_path = Path(data_path)
        self.verbose = verbose
        self.dataset = None
        self.indices_120 = []

    def _load_dataset(self, num_samples: int = 100):
        """加载训练数据集并筛选出规模为120的样本"""
        if self.dataset is None:
            if self.verbose:
                print("Loading training dataset...")
            self.dataset = FloorplanDatasetLite(str(self.data_path))
            if self.verbose:
                print(f"Total training samples: {len(self.dataset):,}")

        # 筛选规模为120的样本
        if not self.indices_120:
            if self.verbose:
                print(f"\nScanning for samples with exactly 120 blocks...")

            # 扫描数据集找到规模为120的样本
            for idx in tqdm(range(len(self.dataset)), desc="Scanning", disable=not self.verbose):
                sample = self.dataset[idx]
                inputs, labels = sample['input'], sample['label']
                area_target = inputs[0]
                block_count = int((area_target != -1).sum().item())

                if block_count == 120:
                    self.indices_120.append(idx)
                    if len(self.indices_120) >= num_samples:
                        break

            if self.verbose:
                print(f"Found {len(self.indices_120)} samples with 120 blocks")
                if len(self.indices_120) < num_samples:
                    print(f"WARNING: Only found {len(self.indices_120)} samples, requested {num_samples}")

    def _load_optimizer(self, optimizer_path: str) -> FloorplanOptimizer:
        """从文件加载优化器"""
        path = Path(optimizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Optimizer file not found: {optimizer_path}")

        spec = importlib.util.spec_from_file_location("optimizer_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and
                issubclass(obj, FloorplanOptimizer) and
                obj.__name__ != 'FloorplanOptimizer'):
                return obj(verbose=self.verbose)

        for name in ['MyOptimizer', 'Optimizer', 'ContestOptimizer']:
            if hasattr(module, name):
                return getattr(module, name)(verbose=self.verbose)

        raise ValueError(f"No optimizer class found in {optimizer_path}")

    def _extract_baseline(self, sample_data, block_count):
        """从训练数据中提取基线指标"""
        inputs, labels = sample_data['input'], sample_data['label']
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        tree_sol, fp_sol, metrics = labels

        # 使用训练数据中的metrics作为基线
        # metrics format: [area, num_pins, num_total_nets, num_b2b_nets,
        #                  num_p2b_nets, num_hardconstraints, b2b_wl, p2b_wl]
        baseline_area = float(metrics[0])
        baseline_b2b_wl = float(metrics[6])
        baseline_p2b_wl = float(metrics[7])
        baseline_hpwl = baseline_b2b_wl + baseline_p2b_wl

        # 从fp_sol提取目标位置 (格式: w, h, x, y)
        target_positions = []
        for i in range(block_count):
            w, h, x, y = fp_sol[i]
            target_positions.append((float(x), float(y), float(w), float(h)))


        return {
            'hpwl_baseline': baseline_hpwl,
            'area_baseline': baseline_area
        }, target_positions

    def evaluate(
        self,
        optimizer_path: str,
        num_samples: int = 100,
        test_ids: Optional[List[int]] = None,
        timeout: float = 300.0
    ) -> EvaluationResult:
        """运行完整评估（不考虑运行时间因素）"""
        self._load_dataset(num_samples)
        optimizer = self._load_optimizer(optimizer_path)

        # 确定要测试的样本
        if test_ids is not None:
            # 使用指定的测试ID（相对于120块样本列表）
            test_indices = [self.indices_120[i] for i in test_ids if i < len(self.indices_120)]
        else:
            # 使用所有找到的120块样本
            test_indices = self.indices_120[:num_samples]

        if not test_indices:
            raise ValueError("No samples with 120 blocks found!")

        results = []
        runtimes = []

        iterator = tqdm(enumerate(test_indices), total=len(test_indices),
                       desc="Evaluating (120 blocks, no runtime)") if self.verbose else enumerate(test_indices)

        for test_idx, dataset_idx in iterator:
            try:
                sample = self.dataset[dataset_idx]
                inputs, labels = sample['input'], sample['label']
                area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
                tree_sol, fp_sol, metrics = labels

                block_count = int((area_target != -1).sum().item())

                if block_count != 120:
                    print(f"WARNING: Sample {dataset_idx} has {block_count} blocks, expected 120")
                    continue

                baseline, target_pos = self._extract_baseline(sample, block_count)

                # 构建target_positions张量
                opt_target_pos = torch.full((block_count, 4), -1.0)
                if target_pos is not None and constraints is not None:
                    nc = constraints.shape[1] if constraints.dim() > 1 else 0
                    for i in range(block_count):
                        is_fixed = nc > 0 and constraints[i, 0] != 0
                        is_preplaced = nc > 1 and constraints[i, 1] != 0
                        if is_preplaced:
                            tx, ty, tw, th = target_pos[i]
                            opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
                        elif is_fixed:
                            _, _, tw, th = target_pos[i]
                            opt_target_pos[i, 2] = tw
                            opt_target_pos[i, 3] = th

                # 运行优化器
                start = time.time()
                positions = optimizer.solve(
                    block_count, area_target, b2b_conn, p2b_conn, pins_pos,
                    constraints, opt_target_pos
                )
                runtime = time.time() - start
                runtimes.append(runtime)

                # 评估（不考虑运行时间）
                solution_metrics = evaluate_solution_no_runtime(
                    {'positions': positions, 'runtime': runtime},
                    baseline,
                    constraints,
                    b2b_conn,
                    p2b_conn,
                    pins_pos,
                    area_target,
                    target_pos
                )

                results.append(TestResult(
                    test_id=test_idx,
                    block_count=block_count,
                    is_feasible=solution_metrics.is_feasible,
                    hpwl_gap=solution_metrics.hpwl_gap,
                    area_gap=solution_metrics.area_gap,
                    violations_relative=solution_metrics.violations_relative,
                    runtime_seconds=runtime,
                    cost=solution_metrics.cost,
                    positions=positions
                ))

            except Exception as e:
                if self.verbose:
                    print(f"\nError on sample {dataset_idx}: {e}")
                results.append(TestResult(
                    test_id=test_idx, block_count=120, is_feasible=False,
                    hpwl_gap=0, area_gap=0, violations_relative=1.0,
                    runtime_seconds=0, cost=M_PENALTY, error=str(e)
                ))

        # 计算总分
        costs = [r.cost for r in results]
        blocks = [r.block_count for r in results]
        total_score = compute_total_score(costs, blocks)

        return EvaluationResult(
            submission_name=Path(optimizer_path).stem,
            timestamp=datetime.now().isoformat(),
            total_score=total_score,
            test_results=results,
            summary={
                'num_tests': len(results),
                'num_feasible': sum(1 for r in results if r.is_feasible),
                'avg_cost': sum(costs) / len(costs) if costs else 0,
                'avg_runtime': sum(runtimes) / len(runtimes) if runtimes else 0,
                'min_cost': min(costs) if costs else 0,
                'max_cost': max(costs) if costs else 0,
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="ICCAD 2026 FloorSet - Test on Training Data (120 blocks only, no runtime factor)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--evaluate', '-e', metavar='OPTIMIZER', required=True,
                       help='Evaluate an optimizer on 120-block training samples')
    parser.add_argument('--data-path', '-d', default='../',
                       help='Path to FloorSet data (default: ../)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output file path')
    parser.add_argument('--num-samples', '-n', type=int, default=100,
                       help='Number of 120-block samples to test (default: 100)')
    parser.add_argument('--test-id', '-t', type=int, default=None,
                       help='Specific sample ID (0-based index in 120-block list)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--save-solutions', '-s', action='store_true',
                       help='Save solutions to separate JSON file')

    args = parser.parse_args()

    evaluator = Training120BlocksEvaluator(args.data_path, verbose=True)
    test_ids = [args.test_id] if args.test_id is not None else None

    result = evaluator.evaluate(args.evaluate, args.num_samples, test_ids)

    # 打印摘要
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS (120 blocks, NO RUNTIME): {result.submission_name}")
    print("=" * 70)
    print(f"\nTotal Score: {result.total_score:.4f}")
    print(f"Tests: {result.summary['num_tests']}")
    print(f"Feasible: {result.summary['num_feasible']}")
    print(f"Avg Cost: {result.summary['avg_cost']:.4f}")
    print(f"Min Cost: {result.summary['min_cost']:.4f}")
    print(f"Max Cost: {result.summary['max_cost']:.4f}")
    print(f"Avg Runtime: {result.summary['avg_runtime']:.2f}s (for info only)")

    # 保存结果
    output = args.output or f"{result.submission_name}_results_120blocks.json"
    with open(output, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"\nResults saved to {output}")

    # 保存解（如果请求）
    if args.save_solutions:
        solutions_file = f"{result.submission_name}_solutions_120blocks.json"
        solutions = {
            'submission': result.submission_name,
            'timestamp': result.timestamp,
            'description': 'Solutions for 120-block training samples',
            'solutions': [
                {
                    'test_id': r.test_id,
                    'block_count': r.block_count,
                    'positions': r.positions,
                    'cost': r.cost,
                    'is_feasible': r.is_feasible
                }
                for r in result.test_results if r.positions is not None
            ]
        }
        with open(solutions_file, 'w') as f:
            json.dump(solutions, f, indent=2)
        print(f"Solutions saved to {solutions_file}")


if __name__ == '__main__':
    main()
