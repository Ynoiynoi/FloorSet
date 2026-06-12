#!/usr/bin/env python3
"""
Analyze cost contributions for an optimizer across all validation test cases.

Computes average HPWLgap, Areagap_bbox, and Violationsrelative (along with
its sub-components Vgrouping/Nsoft, Vboundary/Nsoft, Vmib/Nsoft) to understand
which terms dominate the contest cost function.

Uses the contest's own evaluate_solution() infrastructure so the metrics
exactly match what the official evaluator produces.

Usage:
    python analyze_cost_contributions.py my_optimizer.py
    python analyze_cost_contributions.py my_optimizer.py --data-path ../  --output report.json
    python analyze_cost_contributions.py my_optimizer.py --test-id 0   # single-case debug
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# ── Add contest root to path ──────────────────────────────────────────────
CONTEST_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CONTEST_ROOT.parent
for path in [str(CONTEST_ROOT), str(DATA_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from litetestLoader import FloorplanDatasetLiteTest  # noqa: E402
from iccad2026_evaluate import (  # noqa: E402
    ALPHA, BETA, GAMMA, M_PENALTY, AREA_TOLERANCE,
    FloorplanOptimizer,
    ContestEvaluator,
    evaluate_solution,
    compute_cost,
    compute_total_score,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    SolutionMetrics,
    TestResult,
    check_overlap,
    check_area_tolerance,
    check_dimension_hard_constraints,
)
from utils import (  # noqa: E402
    check_fixed_const,
    check_preplaced_const,
    check_boundary_const,
    check_clust_const,
)

try:
    from shapely.geometry import box
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("WARNING: shapely not installed — grouping violations may not be computed.\n"
          "         Install with: pip install shapely>=2.0.0")


# ============================================================================
# Per-case metrics (extended beyond TestResult to include sub-violations)
# ============================================================================
@dataclass
class CaseContribution:
    """Detailed cost-contribution breakdown for a single test case."""
    test_id: int
    block_count: int
    is_feasible: bool

    # ── Cost components ──
    hpwl_total: float
    hpwl_baseline: float
    hpwl_gap: float                        # max(0, (hpwl - baseline) / baseline)

    bbox_area: float
    bbox_area_baseline: float
    area_gap: float                        # max(0, (area - baseline) / baseline)

    # ── Soft violation sub-components ──
    boundary_violations: int               # Vboundary
    grouping_violations: int               # Vgrouping
    mib_violations: int                    # Vmib
    total_soft_violations: int             # Vboundary + Vgrouping + Vmib
    n_soft: int                            # Nsoft  (max possible soft violations)
    violations_relative: float             # total_soft_violations / n_soft

    # ── Per-component normalized violations ──
    v_boundary_over_nsoft: float           # Vboundary / Nsoft
    v_grouping_over_nsoft: float           # Vgrouping / Nsoft
    v_mib_over_nsoft: float                # Vmib / Nsoft

    # ── Derived cost multipliers (for analysis) ──
    quality_factor: float                  # 1 + α·(HPWLgap + Areagap)
    violation_multiplier: float            # exp(β·Vrel)
    cost: float                            # final contest cost

    runtime_seconds: float
    error: Optional[str] = None


# ============================================================================
# Optimizer loading (same logic as ContestEvaluator._load_optimizer)
# ============================================================================
def _load_optimizer(optimizer_path: str, verbose: bool = False) -> FloorplanOptimizer:
    """Load an optimizer class from a .py file."""
    path = Path(optimizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Optimizer file not found: {optimizer_path}")

    spec = importlib.util.spec_from_file_location("optimizer_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Search for any FloorplanOptimizer subclass (by name, not identity)
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type)
                and issubclass(obj, FloorplanOptimizer)
                and obj.__name__ != 'FloorplanOptimizer'):
            return obj(verbose=verbose)

    # Fallback: try common names
    for name in ['MyOptimizer', 'Optimizer', 'ContestOptimizer']:
        if hasattr(module, name):
            return getattr(module, name)(verbose=verbose)

    raise ValueError(f"No optimizer class found in {optimizer_path}")


# ============================================================================
# Baseline extraction (mirrors ContestEvaluator._extract_baseline)
# ============================================================================
def _extract_baseline(sample: dict, block_count: int) -> Tuple[dict, list]:
    """Return {hpwl_baseline, area_baseline} and target_positions list."""
    inputs, labels = sample['input'], sample['label']
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    polygons, metrics = labels

    positions = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            positions.append((float(x_min), float(y_min),
                              float(x_max - x_min), float(y_max - y_min)))
        else:
            positions.append((0, 0, 1, 1))

    hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
    hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
    area = calculate_bbox_area(positions)

    # Prefer stored metrics when valid
    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0:
            area = float(metrics[0])
        if metrics[-2] > 0:
            hpwl_b2b = float(metrics[-2])
        if metrics[-1] >= 0:
            hpwl_p2b = float(metrics[-1])

    return {'hpwl_baseline': hpwl_b2b + hpwl_p2b, 'area_baseline': area}, positions


# ============================================================================
# Soft-constraint sub-violation computation
# ============================================================================
def _compute_soft_violations(
    positions: List[Tuple[float, float, float, float]],
    constraints: torch.Tensor,
    block_count: int,
    target_positions: Optional[List[Tuple[float, float, float, float]]] = None,
) -> Dict[str, int]:
    """
    Compute per-type soft-constraint violations and N_soft.

    Mirrors evaluate_solution() logic exactly so the numbers match the
    contest scorer.  Returns boundary, grouping, mib counts and n_soft.
    """
    boundary_violations = 0
    grouping_violations = 0
    mib_violations = 0
    n_soft = 0

    if constraints is None or len(constraints) < block_count:
        return {
            'boundary_violations': 0,
            'grouping_violations': 0,
            'mib_violations': 0,
            'n_soft': 0,
        }

    constraints_block = constraints[:block_count]
    ncols = constraints_block.shape[1]

    fixed_const = constraints_block[:, 0] if ncols > 0 else torch.zeros(block_count)
    preplaced_const = constraints_block[:, 1] if ncols > 1 else torch.zeros(block_count)
    mib_const = constraints_block[:, 2] if ncols > 2 else torch.zeros(block_count)
    clust_const = constraints_block[:, 3] if ncols > 3 else torch.zeros(block_count)
    bound_const = constraints_block[:, 4] if ncols > 4 else torch.zeros(block_count)

    n_boundary = int((bound_const != 0).sum().item())

    # ── N_soft ──
    n_soft = n_boundary

    # MIB groups
    n_mib_groups = int(mib_const.max().item()) if mib_const.numel() > 0 else 0
    for g in range(1, n_mib_groups + 1):
        group_size = int((mib_const == g).sum().item())
        n_soft += max(0, group_size - 1)

    # Grouping groups
    n_clust_groups = int(clust_const.max().item()) if clust_const.numel() > 0 else 0
    for g in range(1, n_clust_groups + 1):
        group_size = int((clust_const == g).sum().item())
        n_soft += max(0, group_size - 1)

    # ── Actual violation counts ──

    # V_grouping (requires shapely)
    if SHAPELY_AVAILABLE and n_clust_groups > 0:
        pred_polys = [box(x, y, x + w, y + h) for x, y, w, h in positions]
        for g in range(1, n_clust_groups + 1):
            group_indices = torch.where(clust_const == g)[0].tolist()
            group_polys = [pred_polys[i] for i in group_indices]
            union_result = unary_union(group_polys)
            if union_result.geom_type == 'MultiPolygon':
                grouping_violations += len(union_result.geoms) - 1

    # V_mib
    for g in range(1, n_mib_groups + 1):
        group_indices = torch.where(mib_const == g)[0].tolist()
        distinct_shapes = set()
        for i in group_indices:
            bw, bh = round(positions[i][2], 4), round(positions[i][3], 4)
            distinct_shapes.add((bw, bh))
        mib_violations += len(distinct_shapes) - 1

    # V_boundary
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
                1: abs(bx - x_min_bb) < eps,              # left
                2: abs(bx + bw - x_max_bb) < eps,         # right
                4: abs(by + bh - y_max_bb) < eps,         # top
                8: abs(by - y_min_bb) < eps,              # bottom
            }
            if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                boundary_violations += 1

    return {
        'boundary_violations': boundary_violations,
        'grouping_violations': grouping_violations,
        'mib_violations': mib_violations,
        'n_soft': n_soft,
    }


# ============================================================================
# Evaluation with detailed sub-violation breakdown
# ============================================================================
def analyze_optimizer(
    optimizer_path: str,
    data_path: str = "../",
    test_ids: Optional[List[int]] = None,
    verbose: bool = True,
) -> Tuple[List[CaseContribution], dict]:
    """Run optimizer on all (selected) test cases and collect detailed metrics."""

    # Load dataset + optimizer
    dataset = FloorplanDatasetLiteTest(data_path)
    optimizer = _load_optimizer(optimizer_path, verbose=verbose)

    if test_ids is None:
        test_ids = list(range(len(dataset)))

    cases: List[CaseContribution] = []

    iterator = tqdm(test_ids, desc="Analyzing") if verbose else test_ids

    for idx in iterator:
        try:
            sample = dataset[idx]
            inputs, labels = sample['input'], sample['label']
            area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
            block_count = int((area_target != -1).sum().item())

            # ── Baseline + target positions ──
            baseline_dict, target_pos = _extract_baseline(sample, block_count)

            # ── Build optimizer target_positions tensor ──
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

            # ── Run optimizer ──
            start = time.time()
            positions = optimizer.solve(
                block_count,
                area_target,
                b2b_conn,
                p2b_conn,
                pins_pos,
                constraints,
                opt_target_pos,
            )
            runtime = time.time() - start

            # ── Evaluate with full contest infrastructure ──
            metrics = evaluate_solution(
                {'positions': positions, 'runtime': runtime},
                baseline_dict,
                constraints,
                b2b_conn,
                p2b_conn,
                pins_pos,
                area_target,
                target_pos,
                median_runtime=1.0,
            )

            # Extract sub-violation components from the official metrics
            bv = metrics.boundary_violations
            gv = metrics.grouping_violations
            mv = metrics.mib_violations
            tv = metrics.total_soft_violations
            ns = metrics.max_possible_violations
            vrel = metrics.violations_relative

            # Per-component normalized violations
            v_boundary_ns = bv / max(ns, 1)
            v_grouping_ns = gv / max(ns, 1)
            v_mib_ns = mv / max(ns, 1)

            # Derived cost multipliers
            quality_factor = 1.0 + ALPHA * (max(0.0, metrics.hpwl_gap) + max(0.0, metrics.area_gap))
            violation_multiplier = math.exp(BETA * vrel)

            cases.append(CaseContribution(
                test_id=idx,
                block_count=block_count,
                is_feasible=metrics.is_feasible,

                hpwl_total=metrics.hpwl_total,
                hpwl_baseline=metrics.hpwl_baseline,
                hpwl_gap=max(0.0, metrics.hpwl_gap),

                bbox_area=metrics.bbox_area,
                bbox_area_baseline=metrics.bbox_area_baseline,
                area_gap=max(0.0, metrics.area_gap),

                boundary_violations=bv,
                grouping_violations=gv,
                mib_violations=mv,
                total_soft_violations=tv,
                n_soft=ns,
                violations_relative=vrel,

                v_boundary_over_nsoft=v_boundary_ns,
                v_grouping_over_nsoft=v_grouping_ns,
                v_mib_over_nsoft=v_mib_ns,

                quality_factor=quality_factor,
                violation_multiplier=violation_multiplier,
                cost=metrics.cost,

                runtime_seconds=runtime,
            ))

        except Exception as e:
            cases.append(CaseContribution(
                test_id=idx,
                block_count=0,
                is_feasible=False,
                hpwl_total=0, hpwl_baseline=0, hpwl_gap=0,
                bbox_area=0, bbox_area_baseline=0, area_gap=0,
                boundary_violations=0, grouping_violations=0, mib_violations=0,
                total_soft_violations=0, n_soft=1, violations_relative=1.0,
                v_boundary_over_nsoft=0, v_grouping_over_nsoft=0, v_mib_over_nsoft=0,
                quality_factor=0, violation_multiplier=math.exp(BETA),
                cost=M_PENALTY, runtime_seconds=0, error=str(e),
            ))

    # ── Aggregate statistics ──
    summary = _build_summary(cases)

    return cases, summary


# ============================================================================
# Summary aggregation
# ============================================================================
def _build_summary(cases: List[CaseContribution]) -> dict:
    """Compute averages and breakdowns across all test cases."""

    feasible = [c for c in cases if c.is_feasible]
    n_total = len(cases)
    n_feasible = len(feasible)

    # Pool for averaging (use all cases, including infeasible ones)
    pool = cases if n_total > 0 else []

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    # ── Overall averages ──
    avg_hpwl_gap = mean([c.hpwl_gap for c in pool])
    avg_area_gap = mean([c.area_gap for c in pool])
    avg_vrel = mean([c.violations_relative for c in pool])
    avg_v_boundary_ns = mean([c.v_boundary_over_nsoft for c in pool])
    avg_v_grouping_ns = mean([c.v_grouping_over_nsoft for c in pool])
    avg_v_mib_ns = mean([c.v_mib_over_nsoft for c in pool])
    avg_quality_factor = mean([c.quality_factor for c in pool])
    avg_violation_multiplier = mean([c.violation_multiplier for c in pool])
    avg_cost = mean([c.cost for c in pool])

    # ── Feasible-only averages ──
    avg_hpwl_gap_f = mean([c.hpwl_gap for c in feasible])
    avg_area_gap_f = mean([c.area_gap for c in feasible])
    avg_vrel_f = mean([c.violations_relative for c in feasible])
    avg_v_boundary_ns_f = mean([c.v_boundary_over_nsoft for c in feasible])
    avg_v_grouping_ns_f = mean([c.v_grouping_over_nsoft for c in feasible])
    avg_v_mib_ns_f = mean([c.v_mib_over_nsoft for c in feasible])
    avg_cost_f = mean([c.cost for c in feasible])

    # ── Total violation units (for ratio analysis) ──
    total_boundary = sum(c.boundary_violations for c in pool)
    total_grouping = sum(c.grouping_violations for c in pool)
    total_mib = sum(c.mib_violations for c in pool)
    total_soft = sum(c.total_soft_violations for c in pool)
    total_nsoft = sum(c.n_soft for c in pool)

    # ── Block-count weighted averages (mimic contest Total Score weights) ──
    costs_per_case = [c.cost for c in pool]
    blocks_per_case = [c.block_count for c in pool]
    total_score = compute_total_score(costs_per_case, blocks_per_case)

    # ── Size-group breakdown ──
    size_buckets = [(21, 40), (41, 60), (61, 80), (81, 100), (101, 120)]
    size_breakdown = {}
    for lo, hi in size_buckets:
        bucket = [c for c in pool if lo <= c.block_count <= hi]
        if bucket:
            size_breakdown[f"n{lo}-{hi}"] = {
                "count": len(bucket),
                "avg_hpwl_gap": mean([c.hpwl_gap for c in bucket]),
                "avg_area_gap": mean([c.area_gap for c in bucket]),
                "avg_vrel": mean([c.violations_relative for c in bucket]),
                "avg_v_boundary_ns": mean([c.v_boundary_over_nsoft for c in bucket]),
                "avg_v_grouping_ns": mean([c.v_grouping_over_nsoft for c in bucket]),
                "avg_v_mib_ns": mean([c.v_mib_over_nsoft for c in bucket]),
                "avg_cost": mean([c.cost for c in bucket]),
                "num_feasible": sum(1 for c in bucket if c.is_feasible),
            }

    return {
        "num_cases": n_total,
        "num_feasible": n_feasible,
        "num_infeasible": n_total - n_feasible,

        # ── Core averages (all cases) ──
        "avg_hpwl_gap": avg_hpwl_gap,
        "avg_area_gap": avg_area_gap,
        "avg_violations_relative": avg_vrel,
        "avg_Vboundary_Nsoft": avg_v_boundary_ns,
        "avg_Vgrouping_Nsoft": avg_v_grouping_ns,
        "avg_Vmib_Nsoft": avg_v_mib_ns,

        # ── Cost multipliers ──
        "avg_quality_factor": avg_quality_factor,
        "avg_violation_multiplier": avg_violation_multiplier,
        "avg_cost": avg_cost,

        # ── Feasible-only averages ──
        "feasible_only": {
            "avg_hpwl_gap": avg_hpwl_gap_f,
            "avg_area_gap": avg_area_gap_f,
            "avg_violations_relative": avg_vrel_f,
            "avg_Vboundary_Nsoft": avg_v_boundary_ns_f,
            "avg_Vgrouping_Nsoft": avg_v_grouping_ns_f,
            "avg_Vmib_Nsoft": avg_v_mib_ns_f,
            "avg_cost": avg_cost_f,
        },

        # ── Total violation units ──
        "total_violation_units": {
            "Vboundary": total_boundary,
            "Vgrouping": total_grouping,
            "Vmib": total_mib,
            "total_soft": total_soft,
            "total_Nsoft": total_nsoft,
        },

        # ── Violation composition ratios ──
        "violation_composition": {
            "Vboundary_fraction": total_boundary / max(total_soft, 1),
            "Vgrouping_fraction": total_grouping / max(total_soft, 1),
            "Vmib_fraction": total_mib / max(total_soft, 1),
        } if total_soft > 0 else {},

        # ── Contest total score (exponential weighted) ──
        "total_score_contest_weighted": total_score,

        # ── Size-group breakdown ──
        "size_breakdown": size_breakdown,
    }


# ============================================================================
# Report formatting
# ============================================================================
def format_report(optimizer_name: str, summary: dict) -> str:
    """Produce a human-readable report."""

    s = summary
    fo = s.get("feasible_only", {})

    lines = [
        "=" * 72,
        f"  Cost Contribution Analysis: {optimizer_name}",
        "=" * 72,
        "",
        f"  Test cases evaluated : {s['num_cases']}",
        f"  Feasible             : {s['num_feasible']}",
        f"  Infeasible           : {s['num_infeasible']}",
        "",
        "─" * 72,
        "  AVERAGE METRICS (all cases)",
        "─" * 72,
        "",
        f"  HPWLgap               = {s['avg_hpwl_gap']:.6f}",
        f"  Areagap_bbox          = {s['avg_area_gap']:.6f}",
        f"  Violationsrelative    = {s['avg_violations_relative']:.6f}",
        f"    ├─ Vboundary/Nsoft  = {s['avg_Vboundary_Nsoft']:.6f}",
        f"    ├─ Vgrouping/Nsoft  = {s['avg_Vgrouping_Nsoft']:.6f}",
        f"    └─ Vmib/Nsoft       = {s['avg_Vmib_Nsoft']:.6f}",
        "",
        f"  Quality factor        = {s['avg_quality_factor']:.6f}  (1 + α·(HPWLgap + Areagap))",
        f"  Violation multiplier  = {s['avg_violation_multiplier']:.6f}  (exp(β·Vrel))",
        f"  Avg Cost              = {s['avg_cost']:.6f}",
        f"  Contest Total Score   = {s['total_score_contest_weighted']:.6f}  (exp-weighted)",
        "",
    ]

    if s['num_feasible'] > 0:
        lines += [
            "─" * 72,
            "  AVERAGE METRICS (feasible cases only)",
            "─" * 72,
            "",
            f"  HPWLgap               = {fo['avg_hpwl_gap']:.6f}",
            f"  Areagap_bbox          = {fo['avg_area_gap']:.6f}",
            f"  Violationsrelative    = {fo['avg_violations_relative']:.6f}",
            f"    ├─ Vboundary/Nsoft  = {fo['avg_Vboundary_Nsoft']:.6f}",
            f"    ├─ Vgrouping/Nsoft  = {fo['avg_Vgrouping_Nsoft']:.6f}",
            f"    └─ Vmib/Nsoft       = {fo['avg_Vmib_Nsoft']:.6f}",
            f"  Avg Cost              = {fo['avg_cost']:.6f}",
            "",
        ]

    # ── Violation composition ──
    vc = s.get("violation_composition", {})
    if vc:
        lines += [
            "─" * 72,
            "  VIOLATION COMPOSITION (fraction of total soft violations)",
            "─" * 72,
            "",
            f"  Vboundary / total     = {vc.get('Vboundary_fraction', 0):.6f}",
            f"  Vgrouping / total     = {vc.get('Vgrouping_fraction', 0):.6f}",
            f"  Vmib / total          = {vc.get('Vmib_fraction', 0):.6f}",
            "",
        ]

    # ── Total units ──
    tu = s.get("total_violation_units", {})
    lines += [
        "─" * 72,
        "  TOTAL VIOLATION UNITS (summed across all cases)",
        "─" * 72,
        "",
        f"  Σ Vboundary           = {tu.get('Vboundary', 0)}",
        f"  Σ Vgrouping           = {tu.get('Vgrouping', 0)}",
        f"  Σ Vmib                = {tu.get('Vmib', 0)}",
        f"  Σ total_soft          = {tu.get('total_soft', 0)}",
        f"  Σ Nsoft               = {tu.get('total_Nsoft', 0)}",
        "",
    ]

    # ── Size-group breakdown ──
    sb = s.get("size_breakdown", {})
    if sb:
        lines += [
            "─" * 72,
            "  SIZE-GROUP BREAKDOWN",
            "─" * 72,
            "",
        ]
        header = f"  {'Bucket':>10s}  {'N':>3s}  {'Feas':>4s}  {'HPWLgap':>10s}  {'Areagap':>10s}  {'Vrel':>10s}  {'Vbnd/Ns':>10s}  {'Vgrp/Ns':>10s}  {'Vmib/Ns':>10s}  {'Cost':>10s}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for bucket, stats in sb.items():
            lines.append(
                f"  {bucket:>10s}  {stats['count']:3d}  {stats['num_feasible']:4d}"
                f"  {stats['avg_hpwl_gap']:10.4f}  {stats['avg_area_gap']:10.4f}"
                f"  {stats['avg_vrel']:10.4f}  {stats['avg_v_boundary_ns']:10.4f}"
                f"  {stats['avg_v_grouping_ns']:10.4f}  {stats['avg_v_mib_ns']:10.4f}"
                f"  {stats['avg_cost']:10.4f}"
            )
        lines.append("")

    # ── Interpretation ──
    lines += [
        "─" * 72,
        "  INTERPRETATION GUIDE",
        "─" * 72,
        "",
        "  Cost = quality_factor × violation_multiplier × runtime_factor",
        "         = (1 + 0.5·(HPWLgap + Areagap)) × exp(2.0·Vrel) × max(0.7, R^0.3)",
        "",
        "  - HPWLgap > 0  → wirelength is worse than baseline",
        "  - Areagap > 0  → bounding-box area is larger than baseline",
        "  - Vrel > 0     → soft constraints (boundary/grouping/MIB) are violated",
        "  - The exponential exp(2·Vrel) amplifies even small violation ratios",
        "  - Local evaluation uses RuntimeFactor = 1.0 (neutral)",
        "",
        "  Dominant cost driver: the term with the largest average value.",
        "=" * 72,
    ]

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cost contributions for an optimizer across all "
                    "validation test cases. Reports average HPWLgap, Areagap_bbox, "
                    "and Violationsrelative (with sub-components Vgrouping, "
                    "Vboundary, Vmib each divided by Nsoft).",
    )
    parser.add_argument(
        "optimizer",
        help="Path to the optimizer .py file",
    )
    parser.add_argument(
        "--data-path", "-d",
        default="../",
        help="Path to FloorSet data directory (default: ../)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional output file (.json for structured data, .txt for report)",
    )
    parser.add_argument(
        "--test-id", "-t",
        type=int,
        default=None,
        help="Run on a single test case only (for debugging, 0-99)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bar",
    )
    parser.add_argument(
        "--print-cases",
        action="store_true",
        help="Print per-case breakdown table",
    )

    args = parser.parse_args()

    test_ids = [args.test_id] if args.test_id is not None else None
    verbose = not args.quiet

    cases, summary = analyze_optimizer(
        args.optimizer,
        data_path=args.data_path,
        test_ids=test_ids,
        verbose=verbose,
    )

    optimizer_name = Path(args.optimizer).stem
    report = format_report(optimizer_name, summary)
    print("\n" + report)

    # ── Optional per-case table ──
    if args.print_cases and len(cases) > 1:
        print("\nPer-case breakdown:")
        header = f"{'ID':>4s}  {'N':>3s}  {'OK':>3s}  {'HPWLgap':>10s}  {'Areagap':>10s}  {'Vrel':>10s}  {'Vb/Ns':>10s}  {'Vg/Ns':>10s}  {'Vm/Ns':>10s}  {'Cost':>10s}"
        print(header)
        print("-" * len(header))
        for c in cases:
            ok = "Y" if c.is_feasible else "N"
            print(
                f"{c.test_id:4d}  {c.block_count:3d}  {ok:>3s}"
                f"  {c.hpwl_gap:10.4f}  {c.area_gap:10.4f}"
                f"  {c.violations_relative:10.4f}  {c.v_boundary_over_nsoft:10.4f}"
                f"  {c.v_grouping_over_nsoft:10.4f}  {c.v_mib_over_nsoft:10.4f}"
                f"  {c.cost:10.4f}"
            )

    # ── Optional output file ──
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            payload = {
                "optimizer": optimizer_name,
                "summary": summary,
                "cases": [asdict(c) for c in cases],
            }
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"JSON saved to {args.output}")
        else:
            output_path.write_text(report + "\n", encoding="utf-8")
            print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
