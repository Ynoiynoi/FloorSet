#!/usr/bin/env python3
"""
测试训练数据中是否存在同一个 MIB 组内的块有不同面积约束的情况

MIB (Multi-Instantiation Blocks) 约束：
- 同一 MIB 组内的所有块应该有相同的尺寸 (w, h)
- 但是否应该有相同的面积约束？

这个脚本会检查：
1. 同一 MIB 组内的块是否有不同的 area_target
2. 同一 MIB 组内的块在 fp_sol 中是否有不同的尺寸
"""

import sys
from pathlib import Path
import torch
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026contest.iccad2026_evaluate import get_training_dataloader


def test_mib_area_consistency(num_samples=1000, verbose=True):
    """
    测试 MIB 组内的面积约束一致性

    Args:
        num_samples: 要检查的样本数量
        verbose: 是否打印详细信息
    """
    print("="*80)
    print("测试 MIB 组内的面积约束一致性")
    print("="*80)
    print(f"\n加载 {num_samples} 个训练样本...")

    dataloader = get_training_dataloader(
        batch_size=1,
        num_samples=num_samples,
        shuffle=False
    )

    # 统计信息
    total_samples = 0
    samples_with_mib = 0
    total_mib_groups = 0

    # 问题统计
    area_mismatch_samples = []  # 面积约束不一致的样本
    size_mismatch_samples = []  # fp_sol 中尺寸不一致的样本

    print(f"开始检查...\n")

    for sample_idx, batch in enumerate(dataloader):
        # 解包数据
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, \
            tree_sol, fp_sol, metrics = batch

        # 移除批次维度
        area_target = area_target.squeeze(0)  # [N]
        constraints = constraints.squeeze(0)  # [N, 5]
        fp_sol = fp_sol.squeeze(0)            # [N, 4]: [w, h, x, y]

        # 计算有效块数
        block_count = int((area_target != -1).sum().item())
        total_samples += 1

        # 提取 MIB 约束（第2列）
        mib_groups = constraints[:block_count, 2]

        # 检查是否有 MIB 约束
        has_mib = (mib_groups > 0).any().item()
        if not has_mib:
            continue

        samples_with_mib += 1

        # 按 MIB 组分组
        mib_dict = defaultdict(list)
        for i in range(block_count):
            mib_group = int(mib_groups[i].item())
            if mib_group > 0:
                mib_dict[mib_group].append(i)

        total_mib_groups += len(mib_dict)

        # 检查每个 MIB 组
        sample_has_area_mismatch = False
        sample_has_size_mismatch = False

        for group_id, block_indices in mib_dict.items():
            if len(block_indices) < 2:
                continue  # 单个块的组，跳过

            # 检查面积约束
            areas = [area_target[i].item() for i in block_indices]
            unique_areas = set(areas)

            if len(unique_areas) > 1:
                # 发现面积约束不一致
                sample_has_area_mismatch = True
                if verbose:
                    print(f"样本 {sample_idx}: MIB 组 {group_id} 面积约束不一致")
                    for i in block_indices:
                        print(f"  块 {i}: area_target = {area_target[i].item():.2f}")

            # 检查 fp_sol 中的尺寸
            sizes = [(fp_sol[i, 0].item(), fp_sol[i, 1].item()) for i in block_indices]
            unique_sizes = set(sizes)

            if len(unique_sizes) > 1:
                # 发现尺寸不一致
                sample_has_size_mismatch = True
                if verbose:
                    print(f"样本 {sample_idx}: MIB 组 {group_id} fp_sol 尺寸不一致")
                    for i in block_indices:
                        w, h = fp_sol[i, 0].item(), fp_sol[i, 1].item()
                        area = w * h
                        print(f"  块 {i}: w={w:.2f}, h={h:.2f}, area={area:.2f}")

        if sample_has_area_mismatch:
            area_mismatch_samples.append(sample_idx)

        if sample_has_size_mismatch:
            size_mismatch_samples.append(sample_idx)

        # 进度显示
        if (sample_idx + 1) % 100 == 0:
            print(f"已检查 {sample_idx + 1}/{num_samples} 个样本...")

    # 打印统计结果
    print("\n" + "="*80)
    print("统计结果")
    print("="*80)
    print(f"\n总样本数: {total_samples}")
    print(f"包含 MIB 约束的样本数: {samples_with_mib} ({samples_with_mib/total_samples*100:.1f}%)")
    print(f"总 MIB 组数: {total_mib_groups}")

    print(f"\n【面积约束一致性】")
    if len(area_mismatch_samples) == 0:
        print("✅ 所有 MIB 组内的块都有相同的 area_target")
    else:
        print(f"❌ 发现 {len(area_mismatch_samples)} 个样本的 MIB 组内面积约束不一致")
        print(f"   样本索引: {area_mismatch_samples[:10]}")
        if len(area_mismatch_samples) > 10:
            print(f"   ... 还有 {len(area_mismatch_samples) - 10} 个")

    print(f"\n【fp_sol 尺寸一致性】")
    if len(size_mismatch_samples) == 0:
        print("✅ 所有 MIB 组内的块在 fp_sol 中都有相同的尺寸")
    else:
        print(f"❌ 发现 {len(size_mismatch_samples)} 个样本的 MIB 组内 fp_sol 尺寸不一致")
        print(f"   样本索引: {size_mismatch_samples[:10]}")
        if len(size_mismatch_samples) > 10:
            print(f"   ... 还有 {len(size_mismatch_samples) - 10} 个")

    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if len(area_mismatch_samples) == 0 and len(size_mismatch_samples) == 0:
        print("✅ MIB 约束完全一致：")
        print("   - 同一 MIB 组内的块有相同的 area_target")
        print("   - 同一 MIB 组内的块在 fp_sol 中有相同的尺寸")
    elif len(area_mismatch_samples) > 0 and len(size_mismatch_samples) == 0:
        print("⚠️  MIB 约束部分一致：")
        print("   - 同一 MIB 组内的块可能有不同的 area_target")
        print("   - 但在 fp_sol 中有相同的尺寸（这是正确的）")
        print("\n   这可能意味着：")
        print("   1. area_target 是输入约束，可能有误差")
        print("   2. fp_sol 是参考解，MIB 约束在这里被正确满足")
    elif len(area_mismatch_samples) == 0 and len(size_mismatch_samples) > 0:
        print("❌ 数据异常：")
        print("   - 同一 MIB 组内的块有相同的 area_target")
        print("   - 但在 fp_sol 中有不同的尺寸（违反 MIB 约束！）")
    else:
        print("❌ 数据不一致：")
        print("   - 同一 MIB 组内的块有不同的 area_target")
        print("   - 同一 MIB 组内的块在 fp_sol 中也有不同的尺寸")

    print("="*80)

    return {
        'total_samples': total_samples,
        'samples_with_mib': samples_with_mib,
        'total_mib_groups': total_mib_groups,
        'area_mismatch_samples': area_mismatch_samples,
        'size_mismatch_samples': size_mismatch_samples
    }


def detailed_analysis(sample_idx):
    """
    详细分析某个样本的 MIB 约束

    Args:
        sample_idx: 样本索引
    """
    print("\n" + "="*80)
    print(f"详细分析样本 {sample_idx}")
    print("="*80)

    dataloader = get_training_dataloader(
        batch_size=1,
        num_samples=sample_idx + 1,
        shuffle=False
    )

    # 跳到目标样本
    for i, batch in enumerate(dataloader):
        if i < sample_idx:
            continue

        # 解包数据
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, \
            tree_sol, fp_sol, metrics = batch

        # 移除批次维度
        area_target = area_target.squeeze(0)
        constraints = constraints.squeeze(0)
        fp_sol = fp_sol.squeeze(0)

        block_count = int((area_target != -1).sum().item())

        print(f"\n样本信息:")
        print(f"  块数: {block_count}")

        # 提取 MIB 约束
        mib_groups = constraints[:block_count, 2]

        # 按 MIB 组分组
        mib_dict = defaultdict(list)
        for i in range(block_count):
            mib_group = int(mib_groups[i].item())
            if mib_group > 0:
                mib_dict[mib_group].append(i)

        print(f"  MIB 组数: {len(mib_dict)}")

        # 详细显示每个 MIB 组
        for group_id, block_indices in sorted(mib_dict.items()):
            print(f"\n  MIB 组 {group_id} (包含 {len(block_indices)} 个块):")
            print(f"    {'块ID':<6} {'area_target':<15} {'fp_sol (w, h)':<20} {'fp_sol area':<15} {'差异%':<10}")
            print(f"    {'-'*6} {'-'*15} {'-'*20} {'-'*15} {'-'*10}")

            for idx in block_indices:
                area_tgt = area_target[idx].item()
                w = fp_sol[idx, 0].item()
                h = fp_sol[idx, 1].item()
                area_sol = w * h
                diff_pct = abs(area_sol - area_tgt) / area_tgt * 100 if area_tgt > 0 else 0

                print(f"    {idx:<6} {area_tgt:<15.2f} ({w:.2f}, {h:.2f}){'':<8} {area_sol:<15.2f} {diff_pct:<10.2f}")

            # 检查一致性
            areas = [area_target[i].item() for i in block_indices]
            sizes = [(fp_sol[i, 0].item(), fp_sol[i, 1].item()) for i in block_indices]

            unique_areas = set(areas)
            unique_sizes = set(sizes)

            print(f"\n    一致性检查:")
            if len(unique_areas) == 1:
                print(f"      ✅ area_target 一致: {areas[0]:.2f}")
            else:
                print(f"      ❌ area_target 不一致: {unique_areas}")

            if len(unique_sizes) == 1:
                print(f"      ✅ fp_sol 尺寸一致: w={sizes[0][0]:.2f}, h={sizes[0][1]:.2f}")
            else:
                print(f"      ❌ fp_sol 尺寸不一致: {unique_sizes}")

        break


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='测试 MIB 组内面积约束一致性')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='要检查的样本数量 (默认: 1000)')
    parser.add_argument('--verbose', action='store_true',
                       help='打印详细信息')
    parser.add_argument('--analyze', type=int, metavar='SAMPLE_ID',
                       help='详细分析指定样本')
    args = parser.parse_args()

    if args.analyze is not None:
        # 详细分析模式
        detailed_analysis(args.analyze)
    else:
        # 批量测试模式
        results = test_mib_area_consistency(
            num_samples=args.num_samples,
            verbose=args.verbose
        )

        # 如果发现问题，建议详细分析
        if results['area_mismatch_samples'] or results['size_mismatch_samples']:
            print("\n提示: 使用 --analyze <样本ID> 查看详细信息")
            if results['area_mismatch_samples']:
                print(f"例如: python {Path(__file__).name} --analyze {results['area_mismatch_samples'][0]}")


if __name__ == '__main__':
    main()
