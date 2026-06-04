#!/usr/bin/env python3
"""
示例：如何读取单个训练集样本的输入数据

训练数据格式说明：
每个样本包含 8 个张量：
1. area_target    - 目标面积 [N] (N=最大块数，通常120，-1表示padding)
2. b2b_conn       - 块到块连接矩阵 [N, N] (边权重)
3. p2b_conn       - 引脚到块连接矩阵 [M, N] (M=引脚数)
4. pins_pos       - 引脚位置 [M, 2] (x, y坐标)
5. constraints    - 约束信息 (固定形状、预放置、分组等)
6. tree_sol       - 切片树解 [N-1, 3] (训练标签)
7. fp_sol         - 布图解 [N, 4] (w, h, x, y) (训练标签)
8. metrics        - 基准指标 [8] (HPWL, 面积等)
"""

import sys
from pathlib import Path
import torch

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026contest.iccad2026_evaluate import get_training_dataloader


def read_single_training_sample(sample_index=0):
    """
    读取单个训练样本并打印其输入数据格式

    Args:
        sample_index: 样本索引 (0 到 999,999)
    """
    print("="*80)
    print(f"读取训练集样本 #{sample_index}")
    print("="*80)

    # 创建数据加载器，只加载一个样本
    dataloader = get_training_dataloader(
        batch_size=1,
        num_samples=sample_index + 1,  # 加载到目标索引
        shuffle=False
    )

    # 跳到目标样本
    for i, batch in enumerate(dataloader):
        if i < sample_index:
            continue

        # 解包批次数据 - 8个张量
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch

        # 移除批次维度 (batch_size=1)
        area_target = area_target.squeeze(0)  # [N]
        b2b_conn = b2b_conn.squeeze(0)        # [N, N]
        p2b_conn = p2b_conn.squeeze(0)        # [M, N]
        pins_pos = pins_pos.squeeze(0)        # [M, 2]
        # constraints 保持原样 (字典或None)
        tree_sol = tree_sol.squeeze(0)        # [N-1, 3]
        fp_sol = fp_sol.squeeze(0)            # [N, 4]
        metrics = metrics.squeeze(0)          # [8]

        # 计算有效块数 (非padding的块)
        block_count = int((area_target != -1).sum().item())

        # 计算有效引脚数
        pin_count = int((pins_pos[:, 0] != -1).sum().item())

        print(f"\n【基本信息】")
        print(f"  有效块数: {block_count}")
        print(f"  有效引脚数: {pin_count}")
        print(f"  最大块数 (含padding): {len(area_target)}")

        # =====================================================================
        # 1. area_target - 每个块的目标面积
        # =====================================================================
        print(f"\n【1. area_target - 目标面积】")
        print(f"  形状: {area_target.shape}")
        print(f"  数据类型: {area_target.dtype}")
        print(f"  前5个块的目标面积: {area_target[:5].tolist()}")
        print(f"  说明: 每个块的目标面积，-1表示padding")

        # =====================================================================
        # 2. b2b_conn - 块到块的连接矩阵 (边权重)
        # =====================================================================
        print(f"\n【2. b2b_conn - 块到块连接矩阵】")
        print(f"  形状: {b2b_conn.shape}")
        print(f"  数据类型: {b2b_conn.dtype}")
        print(f"  非零连接数: {(b2b_conn > 0).sum().item()}")
        print(f"  示例 - 块0到其他块的连接权重:")
        nonzero_indices = torch.nonzero(b2b_conn[0] > 0).flatten()
        if len(nonzero_indices) > 0:
            for idx in nonzero_indices[:5]:  # 显示前5个
                print(f"    块0 -> 块{idx}: 权重 = {b2b_conn[0, idx].item():.4f}")
        else:
            print(f"    块0没有连接到其他块")
        print(f"  说明: [N, N]矩阵，b2b_conn[i,j]表示块i到块j的连接权重(线网数)")

        # =====================================================================
        # 3. p2b_conn - 引脚到块的连接矩阵
        # =====================================================================
        print(f"\n【3. p2b_conn - 引脚到块连接矩阵】")
        print(f"  形状: {p2b_conn.shape}")
        print(f"  数据类型: {p2b_conn.dtype}")
        print(f"  非零连接数: {(p2b_conn > 0).sum().item()}")
        if pin_count > 0:
            print(f"  示例 - 引脚0到各块的连接权重:")
            nonzero_indices = torch.nonzero(p2b_conn[0] > 0).flatten()
            if len(nonzero_indices) > 0:
                for idx in nonzero_indices[:5]:
                    print(f"    引脚0 -> 块{idx}: 权重 = {p2b_conn[0, idx].item():.4f}")
            else:
                print(f"    引脚0没有连接到任何块")
        print(f"  说明: [M, N]矩阵，p2b_conn[i,j]表示引脚i到块j的连接权重")

        # =====================================================================
        # 4. pins_pos - 引脚位置
        # =====================================================================
        print(f"\n【4. pins_pos - 引脚位置】")
        print(f"  形状: {pins_pos.shape}")
        print(f"  数据类型: {pins_pos.dtype}")
        if pin_count > 0:
            print(f"  前5个引脚的位置 (x, y):")
            for i in range(min(5, pin_count)):
                x, y = pins_pos[i]
                print(f"    引脚{i}: ({x.item():.2f}, {y.item():.2f})")
        print(f"  说明: [M, 2]矩阵，每行是一个引脚的(x, y)坐标，(-1, -1)表示padding")

        # =====================================================================
        # 5. constraints - 约束信息
        # =====================================================================
        print(f"\n【5. constraints - 约束信息】")
        print(f"  形状: {constraints.shape}")
        print(f"  数据类型: {constraints.dtype}")
        print(f"  说明: [N, 5]矩阵，每行是[fixed, preplaced, mib, cluster, boundary]")
        print(f"  前5个块的约束:")
        constraint_names = ["fixed", "preplaced", "mib", "cluster", "boundary"]
        for i in range(min(5, block_count)):
            constraint_values = constraints[i].tolist()
            active_constraints = []
            for j, (name, val) in enumerate(zip(constraint_names, constraint_values)):
                if val > 0:
                    if j < 2 or j == 4:  # fixed, preplaced, boundary
                        active_constraints.append(name)
                    else:  # mib, cluster (显示组ID)
                        active_constraints.append(f"{name}={int(val)}")
            if active_constraints:
                print(f"    块{i}: {', '.join(active_constraints)}")
            else:
                print(f"    块{i}: 无约束")

        # 统计约束类型
        fixed_count = (constraints[:block_count, 0] > 0).sum().item()
        preplaced_count = (constraints[:block_count, 1] > 0).sum().item()
        mib_count = (constraints[:block_count, 2] > 0).sum().item()
        cluster_count = (constraints[:block_count, 3] > 0).sum().item()
        boundary_count = (constraints[:block_count, 4] > 0).sum().item()

        print(f"\n  约束统计:")
        print(f"    固定形状块数: {fixed_count}")
        print(f"    预放置块数: {preplaced_count}")
        print(f"    MIB约束块数: {mib_count}")
        print(f"    分组约束块数: {cluster_count}")
        print(f"    边界约束块数: {boundary_count}")

        # =====================================================================
        # 6. tree_sol - 切片树解 (训练标签)
        # =====================================================================
        print(f"\n【6. tree_sol - 切片树解 (训练标签)】")
        print(f"  形状: {tree_sol.shape}")
        print(f"  数据类型: {tree_sol.dtype}")
        print(f"  前3行: {tree_sol[:3].tolist()}")
        print(f"  说明: [N-1, 3]矩阵，表示切片树的结构")

        # =====================================================================
        # 7. fp_sol - 布图解 (训练标签)
        # =====================================================================
        print(f"\n【7. fp_sol - 布图解 (训练标签)】")
        print(f"  形状: {fp_sol.shape}")
        print(f"  数据类型: {fp_sol.dtype}")
        print(f"  前5个块的布图 (w, h, x, y):")
        for i in range(min(5, block_count)):
            w, h, x, y = fp_sol[i]
            print(f"    块{i}: 宽={w.item():.2f}, 高={h.item():.2f}, x={x.item():.2f}, y={y.item():.2f}")
        print(f"  说明: [N, 4]矩阵，每行是(宽度, 高度, x坐标, y坐标)")

        # =====================================================================
        # 8. metrics - 基准指标 (训练标签)
        # =====================================================================
        print(f"\n【8. metrics - 基准指标】")
        print(f"  形状: {metrics.shape}")
        print(f"  数据类型: {metrics.dtype}")
        print(f"  值: {metrics.tolist()}")
        print(f"  说明: 包含HPWL、面积等基准指标，用于计算训练损失")

        # =====================================================================
        # 总结：输入数据 vs 标签数据
        # =====================================================================
        print(f"\n{'='*80}")
        print(f"【数据分类总结】")
        print(f"{'='*80}")
        print(f"\n输入数据 (用于模型输入):")
        print(f"  1. area_target  - 目标面积 [N]")
        print(f"  2. b2b_conn     - 块到块连接 [N, N]")
        print(f"  3. p2b_conn     - 引脚到块连接 [M, N]")
        print(f"  4. pins_pos     - 引脚位置 [M, 2]")
        print(f"  5. constraints  - 约束信息")
        print(f"\n标签数据 (用于训练目标):")
        print(f"  6. tree_sol     - 切片树解 [N-1, 3]")
        print(f"  7. fp_sol       - 布图解 [N, 4] (w, h, x, y)")
        print(f"  8. metrics      - 基准指标 [8]")

        print(f"\n{'='*80}")
        print(f"如何使用这些数据训练模型:")
        print(f"{'='*80}")
        print(f"""
# 1. 将输入数据传给你的神经网络
positions = your_model(area_target, b2b_conn, p2b_conn, pins_pos, constraints)
# positions 应该是 [block_count, 4] 的张量，表示 (x, y, w, h)

# 2. 使用可微分的损失函数计算损失
from iccad2026_evaluate import compute_training_loss_differentiable
loss = compute_training_loss_differentiable(
    positions,           # 你的模型输出
    b2b_conn,           # 输入
    p2b_conn,           # 输入
    pins_pos,           # 输入
    area_target[:block_count],  # 输入
    metrics             # 标签 (基准指标)
)

# 3. 反向传播
loss.backward()
optimizer.step()
""")

        break  # 只处理一个样本

    print("="*80)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='读取单个训练样本')
    parser.add_argument('--index', type=int, default=0,
                       help='样本索引 (0-999999)')
    args = parser.parse_args()

    read_single_training_sample(args.index)


if __name__ == '__main__':
    main()
