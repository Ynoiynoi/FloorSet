# PeF_ev1 简化版方案设计

## 目标版本

本版本只解题目的简化目标：

```text
minimize HPWLgap + Areagap_bbox
```

忽略所有软约束的优化收益，包括 `boundary`、`grouping`、`MIB`。这里的“忽略”只表示不主动优化这些软约束；硬约束仍必须严格满足：

- 所有矩形无重叠。
- 普通 soft block 满足面积误差不超过 1%。
- fixed-shape block 的 `w/h` 与输入目标完全一致。
- preplaced block 的 `x/y/w/h` 与输入目标完全一致。
- 如果软约束与硬约束冲突，始终保硬约束。

代码落地文件计划为 `sol/PeF_ev1/my_optimizer.py`。测试完成后按项目约定输出 `result.txt` 和 `analyze.txt`。

## 与 PeFsol1 的区别

`PeFsol1` 试图把软约束转成局部 rigid item，再交给 PeF 主流程。本版本不做这层转换：

- 不把 grouping 压成约束岛。
- 不统一 MIB 模板。
- 不单独后放 boundary block。
- `constraints[:, 2:5]` 仅用于识别“这是被忽略的软约束”，不改变布局逻辑。

保留 PeF 相关主线：

- density map + Poisson potential 做全局扩散。
- connectivity gradient 做线长驱动。
- preplaced block 作为固定障碍参与密度和线长，但位置不更新。
- fixed-shape block 保持尺寸，位置可移动。
- obstacle-aware HCG/VCG + bottom-left packing 做合法化。

## 数据解析

评测器会调用：

```python
solve(block_count, area_targets, b2b_connectivity, p2b_connectivity,
      pins_pos, constraints, target_positions)
```

约束列含义按评测器为：

```text
[fixed, preplaced, mib_id, cluster_id, boundary_code]
```

预处理阶段只使用：

- `fixed != 0`：锁定尺寸。
- `preplaced != 0`：锁定位置和尺寸。
- `target_positions`：读取 fixed/preplaced 的真实 `w/h` 或 `x/y/w/h`。

其余列不进入目标函数和合法化优先级。

## 总体流程

1. **初始化 block 尺寸**
   - preplaced：使用 `target_positions[i]` 的 `x/y/w/h`。
   - fixed-shape：使用 `target_positions[i, 2:4]` 的 `w/h`。
   - 普通 soft：初始用正方形 `sqrt(area)`，后续可按线长方向微调形状，始终保持 `w*h = area`。

2. **构造连接模型**
   - block-to-block edge 直接来自 `b2b_connectivity`。
   - pin-to-block edge 直接来自 `p2b_connectivity + pins_pos`。
   - 对 preplaced block，它的中心作为固定连接端参与梯度；但该 block 本身不移动。

3. **估计 PeF 工作域**
   - 题目没有固定 outline，所以工作域只是数值求解网格，不是硬边界。
   - 初始面积取 `sum(block_area) / target_util`，`target_util` 约为 `0.65~0.72`。
   - 宽高比优先参考有效 pins 的包围盒；没有 pin 信息时用近似正方形。
   - 若存在 preplaced block，工作域必须覆盖所有 preplaced 矩形并留出 padding。

4. **全局 PeF 放置**
   - 根据当前矩形栅格化 density。
   - 解 Poisson 方程得到 potential。
   - 由 potential 得到电场，推动重叠密集区域向空白区域扩散。
   - 用连接梯度近似 HPWL/LSE，让相连 block 和 pin 靠近。
   - preplaced block 的位置梯度清零。
   - fixed-shape block 的尺寸不更新。
   - 普通 soft block 可做少量宽高比搜索，保持面积不变。

5. **obstacle-aware 合法化**
   - 根据全局放置后的中心位置构建 HCG/VCG：
     - 横向更适合分离的 pair 加入 HCG。
     - 纵向更适合分离的 pair 加入 VCG。
     - preplaced 节点作为固定障碍进入图。
   - 先用约束图求推荐中心，再用 obstacle-aware bottom-left packing 放置。
   - 合法化必须保证：
     - preplaced 坐标不变。
     - fixed/preplaced 尺寸不变。
     - 所有 block 无重叠。

6. **最终压缩与局部优化**
   - 在不破坏 preplaced 的前提下，尽量把布局整体向左下压缩。
   - 对普通可移动 block 做有限轮局部移动：
     - 收集相连 block/pin 的坐标，尝试 weighted-median 位置。
     - 对候选位置做合法性检查。
     - 接受能降低内部目标的 move。
   - 对 soft block 尝试少量形状候选：
     - `square`
     - 按连接方向估计的长宽比
     - 该长宽比的轻微放大/缩小版本
   - 每次形状变化后重新局部合法化，只有目标更优且硬约束仍满足才接受。

## 内部目标

评测时真正看的是 gap，但 optimizer 内部拿不到每个 case 的 baseline。因此内部使用未归一化代理目标：

```text
F = HPWL_b2b + HPWL_p2b + lambda_bbox * BBoxArea
```

`lambda_bbox` 用验证集调参。初始建议：

```text
lambda_bbox in {0.01, 0.03, 0.1, 0.3}
```

在本简化版本的结果分析中，主要比较：

```text
avg(HPWLgap) + avg(Areagap_bbox)
```

官方 `Cost`、`Violationsrelative`、runtime 仍会被评测脚本输出，但不作为本版本方案成败的主要判断。

## 失败回退

PeF 全局放置或 HCG/VCG packing 可能在少数 case 上失败。必须有保底合法化：

1. 先固定放入所有 preplaced block。
2. 剩余 block 按面积和连接权重排序。
3. 使用 obstacle-aware shelf/skyline packing：
   - 每次选择不与已放矩形重叠的位置。
   - 优先靠近 PeF 给出的 anchor center。
   - 次优先使 bbox 增量最小。
4. 如果当前候选行放不下，则新开一行。

保底解可能 HPWL 较差，但必须避免硬约束违规导致 `cost = 10`。

## 计划实现顺序

1. 从 `PeFsol1` 裁剪出输入解析、PeF 核心、Poisson helper、obstacle-aware legalization。
2. 删除 grouping/MIB/boundary 的 rigid-item 和后放逻辑。
3. 加入简化目标 `HPWL + lambda_bbox * bbox_area` 的评估函数。
4. 加入最终合法性修复和 fallback packing。
5. 跑：

```bash
python iccad2026_evaluate.py --evaluate sol/PeF_ev1/my_optimizer.py
python analyze_cost_contributions.py sol/PeF_ev1/my_optimizer.py
```

并分别把输出保存到：

```text
sol/PeF_ev1/result.txt
sol/PeF_ev1/analyze.txt
```

## 预期

这个版本的重点不是满足软约束，而是获得一个稳定、可行、PeF 风格的 `HPWLgap + Areagap_bbox` 基线。若该版本确认可行，后续可以再逐步加入：

- 多 restart 参数扫描。
- 更强的 bbox compaction。
- 更精细的 exact HPWL 局部搜索。
- 再把 boundary/grouping/MIB 作为软目标加回。
