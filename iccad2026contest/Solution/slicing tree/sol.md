# Slicing Tree 方案设计

## 1. 目标

在 `iccad2026contest` 现有接口下，实现一个以 slicing tree 为主的 floorplanner，用于替换当前偏贪心的摆放方式。方案优先级如下：

1. 硬约束必须始终满足：无重叠，soft block 面积误差不超过 1%，`fixed` / `preplaced` 的尺寸不可改，`preplaced` 的位置不可改。
2. 在可行前提下，尽量减少 soft violation：`boundary`、`grouping(cluster)`、`MIB`。
3. 再优化质量项：`bbox area`、`b2b HPWL`、`p2b HPWL`。
4. 运行时间控制在验证集 100 个 case 可接受范围内，避免重型 SA。

这里按仓库实际语义处理约束张量：

- `constraints[:, 0]`: `fixed`
- `constraints[:, 1]`: `preplaced`
- `constraints[:, 2]`: `mib_id`
- `constraints[:, 3]`: `cluster_id`
- `constraints[:, 4]`: `boundary_code`
- `boundary_code` 位编码：`1=left, 2=right, 4=top, 8=bottom`，角点是按位或，如 `5=left+top`。

补充约定采用 `readme.txt` 中的解释：当软硬约束冲突时，硬约束优先；`Bpreplaced` 和 `Bfixed` 的 `(w, h)` 不可修改。

## 2. 整体思路

核心不是直接对所有 block 做自由平移，而是把解表示为一棵 slicing tree：

- 叶子结点：一个 block，或者一个必须整体处理的 macro；
- 内部结点：一次水平切分 `H` 或垂直切分 `V`；
- 自底向上计算每个子树的候选尺寸；
- 自顶向下确定每个叶子的 `(x, y, w, h)`。

这样做的好处是：

1. 天然无重叠。
2. 对面积、固定尺寸、MIB 一类约束更容易“按构造满足”。
3. 通过把 cluster 做成子树，可以天然保证组内连通。
4. 后续只需要在树结构和叶子顺序上做小范围搜索，复杂度可控。

## 3. 预处理

### 3.1 block 类型划分

每个 block 分为四类：

1. `preplaced`: 位置和尺寸固定。
2. `fixed`: 尺寸固定，位置可变。
3. `mib-only`: 需要和同组成员共形。
4. 普通 soft block：只需要面积满足。

尺寸初始化规则：

- `preplaced` / `fixed`: 直接使用 `target_positions[i, 2:4]`。
- 其他 block：先取正方形 `sqrt(area)`。
- 对 MIB 组：
  - 若组内存在 `preplaced` 或 `fixed`，全组继承该尺寸。
  - 否则全组统一为同一组公共尺寸，初值取 `sqrt(area_ref)`。

### 3.2 预放置障碍物

所有 `preplaced` block 视为不可移动障碍物。它们不进入普通 slicing tree 搜索，而是在全局 floorplan 中预留固定矩形。

原因：

- slicing tree 对“一个叶子位置绝对固定”支持很差；
- 预放置块数量通常不大，把它们当障碍做增量插入更稳。

### 3.3 cluster 压缩

`cluster_id` 相同的一组 block 先构造成一个局部子树，作为一个 super-node 参与全局布局。

局部子树的目标：

- 组内所有块通过公共边连接成单连通分量；
- 尽量兼顾组内线长和可能存在的 boundary / preplaced 锚点；
- 输出这个 cluster 的若干候选 shape。

实现上，cluster 内部只允许以下两类模板，避免状态爆炸：

1. 单行 / 单列切分；
2. 递归二分得到的小 slicing tree。

这样已经足够保证 grouping，比把 cluster 先拆散再靠后处理修补更稳。

### 3.4 全局对象列表

全局树的叶子不是原始 block，而是以下对象：

1. 每个 cluster super-node；
2. 不属于 cluster 的单个 block。

每个对象维护：

- `members`
- 候选 shape 列表 `[(w, h, meta), ...]`
- `boundary_mask`
- `fixed_anchor`（若内部含 preplaced 锚点）
- `preferred_center`（由 `p2b` 和 `b2b` 估计）

## 4. 候选 shape 生成

### 4.1 单块候选

对普通 soft block，不需要只保留一个正方形，建议保留有限个 shape：

- 面积固定为 `A`
- 宽高比候选取例如  
  `1/4, 1/2, 1, 2, 4`
- 即 `w = sqrt(A*r), h = sqrt(A/r)`

这样不会违反面积约束，又能让树搜索有旋转自由度。

对 `fixed` / `preplaced` 只有一个 shape。

对 MIB 组内可变 block，整组共享同一批候选 shape，后续选中哪一个，全组都用哪一个。

### 4.2 cluster 候选

cluster 局部求解时，不输出单一矩形，而输出前 `K` 个非支配候选，建议 `K=6~12`。  
排序依据：

1. `soft violation` 最小；
2. `bbox area` 最小；
3. 组内加权 HPWL 最小。

非支配的意思是：若候选 A 的面积、长宽比偏差、HPWL 都不差于 B，则删掉 B。

## 5. slicing tree 表示

采用经典 Polish expression / 二叉树均可，推荐内部仍使用节点对象，便于调试：

- `Leaf(node_id, shape_options, members, constraints_meta)`
- `Cut(op='H'|'V', left, right)`

每个节点自底向上计算一个有限候选集：

- 若 `op='H'`：  
  `w = max(w1, w2)`  
  `h = h1 + h2`
- 若 `op='V'`：  
  `w = w1 + w2`  
  `h = max(h1, h2)`

节点候选也只保留前 `K` 个非支配状态，避免组合爆炸。

每个候选状态需要附带：

- `w, h`
- 由哪些子候选组成
- 估计代价 `proxy_cost`
- boundary side 占用信息
- 是否包含固定锚点、锚点在该节点坐标系中的相对位置

## 6. 代价函数设计

真实比赛代价依赖 baseline gap，不适合在构造期精确计算，因此局部搜索使用 proxy cost：

```text
proxy = lambda_area * bbox_area
      + lambda_b2b * estimated_hpwl_b2b
      + lambda_p2b * estimated_hpwl_p2b
      + lambda_soft * soft_penalty
      + lambda_anchor * anchor_penalty
```

建议权重优先级：

- `lambda_soft` 最大，确保先满足 `boundary/grouping/MIB`
- 再是 `lambda_anchor`，防止 preplaced / pin 引导被树结构破坏
- 然后是 `lambda_b2b`, `lambda_p2b`
- 最后是 `lambda_area`

其中：

### 6.1 estimated HPWL

对一个中间节点，只用对象中心估算：

- block/block 边：若两个对象已落在不同子树，可用两个子树 bbox 中心估计；
- pin/block 边：用对象 `preferred_center` 与 shape 中心的偏差估计。

这不是精确值，但足够给局部搜索方向。

### 6.2 soft penalty

`soft_penalty` 分三部分：

1. `boundary_penalty`
2. `group_penalty`
3. `mib_penalty`

其中：

- `group_penalty` 对 cluster 内部应为 0，因为 cluster 作为局部连通子树构造；
- `mib_penalty` 用“组内不同 shape 数 - 1”计算；
- `boundary_penalty` 对未能被当前树边界满足的对象计 1。

### 6.3 boundary 估计

因为 boundary 是相对最终 bbox 的，不适合中途严格判断，所以在树 DP 中只做可满足性标注：

- 对要求 `left` 的对象，优先让其落在某个 `V` 切分链最左端；
- 对要求 `right/top/bottom` 同理；
- 对角点对象，优先约束到对应边界交汇的叶子。

若某个节点候选已经无法让该对象位于所需外侧，则加较大罚分。

## 7. 初始树构造

### 7.1 cluster 内部构造

对每个 cluster，先根据连接权重构图：

- 顶点：组内 block
- 边权：`b2b` 内部权重 + 共享 pin 引导相似度

然后做递归二分：

1. 取加权图的谱二分或简单贪心二分；
2. 比较 `H` / `V` 两种切分；
3. 选择 proxy cost 更小的切分；
4. 递归直到单块。

如果组规模很小（`<=4`），可以直接枚举几种小树拓扑。

### 7.2 全局树构造

全局对象用同样的递归二分，但要显式考虑 preplaced 障碍和 boundary：

1. 先把有固定锚点的对象按锚点坐标分布分区；
2. 再把剩余对象依据连接强度和 pin 偏好加入相近分区；
3. 每次二分都尝试 `H/V` 两种切法；
4. 让 `left/right` 倾向出现在左右分区，`top/bottom` 倾向出现在上下分区。

这样得到一个初始全局 slicing tree。

## 8. preplaced 处理

这是该方案最关键的工程点。

### 8.1 两层布局

不强行把 preplaced 块编进 slicing tree，而是采用“两层布局”：

1. 先对可移动对象生成 slicing tree，相对于局部原点得到布局。
2. 再把整棵树对应的若干子树当作矩形对象，插入到由 preplaced 障碍定义的空白区域中。

### 8.2 空白区域表示

根据所有 preplaced 矩形，维护一组 free rectangles。初始可用 MaxRects 的简化版本：

- 初始空区为覆盖所有 preplaced 的外包框周围扩展区域；
- 每插入一个移动子树，更新剩余空区；
- 只保留不被其它空区完全包含的矩形。

### 8.3 插入策略

优先插入：

1. 含 boundary 约束的子树；
2. 含 fixed block 的子树；
3. 面积大的子树。

每次在可用空区中尝试若干位置：

- 左下角对齐
- 靠左 / 靠右 / 靠上 / 靠下对齐
- 靠最近 pin / anchor 的位置

选 proxy cost 最低且无重叠的位置。

这一步本质上是“树生成形状，packing 决定绝对坐标”。

## 9. boundary 处理

boundary 不应该放到最后纯修补，否则经常会把既有解挤坏。方案里分三层处理：

### 9.1 构树时偏置

- `left` 目标优先进入最左链；
- `right` 优先进入最右链；
- `top` 优先进入最上链；
- `bottom` 优先进入最下链。

### 9.2 候选筛选

节点保留候选时，优先保留能够维持 boundary 可满足性的组合。

### 9.3 收尾 legalization

最终得到全局坐标后，再做一次轻量边界合法化：

1. 计算总 bbox；
2. 找出未触边的 boundary 对象；
3. 仅在不引入重叠、且不破坏 preplaced / fixed 的前提下，把对象向目标边滑移；
4. 若对象属于 cluster，则整体滑移该 cluster 子树。

这个步骤只做平移，不改尺寸。

## 10. MIB 处理

MIB 适合“按构造满足”，不要留给后处理。

实现规则：

1. 为每个 `mib_id` 维护组级 shape index。
2. 若组内存在固定尺寸成员，则 shape index 锁定。
3. 若组内全可变，则在候选集合上统一选一个 shape。
4. 任何局部搜索 move 若改变某个成员 shape，必须同步改整组。

这样 `Vmib` 理论上可以长期保持 0。

## 11. 局部搜索

初始树完成后，做一个轻量局部搜索，时间预算要严格控制。

### 11.1 可用 move

1. `swap-leaf`: 交换两个叶子 / super-node。
2. `flip-cut`: `H <-> V`。
3. `rotate-shape`: 切换单块或 MIB 组的 shape index。
4. `reinsert-subtree`: 把一个小子树移到另一位置。
5. `cluster-internal-refine`: 对 cluster 内部重跑一次小规模优化。

### 11.2 接受准则

不做长时间 SA，建议采用：

- 先做若干轮 greedy improvement；
- 再做少量温和退火，温度快速下降；
- 或直接做 beam search / best-improvement hill climbing。

原因是比赛还看运行时间，不能让搜索吃满大 case。

### 11.3 终止条件

任一条件满足即停：

1. 达到 move 次数上限；
2. 连续若干轮没有改进；
3. 单 case 用时超过预算。

预算建议与 `block_count` 挂钩，例如：

- `N <= 40`: 0.15s
- `40 < N <= 80`: 0.35s
- `N > 80`: 0.60s

这里只是实现起点，后面再根据验证集调整。

## 12. 结果回写

最终输出必须是 `List[(x, y, w, h)]`，顺序与原 block id 一致。

流程：

1. cluster / subtree 坐标展开到成员 block；
2. 覆盖 `preplaced` 的精确 `(x, y, w, h)`；
3. 覆盖 `fixed` 的精确 `(w, h)`；
4. 对普通 block 检查面积误差；
5. 全局检查 overlap；
6. 必要时做一次 deterministic repair。

## 13. repair 策略

即使 slicing tree 理论上无重叠，加入 preplaced packing 和 boundary 滑移后仍可能出问题，所以需要收尾 repair：

### 13.1 overlap repair

按以下顺序移动对象：

1. 普通 block
2. 非固定 cluster
3. `fixed`
4. `preplaced` 不可动

对每个冲突对象，尝试在邻近空区做最小代价平移。

### 13.2 area repair

普通 block 若面积误差超 1%，回退到其最近合法 shape。

### 13.3 infeasible fallback

若局部搜索后的解不可行，则回退到：

1. 最近一个可行树解；
2. 若没有，则回退到初始树解；
3. 再不行，回退到当前仓库已有的贪心摆放器。

这样可以避免单个 case 直接打到 `10.0`。

## 14. 建议代码结构

后续真正写代码时，建议放在 `Soluton/slicing tree/` 下，结构如下：

```text
Soluton/slicing tree/
  sol.md
  my_optimizer.py
  slicing_tree_core.py
  cluster_builder.py
  packer.py
  result.txt
```

其中：

- `my_optimizer.py`: 对接官方 `solve()` 接口；
- `slicing_tree_core.py`: 节点定义、DP、局部搜索；
- `cluster_builder.py`: cluster/MIB 局部构造；
- `packer.py`: preplaced 障碍下的 packing 与 repair。

如果希望先少改代码，也可以先只做两个文件：

```text
my_optimizer.py
slicing_tree_core.py
```

把 packing 和 cluster 逻辑先写在核心文件里，等验证后再拆。

## 15. 实现顺序

建议按下面顺序落地，保证每一步都能运行：

1. 先实现“不含 preplaced 障碍”的纯 slicing tree 版本。
2. 加入 `fixed` 和 MIB 按构造满足。
3. 加入 cluster 压缩，使 grouping 基本为 0。
4. 加入 boundary 偏置和轻量 legalization。
5. 最后加入 preplaced 障碍下的 packing。
6. 用 `iccad2026_evaluate_no_runtime.py` 先看纯质量，再跑正式评估。

## 16. 风险点

### 16.1 预放置块较多时，纯 slicing 结构表达力不足

解决办法：把 slicing tree 只作为“可移动对象内部结构”，绝对坐标交给 packing 层。

### 16.2 boundary 约束本质依赖全局 bbox

解决办法：构树时做外侧偏置，收尾时再做一次 deterministic 滑移合法化。

### 16.3 cluster 内部若含多个冲突 boundary

解决办法：cluster 内优先保证连通，其次满足最多的 boundary；若和硬约束冲突，放弃部分 soft 满足。

### 16.4 运行时间

解决办法：严格限制每个节点候选数 `K` 和局部搜索步数；大 case 只做少量改进，不做重 SA。

## 17. 预期效果

相对于当前仓库里的贪心平铺 + 后处理，这个 slicing tree 方案的预期改进点是：

1. grouping 更稳定，因为 cluster 天然连通；
2. MIB 更容易做到长期 0 违规；
3. 面积和无重叠基本由构造保证；
4. HPWL 会比纯行列堆叠更可控；
5. preplaced / boundary 不再只靠最后补丁式修复。

当前建议先按这份方案实现第一版，再根据 `iccad2026_evaluate_no_runtime.py` 的验证结果决定是否增加更强的局部搜索。
