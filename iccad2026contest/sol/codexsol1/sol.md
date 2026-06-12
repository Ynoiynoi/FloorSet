# codexsol1 方案说明

## 方法类型

纯传统启发式，不使用机器学习。

## 核心思路

目标是先保证硬约束始终可行，再尽量压低三类主要代价：

1. `boundary / grouping / MIB` 软约束违例
2. block-to-block 与 pin-to-block 的 HPWL
3. 总包围盒面积

因此做法采用“先结构化、再合法化、最后后处理”的三段式流程。

## 具体流程

### 1. 先确定每个 block 的合法尺寸

- 普通 soft block：默认取正方形 `w = h = sqrt(area)`
- fixed / preplaced：严格使用题目给定的尺寸
- MIB group：整组统一尺寸；若组内存在 fixed / preplaced，则整组跟随该参考尺寸

这样可以从一开始就避免面积误差和 MIB 形状不一致。

### 2. 把 cluster group 收缩成 connected mini-floorplan

对每个 cluster 先在组内构造一个小布局，再把整个 group 当成一个 item 放置：

- 无边界约束：横向链式排布
- 仅左/右边界：沿该边做纵向堆叠，其余块向内串接
- 仅上/下边界：沿该边做横向排布，其余块向内串接
- 角点约束：构造 L 形小布局，角点块作为拐点

这样做的目的有两个：

- `grouping` 约束在构造时就尽量满足
- 组内带 `boundary` 的块不只是“group 外框贴边”，而是尽量让具体块真的贴到需要的边

这是本方案相对已有简单分组法的主要改动。

### 3. 建 item-level 连通图

把 block-to-block net 聚合到 item 层：

- cluster 内部连接不再重复计入
- 不同 item 之间的边权累加

同时从 pin-to-block net 计算每个 block 的 pin 重心，再聚合成 item 的 pin 重心。

## 4. 粗定位：重心迭代

对每个 item 计算一个粗目标中心：

- 来自 pin 重心
- 来自相邻 item 的连线重心
- 来自 boundary 的边/角吸引
- 对 preplaced item 则直接固定

迭代数轮后得到一个比较稳定的 coarse center，作为后续合法化放置的目标。

## 5. 合法化放置：角点候选 + 代价打分

按优先级逐个放置 item：

- 先放 preplaced / fixed anchor
- 再放 boundary item
- 再放高连通度、大面积 item

每次从以下候选位置里选一个不重叠的最优位置：

- 已放 item 的右侧、上侧、对齐位置
- 连通邻居附近的位置
- coarse center 对应的位置
- boundary item 的目标边/角位置

打分函数综合考虑：

- 与已放连通邻居的加权曼哈顿距离
- 与 coarse center 的偏差
- 当前包围盒面积
- 是否超出估计 outline

## 6. 局部 refinement

初始合法化完成后，对非固定 item 做 2 轮轻量重放：

- 暂时移出当前 item
- 在周围候选位置中重新搜索
- 若代理代价下降则接受

这一步主要用于修正贪心顺序带来的 HPWL 局部劣化。

## 7. boundary 后处理

最后再做一轮 item 级边界整理：

- 左/右 item 重新压到左右边
- 上/下 item 重新压到上下边
- 角点 item 优先占据四角

若后处理后的软约束更优，并且代理总代价没有显著变坏，则接受。

## 为什么这是传统做法

整套方法只用了：

- 分组
- 图聚合
- 重心迭代
- 贪心合法化
- 局部搜索
- 规则化后处理

没有训练、没有参数学习、没有神经网络推理。

## 本地结果

在当前仓库的 validation 100 例上，本方案本地评测结果为：

- Total Score: `3.2786`
- Feasible: `100 / 100`

相对仓库中现有 `my_optimizer_grouping.py` 的 `3.9513` 有明显下降。
