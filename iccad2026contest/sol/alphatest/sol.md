# eazyver2 方案说明

## 本版目标

在 `eazyver` 的“只保硬约束、主压 bbox area”基础上，加入两条额外策略：

1. 对 `grouping`：
   - 把同一个 grouping 组内、且 **不是 preplaced、也没有 boundary 约束** 的模块视为一个子问题
   - 先在组内构造一个连通布局，让该组自己的包围盒尽量小
   - 再把这个组的包围盒视为一个矩形模块参与全局布局

2. 对 `boundary`：
   - 先完成其他模块的布局
   - 再把 boundary 模块放到当前包围盒外侧，使其尽量满足边/角接触要求

## 明确忽略的冲突项

按猊下这次给的约定，做两个裁剪：

1. **忽略 preplaced 的其他软约束**
   - 如果某个模块是 `preplaced`
   - 那么它的 grouping / boundary / MIB 之类的软约束都不再主动处理
   - 只保留其硬约束：位置和尺寸完全固定

2. **忽略带 boundary 模块的 grouping 约束**
   - 若某个模块有 boundary 约束
   - 它不再参加 grouping 子问题
   - 即使原始 `cluster_id` 非零，也从该 grouping 组里剔除

这两个裁剪的目的很直接：避免“先固定位置”与“还要组内连通”或“还要去边界”互相冲突。

## 总体流程

### 第 1 步：分类

把模块分成四类：

1. `preplaced`
2. `boundary` 但不是 `preplaced`
3. 参加 grouping 子问题的模块
4. 其余普通模块

其中：

- `preplaced` 直接锁死
- `boundary` 模块暂不参与核心布局
- grouping 模块先在组内解子问题

### 第 2 步：求 grouping 子问题

对每个 grouping 组，先剔除：

- preplaced 成员
- boundary 成员

如果剔除后组大小仍然至少为 2，就把这个组做成一个子布局。

### 组内布局做法

这版不做复杂图搜索，而是直接构造“连通链”：

1. 如果全是普通软块：
   - 取接近正方形的公共高度 `H = sqrt(sum area)`
   - 所有块按同高横向排开
   - 每对相邻块共享一段边，因此整个组连通
   - 组包围盒面积恰好等于组总面积

2. 如果组内含 fixed-shape：
   - 枚举两种链式布局
     - 横向链：所有块底对齐，依次相接
     - 纵向链：所有块左对齐，依次相接
   - 普通软块的长宽按当前链高/链宽反推
   - 选择组包围盒面积更小、且更接近方形的版本

这样得到一个组内局部坐标系下的解，以及该组对应的外包矩形 `(W, H)`。

之后把整个 grouping 组当成一个“刚性矩形 item”参与全局布局。

## 第 3 步：先做 core 布局

此时参与 core 的只有：

- preplaced 模块
- 非 boundary 的 fixed-shape 单块
- 上一步得到的 grouping item
- 非 grouping、非 boundary 的普通软块

core 布局仍沿用 `eazyver` 的框架：

1. 先放 `preplaced`
2. 对其他刚性 item 做 beam search 压缩包围盒
3. 再把剩余普通软块优先填进空洞
4. 放不下的普通软块挂到 core 包围盒外侧的条带

## 第 4 步：最后放 boundary 模块

boundary 模块不参加 core 优化，而是在 core 完成后单独处理。

### 角点模块

角点编码：

- `5`: top-left
- `6`: top-right
- `9`: bottom-left
- `10`: bottom-right

验证集里每个角点最多只有一个模块，因此可以直接预留四个角。

### 边模块

边编码：

- `1`: left
- `2`: right
- `4`: top
- `8`: bottom

对每条边：

- 若是 fixed-shape，则保持输入尺寸
- 若是普通软块，则按该边条带厚度反推尺寸
- 同一条边上的模块沿边依次排开
- 所有模块统一贴住最终外框对应边

为了让四条边彼此兼容，外框厚度用一个小迭代近似求：

1. 先用角点模块初始化四侧厚度
2. 再根据当前高度/宽度估计左右/上下条带需要的厚度
3. 迭代几轮直到稳定

## 这版相对 eazyver 的变化

相比 `eazyver`，这版主要多了两件事：

1. `grouping` 不再完全忽略，而是先压成组 item
2. `boundary` 不再完全忽略，而是后挂到最终外框

但仍然保留这些简化：

- 不主动优化 HPWL
- 不主动处理 MIB
- preplaced 的软约束忽略
- 带 boundary 的模块不参加 grouping

## 预期效果

理论上这版应该有两个直接改进：

1. `grouping_violations` 会明显下降
2. `boundary_violations` 会明显下降

代价是：

- 组 item 的矩形化会损失一部分全局面积自由度
- boundary 后挂会让 bbox area 比 `eazyver` 略有上升

所以这版本质上是在：

`用一点 bbox area 换更低的 soft violations`

最终是否更优，要交给 `iccad2026_evaluate.py` 实测。
