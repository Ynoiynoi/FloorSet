# PeFsol1 方案说明

## 目标

把原题先转成一个“没有软约束”的核心子问题，再用 `论文方案/PeF` 的思路处理这个子问题。

这里的原则是：

1. **硬约束优先**：`fixed-shape`、`preplaced` 一旦冲突，必须先满足硬约束。
2. **软约束外移**：`grouping`、`MIB`、`boundary` 不进入主求解器的核心优化。
3. **PeF 负责核心布局**：对剥离后的子问题，使用泊松方程式全局布局 + obstacle-aware legalization。

## 约束划分

输入约束按 `constraints = [fixed, preplaced, mib_id, cluster_id, boundary_code]` 解析。

### 硬约束

- `fixed-shape`
- `preplaced`
- no-overlap
- 面积容差

### 软约束

- `grouping`
- `MIB`
- `boundary`

### 冲突规则

- 若 `grouping / MIB / boundary` 与 `fixed / preplaced` 冲突，**以硬约束为准**。
- `preplaced` 模块不参与后续位置更新。
- `fixed-shape` 模块只锁定尺寸，不允许被软模板改写。

## 总体流程

### 第 1 层：把软约束剥离成局部结构

参考 `sol/eazyver2` 的做法，把原图拆成几类对象：

1. `preplaced`：直接锁死。
2. `boundary`：不进核心布局，放到最后处理。
3. `grouping / MIB`：先在局部形成“约束岛”，把岛内软约束消解成少量候选模板。
4. 其余普通模块：进入 PeF 主求解器。

这一层的目标不是做全局最优，而是把“需要满足软约束的关系”尽量转成**固定形状的局部宏块**。

### 第 2 层：把约束岛压成 rigid item

对每个包含 `grouping` 或 `MIB` 的连通分量，生成 1 到 3 个候选模板：

- **纯 grouping 岛**：优先用链式排布
  - 横向链：统一高度，宽度由面积反推
  - 纵向链：统一宽度，高度由面积反推
  - 选包围盒更小的一种
- **纯 MIB 岛**：统一模板尺寸
  - 若岛内有 `fixed-shape`，直接用该固定尺寸做模板
  - 否则优先用近似正方形模板，再做少量长宽比候选
- **grouping + MIB 混合岛**：
  - 先确定 MIB 共享模板
  - 再在模板内部做 grouping 链式连接

每个候选最终都变成一个 `rigid item`：

- 外层：一个矩形 `(W, H)`
- 内层：保存每个原始 block 的局部坐标偏移

这样，顶层 PeF 看到的是“没有软约束的子问题”。

### 第 3 层：PeF 主求解

顶层只处理：

- `preplaced / fixed` 作为锚点或障碍物
- `rigid item`
- 普通 soft block

PeF 的处理方式：

1. 用当前布局生成 density map。
2. 解 Poisson 方程得到 potential。
3. 计算线长梯度与电场梯度。
4. 只更新可动模块的位置。
5. 只更新普通 soft block 的宽度，rigid item 保持尺寸不变。

### 第 4 层：obstacle-aware legalization

用 `PeF` 里的 obstacle-aware HCG/VCG 思路，把硬模块当障碍物：

- `preplaced` 直接固定在原位
- `fixed-shape` 尺寸不动
- `rigid item` 作为普通宏块参与图构造

然后做：

1. 构建水平/垂直约束图
2. 拓扑排序
3. bottom-left 式合法化
4. 局部消重叠修补

### 第 5 层：boundary 后放

`boundary` 模块不进入核心子问题，而是在核心布局完成后单独放置：

- `1/2/4/8` 表示左/右/上/下
- `5/6/9/10` 表示四个角

放置顺序：

1. 先放角点模块
2. 再放边模块
3. 优先沿当前 bbox 边界放，减少额外扩张
4. 若同边冲突，按面积大的先放

### 第 6 层：展开与修复

把 `rigid item` 再展开回原始 block：

- 使用保存的局部偏移恢复每个 block 的绝对坐标
- 若恢复后出现小量重叠，做局部修补
- 若没有 `preplaced`，整体可做一次向原点压缩
- 若有 `preplaced`，只允许在不破坏锚点的前提下微调

## PeF 子问题的具体建模

### 工作域

原题没有固定 outline，所以 PeF 的固定轮廓要改成**工作域**：

- 初值由当前总面积和经验长宽比估计
- 迭代中可根据当前 bbox 做轻量收缩/扩张
- 这个工作域只是数值求解网格，不是最终硬约束

### 目标函数

主目标仍按 PeF 的思路：

- 线长项：HPWL/LSE
- 密度势能项：抑制重叠

另外为了适配本题的“无固定轮廓”特性，最后用 legalization 和压缩去控制 bbox area，而不是把固定 outline 当作硬约束。

### 梯度规则

- `preplaced`：位置梯度清零
- `fixed-shape`：尺寸梯度清零
- `rigid item`：只更新整体位置，不改内部布局
- 普通 soft block：允许按 PeF 方式更新宽度

## 失败回退

如果某个约束岛里出现硬冲突，按下面顺序回退：

1. 保住 `preplaced`
2. 保住 `fixed-shape`
3. 保住 `MIB` 的共享模板尽量不破坏
4. 只牺牲 `grouping / boundary` 的软满足程度

也就是说，**不会为了满足软约束去破坏硬约束**。

## 预期效果

这套做法的目的不是把所有软约束都在主求解器里硬塞进去，而是：

- 先把软约束变成少量局部结构
- 再把局部结构压成 rigid item
- 最后让 PeF 专注处理一个更干净的布局问题

这样实现上更稳，和题目里“软硬冲突时硬约束优先”的要求也一致。
