# BF3 方案说明

## 1. 本版目标

BF3 按 `sol/BF3/idea.docx` 的思路实现，在 `sol/alphatest` 的框架上继续做三类增强：

1. `MIB`：尽量把同一个 MIB 组做成统一正方形模板，降低 `mib_violations`。
2. `grouping`：仍把 grouping 组先打包成连通子布局，再作为一个矩形 item 参与全局布局。
3. 普通 soft block：不再只按面积塞入空洞，而是先分配到空白矩形区域，再在区域内用固定顺序的一维 PAVA 优化 x 坐标。
4. `boundary`：核心布局完成后，在外围单独放置 boundary block；角点模块枚举横向/纵向延展方式，选择最终 bbox 面积最小的方案。

目标优先级：

1. 必须满足硬约束：
   - preplaced 的 `x/y/w/h` 完全等于输入目标。
   - fixed-shape 的 `w/h` 完全等于输入目标。
   - 普通 soft block 的面积误差在 1% 内。
   - 所有 block 无重叠。
2. 在硬约束安全的前提下，尽量降低 soft violations：
   - MIB 同形状。
   - grouping 连通。
   - boundary 贴边/贴角。
3. 再考虑 bbox area 和 HPWL。

若软约束和硬约束冲突，以硬约束为准。

## 2. 冲突取舍

按 `idea.docx` 的约定，本版采用以下裁剪：

1. `preplaced + grouping`：
   - preplaced 的位置和尺寸是硬约束。
   - 若 preplaced block 同时有 grouping 约束，则该 block 不参与 grouping 子问题。
   - 若剔除 preplaced 后组大小不足 2，则不再主动处理该 grouping 组。

2. `boundary + grouping`：
   - 当前主版本优先 boundary。
   - 若某个 block 同时有 boundary 和 grouping 约束，则把它从 grouping 子问题中剔除，并放入外围 boundary 队列。
   - 这样优先保证该 block 贴住指定边/角，允许对应 grouping 组出现额外连通分量。
   - 原 grouping 优先实现备份在 `my_optimizer_grouping_priority.py`。

3. `MIB + fixed/preplaced`：
   - fixed/preplaced 尺寸或位置不能为了 MIB 被改变。
   - 如果一个 MIB 组存在 fixed/preplaced 成员，只有在不破坏任何成员面积/固定尺寸的情况下才把其他 movable block 改成同一模板。
   - 若 MIB 与硬约束天然矛盾，则保硬约束，接受残余 `mib_violations`。

## 3. 输入解析和基础数据结构

继续沿用 alphatest 的接口和基础类：

- `BlockSpec`
  - `block_id`
  - `area`
  - `fixed`
  - `preplaced`
  - `width/height`
  - `x/y`
  - `group_id`
  - `boundary_code`
  - `mib_id`
- `LayoutItem`
  - 一个可整体移动的矩形 item。
  - 可以代表单个 fixed/MIB block，也可以代表一个 grouping 子布局。
- `FreeRect`
  - 全局布局中由已放置刚性矩形切出来的空白矩形区域。

`solve()` 保持官方评测器可识别的签名：

```python
def solve(
    self,
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions=None,
):
```

其中 `constraints[:, 0:5]` 分别表示：

- fixed
- preplaced
- MIB id
- grouping/cluster id
- boundary bitmask

`target_positions` 中 fixed block 有目标 `w/h`，preplaced block 有目标 `x/y/w/h`。

## 4. MIB 处理

先按 `mib_id` 收集 MIB 组。

### 4.1 全部为普通 movable soft block

若组内没有 fixed/preplaced，且组内面积目标可以共用一个尺寸：

- 优先取公共正方形：

```text
side = sqrt(common_area)
w = h = side
```

- 若面积略有数值噪声，用组内面积的代表值做模板，并检查每个成员面积误差是否仍在 1% 内。
- 通过检查后，把这些 block 视为形状已经确定的 item，后续不再单独改 shape。

这样可以直接让该 MIB 组的 distinct shape 数为 1。

### 4.2 含 fixed/preplaced 或面积不一致

若存在硬约束成员：

- fixed/preplaced 成员保持输入尺寸。
- 如果所有成员都能合法使用同一 `(w, h)`，则 movable 成员跟随该模板。
- 否则 movable 成员保持面积合法的正方形或区域内形状，不强行破坏硬约束。

这类 MIB 可能仍有 soft violation，但不会产生不可行解。

## 5. grouping 打包

对每个 `group_id > 0` 的组：

1. 删除 preplaced 成员。
2. 保留 boundary 成员，但后续不再把这些成员放进 boundary 队列。
3. 若剩余成员数小于 2，则不建 group item。
4. 若剩余成员数至少 2，则构造组内连通链。

### 5.1 全部为普通 soft block

设组内总面积为 `A`，取公共高度：

```text
H = sqrt(A)
```

每个 block 的宽度：

```text
w_i = area_i / H
h_i = H
```

所有 block 底对齐，按顺序横向相接：

```text
x_0 = 0
x_i = sum_{j < i} w_j
y_i = 0
```

这样每对相邻 block 共享一条边，整个 group 连通，组 item 的面积等于组内总面积。

### 5.2 组内含 fixed/MIB 形状块

枚举两种链式布局：

1. 横向链：
   - fixed/MIB 块保持尺寸。
   - 所有块底对齐，依次相接。
   - 普通 soft block 使用公共链高 `H`，宽度由 `area / H` 反推。

2. 纵向链：
   - fixed/MIB 块保持尺寸。
   - 所有块左对齐，依次相接。
   - 普通 soft block 使用公共链宽 `W`，高度由 `area / W` 反推。

在候选中选择：

```text
(组 bbox 面积, |W/H - 1|, HPWL proxy)
```

更小的方案。

最终得到：

- group item 外包矩形 `(W, H)`。
- 每个原始 block 在 group item 局部坐标内的 `(lx, ly, w, h)`。

后续全局布局只移动 group item，不打散组内相对位置。

## 6. core 刚性布局

参与 core 刚性布局的对象：

- preplaced block：直接作为 occupied rectangle。
- 非 boundary、非 grouping 的 fixed-shape block。
- MIB 中已经确定统一模板的单块 item。
- grouping item。

core 布局沿用 alphatest 的 beam-search 装箱框架：

1. 先固定所有 preplaced。
2. 对刚性 item 按多种顺序尝试放置。
3. 候选点来自已有矩形的左/右/上/下边界贴靠位置。
4. 每个候选必须不与 occupied rectangles 重叠。
5. 评分先以 bbox area 为主，tie-break 用周长、最大边长、简单 HPWL proxy。

如果没有 preplaced，可以在最后把整体平移到原点；如果有 preplaced，绝不整体平移。

## 7. 空白矩形区域划分

core 刚性布局完成后，基于当前 occupied rectangles 划分空白矩形区域。

实现上可复用 alphatest 的 free-rectangle 思路：

1. 收集当前 bbox 内所有 occupied rectangle 的 x/y 边界。
2. 用这些边界切分成网格 cell。
3. 保留完全不与 occupied 相交的 cell。
4. 合并或筛选出若干可用 `FreeRect`。

如果 core bbox 内没有足够空洞，则额外创建外围 fallback region，例如：

- 当前 bbox 右侧 strip。
- 当前 bbox 顶部 strip。

fallback region 仍作为后续一维 PAVA 的矩形区域处理。

## 8. 普通 soft block 分配到区域

这里的普通 soft block 指：

- 非 preplaced。
- 非 fixed。
- 非已打包进 grouping item。
- 非已固定模板的 MIB item。
- 非待外围处理的 boundary block。

逐个贪心分配：

1. 为每个 soft block 建立 HPWL target：
   - p2b pin 的 x/y 坐标作为固定 target。
   - 已放置邻居 block 的中心作为固定 target。
   - 未放置邻居只用于权重排序，不作为 PAVA 固定 target。
2. soft block 排序：
   - 连接权重高的优先。
   - 面积大的优先。
   - 有 pin/preplaced 邻居的优先。
3. 对每个 candidate region，若剩余面积能容纳该 block，则假设 block 放在 region 中心，估计该 block 到 pins/已放置邻居的 HPWL 增量。
4. 选择 HPWL 增量最小的 region。
5. 更新 region 的剩余面积和成员列表。

只用面积判断可容纳性：

```text
sum(area_i assigned to region) <= region.w * region.h
```

因为后续区域内统一底对齐链式布局时，只要总面积不超过矩形面积，就能通过设定公共高度或公共宽度保证放下。

## 9. 区域内一维 PAVA 优化

对每个 region 单独处理。以横向底对齐链为默认形态：

```text
L = region.x
R = region.x + region.w
Y = region.y
H = region.h
```

区域内每个 block：

```text
h_i = H
w_i = area_i / H
```

只要分配时满足总面积不超过 region 面积，就有：

```text
sum_i w_i <= R - L
```

因此可在该 region 内无重叠放下。

### 9.1 固定顺序

对每个 block 收集 x 方向 target：

- p2b pin 的 `pin_x`。
- 已放置邻居 block 的 center_x。

先把所有 target clip 到 `[L, R]`：

```text
t = min(max(t, L), R)
```

对全局 target 值排序，得到 rank；相同 target 使用相同 rank。

每个 block 的顺序分数为它的 target rank 加权平均值：

```text
order_score_i = weighted_average(rank(t_i,j))
```

没有 target 的 block 用 region center 作为虚拟 target。

按 `order_score_i` 从小到大确定固定顺序。

### 9.2 PAVA 求 x 坐标

固定顺序后，问题变成 `sol/BF3/PAVA.md` 中的一维线段问题：

```text
min sum_i sum_j |x_i - t_i,j|
```

满足：

```text
L + w_i/2 <= x_i <= R - w_i/2
x_{next} - x_i >= (w_i + w_next) / 2
```

令顺序中第 `a` 个 block 为 `s_a`，定义：

```text
d_a = (w_{s_a} + w_{s_{a+1}}) / 2
c_1 = 0
c_a = sum_{q<a} d_q
y_a = x_{s_a} - c_a
```

则不重叠约束变成：

```text
y_1 <= y_2 <= ... <= y_n
```

上下界变成：

```text
L'_a = L + w_{s_a}/2 - c_a
R'_a = R - w_{s_a}/2 - c_a
```

每个 target 平移为：

```text
q_{a,j} = t_{s_a,j} - c_a
```

然后用带上下界的 L1 isotonic regression / PAVA：

- 每个位置初始为一个块。
- 块的最优值为块内所有 `q` 的中位数，再投影到块可行区间。
- 若相邻块最优值违反单调性，则合并。
- 结束后还原 `x_i = y_a + c_a`。

PAVA 中忽略 region 内 soft block 之间的 b2b 项：

```text
sum |x_u - x_v|
```

这样牺牲一部分精度，但能在线性/近线性时间内得到固定顺序下的稳健解。

若 PAVA 因数值误差判定不可行，则回退为按固定顺序从左到右紧密排列。

## 10. boundary 外围放置

core 和普通 soft region 全部完成后，再处理剩余 boundary block。

注意：已经进入 grouping item 的 boundary block 不再参与本步骤。

boundary bitmask：

- `1`: left
- `2`: right
- `4`: top
- `8`: bottom
- `5 = 4 + 1`: top-left
- `6 = 4 + 2`: top-right
- `9 = 8 + 1`: bottom-left
- `10 = 8 + 2`: bottom-right

### 10.1 角点模块枚举

每个角点模块有两种延展方式：

1. 沿水平边延展，占用 top/bottom strip。
2. 沿垂直边延展，占用 left/right strip。

对最多四个角点做笛卡尔枚举。

每个枚举方案计算四侧最小扩展量：

```text
left_ext, right_ext, top_ext, bottom_ext
```

要求：

- 角点 block 接触对应两条最终 bbox 边。
- fixed 角点保持输入尺寸。
- soft 角点保持面积合法，并按当前 strip 厚度反推另一维。

### 10.2 边模块条带

对四条边分别收集 edge boundary block。

left/right 边：

- block 必须贴住最终 left/right 边。
- fixed block 保持输入尺寸。
- soft block 由条带厚度反推宽度/高度。
- 沿 y 方向依次排开。

top/bottom 边：

- block 必须贴住最终 top/bottom 边。
- fixed block 保持输入尺寸。
- soft block 由条带厚度反推宽度/高度。
- 沿 x 方向依次排开。

条带厚度通过小迭代估计：

1. 用角点模块初始化四侧扩展。
2. 根据当前最终宽/高，估算左右边和上下边容纳所有 edge block 所需厚度。
3. 更新 `left_ext/right_ext/top_ext/bottom_ext`。
4. 重复 3-5 轮，直到面积变化很小或达到固定轮数。

每个枚举方案完成后，计算：

```text
score = (final_bbox_area, boundary_violations, HPWL_proxy)
```

选择 score 最小的合法方案。

## 11. 最终安全检查和回退

返回前做一轮硬约束安全检查：

1. preplaced 是否完全保持 `x/y/w/h`。
2. fixed 是否完全保持 `w/h`。
3. 普通 soft block 面积误差是否在 1% 内。
4. 是否有任意正面积 overlap。

若发现 overlap：

- 只移动非-preplaced block。
- 不改变 fixed/MIB/template/grouping 内部尺寸。
- 优先把冲突 block 推到当前 bbox 外侧最近可行位置。

若 BF3 主流程仍不能得到合法解，则回退：

1. 先尝试 alphatest 风格的合法装箱结果。
2. 再退到简单 strip packing，保证所有硬约束满足。

## 12. 预期效果和风险

预期改善：

- MIB 正方形模板会降低 `mib_violations`。
- `boundary + grouping` 优先 boundary 后，相关 block 的贴边/贴角约束更稳定。
- soft block 分区域后再 PAVA，可以比单纯填洞更靠近 pin/preplaced/已放置邻居，降低一部分 HPWL。
- 角点 boundary 枚举能减少外围条带面积浪费。

主要风险：

- MIB 与 fixed/preplaced 或不同面积目标冲突时，不能强行统一形状，只能保硬约束。
- PAVA 固定顺序由 target rank 决定，若排序不准，HPWL 可能不如局部搜索。
- 忽略 region 内 soft block 之间的 b2b 项，会损失一部分线长优化。
- boundary 外围后挂通常会增大 bbox area。

控制方式：

- 保留 alphatest 的合法结果作为 fallback。
- 每个阶段只接受不会破坏硬约束的 move。
- PAVA 失败时回退紧密排列。
- 最终以 `iccad2026_evaluate.py` 和 `analyze_cost_contributions.py` 实测调整权重。

## 13. 代码落地顺序

确认方案后，在 `sol/BF3/my_optimizer.py` 中按以下顺序实现：

1. 从 `sol/alphatest/my_optimizer.py` 拷贝基础结构。
2. 修正冲突取舍：
   - `boundary + grouping` 优先 boundary。
   - `preplaced + grouping` 优先 preplaced。
   - MIB 永远不破坏 fixed/preplaced 硬约束。
3. 完善 `_apply_mib_templates()`。
4. 完善 `_build_group_items()`，在建组时剔除 boundary 成员。
5. 保留 alphatest 的 core rigid item beam-search。
6. 新增 `_extract_free_regions()` 或复用 `_extract_free_rectangles()`。
7. 新增 `_assign_soft_blocks_to_regions()`。
8. 新增 `_solve_region_with_pava()`。
9. 改造 `_place_boundary_blocks()`，加入角点延展枚举。
10. 增加硬约束检查和 fallback。
11. 运行评测并输出：
    - `sol/BF3/result.txt`
    - `sol/BF3/analyze.txt`

测试命令计划：

```powershell
python .\iccad2026_evaluate.py --evaluate .\sol\BF3\my_optimizer.py *> .\sol\BF3\result.txt
python .\analyze_cost_contributions.py .\sol\BF3\my_optimizer.py *> .\sol\BF3\analyze.txt
```

## 14. soft region 内部改为 Slicing Tree 的设计

本节设计只替换当前 `_solve_region_with_pava()` 的“统一高度横向链”，不改动 BF3 的外层框架：

1. preplaced、fixed、MIB template、grouping item 仍先作为刚性对象完成 core 布局。
2. boundary block 仍在 core 完成后做外围放置。
3. 普通 soft block 仍先按容量和 HPWL guide 分配到互不相交的 `FreeRect`。
4. 每个 `FreeRect` 内部改用局部 slicing tree 生成二维布局。
5. 当前 PAVA 实现保留为候选和失败回退，不直接删除。

这样改动的边界比较清楚：slicing tree 只负责“已分配到同一空白矩形的一组普通 soft block 如何在二维空间内排列”，不会干扰现有硬约束和 boundary/grouping/MIB 处理。

### 14.1 为什么局部 slicing tree 可行

进入该阶段的 block 原则上都是可自由变形的普通 soft block，只要求面积满足：

```text
w_i * h_i = area_i
```

题目没有 aspect ratio 硬限制。因此只要某个 region 满足：

```text
sum(area_i) <= region.w * region.h
```

就可以选择一个完全位于 region 内、面积等于 `sum(area_i)` 的 active root rectangle，再递归按子树面积比例切分。每个叶子最终获得的矩形面积会严格等于自己的目标面积。

相比当前 PAVA：

- PAVA 强制所有 block 使用同一个高度，只能形成一条横向链。
- slicing tree 可以递归混合水平切分和垂直切分，形成多行、多列和不等宽子结构。
- 两者都能按构造保证无重叠和面积合法。

抽样统计当前 BF3 的 region 规模：

```text
N=40:  region 内平均 1.14 个 block，最大 2
N=60:  region 内平均 1.62 个 block，最大 3
N=80:  region 内平均 2.49 个 block，最大 7
N=100: region 内平均 2.18 个 block，最大 9
N=120: region 内平均 3.26 个 block，最大 19
```

因此局部树通常很小，适合做有限候选搜索，不需要全局模拟退火。

### 14.2 数据结构

新增两个轻量结构，先直接放在 `my_optimizer.py` 内：

```python
class SliceNode:
    block_id: Optional[int]
    cut: Optional[str]       # "H" / "V" / None
    left: Optional[SliceNode]
    right: Optional[SliceNode]
    area: float

class RegionLayoutCandidate:
    root: SliceNode
    root_rect: Tuple[float, float, float, float]
    positions: Dict[int, Tuple[float, float, float, float]]
    score: float
```

叶子结点对应一个 soft block；内部结点只记录 `H` 或 `V` 切分以及左右子树。

### 14.3 active root rectangle 候选

设 region 尺寸为 `(W_R, H_R)`，已分配 block 总面积为：

```text
A = sum(area_i)
```

不要求 slicing tree 填满整个 region，而是在 region 内选一个面积恰好为 `A` 的 active root rectangle。设 root 宽高比为 `rho = W/H`，则：

```text
W = sqrt(A * rho)
H = sqrt(A / rho)
```

能放入 region 的宽高比区间为：

```text
rho_min = A / H_R^2
rho_max = W_R^2 / A
```

root shape 从下列值投影到 `[rho_min, rho_max]` 后去重：

- `1.0`：接近正方形。
- `W_R / H_R`：匹配 region 形状。
- `target_span_x / target_span_y`：匹配 pin/邻居 target 的二维分布。
- `0.5`、`2.0`：保留两个温和的长宽变体。
- 区间端点：对应铺满 region 宽度或高度。

root 的绝对位置枚举少量候选：

1. root 中心对齐所有成员的加权 HPWL guide，并 clamp 到 region 内。
2. region 居中。
3. 分别贴近 guide 所在方向的最近边。

这样未使用面积会留在 active root 外部，不需要引入会参与输出的虚拟 block。

### 14.4 面积比例切分

设某内部结点覆盖矩形 `(x, y, W, H)`，左右子树面积分别为 `A_L`、`A_R`。

若采用垂直切分 `V`：

```text
W_L = W * A_L / (A_L + A_R)
W_R = W - W_L
```

两个子树共享父结点高度 `H`，左右放置。

若采用水平切分 `H`：

```text
H_L = H * A_L / (A_L + A_R)
H_R = H - H_L
```

两个子树共享父结点宽度 `W`，上下放置。

递归到叶子时，叶子矩形面积严格等于 `area_i`。因此：

- 普通 soft block 面积误差仅来自浮点误差。
- sibling 子树内部无重叠。
- root 完全位于 free region 内，所以不会与 core 或其他 region 重叠。

### 14.5 target 和线网预处理

沿用当前 `_precompute_soft_guides()`，但把线网进一步整理成普通 Python 邻接表，避免在每个树候选中反复遍历 PyTorch tensor：

- `incident_b2b[block_id] = [(neighbor_id, weight), ...]`
- `incident_p2b[block_id] = [(pin_x, pin_y, weight), ...]`
- `guide[block_id] = (gx, gy, total_weight)`

guide 来源：

1. 固定 pin。
2. 已放置的 core/grouping/MIB/fixed block 中心。
3. 同一 region 内相连 soft block 的 guide 迭代值。

对只有 soft-soft 连线、没有固定锚点的连通分量，初值使用 region 中心，再做 3–4 轮加权邻居平均或加权中位数传播。

### 14.6 初始树生成

对一个含 `m` 个 block 的 region，生成多棵小规模候选树。

每个递归结点同时尝试：

1. `V`：按 `guide_x` 排序。
2. `H`：按 `guide_y` 排序。

排序后不枚举所有切点，只考虑面积前缀最接近下列比例的位置：

```text
1/3, 1/2, 2/3
```

切分代理代价：

```text
split_score =
    lambda_cut * 跨越左右子集的 b2b 权重
  + lambda_balance * abs(A_L - A_R) / (A_L + A_R)
  + lambda_span * 子集内部 target spread
```

含义：

- 尽量不把强连接 block 切到两个子树。
- 避免产生极端不平衡的树。
- 让 target 靠近的 block 落在同一子树。

规模策略：

- `m == 1`：直接生成单叶布局，不建树。
- `2 <= m <= 6`：保留更多 H/V 和切点组合，beam width 取 8。
- `7 <= m <= 20`：每层只保留代理代价最小的 4–6 个组合。
- `m > 20`：使用确定性的 target-aware balanced tree，不做拓扑 beam。

当前抽样最大 region 为 19，因此前两种路径会覆盖绝大多数情况。

### 14.7 叶子顺序和子树方向

对 `V` 切分：

- guide_x 较小的子树放左侧。
- guide_x 较大的子树放右侧。

对 `H` 切分：

- guide_y 较小的子树放下侧。
- guide_y 较大的子树放上侧。

每个内部结点还保留一次 sibling swap 候选。虽然集合划分不变，但左右/上下翻转可能明显降低外部 HPWL。

### 14.8 候选布局评分

树 materialize 后，使用实际 block 中心计算 region 相关线长，而不是只用 guide 距离。

需要计入：

1. region 内部 block-block 边。
2. region block 到已放置 core block 的边。
3. region block 到 pin 的边。
4. region block 到其他尚未确定 region 的边，暂时使用对方 guide。

第一版评分：

```text
score =
    HPWL_incident / hpwl_scale
  + lambda_shape * mean(log(w_i / h_i)^2)
  + lambda_extreme * extreme_aspect_penalty
```

建议初值：

```text
lambda_shape = 0.02
lambda_extreme = 0.05
```

题目允许任意长宽比，因此 shape penalty 不是硬限制，只用于避免出现大量极细长 block。bbox 不需要单独罚，因为 candidate 始终位于既有 free region 内，不会扩大 core bbox。

每个 region 保留评分最好的 3–4 个布局候选。

### 14.9 跨 region 精确选择

若每个 region 只独立选择，会忽略不同 region 之间的 b2b 边。完成所有 region 的候选生成后，做一轮 coordinate descent：

1. 先为每个 region 选择局部最优候选。
2. 固定其他 region，依次枚举当前 region 的 3–4 个候选。
3. 用当前完整布局的真实 HPWL 选择更优候选。
4. 最多做 1–2 轮，若没有改善则提前停止。

所有候选都在原 region 内且面积合法，因此这一步只改变 HPWL，不会引入 overlap 或改变 soft constraint 处理结果。

### 14.10 与 PAVA 的关系

当前 PAVA 不删除，而是作为每个 region 的额外候选：

- 对 target 基本呈一维分布的 region，PAVA 可能仍然更好。
- 对二维 target 分布，slicing tree 通常有更强表达力。
- 最终按同一个实际 HPWL + shape penalty 评分选取。

如果 slicing tree 出现下列情况，直接回退 PAVA：

1. 没有 root shape 能放入 region。
2. materialize 后因浮点误差越过 region 边界。
3. 任一 leaf 面积误差超过 `1e-6` 相对误差。
4. 候选搜索超过该 case 的时间或次数预算。

### 14.11 特殊约束保护

局部 slicing tree 的输入需要再次过滤：

- preplaced、fixed：不进入。
- 已统一模板的 MIB：不进入。
- grouping item 成员：不进入。
- boundary block：不进入。
- 无法合法统一模板的残余 MIB：第一版继续走原 PAVA/刚性 fallback，不允许 slicing tree 单独改变其 MIB shape。

因此 slicing tree 不会改变当前 boundary 优先策略，也不会破坏已经达到 0 的 MIB violation。

### 14.12 运行时间控制

当前 boundary 优先 BF3 的正式评测平均运行时间约为 `0.64s`，所以不能加入重型全局 SA。

建议限制：

- 单 region tree beam width：最大 8。
- root shape：最多 6 个。
- root absolute placement：每个 shape 最多 3 个。
- 每个 region 最终候选：最多 4 个。
- 每次 region-fill tree materialize/score 次数：最多 60 次；当前 `solve()` 最多评估 3 个 core layout，因此全 case 上限约 180 次。
- `N > 100` 时关闭通用 leaf swap，只保留 sibling flip。

如果达到预算，剩余 region 直接使用 PAVA。

### 14.13 正确性和回退

每个 slicing candidate 必须通过局部检查：

1. 所有 leaf 坐标位于所属 `FreeRect` 内。
2. leaf 两两无正面积 overlap。
3. 每个 leaf 的 `w*h` 与目标面积相对误差不超过 `1e-6`。
4. 输出 block id 集合与分配给 region 的集合完全相同。

全局仍保留现有硬约束检查和 fallback。若 slicing 版本未通过完整可行性检查，返回同一个 item layout 下的 PAVA 结果；不能通过移动 preplaced 或修改 fixed 尺寸来修复。

### 14.14 预期效果

预期主要改善 `HPWLgap`：

- 当前横链只能优化 x，所有 region 内 block 的 y 中心相同。
- slicing tree 可以同时响应 x/y target 和 soft-soft b2b 连接。
- active root 仍位于已有 free region 内，理论上不会恶化 bbox area。
- boundary、grouping、MIB 的处理路径不变，违规率应基本保持。

主要风险：

- slicing floorplan 仍不能表达所有非 slicing 拓扑。
- shape 自由度过高时可能产生细长 block，需要温和 shape penalty。
- region 分配本身若不合理，单纯优化 region 内部无法修复跨 region 错配。
- 候选过多会抵消运行时间得分。

如果首版有效，第二阶段再增加一次容量允许下的跨 region block relocation；首版不同时改 assignment 和内部布局，便于确定收益来源。

### 14.15 实现及 A/B 测试顺序

收到确认后按以下顺序实现：

1. 备份当前 boundary 优先 PAVA 版本：
   - `my_optimizer_pava.py`
   - `result_pava.txt`
   - `analyze_pava.txt`
2. 新增 `SliceNode`、root shape 生成和面积比例 materialize。
3. 新增 target-aware 递归树和受限 beam。
4. 将 PAVA 作为 region 候选及 fallback 接入。
5. 加一轮跨 region candidate coordinate descent。
6. 先跑 `--validate` 和 case `59/79/99`。
7. 再跑完整评测并更新主版本 `result.txt`、`analyze.txt`。

当前 boundary 优先 PAVA 基线为：

```text
Feasible             = 100 / 100
正式 Total Score     = 2.0566
分析 Total Score     = 2.012567
HPWLgap              = 0.918604
Areagap_bbox         = 0.297790
Violationsrelative   = 0.155248
Vboundary/Nsoft      = 0.028333
Vgrouping/Nsoft      = 0.126915
Vmib/Nsoft           = 0.000000
Avg Runtime          = 0.64s
```

验收条件：

1. 必须保持 100/100 可行。
2. MIB violation 保持 0，boundary/grouping 不因内部布局明显恶化。
3. 正式 Total Score 优于 `2.0566`；若只改善 HPWL 但运行时间导致总分变差，则保留 per-region 的 PAVA/slicing 自适应选择，或回退 PAVA 主版本。

### 14.16 实现结果

局部 slicing tree 已按本节方案实现到主版本 `my_optimizer.py`，原 boundary 优先 PAVA 版本备份为：

- `my_optimizer_pava.py`
- `sol_pava.md`
- `result_pava.txt`
- `analyze_pava.txt`

实际实现包括：

1. active root rectangle 的面积/长宽比/位置候选。
2. target-aware 的 H/V 递归树和 `1/3、1/2、2/3` 面积分割候选。
3. 子树面积比例 materialize，叶子面积按构造精确满足。
4. region incident edges 预过滤和 Python 边缓存，避免候选内反复扫描 PyTorch tensor。
5. PAVA 作为每个 region 的固定候选和失败回退。
6. 每个 region 最多保留 4 个候选，并做最多两轮跨 region coordinate descent。
7. residual MIB 或 `group_id > 0` 的 soft block 继续使用 PAVA，防止 MIB/grouping 回归。
8. 候选预算经 case 99 扫描后设为 60；该预算已取得与 96 相同的 case 99 HPWL，而开销更低。

完整验证集 A/B 结果：

```text
指标                         PAVA 基线       Slicing + PAVA
Feasible                     100 / 100       100 / 100
正式 Total Score             2.0566          2.0564
正式 Avg Cost                2.2052          2.2033
正式 Avg Runtime             0.64s           0.32s
HPWLgap                      0.918604        0.916014
Areagap_bbox                 0.297790        0.297790
Violationsrelative           0.155248        0.155248
Vboundary/Nsoft              0.028333        0.028333
Vgrouping/Nsoft              0.126915        0.126915
Vmib/Nsoft                   0.000000        0.000000
分析脚本 Contest Total Score 2.012567        1.712835
```

结论：第一版 slicing tree 的质量收益较小，但它保持了全部硬约束和原有软约束水平，并使 HPWL 稳定下降。当前主版本保留自适应选择：只有 slicing candidate 的 region 评分优于 PAVA 时才采用，否则仍返回 PAVA 布局。

## 15. Boundary 条带压缩与沿边 PAVA

### 15.1 原问题

在原实现中，四个角块先单独选择形状，然后才迭代求 left/right/top/bottom 四侧扩展量。顶边和底边的普通 soft boundary block 又直接使用整个 `T/B` 作为统一高度。

这会产生两个问题：

1. 某个角块很高时，整条 top/bottom side 都被迫继承同样高度，即使普通边模块沿 x 方向有大量空余宽度。
2. 角块方向选择只看 core 尺寸，没有考虑左右边条带本身已经需要的宽度。例如 left strip 已经需要宽度 `L` 时，把 top-left 角块加宽到 `L` 往往不会增加左扩展，却能明显降低 top 扩展。

case 97 的原布局中：

```text
top side 理论软条带高度约 10.63
实际统一使用高度约 40.13
```

主要原因是 top-left 角块高度约 `40.13`，被错误传播为整条顶边高度。

### 15.2 联合角块与四边厚度求解

当前版本对每个软角块生成有限形状候选：

1. 按 core 长宽比生成的方向。
2. 上述形状旋转 90 度。
3. 正方形。
4. 宽度匹配相邻 left/right side 的估计厚度。
5. 高度匹配相邻 top/bottom side 的估计厚度。

对最多四个角做组合枚举。每个组合不再只看角块自身 bbox，而是完整运行四边厚度固定点：

```text
corner dimensions
    -> L/R/T/B 初值
    -> 计算四边剩余可用长度
    -> 计算 side block 所需最小厚度
    -> 更新 L/R/T/B
    -> 收敛后计算最终 bbox area
```

选择最终 bbox area 最小的角块形状组合。

### 15.3 side block 使用独立最小厚度

最终外框 `L/R/T/B` 可能由角块决定，但普通边模块不再继承整个外框厚度。

对 top/bottom：

```text
side_h = max(
    fixed block 最大高度,
    soft block 总面积 / 扣除 fixed 宽度后的可用宽度
)
```

对 left/right 对称计算 `side_w`。

模块仍贴住最终指定边，只在模块与 core 之间留下合法空隙。因此 boundary 约束保持满足，同时 soft block 通过增加沿边长度来降低垂直边框厚度。

### 15.4 沿边 PAVA

单纯压薄条带后，如果仍按 fixed/area 顺序从一端紧排，会让 block 中心大幅偏离 pin 和邻居，导致 HPWL 上升。

当前版本对四条边分别做一维 PAVA：

- top/bottom 在 x 方向求解。
- left/right 在 y 方向求解。
- 每个 block 的线段长度由 fixed 尺寸或 `area / strip_thickness` 决定。
- target 来自 pin、已放置 b2b 邻居和同 grouping 的已放置成员。
- grouping target 使用较高权重，尽量在保持 boundary 的同时恢复组连通。

PAVA 保证边模块在线段范围内互不重叠，并允许模块之间保留空隙，以降低沿边 HPWL。

### 15.5 case 97 效果

```text
指标                    修改前       修改后
local cost              2.2221       1.9587
HPWLgap                 0.9664       0.8433
Areagap                 0.2180       0.1572
Vrel                    0.1667       0.1333
top soft strip height   40.13        10.63
final top               297.13       280.92
```

更新后的图为 `my_optimizer_test_97.png`。

### 15.6 完整验证集结果

与 boundary 压缩前的 slicing + PAVA 版本相比：

```text
指标                         修改前          修改后
Feasible                     100 / 100       100 / 100
正式 Total Score             2.0564          1.8790
正式 Avg Cost                2.2033          2.0424
正式 Avg Runtime             0.32s           0.47s
HPWLgap                      0.916014        0.841005
Areagap_bbox                 0.297790        0.270279
Violationsrelative           0.155248        0.132951
Vboundary/Nsoft              0.028333        0.028333
Vgrouping/Nsoft              0.126915        0.104617
Vmib/Nsoft                   0.000000        0.000000
grouping violation units     564             455
```

结论：联合角块/边厚度优化解决了 top/bottom soft boundary block 过高的问题；沿边 PAVA 同时避免了压缩后的 HPWL 回归，并额外降低了 grouping violation。硬约束和 MIB 均未恶化。

## 16. 角块沿边延伸

### 16.1 原几何模型的问题

上一版虽然允许软角块改变长宽比，但仍把角块的完整宽度计入 L/R，同时把完整高度计入 T/B。以右下角块为例，相当于强制：

    R >= corner_width
    B >= corner_height

实际上角块只要贴住外框的右边和下边，并且不与 core 相交即可。合法条件应为：

    corner_width <= R  or  corner_height <= B

因此，右下角块可以采用两种模式：

1. horizontal：由高度决定 B，宽度沿底边向左延伸。
2. vertical：由宽度决定 R，高度沿右边向上延伸。

其余三个角完全对称。旧模型错误地要求两个方向都在 core 外部，因而会留下大块空白角区。

### 16.2 双模式联合枚举

当前版本对每个软角块的形状候选继续枚举，并为每个形状同时尝试 horizontal 与 vertical：

    corner shapes × corner modes
        -> 计算模式对应的 L/R/T/B 最小值
        -> 与四条 side strip 的厚度做固定点迭代
        -> 检查同一条边两端角块不相交
        -> 按最终 bbox area 选择最优组合

同一条边上的角块额外满足：

    w_tl + w_tr <= total_width
    w_bl + w_br <= total_width
    h_tl + h_bl <= total_height
    h_tr + h_br <= total_height

所以角块可以沿边进入原先的空白区域，但不会彼此重叠。

### 16.3 case 97

块 32 和块 39 都是 soft corner block。求解器也评估了将它们拉宽成扁块的方案，但最终面积更小的是 vertical 模式：

    block 32: 14.9024 x 40.1277，向下沿左边延伸
    block 39: 14.8482 x 38.7925，向上沿右边延伸
    top extent T    = 10.6346
    bottom extent B = 7.7323

这与向左右拉宽的目标等价：角块不再决定上下边界厚度；同时窄高形状可以利用左右边已有的厚度，所以最终更优。

    指标                 修改前          修改后
    local cost           1.9587          1.8478
    HPWLgap              0.8433          0.8188
    Areagap              0.1572          0.0117
    Vrel                 0.1333          0.1333
    bbox width           135.2918        137.3330
    bbox height          319.7125        275.3669
    bbox area            43254.48        37816.97

### 16.4 完整验证集结果

    指标                         修改前          修改后
    Feasible                     100 / 100       100 / 100
    正式 Total Score             1.8790          1.7886
    正式 Avg Cost                2.0424          1.9200
    正式 Avg Runtime             0.47s           1.12s
    HPWLgap                      0.841005        0.807605
    Areagap_bbox                 0.270279        0.142459
    Violationsrelative           0.132951        0.129294
    Vboundary/Nsoft              0.028333        0.028333
    Vgrouping/Nsoft              0.104617        0.100961
    Vmib/Nsoft                   0.000000        0.000000
    grouping violation units     455             438

角块模式枚举显著降低了边界留白，且 100 例均保持可行。代价是角块组合数增加，平均运行时间有所上升。
