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
   - BF3 优先 grouping。
   - 若某个 block 同时有 boundary 和 grouping 约束，则把它放进 grouping 子问题，并忽略它的 boundary 约束。
   - 这样可以保证 group item 内部连通，不再把该 block 后挂到外围。

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
- `boundary + grouping` 优先 grouping 后，group 连通性更稳定。
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
   - `boundary + grouping` 优先 grouping。
   - `preplaced + grouping` 优先 preplaced。
   - MIB 永远不破坏 fixed/preplaced 硬约束。
3. 完善 `_apply_mib_templates()`。
4. 完善 `_build_group_items()`，支持 boundary 成员进入 grouping item。
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
