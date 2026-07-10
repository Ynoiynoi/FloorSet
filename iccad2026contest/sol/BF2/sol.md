# BF2：基于 eazyver2 的 HPWL-aware 做法设计

## 1. 本版目标

BF2 以 `sol/eazyver2/my_optimizer_mib_square.py` 的框架为基础，不重写整体求解器，而是在现有“grouping item + MIB square template + boundary 后处理 + 面积优先装箱”的流程中加入线长信号。

目标优先级：

1. 必须满足硬约束：
   - preplaced 的 `x/y/w/h` 完全不变
   - fixed-shape 的 `w/h` 完全不变
   - 普通 soft block 面积满足 1% 容差
   - 所有 block 无重叠
2. 保留 eazyver2 已有的软约束处理：
   - MIB 组统一模板
   - 非 preplaced、非 boundary 的 grouping 组压成连通 item
   - boundary block 在 core 完成后贴边放置
3. 在不破坏前两点的前提下，降低 `HPWLgap`，同时控制 `Areagap_bbox` 不明显恶化。

现有 BF1/eazyver2 系列的主要短板是主搜索阶段只按 bbox 面积排序，HPWL 只在最后做很弱的局部修补。BF2 的核心改动是：从 item 排序、候选点选择、soft block 填洞、boundary 顺序到最终局部搜索，都把 HPWL 作为显式评分项。

## 2. 现有流程中需要改的位置

`my_optimizer_mib_square.py` 的关键流程如下：

1. `_build_blocks`
2. `_apply_mib_templates`
3. `_build_group_items`
4. `_search_item_layouts`
5. `_fill_soft_singles`
6. `_finalize_layout`
7. `_place_boundary_blocks`
8. `_resolve_all_overlaps`

BF2 不改变 1-3 的语义，主要修改 4-7，并把最后的 overlap repair 改成硬约束安全版本：不能移动 preplaced，也不能修改 fixed/MIB 尺寸。

## 3. 新增数据结构：NetModel

在 `solve()` 中，完成 blocks 和 group_items 后，先建立一个轻量线网模型：

- `b2b_edges`: `(block_i, block_j, weight)`
- `p2b_edges`: `(pin_idx, block_id, weight)`
- `incident_b2b[block_id]`
- `incident_p2b[block_id]`
- `block_weight[block_id]`: 与该 block 相连的总权重
- `pin_weight[block_id]`: 与固定 pin 相连的总权重
- `item_members[item_id]`: 该 item 包含的原始 block 列表
- `block_to_item[block_id]`

这样候选点评分时不需要全局重算 HPWL，只看当前 item 内 block 的 incident edges。

对于 grouping item：

- item 的外部移动会同时移动组内所有 member
- 组内 b2b 边在 item 平移时长度不变，可在候选排序时忽略
- 组内到外部 block / pin 的边要计入 HPWL proxy

## 4. 线长锚点估计

主装箱是增量过程，放某个 item 时，很多相连 block 可能还没放置。因此需要给未放置 block 一个临时目标中心。

BF2 使用 weighted-median 风格的目标中心估计：

1. preplaced block 的中心固定为输入中心。
2. pin 作为固定锚点。
3. 其他 block 初始放在当前固定 bbox 中心；如果没有 preplaced，则用原点附近的紧凑初值。
4. 做 6-8 轮迭代：
   - 对每个 movable block，收集相连 block 上一轮中心和相连 pin 坐标
   - 分别对 x/y 求加权中位数
   - 没有线网的 block 保持在全局中心

得到 `target_center[block_id]` 后，再推导 item 的目标左下角：

`target_item_xy = weighted_average(target_center[member] - local_center[member])`

这个目标不会直接作为最终位置，只作为候选排序和 tie-break 的 wire anchor。

## 5. HPWL-aware item 排序

eazyver2 目前的 `_make_item_orders()` 主要按面积、长边和接近方形程度排序。BF2 增加线网优先顺序：

1. 面积优先：保留原排序，保证 packing 稳定。
2. 线网权重优先：`-(sum block_weight)`，高连接度 item 先放。
3. pin/preplaced 锚点优先：与固定锚点连接强的 item 先放。
4. target sweep：
   - 按目标 x 排
   - 按目标 y 排

最后去重，作为多种 beam-search order。这样仍保留面积基线，但给 HPWL 更强的早期决策权。

## 6. HPWL-aware 候选点生成

保留 eazyver2 的边界候选点：

- `x0`, `x1`, `x0 - w`, `x1 - w`
- 已放矩形的左边、右边、贴左、贴右
- y 方向同理

新增目标附近候选点：

- `target_x - w/2`
- `target_y - h/2`
- target x 与已有 y 边界组合
- target y 与已有 x 边界组合
- target 附近最近的若干 snap-to-edge 位置

每个候选点都必须通过 overlap 检查；preplaced 会作为 occupied obstacle 参与检查。

建议参数：

- `beam_width = 3` 或 `4`
- `state_candidate_limit = 6` 或 `8`
- 大 case 可降低到 `beam_width = 2`，避免 runtime 过高

## 7. 候选点评分函数

原 eazyver2 使用：

`score = (bbox_area, bbox_perimeter, max_side)`

BF2 改为归一化混合目标：

```text
score =
    hpwl_weight * HPWL_proxy_norm
  + bbox_weight * BBox_norm
  + anchor_weight * AnchorDist_norm
```

其中：

- `HPWL_proxy_norm`：
  - 对已放置邻居和 pin，使用真实中心距离
  - 对未放置邻居，使用 `target_center`
  - 只计算当前 item 相关边，降低复杂度
- `BBox_norm = new_bbox_area / total_block_area`
- `AnchorDist_norm = item_center 到 item target_center 的加权曼哈顿距离 / sqrt(total_block_area)`

初始权重建议：

- `hpwl_weight = 1.0`
- `bbox_weight = 0.20`
- `anchor_weight = 0.15`

排序 tuple 建议为：

```text
(
    mixed_score,
    new_bbox_area,
    exact_known_hpwl_delta,
    distance_to_target,
    new_bbox_perimeter,
)
```

这样 HPWL 是主目标，但 bbox 仍能兜住面积。

## 8. soft single 填洞改造

eazyver2 的 `_fill_soft_singles()` 当前按面积找 free rectangle，并按 leftover 最小选择 bin。BF2 改成：

1. soft single 排序：
   - 高 `block_weight` 优先
   - 面积大优先
   - 与 pin/preplaced 连接强优先
2. 对每个 free rectangle，枚举少量形状：
   - square
   - net-direction aspect ratio
   - `0.5x` / `2x` 的温和变体
3. 在 free rectangle 内把 block center 尽量放到 `target_center`
   - 位置用 clamp，保证完整落在 free rectangle 内
   - 同时枚举贴 free rectangle 四边的 snap 候选
4. 用同一个混合评分选择 `(bin, shape, x, y)`

如果所有洞都放不下，保留 strip fallback，但 strip 内顺序不再按面积，而是按目标坐标排序：

- 横向 strip 按 `target_x`
- 纵向 strip 按 `target_y`

这样不会牺牲“总能给出合法布局”的保底能力。

## 9. boundary 放置的 HPWL 改造

boundary block 的边/角由约束决定，不能为了 HPWL 改边。可优化的是同一条边上的顺序和沿边坐标。

改造方式：

1. corner block 仍按 eazyver2 固定在四角。
2. left/right 边：
   - 根据 `target_y` 排序，而不是只按 fixed/area 排序
   - 固定尺寸保持不变；soft block 用 strip 宽度反推高度
3. top/bottom 边：
   - 根据 `target_x` 排序
   - soft block 用 strip 高度反推宽度
4. 放完后做同边相邻 swap：
   - 只交换同一条边上相邻 block 的顺序
   - 必须保持贴边、无重叠
   - 若 exact HPWL 下降且 boundary violation 不增加，则接受

这一步代价小，但对 pin-to-block HPWL 通常有帮助。

## 10. 完整布局后的局部优化

完成 core + soft singles + boundary 后，做 1-2 轮安全局部优化。

### 10.1 item 级 remove-and-reinsert

对非 preplaced item：

1. 临时移除该 item。
2. 用当前其余 block 作为 occupied。
3. 重新生成候选点。
4. 用完整目标 `F` 评分。
5. 如果更好且不新增硬约束违规，则接受。

完整目标：

```text
F =
    (HPWL_b2b + HPWL_p2b) / hpwl_scale
  + lambda_bbox * bbox_area / total_block_area
```

其中：

- `hpwl_scale = total_net_weight * sqrt(total_block_area)`
- `lambda_bbox = 0.20` 起步

### 10.2 standalone soft shape retry

只对非 fixed、非 preplaced、非 MIB、非 grouping item 的 standalone soft block 做：

- 固定面积不变
- 尝试少量 aspect ratio
- 每次先在原中心附近合法化
- exact `F` 更好才接受

MIB block 不参与该步骤，避免破坏 MIB shape consistency。

### 10.3 adjacent swap 保留

可以保留 BF1 里已经存在的相邻交换思路，但作为最后的小修：

- 不交换 preplaced
- 不改变 fixed 尺寸
- 不引入 overlap
- 不增加 boundary/grouping/MIB violation
- exact HPWL 下降才接受

## 11. 硬约束安全策略

BF2 的所有 move 都必须走同一套 guard：

1. `preplaced` 永远不可移动。
2. `fixed` 只允许移动 `x/y`，不允许改 `w/h`。
3. MIB template 一旦确定，不再被单独改 shape。
4. grouping item 内部局部坐标不被 item-level move 打散。
5. 若没有 preplaced，可以整体 shift 到原点；若有 preplaced，不能整体平移。
6. overlap repair 不能移动 preplaced；如果需要修复，只移动非 preplaced block。
7. 如果 HPWL-aware 结果不合法，回退到未做 HPWL 改造前的合法基线结果。

实际编码时，建议在 `solve()` 末尾保留两个候选：

- `base_positions`: 原 eazyver2/BF1 风格结果
- `hpwl_positions`: BF2 改良结果

最终只在 `hpwl_positions` 通过硬约束检查且内部目标更优时返回它，否则返回 `base_positions`。

## 12. 预期效果与风险

预期改善：

- 主装箱阶段靠近相连 pin/preplaced/neighbor，`HPWLgap` 应比只做 bbox packing 更低。
- boundary block 沿边排序后，p2b HPWL 会更合理。
- soft single 填洞不再只看 leftover，减少“洞填得很满但线长很差”的情况。

主要风险：

- HPWL 权重过高会拉大 bbox，导致 `Areagap_bbox` 上升。
- beam/candidate 增大后 runtime 会增加。
- target 估计只是 proxy，不能保证每次都比面积优先更好。

控制方式：

- 保留面积优先 order 作为候选之一。
- 最终用 exact full objective 选择布局。
- 若可行性或目标恶化，回退到 base。
- `bbox_weight` 和 beam 参数按 `analyze_cost_contributions.py` 结果微调。

## 13. 代码落地顺序

等方案确认后，建议按下面顺序写 `sol/BF2/my_optimizer.py`：

1. 从 `sol/eazyver2/my_optimizer_mib_square.py` 拷贝基础版本。
2. 补入 BF1 中已经验证过的硬约束检查和 adjacent swap guard。
3. 新增 `NetModel` 和 target center 估计。
4. 改 `_make_item_orders()`，加入 net-aware orders。
5. 改 `_rank_item_candidates()`，加入 target 候选和混合评分。
6. 改 `_fill_soft_singles()`，做 HPWL-aware bin/shape/position 选择。
7. 改 `_place_boundary_blocks()`，按 target coordinate 排序并做同边 swap。
8. 加完整布局后的 remove-and-reinsert refine。
9. 运行评测并写入：
   - `sol/BF2/result.txt`
   - `sol/BF2/analyze.txt`

测试命令计划：

```powershell
python .\iccad2026_evaluate.py --evaluate .\sol\BF2\my_optimizer.py *> .\sol\BF2\result.txt
python .\analyze_cost_contributions.py .\sol\BF2\my_optimizer.py *> .\sol\BF2\analyze.txt
```

