# HPWL 简化版做法设计

## 1. 本文只解决的版本

只考虑下面这一个简化版本：

- 约束：只保证所有硬约束
  - 无重叠
  - 普通软块面积误差不超过 1%
  - fixed-shape 块的 `w/h` 必须与给定值完全一致
  - preplaced 块的 `x/y/w/h` 必须与给定值完全一致
- 目标：只优化 `HPWLgap + Areagap_bbox`
- 测试口径：只看 `analyze_cost_contributions.py` 输出里的平均 `HPWLgap` 和平均 `Areagap_bbox`，其余项不作为本版本目标

因此，这一版可以明确忽略：

- grouping / boundary / MIB 的满足情况
- 运行时间因子
- 总 cost 里的软约束指数惩罚

但要注意：评测脚本本身仍然会读取这些字段，所以实现时要保证我们只是“不优化它们”，不是把 fixed / preplaced 这些硬约束也一起忽略掉。

---

## 2. 先验观察

我先看了验证集和评测接口，有几个结论直接影响做法：

1. `fixed` 和 `preplaced` 在 100/100 个验证样例里都出现。
2. 平均每个样例大约有：
   - `7.12` 个 fixed-shape 块
   - `2.59` 个 preplaced 块
3. 平均每个样例大约有：
   - `994` 条 b2b 边
   - `705` 条 p2b 边
   所以 HPWL 信号很强，不能只做单纯 packing。
4. preplaced 块形成的已占区域并不小，平均约占总面积的 `29%`。因此 legalizer 必须把它们当障碍物处理，不能假设是一张空白画布。
5. 评测器对 hard constraint 的判定很直接：
   - soft block 只查面积
   - fixed 只查尺寸
   - preplaced 查位置和尺寸
   - 容差约为 `1e-4`

这说明本版本最合理的方向不是去管软约束，而是做一个：

- **连线驱动的全局摆放**
- **preplaced-aware 的合法化**
- **围绕 exact HPWL 做局部改良**

---

## 3. 总体框架

我建议做成一个多阶段、可多次重启的流程：

1. **硬约束预处理**
2. **全局中心点求解**：先得到每个可动块的理想中心
3. **形状分配**：给软块选 `w/h`
4. **合法化**：在 fixed/preplaced 约束下消除重叠
5. **精确目标局部优化**：直接下降 `HPWL + λ * BBoxArea`
6. **多启动保底**：保留最好的合法解

其中：

- preplaced 块始终固定不动
- fixed-shape 块可移动，但尺寸固定
- 普通 soft 块尺寸可变，但必须满足面积

因为这次不看 runtime，可以适当多做几次 restart 和局部搜索，把更多算力换成更低的 `HPWLgap + Areagap_bbox`。

---

## 4. 具体设计

### 4.1 硬约束预处理

把块分成三类：

1. `preplaced`
   - 位置和尺寸全固定
   - 直接作为障碍物放入版图
2. `fixed-shape`
   - `w/h` 固定
   - `x/y` 需要求
3. `soft`
   - 只要求 `w*h = area_target`
   - `x/y/w/h` 都可调

同时预先建好：

- `b2b` 邻接表
- `p2b` 邻接表
- 每个块的总连接权重
- 与 pin / preplaced 相连的锚点信息

这一步的目标是把后面每个局部 move 的增量 HPWL 计算压到只看邻居，而不是全局重算。

---

### 4.2 全局中心点初始化

先不考虑重叠，只求每个可动块的理想中心 `(cx, cy)`。

建议先解一个二次型全局摆放：

\[
\min \sum_{(i,j)\in E_{b2b}} w_{ij} \left[(cx_i-cx_j)^2 + (cy_i-cy_j)^2\right]
 + \sum_{(p,i)\in E_{p2b}} u_{pi} \left[(cx_i-px_p)^2 + (cy_i-py_p)^2\right]
\]

其中：

- pin 坐标是固定锚点
- preplaced 块中心也是固定锚点
- 其他块是变量

这一步不直接优化 HPWL，而是用 L2 版本先拿一个稳定、全局一致的种子。原因是：

- L2 目标可以直接拆成 `x/y` 两个线性系统
- 在连线很密的时候，初始拓扑会比随机摆放稳定很多
- 后面再用 weighted median 局部下降去逼近真正的 L1 HPWL

实现上：

- 对 `x` 和 `y` 分别解一次稀疏线性方程
- preplaced 从变量中消掉，贡献进右端项
- 对没有任何连线的块，直接把中心放到当前已知块包围盒中心附近

---

### 4.3 软块形状分配

对 soft block，面积固定，但长宽比可选。

这里不建议全都用正方形，而是按局部连线方向做各向异性分配。核心想法：

- 如果一个块主要在水平方向上拉线，就让它更扁、更宽
- 如果主要在竖直方向上拉线，就让它更高、更窄

对每个 soft 块 `i` 计算：

\[
H_i = \sum w \cdot |x\text{-distance}|
\]

\[
V_i = \sum w \cdot |y\text{-distance}|
\]

然后取

\[
r_i = \text{clip}\left(\sqrt{\frac{H_i+\epsilon}{V_i+\epsilon}},\ r_{min},\ r_{max}\right)
\]

再令

\[
w_i = \sqrt{A_i \cdot r_i}, \quad h_i = \sqrt{A_i / r_i}
\]

建议：

- `r_min = 0.25`
- `r_max = 4.0`

这样可以避免极端细条，减轻 legalizer 难度。

此外，这一步不要只保留一个比例。实际实现时可给每个 soft 块保留少量候选：

- `square`
- `r_i`
- `0.5 * r_i`
- `2.0 * r_i`

后面在局部搜索中再做选择。

---

### 4.4 主合法化器：交替约束投影

全局摆放和形状确定后，会得到一批可能重叠的矩形。接下来要把它们变成合法解。

主 legalizer 建议做成“横向一遍 + 纵向一遍”的交替过程。

#### X-pass

冻结所有块的 `y/h`，只调 `x`。

做法：

1. 找出所有在 `y` 上有重叠的块对
2. 按当前理想中心的 `x` 顺序给这些块对定左右关系
3. 形成一张差分约束图：
   - 若 `i` 必须在 `j` 左边，则要求  
     `x_j >= x_i + w_i`
4. preplaced 块的 `x` 是硬锚点，不能改
5. 在这个顺序下做 longest-path / 最早可行位置求解

这样做完以后，所有“当前在 y 上相交的块”都会在 `x` 方向被分开。

#### Y-pass

同理，冻结 `x/w`，只调 `y`：

1. 找出所有在 `x` 上有重叠的块对
2. 按当前理想中心的 `y` 顺序定上下关系
3. 建差分约束：
   - `y_j >= y_i + h_i`
4. preplaced 的 `y` 固定

#### 交替迭代

`X-pass -> Y-pass` 反复做几轮，直到：

- 没有 overlap
- 或者 overlap 已经不再下降

这个 legalizer 的优点是：

- 和 wirelength seed 一致，不是纯 bottom-left 乱塞
- 能自然处理 preplaced 障碍，因为它们就是图里的固定节点
- fixed-shape 块和 soft 块都能统一处理

---

### 4.5 保底合法化器：obstacle-aware shelf packing

交替投影不一定对每个 case 都足够稳，尤其是：

- preplaced 形成复杂孔洞
- 某些块长宽比比较极端
- 当前顺序本身就不适合

因此需要一个**一定能产出合法解**的 fallback。

保底方案建议用 obstacle-aware shelf packing：

1. 先把所有 preplaced 块放进去，作为已占矩形
2. 剩余块按下面的优先级排序：
   - fixed-shape 优先于 soft
   - 连接权重大、面积大的优先
   - 同优先级按理想中心 `(cy, cx)` 排
3. 维护一组可放置的 shelf / row
   - 起始行来自 `y=0`
   - 新行高度来自已有块顶边和 preplaced 顶边
4. 对当前块，枚举若干最接近理想 `cy` 的 row
5. 在该 row 内扣掉障碍物区间，找到能放下块的空闲 `x` 区间
6. 选择“最接近理想中心，且对 bbox 增量最小”的位置
7. 如果所有现有 row 都放不下，就在当前最高处新开一行

这个 fallback 的作用不是为了极致，而是保证：

- 始终能满足无重叠
- preplaced 一定不被撞
- fixed / preplaced 的硬约束不失效

只要 fallback 是稳定的，整个优化器就不会因为某个 case legalize 失败而直接崩掉。

---

### 4.6 精确目标局部优化

得到合法解后，优化重点就转成：

- 降低 exact HPWL
- 同时压缩 bbox area

这一阶段直接对合法状态做 hill-climbing / coordinate descent。

#### 目标函数

求解器内部不直接知道 baseline，所以不能直接算 gap。内部建议优化：

\[
F = HPWL_{b2b} + HPWL_{p2b} + \lambda_{bbox} \cdot Area_{bbox}
\]

然后用验证集调 `\lambda_{bbox}`。

推荐把 `\lambda_{bbox}`` 作为超参数扫一遍，比如：

- `0.01`
- `0.03`
- `0.1`
- `0.3`
- `1.0`

最后按 `analyze_cost_contributions.py` 的平均

\[
\text{avg HPWLgap} + \text{avg Areagap\_bbox}
\]

选最优。

#### move 1: 加权中位数平移

对每个 movable block：

1. 固定其他块
2. 先看 `x` 方向
   - 把所有相连块中心和 pin 的 `x` 收集起来
   - 用连接权重求 weighted median，得到理想 `cx`
   - 再把它裁到当前合法顺序允许的区间内
3. `y` 方向同理

因为 HPWL 是 L1 距离，weighted median 比均值更贴近真实目标。

#### move 2: 形状重选

只对 soft block 做：

- 在几组候选比例里切换
- 每切一次都保持面积精确不变
- 只要 legal 且 `F` 更小就接受

直觉上：

- 横向 net 密的块变宽，有利于降 HPWL
- 纵向 net 密的块变高
- 但形状变化会影响 packing，所以必须和合法性一起看

#### move 3: 邻近吸附

对每个块，尝试把它的某条边吸到：

- 邻块边界
- preplaced 边界
- 当前 bbox 边界附近

这样经常可以：

- 少量降低 HPWL
- 顺手吃掉一部分空白
- 进一步减小 bbox

#### move 4: 小规模交换顺序

如果两个相邻块在 legalizer 顺序里互换后更优，就接受交换并重新局部合法化。

这个 move 主要用于修复：

- 全局 seed 的相对顺序不准
- 某个 row 内局部线序不合理

---

### 4.7 多启动

因为这一版不关心 runtime，所以很适合做少量 multi-start。

建议每个 case 跑 `4~8` 次，变化来源包括：

- 全局摆放初始正则项不同
- soft block 初始比例不同
- legalizer tie-break 随机种子不同
- 局部搜索块遍历顺序不同

最后只保留：

- 完全满足硬约束
- 内部目标 `F` 最小

的那个解。

---

## 5. 为什么这个方案适合当前简化版

### 5.1 它优先解决了真正必须解决的问题

这版不是“软约束很多怎么折中”，而是：

- preplaced/fixed 每个 case 都有
- 连线非常密
- 只看 HPWL 和 bbox

因此最关键的是：

1. 把锚点约束处理正确
2. 把线长信号吃进去
3. 用一个稳定 legalizer 落成合法解

这个方案正好按这三件事来分层。

### 5.2 它比纯 packing 更贴目标

如果只做 row packing / skyline，通常能保面积，但 HPWL 会很差；  
如果只做解析摆放，不做强 legalize，又很容易撞到 preplaced 障碍。

这里的组合是：

- 前段先用连线信息决定大致拓扑
- 中段再做带锚点的 legalize
- 后段用 exact HPWL 做局部修正

比单一技术更贴这个简化目标。

### 5.3 它保留了后续扩展空间

虽然这版明确不考虑 soft constraints，但如果后面要逐步加回：

- boundary
- grouping
- MIB

也可以继续沿用这套框架，只是在 legalizer 和 local move 里多加约束项即可，不需要推倒重写。

---

## 6. 预期实现顺序

真正写代码时，我建议按下面顺序落地，而不是一口气全写完：

1. **先写一个稳定合法版本**
   - 正确处理 preplaced / fixed
   - 先用 square soft blocks
   - 先把 fallback shelf legalizer 写稳
2. **再补全局摆放**
   - 引入 quadratic placement seed
3. **再补局部 HPWL 优化**
   - weighted median 平移
   - 邻近吸附
4. **最后补形状搜索和 multi-start**

这样能确保每一步都能单独验证，不会一开始就陷入“目标更优但经常非法”的状态。

---

## 7. 最终结论

这个简化版我建议采用：

- **preplaced-aware 全局摆放**
- **交替约束投影合法化**
- **exact HPWL + bbox 的局部搜索**
- **fallback shelf legalizer 保底**
- **multi-start 选最优**

这套做法的核心不是去碰软约束，而是把：

- 锚点
- 连线
- 合法性
- 紧凑度

这四件事拆开，各自做对，然后在合法状态上继续压 `HPWL + bbox`。

如果后续确认这个方向，就按这份 `sol.md` 写 `sol/HPWL/my_optimizer.py`。
