# PeF Python 简化复现

这是一个可直接运行的 Python 版本，用随机生成的数据复现论文《PeF: Poisson’s Equation Based Large-Scale Fixed-Outline Floorplanning》的核心思路，并输出可视化结果。

## 复现范围

该实现是**简化复现**，目标是：

1. 随机生成软/硬模块与网络；
2. 构建固定轮廓平面规划实例；
3. 用基于密度与泊松方程的数值方法做全局平面规划；
4. 对软模块宽度做近似优化；
5. 用一个可运行的 bottom-left 合法化过程消除重叠；
6. 输出最终布局图与指标。

它没有逐项一比一实现论文中的所有解析公式和约束图合法化细节，但整体流程与论文主线一致，适合做原型验证和可视化演示。

## 依赖

- Python 3.10+
- `numpy`
- `matplotlib`

## 运行

在当前目录执行：

```bash
python pef_repro.py --modules 32 --nets 70 --iters 140 --grid 44 --outdir output
```

可选参数示例：

```bash
python pef_repro.py --modules 48 --nets 120 --soft-ratio 0.7 --whitespace 0.35 --aspect 1.2 --iters 220 --grid 56 --outdir output_big --show-ids
```

## 输出

运行后会生成：

- `output/pef_random_floorplan.png`
  - 四宫格图：初始布局、全局平面规划后、合法化后、密度图
- `output/pef_random_floorplan_final.png`
  - 最终合法布局图
- `output/pef_random_metrics.json`
  - HPWL、重叠面积等指标

## 说明

- `soft-ratio` 控制软模块比例；
- `whitespace` 越大，合法化越容易；
- `grid` 是泊松求解栅格大小；
- `iters` 是全局平面规划迭代次数。

如果把模块数提得很高、空白率压得很低，简化合法化器可能失败。这不是论文思路本身的问题，而是这里用的是一个更容易运行的原型合法化实现。
