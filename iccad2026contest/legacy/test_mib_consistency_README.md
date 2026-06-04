# MIB 约束一致性测试

## 测试目的

测试训练数据中是否存在**同一个 MIB 组内的块有不同面积约束**的情况。

MIB (Multi-Instantiation Blocks) 约束要求：
- 同一 MIB 组内的所有块应该有**相同的尺寸** (width, height)
- 但是否应该有**相同的面积约束** (area_target)？

## 测试内容

脚本会检查两个方面：

1. **area_target 一致性**：同一 MIB 组内的块是否有相同的 `area_target`
2. **fp_sol 尺寸一致性**：同一 MIB 组内的块在 `fp_sol` 中是否有相同的尺寸

## 使用方法

### 基本测试（检查 1000 个样本）

```bash
cd FloorSet/iccad2026contest
python test_mib_consistency.py
```

### 检查更多样本

```bash
# 检查 5000 个样本
python test_mib_consistency.py --num-samples 5000
```

### 显示详细信息

```bash
# 显示每个不一致的 MIB 组的详细信息
python test_mib_consistency.py --verbose
```

### 详细分析特定样本

```bash
# 详细分析样本 #42
python test_mib_consistency.py --analyze 42
```

## 输出示例

### 正常情况（一致）

```
================================================================================
统计结果
================================================================================

总样本数: 1000
包含 MIB 约束的样本数: 234 (23.4%)
总 MIB 组数: 456

【面积约束一致性】
✅ 所有 MIB 组内的块都有相同的 area_target

【fp_sol 尺寸一致性】
✅ 所有 MIB 组内的块在 fp_sol 中都有相同的尺寸

================================================================================
结论
================================================================================
✅ MIB 约束完全一致：
   - 同一 MIB 组内的块有相同的 area_target
   - 同一 MIB 组内的块在 fp_sol 中有相同的尺寸
```

### 发现不一致

```
================================================================================
统计结果
================================================================================

总样本数: 1000
包含 MIB 约束的样本数: 234 (23.4%)
总 MIB 组数: 456

【面积约束一致性】
❌ 发现 15 个样本的 MIB 组内面积约束不一致
   样本索引: [42, 87, 123, 156, 234, 289, 345, 401, 456, 512]
   ... 还有 5 个

【fp_sol 尺寸一致性】
✅ 所有 MIB 组内的块在 fp_sol 中都有相同的尺寸

================================================================================
结论
================================================================================
⚠️  MIB 约束部分一致：
   - 同一 MIB 组内的块可能有不同的 area_target
   - 但在 fp_sol 中有相同的尺寸（这是正确的）

   这可能意味着：
   1. area_target 是输入约束，可能有误差
   2. fp_sol 是参考解，MIB 约束在这里被正确满足

提示: 使用 --analyze <样本ID> 查看详细信息
例如: python test_mib_consistency.py --analyze 42
```

### 详细分析输出

```bash
python test_mib_consistency.py --analyze 42
```

```
================================================================================
详细分析样本 42
================================================================================

样本信息:
  块数: 85
  MIB 组数: 3

  MIB 组 1 (包含 3 个块):
    块ID   area_target     fp_sol (w, h)        fp_sol area     差异%     
    ------ --------------- -------------------- --------------- ----------
    5      1234.50         (35.20, 35.07)       1234.46         0.00      
    12     1234.50         (35.20, 35.07)       1234.46         0.00      
    23     1234.50         (35.20, 35.07)       1234.46         0.00      

    一致性检查:
      ✅ area_target 一致: 1234.50
      ✅ fp_sol 尺寸一致: w=35.20, h=35.07

  MIB 组 2 (包含 4 个块):
    块ID   area_target     fp_sol (w, h)        fp_sol area     差异%     
    ------ --------------- -------------------- --------------- ----------
    8      2345.60         (48.42, 48.45)       2346.15         0.02      
    15     2346.80         (48.42, 48.45)       2346.15         0.03      
    28     2345.60         (48.42, 48.45)       2346.15         0.02      
    41     2347.20         (48.42, 48.45)       2346.15         0.04      

    一致性检查:
      ❌ area_target 不一致: {2345.6, 2346.8, 2347.2}
      ✅ fp_sol 尺寸一致: w=48.42, h=48.45
```

## 可能的结果解释

### 情况 1: 完全一致

```
✅ area_target 一致
✅ fp_sol 尺寸一致
```

**解释**：数据完美，MIB 约束在输入和输出中都被正确满足。

### 情况 2: area_target 不一致，但 fp_sol 一致

```
❌ area_target 不一致
✅ fp_sol 尺寸一致
```

**解释**：
- `area_target` 是输入约束，可能有轻微误差或来自不同来源
- `fp_sol` 是参考解，MIB 约束在这里被正确满足
- **这是可以接受的**，因为最终评估时会检查 fp_sol 中的尺寸

**对你的影响**：
- 训练时，可以使用 `fp_sol` 中的尺寸作为 MIB 组的目标尺寸
- 不要完全依赖 `area_target` 来判断 MIB 组的尺寸

### 情况 3: area_target 一致，但 fp_sol 不一致

```
✅ area_target 一致
❌ fp_sol 尺寸不一致
```

**解释**：数据异常！`fp_sol` 应该满足 MIB 约束但没有满足。

**对你的影响**：
- 这可能是数据错误
- 需要进一步调查

### 情况 4: 都不一致

```
❌ area_target 不一致
❌ fp_sol 尺寸不一致
```

**解释**：数据严重不一致，可能是数据生成问题。

## 实际应用建议

### 如果发现 area_target 不一致

在你的优化器中处理 MIB 约束时：

```python
def solve(self, block_count, area_targets, b2b_connectivity, 
          p2b_connectivity, pins_pos, constraints, target_positions):
    
    # 收集 MIB 组
    mib_groups = {}
    for i in range(block_count):
        mib_id = int(constraints[i, 2].item())
        if mib_id > 0:
            if mib_id not in mib_groups:
                mib_groups[mib_id] = []
            mib_groups[mib_id].append(i)
    
    # 为每个 MIB 组确定统一的尺寸
    mib_sizes = {}
    for mib_id, block_indices in mib_groups.items():
        # 方法 1: 使用 target_positions（如果有 fixed 块）
        for i in block_indices:
            if target_positions[i, 2] > 0:  # 有固定尺寸
                w = target_positions[i, 2]
                h = target_positions[i, 3]
                mib_sizes[mib_id] = (w, h)
                break
        
        # 方法 2: 使用平均面积
        if mib_id not in mib_sizes:
            areas = [area_targets[i].item() for i in block_indices]
            avg_area = sum(areas) / len(areas)
            # 使用正方形或其他策略
            w = h = (avg_area ** 0.5)
            mib_sizes[mib_id] = (w, h)
    
    # 应用 MIB 尺寸
    positions = []
    for i in range(block_count):
        mib_id = int(constraints[i, 2].item())
        if mib_id > 0:
            w, h = mib_sizes[mib_id]
            # 优化位置
            x, y = optimize_position(i, w, h)
        else:
            # 正常处理
            x, y, w, h = optimize_block(i)
        
        positions.append((x, y, w, h))
    
    return positions
```

## 总结

这个测试脚本帮助你：

1. **理解数据**：了解 MIB 约束在训练数据中的实际情况
2. **发现问题**：识别潜在的数据不一致
3. **指导实现**：根据测试结果调整你的优化器实现

**建议**：在开始实现 MIB 约束处理之前，先运行这个测试了解数据特性。
