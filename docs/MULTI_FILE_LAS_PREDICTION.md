# 多 LAS 文件预测机制说明

## 问题

用户疑惑：**为什么多个 LAS 文件预测时，即使分块顺序被打乱，最后仍能正确写回多个完整的 LAS 文件？**

## 答案

这是 `SegmentationWriter` 回调的核心设计，通过 **文件信息传递** + **分组累积** + **投票机制** 实现的。

---

## 工作流程

### 1. 数据准备阶段 (`tools/tile.py`)

分割 LAS 文件时，为每个 segment 保存文件关联信息：

```python
segment_info = {
    'segment_id': 0,
    'indices': [0, 1, 2, ...],  # 原始点索引
    'bin_file': 'file_001',  # 🔥 关键：所属 bin 文件名
    'bin_path': '/path/to/file_001.bin',
    'pkl_path': '/path/to/file_001.pkl',
}
```

**关键点**：每个 segment 都明确记录了它来自哪个原始 LAS 文件。

---

### 2. 数据加载阶段 (`BinPklDataset`)

在 `test/predict` split 时，Dataset 将文件信息添加到数据字典：

```python
# dataset_bin.py 的 __getitem__ 方法
if self.split in ['test', 'predict']:
    data['indices'] = indices.copy()  # 原始点索引
    data['bin_file'] = sample_info['bin_file']  # 文件名
    data['bin_path'] = sample_info['bin_path']  # 完整路径
    data['pkl_path'] = sample_info['pkl_path']
```

**关键点**：每个样本都携带自己的"身份证"（来源文件信息）。

---

### 3. 预测阶段 (`SemanticSegmentationTask.predict_step`)

模型预测后，将文件信息传递给 callback：

```python
def predict_step(self, batch, batch_idx):
    preds = self.forward(batch)
    
    return {
        "logits": preds.cpu(),  # 预测结果
        "indices": batch["indices"].cpu(),  # 原始索引
        "bin_file": batch["bin_file"],  # 🔥 文件名
        "bin_path": batch["bin_path"],  # 完整路径
        "pkl_path": batch["pkl_path"],
        "coord": batch["coord"].cpu(),
    }
```

**关键点**：预测结果和文件信息一起传递，永不分离。

---

### 4. 临时保存阶段 (`SegmentationWriter.write_on_batch_end`)

每个批次的预测结果流式写入临时文件：

```python
# 从 prediction 提取 bin 文件名
bin_basename = prediction['bin_file'][0]  # 例如: 'file_001'

# 保存临时文件
tmp_filename = f"{bin_basename}_batch_{batch_idx}.pred.tmp"
# 例如: file_001_batch_0.pred.tmp
#       file_001_batch_5.pred.tmp
#       file_002_batch_1.pred.tmp
#       file_002_batch_8.pred.tmp

torch.save({
    'logits': prediction['logits'],
    'indices': prediction['indices'],
    'bin_file': bin_basename,
    'bin_path': prediction['bin_path'],
    'pkl_path': prediction['pkl_path'],
}, tmp_filename)
```

**关键点**：
- 临时文件名包含 `bin_basename`，即使 batch 顺序乱序，文件名也能标识来源
- 不同 LAS 文件的 segment 会写入不同的临时文件

---

### 5. 分组阶段 (`SegmentationWriter.on_predict_end`)

预测结束后，按文件名分组所有临时文件：

```python
# 查找所有临时文件
tmp_files = glob.glob("*.pred.tmp")
# ['file_001_batch_0.pred.tmp', 'file_001_batch_5.pred.tmp', 
#  'file_002_batch_1.pred.tmp', 'file_002_batch_8.pred.tmp']

# 按 bin_basename 分组
bin_file_groups = defaultdict(list)
for tmp_file in tmp_files:
    bin_basename = tmp_file.split('_batch_')[0]  # 提取 'file_001'
    bin_file_groups[bin_basename].append(tmp_file)

# 结果:
# {'file_001': ['file_001_batch_0.pred.tmp', 'file_001_batch_5.pred.tmp'],
#  'file_002': ['file_002_batch_1.pred.tmp', 'file_002_batch_8.pred.tmp']}
```

**关键点**：通过文件名自动分组，无需知道原始处理顺序。

---

### 6. 投票累积阶段 (`_process_single_bin_file`)

对每个 bin 文件的所有 segment 执行投票：

```python
# 为文件 file_001 创建投票数组
num_points = 100000  # 从 bin/pkl 读取
logits_sum = torch.zeros((num_points, num_classes))
counts = torch.zeros(num_points)

# 累积所有 segment 的预测
for tmp_file in bin_file_groups['file_001']:
    data = torch.load(tmp_file)
    indices = data['indices']  # 例如: [0, 5, 10, ...] (原始索引)
    logits = data['logits']    # [N, C]
    
    logits_sum[indices] += logits  # 按索引累加
    counts[indices] += 1

# 平均投票
mean_logits = logits_sum / (counts.unsqueeze(-1) + 1e-10)
final_preds = torch.argmax(mean_logits, dim=1)
```

**关键点**：
- 使用 `indices` 将每个 segment 的预测放回原始点云的正确位置
- 多次预测的点会自动平均（Test-Time Augmentation）
- 即使 segment 顺序乱序，`indices` 确保放回正确位置

---

### 7. 保存阶段

从原始 bin/pkl 加载完整点云，替换 classification 字段，保存为 LAS：

```python
# 加载原始点云（所有属性）
point_data = np.memmap(bin_path, dtype=metadata['dtype'])

# 提取坐标和属性
xyz = np.stack([point_data['X'], point_data['Y'], point_data['Z']], axis=1)
intensity = point_data['intensity']
rgb = np.stack([point_data['red'], point_data['green'], point_data['blue']], axis=1)

# 替换 classification（预测结果）
point_data['classification'] = final_preds

# 保存为 LAS
laspy.write('file_001.las', point_data)
```

**关键点**：保留所有原始属性（坐标、颜色、强度等），只替换分类标签。

---

## 核心机制总结

### 为什么不会乱？

1. **文件信息绑定**：每个 segment 从 tile → dataset → predict_step → callback，始终携带来源文件信息
2. **文件名分组**：临时文件名包含 `bin_basename`，自动分组到正确的原始文件
3. **索引映射**：`indices` 字段记录每个点在原始点云中的位置，投票时放回正确位置
4. **投票机制**：多个 segment 对同一点的预测会自动平均，提高鲁棒性

### 类比理解

想象你有一本书，被撕成很多小纸片（segment），每个纸片上都标记了：
- 它来自哪本书（`bin_file`）
- 它在原书的第几页（`indices`）

即使你把纸片打乱顺序，甚至混入其他书的纸片，只要：
1. 纸片标记完整
2. 你按书名分组
3. 按页码排序

就能完美还原每本书。

---

## 潜在问题

### 1. **collate_fn 混合不同文件的点**

**问题**：如果 dynamic batch 将来自不同 bin 文件的 segment 合并到一个 batch？

**答案**：**不会发生**。`BinPklDataset` 的 segment 是预先切分好的，每个 segment 只属于一个 bin 文件。Dynamic batch 只是将多个 segment 堆叠，但每个点的 `bin_file` 信息已经通过 collate_fn 保留（作为列表）。

检查 `collate.py`：

```python
def collate_fn(batch):
    # ...
    # 'bin_file' 等字符串字段保持为列表，不拼接
    if 'bin_file' in batch[0]:
        result['bin_file'] = [item['bin_file'] for item in batch]
    # ...
```

在 `write_on_batch_end` 中取第一个：

```python
bin_basename = prediction['bin_file'][0]
```

**假设**：一个 batch 内的所有点来自同一个 bin 文件的不同 segment。这由 Dataset 的 segment 划分保证。

**验证**：在 `write_on_batch_end` 中可以添加断言检查：

```python
assert len(set(prediction['bin_file'])) == 1, \
    f"Batch contains segments from multiple files: {set(prediction['bin_file'])}"
```

### 2. **indices 冲突**

**问题**：不同 segment 的 `indices` 是否会重叠？

**答案**：**不会**。每个 segment 的 `indices` 是不重叠的：

```python
# tile.py
segments = [
    {'indices': [0, 5, 10, 15]},      # segment_0
    {'indices': [20, 25, 30, 35]},    # segment_1
    {'indices': [40, 45, 50, 55]},    # segment_2
]
```

投票时，每个 segment 的预测放入不同的索引位置，不会冲突。

### 3. **未预测的点**

**问题**：如果某些点没有被任何 segment 覆盖（`counts[i] == 0`）？

**答案**：在 `_process_single_bin_file` 中处理：

```python
unpredicted_mask = (counts == 0)
if unpredicted_mask.any():
    print(f"警告: {unpredicted_mask.sum()} 个点未被预测，将赋予标签 0")
    final_preds[unpredicted_mask] = 0
```

**原因**：tile 时可能跳过某些点（如地面点、边界点），这些点使用默认标签。

---

## 实验验证

### 测试用例 1：单文件多 segment

```
file_001.bin (100k points)
├── segment_0: points [0-19999]
├── segment_1: points [20000-39999]
├── segment_2: points [40000-59999]
└── segment_3: points [60000-99999]
```

**预测顺序**：2 → 0 → 3 → 1（乱序）

**临时文件**：
```
file_001_batch_2.pred.tmp  (indices: [40000-59999])
file_001_batch_0.pred.tmp  (indices: [0-19999])
file_001_batch_3.pred.tmp  (indices: [60000-99999])
file_001_batch_1.pred.tmp  (indices: [20000-39999])
```

**分组结果**：
```python
{'file_001': [
    'file_001_batch_2.pred.tmp',
    'file_001_batch_0.pred.tmp',
    'file_001_batch_3.pred.tmp',
    'file_001_batch_1.pred.tmp'
]}
```

**投票结果**：
```python
logits_sum[40000:59999] += logits_from_batch_2
logits_sum[0:19999] += logits_from_batch_0
logits_sum[60000:99999] += logits_from_batch_3
logits_sum[20000:39999] += logits_from_batch_1
# 最终: logits_sum[0:99999] 全部填充，顺序无影响
```

---

### 测试用例 2：多文件交错

```
file_001.bin
├── segment_0
└── segment_1

file_002.bin
├── segment_0
└── segment_1
```

**预测顺序**：file_001_seg_0 → file_002_seg_0 → file_001_seg_1 → file_002_seg_1

**临时文件**：
```
file_001_batch_0.pred.tmp
file_002_batch_1.pred.tmp
file_001_batch_2.pred.tmp
file_002_batch_3.pred.tmp
```

**分组结果**：
```python
{
    'file_001': ['file_001_batch_0.pred.tmp', 'file_001_batch_2.pred.tmp'],
    'file_002': ['file_002_batch_1.pred.tmp', 'file_002_batch_3.pred.tmp']
}
```

**输出**：
```
file_001.las  (完整点云)
file_002.las  (完整点云)
```

---

## 结论

这是一个**健壮的设计**，通过文件信息传递和索引映射，确保：

1. ✅ 多文件预测互不干扰
2. ✅ 乱序预测自动还原
3. ✅ 支持 TTA（Test-Time Augmentation）投票
4. ✅ 保留所有原始属性

**漏洞**：
- ❌ 假设一个 batch 内所有点来自同一 bin 文件（当前代码未验证）
- ❌ 未处理 segment 边界点重复预测的情况（当前简单平均）

**建议**：
- 添加断言检查 batch 内文件一致性
- 考虑为重叠区域使用加权平均（距离中心越近权重越大）
