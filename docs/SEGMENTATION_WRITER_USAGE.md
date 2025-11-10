# SegmentationWriter 使用指南

## 概述

`SegmentationWriter` 是一个专为 PointSuite bin+pkl 数据格式设计的 PyTorch Lightning 回调，用于在语义分割任务中进行预测结果的保存。

## 主要特性

1. **流式处理**: 批次预测结果先写入临时文件，避免 OOM
2. **投票机制**: 对重叠采样的点进行 logits 平均投票
3. **完整点云恢复**: 从原始 bin/pkl 文件加载完整点云数据
4. **LAS 头保留**: 保留原始 LAS 文件的坐标系统、缩放和偏移
5. **类别映射**: 支持将连续标签映射回原始标签
6. **Logits 保存**: 可选保存 logits 用于后处理或集成学习

## 数据流程

```
BinPklDataset (test) 
    ↓ {'coord', 'feat', 'indices', ...}
SemanticSegmentationTask.predict_step 
    ↓ {'logits', 'indices', 'coord'}
SegmentationWriter.write_on_batch_end 
    ↓ 保存到临时文件
SegmentationWriter.on_predict_end 
    ↓ 投票 + 从 bin/pkl 加载完整数据
保存为 .las 文件
```

## 使用方法

### 1. 基础使用

```python
import pytorch_lightning as pl
from pointsuite.utils.callbacks import SegmentationWriter
from pointsuite.tasks.semantic_segmentation import SemanticSegmentationTask
from pointsuite.data.datamodule_binpkl import BinPklDataModule

# 创建回调
writer = SegmentationWriter(
    output_dir='path/to/predictions',
    num_classes=8,  # 或 -1 自动推断
)

# 创建 DataModule
datamodule = BinPklDataModule(
    data_root='path/to/test/data',
    test_files=['test_file1.pkl', 'test_file2.pkl'],
    assets=['coord', 'intensity', 'classification'],
    batch_size=4,
    num_workers=4,
)

# 创建 Trainer 并预测
trainer = pl.Trainer(
    callbacks=[writer],
    accelerator='gpu',
    devices=1,
)

# 加载模型并预测
model = SemanticSegmentationTask.load_from_checkpoint('path/to/checkpoint.ckpt')
trainer.predict(model, datamodule=datamodule)
```

### 2. 带类别映射的使用

如果在训练时使用了类别映射（将稀疏标签映射为连续标签），在预测时需要反向映射：

```python
# 训练时的映射: {原始标签: 连续标签}
class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}

# 创建反向映射: {连续标签: 原始标签}
reverse_class_mapping = {v: k for k, v in class_mapping.items()}

# 创建回调时传入反向映射
writer = SegmentationWriter(
    output_dir='path/to/predictions',
    num_classes=5,  # 连续标签的数量
    reverse_class_mapping=reverse_class_mapping,
)

# DataModule 也需要使用相同的 class_mapping
datamodule = BinPklDataModule(
    data_root='path/to/test/data',
    class_mapping=class_mapping,  # 训练时用的映射
    # ... 其他参数
)
```

### 3. 保存 Logits 用于后处理

```python
writer = SegmentationWriter(
    output_dir='path/to/predictions',
    num_classes=8,
    save_logits=True,  # 保存 logits 到 .npz 文件
)

# 预测后会生成:
# - {bin_basename}_predicted.las  (最终预测的 LAS 文件)
# - {bin_basename}_logits.npz     (包含 logits, predictions, counts)
```

### 4. 从 YAML 配置使用

在配置文件中定义回调：

```yaml
# config.yaml
trainer:
  callbacks:
    - class_path: pointsuite.utils.callbacks.SegmentationWriter
      init_args:
        output_dir: "predictions"
        num_classes: 8
        save_logits: false
        reverse_class_mapping:
          0: 0
          1: 1
          2: 2
          3: 6
          4: 9

# 运行预测
# python main.py predict --config config.yaml --ckpt_path checkpoint.ckpt
```

## Task.predict_step 要求

为了使 `SegmentationWriter` 正常工作，`SemanticSegmentationTask.predict_step` 必须返回包含以下键的字典：

```python
def predict_step(self, batch, batch_idx, dataloader_idx=0):
    logits = self.forward(batch)  # [N, C]
    
    return {
        'logits': logits.cpu(),          # 必需: [N, C] 类别 logits
        'indices': batch['indices'].cpu(),  # 必需: [N] 原始点索引
        'coord': batch['coord'].cpu(),   # 可选: [N, 3] 坐标 (用于调试)
    }
```

**注意**: `indices` 字段由 `BinPklDataset` 在 test split 时自动提供。

## 输出文件格式

### 预测的 LAS 文件

```
predictions/
├── 5080_54400_predicted.las  # 完整点云 + 预测标签
├── 5080_54450_predicted.las
└── ...
```

LAS 文件包含：
- **坐标** (X, Y, Z): 从原始 bin 文件恢复，保留原始精度
- **分类** (classification): 预测的类别标签
- **LAS 头**: 保留原始 LAS 文件的 scale、offset、point_format 等

### Logits 文件 (可选)

```python
# 加载 logits 文件
import numpy as np
data = np.load('predictions/5080_54400_logits.npz')

logits = data['logits']        # [N, C] 平均后的 logits
predictions = data['predictions']  # [N] 预测标签 (argmax)
counts = data['counts']        # [N] 每个点的投票次数
```

## 工作原理

### 1. 批次处理阶段 (write_on_batch_end)

- 每个批次的预测结果 (logits, indices) 写入临时文件
- 临时文件命名: `{bin_basename}_batch_{batch_idx}.pred.tmp`
- 自动从 indices 推断属于哪个 bin 文件

### 2. 投票和保存阶段 (on_predict_end)

```python
# 伪代码
for each bin_file:
    # 1. 加载所有该文件的临时预测
    for tmp_file in tmp_files:
        logits_sum[indices] += logits
        counts[indices] += 1
    
    # 2. 计算平均 logits
    mean_logits = logits_sum / counts
    
    # 3. Argmax 获取类别
    predictions = argmax(mean_logits, dim=-1)
    
    # 4. 从原始 bin/pkl 加载完整点云
    point_data = np.memmap(bin_path, ...)
    xyz = extract_xyz(point_data)
    las_header = load_from_pkl(pkl_path)
    
    # 5. 保存为 LAS
    save_las(xyz, predictions, las_header)
```

### 3. 投票机制

当使用重叠采样时，同一个点可能出现在多个 segment 中：

```
Segment 1: 点 [0, 1, 2, 3] → 预测 logits_1
Segment 2: 点 [2, 3, 4, 5] → 预测 logits_2

投票后:
- 点 0, 1:    logits_1
- 点 2, 3:    (logits_1 + logits_2) / 2  # 平均
- 点 4, 5:    logits_2
```

这种投票机制可以提高预测的稳定性和准确性。

## 常见问题

### Q1: 为什么需要 `indices` 字段？

`indices` 记录了每个点在原始 bin 文件中的全局索引，用于投票时将预测结果映射回正确的位置。

### Q2: 如何处理未被预测的点？

如果某些点从未出现在任何 segment 中（counts == 0），它们会被赋予标签 0，并打印警告信息。

### Q3: 可以跨多个 bin 文件预测吗？

可以！回调会自动按 bin 文件分组处理。只需要在 DataModule 中包含多个 pkl 文件即可。

### Q4: LAS 头信息从哪里来？

LAS 头信息存储在 pkl 文件的 `metadata['las_header']` 中，由 `tile.py` 在处理原始 LAS 文件时保存。

### Q5: 如何验证预测结果？

```python
import laspy

# 读取预测结果
las = laspy.read('predictions/5080_54400_predicted.las')
print(f"点数: {len(las.points)}")
print(f"类别: {np.unique(las.classification)}")
print(f"坐标范围: X=[{las.x.min()}, {las.x.max()}]")
```

## 与 tile.py 的配合

`tile.py` 生成的 bin+pkl 文件包含完整的元数据：

```python
# tile.py 保存的 pkl 结构
metadata = {
    'dtype': np.dtype([('X', '<f8'), ('Y', '<f8'), ...]),
    'las_header': {
        'point_format': 3,
        'version': '1.2',
        'offsets': [x_offset, y_offset, z_offset],
        'scales': [x_scale, y_scale, z_scale],
        # ... 其他头信息
    },
    'segments': [
        {
            'segment_id': 0,
            'indices': [0, 1, 2, ...],  # 点在 bin 文件中的索引
            'num_points': 10000,
            'bounds': {...},
            # ... 其他片段信息
        },
        # ...
    ]
}
```

`SegmentationWriter` 利用这些信息来恢复完整的 LAS 文件。

## 性能优化建议

1. **使用 SSD**: 临时文件 I/O 频繁，SSD 可以显著提升速度
2. **调整 batch_size**: 较大的 batch 可以减少临时文件数量
3. **使用 pin_memory**: 在 DataModule 中启用 `pin_memory=True`
4. **并行预测**: 可以使用多 GPU 加速预测，但注意临时文件的命名冲突

## 完整示例脚本

```python
# predict.py
import pytorch_lightning as pl
from pointsuite.utils.callbacks import SegmentationWriter
from pointsuite.tasks.semantic_segmentation import SemanticSegmentationTask
from pointsuite.data.datamodule_binpkl import BinPklDataModule

# 类别映射 (与训练时一致)
class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
reverse_mapping = {v: k for k, v in class_mapping.items()}

# 创建回调
writer = SegmentationWriter(
    output_dir='predictions',
    num_classes=5,
    save_logits=True,
    reverse_class_mapping=reverse_mapping,
)

# 创建 DataModule
datamodule = BinPklDataModule(
    data_root='data/test',
    test_files=['5080_54400.pkl', '5080_54450.pkl'],
    assets=['coord', 'intensity', 'classification'],
    class_mapping=class_mapping,
    batch_size=4,
    num_workers=4,
)

# 设置数据
datamodule.setup('predict')

# 创建 Trainer
trainer = pl.Trainer(
    callbacks=[writer],
    accelerator='gpu',
    devices=1,
    logger=False,
)

# 加载模型
model = SemanticSegmentationTask.load_from_checkpoint('checkpoints/best.ckpt')

# 预测
trainer.predict(model, datamodule=datamodule)

print("预测完成！结果保存在 predictions/ 目录")
```

## 参考

- [BinPklDataset 文档](../pointsuite/data/datasets/dataset_bin.py)
- [SemanticSegmentationTask 文档](../pointsuite/tasks/semantic_segmentation.py)
- [tile.py 使用指南](../tools/tile.py)
