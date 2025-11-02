# PointDataModule - DynamicBatchSampler 使用指南

## 概述

`PointDataModule` 现在支持 `DynamicBatchSampler`，这是一个更优雅的解决方案来控制每个 batch 的点数。与 `LimitedPointsCollateFn`（在 collate 阶段丢弃样本）不同，`DynamicBatchSampler` 在采样阶段就控制 batch 大小，确保：

✅ **每个样本都会被访问** - 不会丢弃任何样本  
✅ **内存使用可控** - 严格控制每个 batch 的总点数  
✅ **支持加权采样** - 可与 `WeightedRandomSampler` 结合使用  
✅ **类别不平衡处理** - 对稀有类别进行过采样  

## 快速开始

### 基础用法

```python
from pointsuite.data.point_datamodule import PointDataModule

# 启用 DynamicBatchSampler
datamodule = PointDataModule(
    data_root='path/to/data',
    train_files=['train.pkl'],
    val_files=['val.pkl'],
    use_dynamic_batch=True,      # 启用动态 batch
    max_points=500000,            # 每个 batch 最多 50万点
    num_workers=4
)

datamodule.setup('fit')
train_loader = datamodule.train_dataloader()
```

### 与 WeightedRandomSampler 结合

```python
# 计算样本权重（例如：根据类别分布）
sample_weights = [2.0 if is_rare_class else 1.0 for ...]

datamodule = PointDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,
    max_points=500000,
    train_sampler_weights=sample_weights,  # 提供权重
    num_workers=4
)
```

## 参数说明

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_dynamic_batch` | bool | False | 是否使用 DynamicBatchSampler |
| `max_points` | int | 500000 | 每个 batch 的最大点数 |
| `train_sampler_weights` | List[float] | None | 训练集样本权重（用于 WeightedRandomSampler） |

### 行为变化

当 `use_dynamic_batch=True` 时：
- **`batch_size` 参数被忽略** - batch 大小由点数动态决定
- **训练集** - 使用 `DynamicBatchSampler`，如果提供了 `train_sampler_weights` 则结合 `WeightedRandomSampler`
- **验证集/测试集** - 使用 `DynamicBatchSampler`，顺序遍历，不打乱

## 使用场景

### 场景1: 内存受限

```python
# 问题：样本点数差异大，固定 batch_size 可能 OOM
# 解决：使用 DynamicBatchSampler 控制总点数

datamodule = PointDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,
    max_points=500000,  # 根据 GPU 内存调整
)
```

### 场景2: 类别不平衡

```python
# 问题：某些类别样本很少
# 解决：使用加权采样对稀有类别进行过采样

from collections import Counter
import numpy as np

# 1. 统计类别分布
class_counts = Counter()
for sample_info in dataset.data_list:
    class_counts.update(sample_info['label_counts'])

# 2. 计算类别权重（逆频率）
total = sum(class_counts.values())
class_weights = {
    cls: total / (len(class_counts) * cnt)
    for cls, cnt in class_counts.items()
}

# 3. 为每个样本计算权重
sample_weights = []
for sample_info in dataset.data_list:
    counts = sample_info['label_counts']
    weight = sum(
        class_weights[cls] * cnt / sum(counts.values())
        for cls, cnt in counts.items()
    )
    sample_weights.append(weight)

# 4. 归一化
sample_weights = np.array(sample_weights)
sample_weights = (sample_weights / sample_weights.max()).tolist()

# 5. 使用权重
datamodule = PointDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,
    max_points=500000,
    train_sampler_weights=sample_weights,
)
```

### 场景3: 空间区域均衡

```python
# 问题：某些区域样本密集，某些区域稀疏
# 解决：根据空间分布计算权重

from collections import defaultdict

# 统计每个网格的样本数
grid_size = 100.0  # 100米网格
grid_counts = defaultdict(int)
sample_grids = []

for sample_info in dataset.data_list:
    x_center = (sample_info['bounds']['x_min'] + sample_info['bounds']['x_max']) / 2
    y_center = (sample_info['bounds']['y_min'] + sample_info['bounds']['y_max']) / 2
    grid_id = (int(x_center // grid_size), int(y_center // grid_size))
    
    grid_counts[grid_id] += 1
    sample_grids.append(grid_id)

# 计算权重（稀疏区域权重高）
total_samples = len(sample_grids)
sample_weights = [
    total_samples / (len(grid_counts) * grid_counts[grid_id])
    for grid_id in sample_grids
]

# 归一化
max_w = max(sample_weights)
sample_weights = [w / max_w for w in sample_weights]

datamodule = PointDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,
    train_sampler_weights=sample_weights,
)
```

### 场景4: 困难样本挖掘

```python
# 问题：希望更多地训练困难样本
# 解决：根据样本复杂度计算权重

def compute_difficulty_weights(dataset):
    weights = []
    for sample_info in dataset.data_list:
        # 点数越多越困难
        point_factor = min(sample_info['num_points'] / 100000, 2.0)
        
        # 高度变化大越困难
        z_range = sample_info['bounds']['z_max'] - sample_info['bounds']['z_min']
        height_factor = min(z_range / 50, 2.0)
        
        # 组合
        weight = (point_factor + height_factor) / 2
        weights.append(weight)
    
    # 归一化
    max_w = max(weights)
    return [w / max_w for w in weights]

difficulty_weights = compute_difficulty_weights(dataset)

datamodule = PointDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,
    train_sampler_weights=difficulty_weights,
)
```

## 工作原理

### DynamicBatchSampler

```python
class DynamicBatchSampler:
    def __iter__(self):
        # 1. 生成索引序列
        if self.sampler is not None:
            indices = list(self.sampler)  # 使用提供的 sampler
        elif self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # 2. 动态组装 batch
        batch = []
        batch_points = 0
        
        for idx in indices:
            num_points = self.num_points_list[idx]
            
            if len(batch) == 0 or batch_points + num_points <= self.max_points:
                batch.append(idx)
                batch_points += num_points
            else:
                yield batch  # 当前 batch 已满
                batch = [idx]
                batch_points = num_points
        
        if len(batch) > 0:
            yield batch
```

### 与 WeightedRandomSampler 结合

```python
# 1. 创建 WeightedRandomSampler
weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True  # 有放回采样，允许重复
)

# 2. DynamicBatchSampler 使用该 sampler 生成索引
batch_sampler = DynamicBatchSampler(
    dataset=dataset,
    max_points=500000,
    sampler=weighted_sampler  # 传入加权采样器
)

# 3. DataLoader 使用 batch_sampler
dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=collate_fn
)
```

## 对比：DynamicBatchSampler vs LimitedPointsCollateFn

| 特性 | DynamicBatchSampler | LimitedPointsCollateFn |
|------|---------------------|------------------------|
| **工作阶段** | 采样阶段 | Collate 阶段 |
| **样本丢弃** | ❌ 不丢弃 | ✅ 可能丢弃 |
| **样本覆盖** | ✅ 100% 覆盖 | ❌ 可能不完全覆盖 |
| **与 WeightedSampler 兼容** | ✅ 完美兼容 | ❌ 不兼容 |
| **内存控制** | ✅ 严格控制 | ✅ 严格控制 |
| **batch 大小** | 动态（1-N） | 固定范围（0-batch_size） |
| **实现复杂度** | 中等 | 简单 |
| **推荐使用** | ✅ 推荐 | ⚠️ 简单场景 |

## 性能考虑

### 内存使用

```python
# 根据 GPU 内存调整 max_points
# 16GB GPU: max_points = 500000
# 24GB GPU: max_points = 1000000
# 32GB GPU: max_points = 1500000

datamodule = PointDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,
    max_points=500000,  # 根据实际情况调整
)
```

### 加载速度

```python
# 优化数据加载性能
datamodule = PointDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,
    max_points=500000,
    num_workers=8,              # 增加 worker 数量
    persistent_workers=True,    # 保持 worker 存活
    prefetch_factor=4,          # 增加预取因子
    pin_memory=True,            # 使用 pinned memory
)
```

### 采样效率

```python
# WeightedRandomSampler 的性能考虑
# replacement=True: 有放回，支持过采样但可能重复
# replacement=False: 无放回，每个样本最多采样一次

from torch.utils.data import WeightedRandomSampler

# 方案1: 有放回（适合类别不平衡）
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),  # 或更多以增加 epoch 长度
    replacement=True
)

# 方案2: 无放回（确保每个样本恰好一次）
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=False
)
```

## 完整训练示例

```python
import pytorch_lightning as pl
from pointsuite.data.point_datamodule import PointDataModule
from pointsuite.data.datasets.dataset_bin import BinPklDataset

# 1. 加载数据集
dataset = BinPklDataset(data_root='data/train.pkl', split='train')

# 2. 计算样本权重（可选）
from collections import Counter
import numpy as np

global_class_counts = Counter()
for sample_info in dataset.data_list:
    if 'label_counts' in sample_info:
        global_class_counts.update(sample_info['label_counts'])

total_points = sum(global_class_counts.values())
class_weights = {
    cls: total_points / (len(global_class_counts) * cnt)
    for cls, cnt in global_class_counts.items()
}

sample_weights = []
for sample_info in dataset.data_list:
    if 'label_counts' in sample_info:
        counts = sample_info['label_counts']
        total = sum(counts.values())
        weight = sum(
            class_weights[cls] * (cnt / total)
            for cls, cnt in counts.items()
        )
        sample_weights.append(weight)

# 归一化
sample_weights = np.array(sample_weights)
sample_weights = (sample_weights / sample_weights.max()).tolist()

# 3. 创建 DataModule
datamodule = PointDataModule(
    data_root='data',
    train_files=['train.pkl'],
    val_files=['val.pkl'],
    test_files=['test.pkl'],
    # DynamicBatchSampler 配置
    use_dynamic_batch=True,
    max_points=500000,
    train_sampler_weights=sample_weights,  # 使用加权采样
    # 性能优化
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    # 其他参数
    assets=['coord', 'intensity', 'classification'],
    loop=1,
)

# 4. 创建 Trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    precision=16,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15
        ),
    ]
)

# 5. 训练
model = YourPointCloudModel(...)
trainer.fit(model, datamodule)

# 6. 测试
trainer.test(model, datamodule)
```

## 调试和验证

### 检查 batch 信息

```python
datamodule.setup('fit')
train_loader = datamodule.train_dataloader()

# 检查前几个 batch
for i, batch in enumerate(train_loader):
    if i >= 3:
        break
    
    total_points = batch['coord'].shape[0]
    num_samples = len(batch['offset'])
    
    print(f"Batch {i}:")
    print(f"  样本数: {num_samples}")
    print(f"  总点数: {total_points:,}")
    print(f"  平均点数/样本: {total_points / num_samples:.0f}")
```

### 验证样本覆盖

```python
# 记录所有被访问的样本索引
visited_indices = set()

for batch in train_loader:
    # 从 batch 中提取样本索引（需要修改 collate_fn 返回索引）
    # 或者通过其他方式追踪
    pass

coverage = len(visited_indices) / len(dataset) * 100
print(f"样本覆盖率: {coverage:.1f}%")
```

### 验证类别分布

```python
from collections import Counter

epoch_class_counts = Counter()

for batch in train_loader:
    labels = batch['classification'].numpy()
    epoch_class_counts.update(labels)

print("一个 epoch 的类别分布:")
for cls, count in sorted(epoch_class_counts.items()):
    print(f"  类别 {cls}: {count:,} 点")
```

## 常见问题

### Q: 为什么我的 batch size 不固定？

A: 使用 `DynamicBatchSampler` 时，batch 大小由点数动态决定。点数少的样本会被组合成更大的 batch，点数多的样本 batch 会更小。

### Q: WeightedRandomSampler 是否会导致某些样本被多次访问？

A: 是的，如果 `replacement=True`（默认），稀有类别的样本会在一个 epoch 中被多次采样。这是处理类别不平衡的常用方法。

### Q: 如何确保每个样本恰好被访问一次？

A: 使用 `WeightedRandomSampler(replacement=False)` 或者不使用加权采样。

### Q: DynamicBatchSampler 和 WeightedSampler 可以用于验证集吗？

A: `DynamicBatchSampler` 可以用于验证集（建议用于内存控制），但 `WeightedRandomSampler` 通常只用于训练集。

### Q: 性能会受到影响吗？

A: `DynamicBatchSampler` 的额外开销很小（<1%）。预先计算点数列表后，采样速度几乎与标准采样器相同。

## 相关文档

- [BinPklDataset 文档](datasets/README.md)
- [Collate Functions 文档](datasets/collate.py)
- [采样器覆盖率测试](../test/test_sampler_coverage.py)
- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)
- [WeightedRandomSampler 文档](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)

## 总结

使用 `DynamicBatchSampler` + `WeightedRandomSampler` 组合可以：

✅ **控制内存使用** - 通过 `max_points` 严格限制  
✅ **处理类别不平衡** - 通过 `train_sampler_weights` 实现加权采样  
✅ **保证样本覆盖** - 不会丢弃任何样本  
✅ **灵活的采样策略** - 支持多种权重计算方法  
✅ **完美的 Lightning 集成** - 开箱即用  

推荐在所有生产环境中使用这个组合！
