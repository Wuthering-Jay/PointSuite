# DataModule 重构迁移指南

## 🎯 重构目标

解决以下问题：
1. ✅ **文件路径更灵活**：支持跨目录文件配置
2. ✅ **独立的预测数据集**：test 和 predict 可以不同
3. ✅ **WeightedRandomSampler 独立控制**：不依赖 DynamicBatchSampler
4. ✅ **loop 与 weights 匹配**：明确要求 weights 长度考虑 loop
5. ✅ **weights 不保存超参数**：避免超参数膨胀

## 📋 API 变化

### 旧 API（已废弃）

```python
BinPklDataModule(
    data_root='data/dales',           # 单一根目录
    train_files=['train.pkl'],        # 文件名列表
    val_files=['val.pkl'],
    test_files=['test.pkl'],
    # ... 其他参数
)
```

**限制**：
- 所有文件必须在同一个 `data_root` 下
- test 和 predict 共享相同数据
- WeightedRandomSampler 只能与 DynamicBatchSampler 一起使用

### 新 API（推荐）

```python
BinPklDataModule(
    train_data='data/train',          # 完整路径（目录或文件列表）
    val_data='data/val',
    test_data='data/test',
    predict_data='data/predict',      # 独立的预测数据（可选）
    use_weighted_sampler=True,        # 独立控制
    train_sampler_weights=weights,    # 长度 = len(dataset) * loop
    # ... 其他参数
)
```

**优势**：
- ✅ 文件可以在任意目录
- ✅ 预测数据集独立配置
- ✅ WeightedRandomSampler 独立启用
- ✅ 明确 weights 与 loop 的关系

## 🔄 迁移步骤

### 情况 1: 所有文件在同一目录

**旧代码**：
```python
datamodule = BinPklDataModule(
    data_root='data/dales',
    train_files=['train.pkl'],
    val_files=['val.pkl'],
    test_files=['test.pkl'],
    batch_size=8,
)
```

**新代码**：
```python
datamodule = BinPklDataModule(
    train_data='data/dales/train.pkl',     # 或 'data/dales' 自动发现
    val_data='data/dales/val.pkl',
    test_data='data/dales/test.pkl',
    batch_size=8,
)
```

### 情况 2: 文件跨目录

**旧代码**（不支持）：
```python
# 无法实现跨目录
```

**新代码**：
```python
datamodule = BinPklDataModule(
    train_data=[
        'data/scene1/train.pkl',
        'data/scene2/train.pkl',
        'other_data/extra_train.pkl',
    ],
    val_data='data/scene1/val.pkl',
    test_data='data/test_set',  # 整个目录
    batch_size=8,
)
```

### 情况 3: 使用 DynamicBatchSampler + WeightedRandomSampler

**旧代码**：
```python
# 假设 original_weights 长度 = 原始样本数（不考虑 loop）
datamodule = BinPklDataModule(
    data_root='data/dales',
    use_dynamic_batch=True,
    max_points=500000,
    train_sampler_weights=original_weights,  # ❌ 错误：未考虑 loop
    loop=2,  # 数据集循环 2 次
)
```

**新代码**：
```python
# 正确处理 loop
original_weights = [...]  # 长度 = 原始样本数
loop = 2

# weights 必须重复 loop 次
train_weights = original_weights * loop  # 长度 = 原始样本数 * loop

datamodule = BinPklDataModule(
    train_data='data/dales',
    use_dynamic_batch=True,
    max_points=500000,
    use_weighted_sampler=True,           # ✅ 显式启用
    train_sampler_weights=train_weights, # ✅ 正确长度
    loop=2,
)
```

### 情况 4: 不使用 DynamicBatchSampler，但需要 WeightedRandomSampler

**旧代码**（不支持）：
```python
# WeightedRandomSampler 只能与 DynamicBatchSampler 一起使用
```

**新代码**：
```python
# 现在可以独立使用
datamodule = BinPklDataModule(
    train_data='data/dales',
    batch_size=8,                        # 固定 batch size
    use_dynamic_batch=False,             # 不使用动态批次
    use_weighted_sampler=True,           # ✅ 独立启用加权采样
    train_sampler_weights=weights,       # 考虑 loop
    loop=1,
)
```

### 情况 5: 独立的预测数据集

**旧代码**：
```python
# test 和 predict 共享相同数据
datamodule = BinPklDataModule(
    data_root='data',
    test_files=['test.pkl'],
)
# predict 会使用 test 数据
```

**新代码**：
```python
datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    test_data='data/test_labeled',      # 有标签的测试集
    predict_data='data/new_scenes',     # ✅ 无标签的新场景
)
```

## 📖 完整示例

### 示例 1: DALES 数据集（基本配置）

```python
from pointsuite.data.datamodule_bin import BinPklDataModule
from pointsuite.data import transforms as T

# 定义 transforms
train_transforms = [
    T.RandomRotate(angle=[-180, 180], axis='z'),
    T.RandomScale([0.95, 1.05]),
    T.RandomFlip(p=0.5),
    T.AutoNormalizeIntensity(),
    T.Collect(keys=['coord', 'intensity', 'class'], feat_keys=['intensity']),
]

val_transforms = [
    T.AutoNormalizeIntensity(),
    T.Collect(keys=['coord', 'intensity', 'class'], feat_keys=['intensity']),
]

# 创建 DataModule
datamodule = BinPklDataModule(
    train_data='data/dales/train',
    val_data='data/dales/val',
    test_data='data/dales/test',
    batch_size=8,
    num_workers=4,
    assets=['coord', 'intensity', 'class'],
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=val_transforms,
    class_mapping={0: 0, 1: 1, 2: 2, 6: 3, 8: 4},  # 5 类
    ignore_label=-1,
)

# 使用
datamodule.setup()
datamodule.print_info()
```

### 示例 2: 使用 DynamicBatchSampler（内存控制）

```python
datamodule = BinPklDataModule(
    train_data=[
        'data/scene1/train.pkl',
        'data/scene2/train.pkl',
    ],
    val_data='data/scene1/val.pkl',
    test_data='data/test',
    use_dynamic_batch=True,       # 启用动态批次
    max_points=500000,            # 每批次最多 50 万点
    num_workers=8,
    assets=['coord', 'intensity', 'h_norm', 'class'],
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    class_mapping={0: 0, 1: 1, 2: 2, 6: 3, 8: 4},
)
```

### 示例 3: 加权采样处理类别不平衡

```python
import numpy as np

# 假设你有类别不平衡的数据集
# 原始样本数 = 1000，使用 loop=2
num_samples = 1000
loop = 2

# 计算样本权重（基于类别频率的倒数）
# 这里假设你已经从 dataset 中获取了每个样本的主要类别
sample_labels = [...]  # 长度 = 1000

# 计算类别权重
from collections import Counter
label_counts = Counter(sample_labels)
class_weights = {label: 1.0 / count for label, count in label_counts.items()}

# 为每个样本分配权重
original_weights = [class_weights[label] for label in sample_labels]

# ⚠️ 关键：考虑 loop，重复权重
train_weights = original_weights * loop  # 长度 = 2000

datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    test_data='data/test',
    use_dynamic_batch=True,
    max_points=500000,
    use_weighted_sampler=True,        # 启用加权采样
    train_sampler_weights=train_weights,  # 长度必须 = 2000
    loop=loop,                        # loop = 2
    num_workers=8,
    # ... 其他参数
)
```

### 示例 4: 独立预测数据集

```python
datamodule = BinPklDataModule(
    train_data='data/dales/train',
    val_data='data/dales/val',
    test_data='data/dales/test',           # 有标签的测试集
    predict_data='data/new_scenes',        # 无标签的新场景
    predict_transforms=[
        T.AutoNormalizeIntensity(),
        T.Collect(keys=['coord', 'intensity'], feat_keys=['intensity']),
    ],
    # ... 其他参数
)

# 测试阶段：使用有标签的 test 数据
trainer.test(model, datamodule)

# 预测阶段：使用无标签的 predict 数据
trainer.predict(model, datamodule)
```

## ⚠️ 常见错误

### 错误 1: weights 长度不匹配

```python
# ❌ 错误
original_weights = [1.0] * 1000  # 长度 = 1000
datamodule = BinPklDataModule(
    train_data='data/train',
    train_sampler_weights=original_weights,  # ❌ 长度不对
    loop=2,  # dataset 长度会变成 2000
)
```

**错误信息**：
```
ValueError: train_sampler_weights 长度 (1000) 与 dataset 长度 (2000) 不匹配。
提示：如果使用 loop > 1，weights 需要重复 loop 次。
例如：weights = original_weights * loop
```

**修复**：
```python
# ✅ 正确
original_weights = [1.0] * 1000
train_weights = original_weights * 2  # 长度 = 2000
datamodule = BinPklDataModule(
    train_data='data/train',
    train_sampler_weights=train_weights,  # ✅ 长度正确
    loop=2,
)
```

### 错误 2: 忘记启用 use_weighted_sampler

```python
# ❌ 不会生效
datamodule = BinPklDataModule(
    train_data='data/train',
    train_sampler_weights=[1.0] * 2000,  # 提供了 weights
    # use_weighted_sampler=True,  # ❌ 忘记启用
)
# weights 会被忽略！
```

**修复**：
```python
# ✅ 正确
datamodule = BinPklDataModule(
    train_data='data/train',
    use_weighted_sampler=True,           # ✅ 显式启用
    train_sampler_weights=[1.0] * 2000,
)
```

### 错误 3: 使用旧的 data_root API

```python
# ❌ 旧 API（已废弃）
datamodule = BinPklDataModule(
    data_root='data/dales',  # ❌ 参数不存在
    train_files=['train.pkl'],
)
```

**修复**：
```python
# ✅ 新 API
datamodule = BinPklDataModule(
    train_data='data/dales/train.pkl',  # 或 'data/dales' 自动发现
    val_data='data/dales/val.pkl',
    test_data='data/dales/test.pkl',
)
```

## 🔧 实用工具函数

### 计算类别权重

```python
import numpy as np
from collections import Counter

def compute_sample_weights(dataset, loop=1):
    """
    为数据集计算样本权重以处理类别不平衡
    
    Args:
        dataset: BinPklDataset 实例
        loop: 数据集循环次数
        
    Returns:
        weights: 样本权重列表，长度 = len(dataset) * loop
    """
    # 统计每个样本的主要类别
    sample_labels = []
    for i in range(len(dataset.data_list)):
        data = dataset[i]
        labels = data['class'].numpy()
        # 使用最频繁的类别作为样本标签
        most_common = Counter(labels).most_common(1)[0][0]
        sample_labels.append(most_common)
    
    # 计算类别权重（频率的倒数）
    label_counts = Counter(sample_labels)
    total = len(sample_labels)
    class_weights = {label: total / count for label, count in label_counts.items()}
    
    # 为每个样本分配权重
    original_weights = [class_weights[label] for label in sample_labels]
    
    # 考虑 loop
    final_weights = original_weights * loop
    
    print(f"类别权重: {class_weights}")
    print(f"原始样本数: {len(original_weights)}")
    print(f"最终权重数量: {len(final_weights)} (loop={loop})")
    
    return final_weights

# 使用示例
from pointsuite.data.datasets.dataset_bin import BinPklDataset

dataset = BinPklDataset(
    data_root='data/train',
    split='train',
    loop=2,
)

weights = compute_sample_weights(dataset, loop=2)

datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    use_weighted_sampler=True,
    train_sampler_weights=weights,
    loop=2,
)
```

## 📊 对比总结

| 特性 | 旧 API | 新 API |
|------|--------|--------|
| **文件路径配置** | `data_root + files` | 直接完整路径或列表 |
| **跨目录文件** | ❌ 不支持 | ✅ 支持 |
| **独立预测数据** | ❌ 与 test 共享 | ✅ 独立 `predict_data` |
| **WeightedRandomSampler** | 仅与 DynamicBatch | ✅ 独立控制 |
| **loop 与 weights** | ⚠️ 容易出错 | ✅ 明确要求匹配 |
| **weights 保存** | ✅ 保存到超参数 | ✅ 不保存（避免膨胀）|
| **灵活性** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎯 最佳实践

1. **总是使用完整路径**：避免依赖工作目录
2. **明确 loop 与 weights**：确保长度匹配
3. **独立控制采样策略**：根据需要启用 `use_weighted_sampler`
4. **预测数据独立配置**：如果与测试不同，使用 `predict_data`
5. **利用类型提示**：IDE 会提供更好的自动补全

需要帮助迁移现有代码吗？
