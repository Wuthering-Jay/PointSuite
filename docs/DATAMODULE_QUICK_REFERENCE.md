# DataModule 快速参考

## 📦 导入

```python
# 向后兼容
from pointsuite.data.point_datamodule import PointDataModule

# 新方式（推荐）
from pointsuite.data.datamodule_binpkl import BinPklDataModule

# 基类（用于自定义）
from pointsuite.data.datamodule_base import DataModuleBase

# 从包导入
from pointsuite.data import PointDataModule, BinPklDataModule, DataModuleBase
```

## 🎯 基础用法

```python
datamodule = BinPklDataModule(
    data_root='path/to/data',
    train_files=['train.pkl'],  # 可选，None 则自动发现
    val_files=['val.pkl'],
    test_files=['test.pkl'],
    batch_size=8,               # use_dynamic_batch=False 时使用
    num_workers=4,
)

datamodule.setup('fit')
trainer.fit(model, datamodule)
```

## 🔥 DynamicBatchSampler

```python
datamodule = BinPklDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,     # ← 启用动态批次
    max_points=500000,          # ← 每批最多50万点
    num_workers=8,
)
```

## ⚖️ 加权采样（处理类别不平衡）

```python
# 计算样本权重
sample_weights = compute_weights_from_class_distribution(dataset)

datamodule = BinPklDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,
    max_points=500000,
    train_sampler_weights=sample_weights,  # ← 加权采样
)
```

## 🛠️ 完整配置

```python
datamodule = BinPklDataModule(
    # 数据路径
    data_root='path/to/data',
    train_files=['train.pkl'],
    val_files=['val.pkl'],
    test_files=['test.pkl'],
    
    # BinPkl 特定参数
    assets=['coord', 'intensity', 'classification'],
    ignore_label=-1,
    loop=2,                     # 训练数据循环2次
    cache_data=False,
    class_mapping={0: 0, 1: 1, 2: 2, 6: 3, 9: 4},
    
    # 采样策略
    use_dynamic_batch=True,     # 动态批次
    max_points=500000,
    train_sampler_weights=None, # 可选：样本权重
    
    # DataLoader 参数
    batch_size=8,               # 仅当 use_dynamic_batch=False
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    
    # 数据增强
    train_transforms=[...],
    val_transforms=[...],
    test_transforms=[...],
)
```

## 🎨 创建自定义 DataModule

```python
from pointsuite.data.datamodule_base import DataModuleBase

class MyDataModule(DataModuleBase):
    def __init__(self, data_root, my_param=None, **kwargs):
        self.my_param = my_param
        super().__init__(data_root=data_root, **kwargs)
    
    def _create_dataset(self, data_paths, split, transforms):
        return MyDataset(
            data_paths=data_paths,
            split=split,
            my_param=self.my_param,
            transform=transforms
        )

# 使用
datamodule = MyDataModule(
    data_root='path/to/data',
    my_param='value',
    use_dynamic_batch=True,
    max_points=500000
)
```

## 🔍 方法

### 设置和生命周期
- `setup(stage)` - 设置数据集
- `prepare_data()` - 数据准备（单进程）
- `teardown(stage)` - 清理资源

### DataLoader 创建
- `train_dataloader()` - 训练 DataLoader
- `val_dataloader()` - 验证 DataLoader
- `test_dataloader()` - 测试 DataLoader
- `predict_dataloader()` - 预测 DataLoader

### 工具方法
- `get_dataset_info(split)` - 获取数据集信息
- `print_info()` - 打印详细信息

## 📊 数据集信息

```python
datamodule.setup('fit')

# 获取信息
info = datamodule.get_dataset_info('train')
print(info)
# {
#     'split': 'train',
#     'num_samples': 100,
#     'total_length': 200,  # with loop=2
#     'assets': ['coord', 'intensity', 'classification'],
#     'loop': 2,
#     'cache_enabled': False,
#     'class_mapping': {...}
# }

# 打印所有信息
datamodule.print_info()
```

## ⚡ 性能优化

### 小数据集（< 10GB）
```python
datamodule = BinPklDataModule(
    data_root='...',
    batch_size=16,
    num_workers=4,
    cache_data=True,            # ← 缓存到内存
    persistent_workers=True,
    prefetch_factor=4,
)
```

### 大数据集（> 50GB）
```python
datamodule = BinPklDataModule(
    data_root='...',
    use_dynamic_batch=True,     # ← 动态批次控制内存
    max_points=500000,
    num_workers=8,
    cache_data=False,           # ← 不缓存
    persistent_workers=True,
    prefetch_factor=2,
)
```

### 多GPU训练
```python
datamodule = BinPklDataModule(
    data_root='...',
    batch_size=2,               # 每GPU批次大小
    num_workers=8,              # 每GPU worker数
    persistent_workers=True,
    pin_memory=True,
)

trainer = pl.Trainer(
    devices=4,                  # 4个GPU
    strategy='ddp',
)
trainer.fit(model, datamodule)
```

## 🐛 常见问题

### Q: PointDataModule 和 BinPklDataModule 有什么区别？
A: 没有区别！`PointDataModule` 是 `BinPklDataModule` 的别名，用于向后兼容。

### Q: 如何启用 DynamicBatchSampler？
A: 设置 `use_dynamic_batch=True` 和 `max_points=500000`

### Q: 如何处理类别不平衡？
A: 计算样本权重并传入 `train_sampler_weights` 参数

### Q: 如何创建支持新格式的 DataModule？
A: 继承 `DataModuleBase` 并实现 `_create_dataset()` 方法

## 📚 更多文档

- [完整重构文档](docs/DATAMODULE_REFACTOR.md)
- [DynamicBatchSampler 详细指南](docs/DYNAMIC_BATCH_SAMPLER.md)
- [示例代码](examples/datamodule_usage_example.py)
