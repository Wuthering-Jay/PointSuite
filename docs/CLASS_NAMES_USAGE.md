# 类别名称 (Class Names) 使用指南

## 概述

类别名称功能允许你为每个类别指定有意义的名称（如 "Ground", "Vegetation"），使得评估指标更加直观易读。

**设计原则：**
- 类别名称应该在 `DataModule` 中定义（作为数据的元信息）
- `PerClassIoU` 等指标会自动从 `DataModule` 获取类别名称
- `MeanIoU` 是平均值，不需要类别名称

## 快速开始

### 方法 1：在 DataModule 中指定类别名称（推荐）

```python
from pointsuite.data import BinPklDataModule

# 定义类别映射和类别名称
datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    class_mapping={0: 0, 1: 1, 2: 2, 6: 3, 9: 4},  # 原始标签 -> 连续标签
    class_names=['Ground', 'Vegetation', 'Building', 'Wire', 'Pole'],  # 按连续标签顺序
    batch_size=8
)

# 创建指标（会自动从 datamodule 获取类别名称）
from pointsuite.utils.metrics import PerClassIoU

metric = PerClassIoU(num_classes=5, ignore_index=-1)
# 在训练过程中，metric 会自动使用 datamodule.class_names

# 打印时会显示：
# IoU/Ground: 0.85
# IoU/Vegetation: 0.78
# IoU/Building: 0.92
# IoU/Wire: 0.65
# IoU/Pole: 0.71
```

### 方法 2：手动指定类别名称

```python
# 如果需要在指标中显式指定类别名称
metric = PerClassIoU(
    num_classes=5,
    class_names=['Ground', 'Vegetation', 'Building', 'Wire', 'Pole'],
    ignore_index=-1
)
```

### 方法 3：使用原始标签号（有 class_mapping 但无 class_names）

```python
# 如果只提供 class_mapping，没有 class_names
datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    class_mapping={0: 0, 1: 1, 2: 2, 6: 3, 9: 4},
    # 注意：没有 class_names
    batch_size=8
)

# 指标会自动使用原始标签号作为名称
# 打印时会显示：
# IoU/Class 0: 0.85
# IoU/Class 1: 0.78
# IoU/Class 2: 0.92
# IoU/Class 6: 0.65  # 注意：使用原始标签号 6，不是连续标签 3
# IoU/Class 9: 0.71
```

### 方法 4：默认行为（无 class_mapping 和 class_names）

```python
datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    batch_size=8
)

# 指标会使用连续标签号作为名称
# 打印时会显示：
# IoU/Class 0: 0.85
# IoU/Class 1: 0.78
# IoU/Class 2: 0.92
# IoU/Class 3: 0.65
# IoU/Class 4: 0.71
```

## 优先级规则

类别名称按以下优先级确定：

1. **用户手动指定**：在 `PerClassIoU(..., class_names=[...])` 中指定
2. **DataModule 提供**：从 `datamodule.class_names` 获取
3. **从 class_mapping 生成**：使用原始标签号（如 "Class 6"）
4. **默认生成**：使用连续标签号（如 "Class 0", "Class 1", ...）

## 完整示例

```python
import pytorch_lightning as pl
from pointsuite.data import BinPklDataModule
from pointsuite.tasks import SemanticSegmentationTask

# 1. 创建 DataModule（定义类别名称）
datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    test_data='data/test',
    class_mapping={0: 0, 1: 1, 2: 2, 6: 3, 9: 4},
    class_names=['Ground', 'Vegetation', 'Building', 'Wire', 'Pole'],
    batch_size=8,
    num_workers=4
)

# 2. 创建任务（指标会自动从 datamodule 获取类别名称）
task = SemanticSegmentationTask(
    backbone='PointNet++',
    num_classes=5,
    ignore_label=-1,
    learning_rate=0.001
)

# 3. 训练
trainer = pl.Trainer(max_epochs=100, devices=1)
trainer.fit(task, datamodule)

# 训练日志会显示：
# Epoch 1: train_loss=0.5, val_MeanIoU=0.72
# Epoch 1 Val Metrics:
#   IoU/Ground: 0.85
#   IoU/Vegetation: 0.78
#   IoU/Building: 0.92
#   IoU/Wire: 0.65
#   IoU/Pole: 0.71
```

## 注意事项

### 1. class_names 的长度必须与 num_classes 匹配

```python
# ❌ 错误：长度不匹配
datamodule = BinPklDataModule(
    class_mapping={0: 0, 1: 1, 2: 2},  # 3 个类别
    class_names=['Ground', 'Vegetation']  # 只有 2 个名称 -> 错误！
)

# ✅ 正确
datamodule = BinPklDataModule(
    class_mapping={0: 0, 1: 1, 2: 2},
    class_names=['Ground', 'Vegetation', 'Building']  # 3 个名称
)
```

### 2. class_names 的顺序必须与映射后的连续标签对应

```python
# class_mapping: 原始 -> 连续
# {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
#
# class_names 的顺序应该是：
# [连续标签0的名称, 连续标签1的名称, 连续标签2的名称, 连续标签3的名称, 连续标签4的名称]

class_names = [
    'Ground',      # 连续标签 0（原始标签 0）
    'Vegetation',  # 连续标签 1（原始标签 1）
    'Building',    # 连续标签 2（原始标签 2）
    'Wire',        # 连续标签 3（原始标签 6）
    'Pole'         # 连续标签 4（原始标签 9）
]
```

### 3. MeanIoU 不需要类别名称

```python
from pointsuite.utils.metrics import MeanIoU

# MeanIoU 只返回平均值，不需要类别名称
mean_iou = MeanIoU(num_classes=5, ignore_index=-1)
# 输出: val_MeanIoU=0.72
```

### 4. PerClassIoU 自动显示每个类别的指标

```python
from pointsuite.utils.metrics import PerClassIoU

per_class_iou = PerClassIoU(num_classes=5, ignore_index=-1)
# 输出:
#   IoU/Ground: 0.85
#   IoU/Vegetation: 0.78
#   IoU/Building: 0.92
#   IoU/Wire: 0.65
#   IoU/Pole: 0.71
```

## 在任务中使用

```python
from pointsuite.tasks import SemanticSegmentationTask

task = SemanticSegmentationTask(
    backbone='PointNet++',
    num_classes=5,
    ignore_label=-1,
    learning_rate=0.001,
    # 注意：不需要在这里指定 class_names
    # 任务会自动从 datamodule 获取
)
```

## 预测时的类别名称

在预测阶段，类别名称会自动从 checkpoint 中恢复：

```python
# 训练时
task = SemanticSegmentationTask(num_classes=5, ...)
trainer.fit(task, datamodule)  # datamodule.class_names 自动保存到 checkpoint

# 预测时
task = SemanticSegmentationTask.load_from_checkpoint('checkpoint.ckpt')
# task.hparams 包含 class_mapping 和其他信息
# 如果需要类别名称，可以通过 datamodule 传递
trainer.predict(task, datamodule=datamodule_predict)
```

## 高级用法：动态修改类别名称

```python
# 如果需要在不同语言环境显示不同名称
datamodule_en = BinPklDataModule(
    train_data='data/train',
    class_names=['Ground', 'Vegetation', 'Building', 'Wire', 'Pole']
)

datamodule_zh = BinPklDataModule(
    train_data='data/train',
    class_names=['地面', '植被', '建筑', '线缆', '杆塔']
)

# 指标会根据使用的 datamodule 显示对应语言的名称
```

## 总结

- ✅ **推荐做法**：在 `DataModule` 中定义 `class_names`，让指标自动获取
- ✅ 类别名称是数据的元信息，应该与数据一起定义
- ✅ `MeanIoU` 不需要类别名称（它只是平均值）
- ✅ 如果没有提供 `class_names`，会自动使用原始标签号或连续标签号
- ❌ 不要在每个指标中重复指定 `class_names`（除非有特殊需求）
