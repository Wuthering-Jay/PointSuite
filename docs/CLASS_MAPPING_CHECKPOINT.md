# 类别映射自动保存和加载机制

## 📌 问题背景

在训练时使用 `class_mapping` 将不连续的原始类别标签映射为连续标签（如 `{0: 0, 1: 1, 2: 2, 6: 3, 9: 4}`）。但在**单独预测**场景下，需要将模型输出的连续标签反向映射回原始标签，用户可能：
- 忘记记录训练时的 `class_mapping`
- 手动构造 `reverse_class_mapping` 时出错
- 在不同实验中混淆不同的映射关系

## ✅ 解决方案

### 自动保存机制

**class_mapping 保存到模型 checkpoint**，在预测时自动加载并应用反向映射。

---

## 🔧 使用方法

### 方案 1: 完全自动（推荐）⭐

#### 训练时：
```python
# 定义类别映射
class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}

# 1. 创建 DataModule（应用正向映射）
datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    class_mapping=class_mapping,  # Dataset 使用
)

# 2. 创建 Model（保存到 checkpoint）
model = SemanticSegmentationTask(
    backbone=backbone,
    head=head,
    class_mapping=class_mapping,  # 🔥 自动保存到 checkpoint
    learning_rate=0.001,
    ...
)

# 3. 训练
trainer.fit(model, datamodule)
```

#### 预测时：
```python
# 1. 加载模型（class_mapping 自动从 checkpoint 加载）
model = SemanticSegmentationTask.load_from_checkpoint('checkpoints/best.ckpt')

# 2. 创建 Writer（自动从模型获取 class_mapping）
writer = SegmentationWriter(
    output_dir='predictions',
    # ✅ 无需手动指定 reverse_class_mapping！
    # auto_infer_reverse_mapping=True（默认）会自动从 model.hparams.class_mapping 构建
)

# 3. 预测
trainer.predict(model, datamodule, callbacks=[writer])

# 输出：
# [SegmentationWriter] 自动加载 reverse_class_mapping 从模型 checkpoint:
#   - class_mapping: {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
#   - reverse_class_mapping: {0: 0, 1: 1, 2: 2, 3: 6, 4: 9}
```

---

### 方案 2: 手动指定

如果你想覆盖自动行为：

```python
from pointsuite.utils.callbacks import create_reverse_class_mapping

# 手动创建反向映射
class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
reverse_mapping = create_reverse_class_mapping(class_mapping)

writer = SegmentationWriter(
    output_dir='predictions',
    reverse_class_mapping=reverse_mapping,  # 手动指定（最高优先级）
)
```

---

## 🔍 优先级机制

`SegmentationWriter` 在 `on_predict_start` 时按以下优先级查找 `reverse_class_mapping`：

| 优先级 | 来源 | 说明 |
|--------|------|------|
| **1** | 用户手动指定 | `SegmentationWriter(reverse_class_mapping=...)` |
| **2** | 模型 checkpoint | `model.hparams.class_mapping` |
| **3** | DataModule | `datamodule.class_mapping` |
| **4** | 无映射 | 使用模型输出的连续标签（不转换） |

---

## 📊 完整示例

### 训练阶段

```python
import pytorch_lightning as pl
from pointsuite.data import BinPklDataModule
from pointsuite.tasks import SemanticSegmentationTask

# 1. 定义类别映射（原始 -> 连续）
class_mapping = {
    0: 0,  # 噪声
    1: 1,  # 地面
    2: 2,  # 植被
    6: 3,  # 建筑
    9: 4   # 电线
}

# 2. 创建 DataModule
datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    test_data='data/test',
    class_mapping=class_mapping,  # Dataset 应用正向映射
    batch_size=8,
    num_workers=4,
)

# 3. 创建 Model
model = SemanticSegmentationTask(
    backbone=backbone,
    head=head,
    learning_rate=0.001,
    class_mapping=class_mapping,  # 🔥 保存到 checkpoint
    loss_configs=[...],
    metric_configs=[...],
)

# 4. 训练
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/',
            monitor='val/total_loss',
            save_top_k=3,
        )
    ]
)
trainer.fit(model, datamodule)

# ✅ checkpoints/best.ckpt 现在包含 class_mapping
```

### 预测阶段（单独运行）

```python
import pytorch_lightning as pl
from pointsuite.data import BinPklDataModule
from pointsuite.tasks import SemanticSegmentationTask
from pointsuite.utils.callbacks import SegmentationWriter

# 1. 加载模型（class_mapping 自动恢复）
model = SemanticSegmentationTask.load_from_checkpoint('checkpoints/best.ckpt')
print(f"模型中的 class_mapping: {model.hparams.class_mapping}")
# 输出: {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}

# 2. 创建 DataModule（无需 class_mapping）
datamodule = BinPklDataModule(
    predict_data='data/new_scenes',
    batch_size=8,
    num_workers=4,
    # 注意：这里不需要 class_mapping，因为预测阶段无真值标签
)

# 3. 创建 Writer（自动推断）
writer = SegmentationWriter(
    output_dir='predictions',
    # ✅ 自动从 model.hparams.class_mapping 构建 reverse_mapping
)

# 4. 预测
trainer = pl.Trainer(callbacks=[writer])
trainer.predict(model, datamodule)

# 控制台输出：
# [SegmentationWriter] 自动加载 reverse_class_mapping 从模型 checkpoint:
#   - class_mapping: {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
#   - reverse_class_mapping: {0: 0, 1: 1, 2: 2, 3: 6, 4: 9}
#
# ✅ 保存的 .las 文件中的类别标签已经是原始标签（0, 1, 2, 6, 9）
```

---

## 🎯 关键要点

1. **训练时**：
   - DataModule 使用 `class_mapping` 将原始标签映射为连续标签
   - Model 接收 `class_mapping` 并保存到 checkpoint
   - 两者使用**相同的** `class_mapping`

2. **预测时**：
   - Model 自动从 checkpoint 恢复 `class_mapping`
   - SegmentationWriter 自动构建 `reverse_class_mapping`
   - 保存的 .las 文件使用**原始标签**

3. **优势**：
   - ✅ 无需手动记录映射关系
   - ✅ 避免映射错误
   - ✅ 不同实验互不干扰
   - ✅ 单文件包含所有信息（checkpoint）

---

## ⚠️ 注意事项

### 1. class_mapping 必须一致

训练时 DataModule 和 Model 应使用相同的 `class_mapping`：

```python
# ✅ 正确
class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}

datamodule = BinPklDataModule(class_mapping=class_mapping)
model = SemanticSegmentationTask(class_mapping=class_mapping)

# ❌ 错误：不一致
datamodule = BinPklDataModule(class_mapping={0: 0, 1: 1})
model = SemanticSegmentationTask(class_mapping={0: 0, 1: 1, 2: 2})
```

### 2. 禁用自动推断

如果不想使用自动推断（如想使用原始连续标签）：

```python
writer = SegmentationWriter(
    output_dir='predictions',
    auto_infer_reverse_mapping=False,  # 禁用
)
```

### 3. 手动覆盖

即使 checkpoint 中有 `class_mapping`，也可以手动覆盖：

```python
# 使用不同的映射关系（例如修正错误）
custom_mapping = {0: 0, 1: 1, 2: 2, 3: 7, 4: 10}

writer = SegmentationWriter(
    output_dir='predictions',
    reverse_class_mapping=custom_mapping,  # 最高优先级
)
```

---

## 🔧 辅助函数

```python
from pointsuite.utils.callbacks import create_reverse_class_mapping

# 快速创建反向映射
class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
reverse_mapping = create_reverse_class_mapping(class_mapping)
print(reverse_mapping)
# {0: 0, 1: 1, 2: 2, 3: 6, 4: 9}
```

---

## 📝 YAML 配置示例

```yaml
# configs/experiments/my_experiment.yaml

model:
  class_path: pointsuite.tasks.SemanticSegmentationTask
  init_args:
    learning_rate: 0.001
    
    # 🔥 类别映射（会保存到 checkpoint）
    class_mapping:
      0: 0  # 噪声 -> 0
      1: 1  # 地面 -> 1
      2: 2  # 植被 -> 2
      6: 3  # 建筑 -> 3
      9: 4  # 电线 -> 4
    
    backbone:
      class_path: pointsuite.models.backbones.PointTransformerV2m5
      init_args:
        num_classes: 5  # 映射后的类别数
    
    head:
      class_path: pointsuite.models.heads.SegmentationHead
      init_args:
        num_classes: 5

data:
  class_path: pointsuite.data.BinPklDataModule
  init_args:
    train_data: data/train
    val_data: data/val
    
    # DataModule 也使用相同的映射
    class_mapping:
      0: 0
      1: 1
      2: 2
      6: 3
      9: 4
```

---

## ✅ 总结

| 场景 | 需要做什么 | 自动处理 |
|------|-----------|---------|
| **训练** | 传入 `class_mapping` 到 Model 和 DataModule | ✅ 自动保存到 checkpoint |
| **预测（单独）** | 加载 checkpoint | ✅ 自动构建 reverse_mapping |
| **修改映射** | 手动指定 `reverse_class_mapping` | ✅ 覆盖自动行为 |

**推荐做法**：始终在训练时传入 `class_mapping` 到 Model，让框架自动处理其余部分。🎉
