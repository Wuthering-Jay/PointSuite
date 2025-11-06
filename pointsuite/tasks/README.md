# PointSuite Task Framework 使用指南

## 概述

新的 Task 框架采用**配置驱动**的设计，通过 YAML 文件定义损失函数、指标和优化器，使代码更加模块化和可维护。

## 主要特点

### 1. 配置驱动的损失函数和指标
- 通过 YAML 配置动态实例化损失函数和指标
- 支持多损失加权组合
- 自动计算和记录所有指标

### 2. LightningCLI 友好
- 优化器和调度器由 PyTorch Lightning CLI 自动配置
- 不需要在 Task 中实现 `configure_optimizers`

### 3. 模块化设计
- BaseTask 处理通用训练逻辑
- 子类只需实现 `forward()` 方法
- 可以覆盖 `_calculate_total_loss()` 处理复杂损失

### 4. 适配我们的数据格式
- 自动从 `offset` 或 `batch_index` 推断 batch_size
- 兼容拼接格式的 collate_fn
- 支持 `coord`、`feat`、`class` 等字段

## 使用示例

### 示例 1: 基础语义分割（纯 Python）

```python
from pointsuite.models.pointnet2 import PointNet2SemanticSegmentation
from pointsuite.tasks.semantic_segmentation import SemanticSegmentationTask
from pointsuite.losses import SemanticSegmentationLoss
from pointsuite.metrics import OverallAccuracy, MeanIoU

# 1. 创建模型组件
backbone_and_head = PointNet2SemanticSegmentation(
    num_classes=8,
    in_channel=3,
    feature_dim=0
)

# 我们可以将完整模型作为 backbone，head 设为 Identity
import torch.nn as nn

# 2. 定义损失配置（模拟 YAML）
loss_configs = [
    {
        "name": "ce_loss",
        "class_path": "pointsuite.losses.SemanticSegmentationLoss",
        "init_args": {
            "loss_type": "cross_entropy",
            "ignore_index": -1,
            "label_smoothing": 0.1
        },
        "weight": 1.0
    }
]

# 3. 定义指标配置
metric_configs = [
    {
        "name": "overall_acc",
        "class_path": "pointsuite.metrics.OverallAccuracy",
        "init_args": {"ignore_index": -1}
    },
    {
        "name": "miou",
        "class_path": "pointsuite.metrics.MeanIoU",
        "init_args": {"num_classes": 8, "ignore_index": -1}
    }
]

# 4. 创建 Task
task = SemanticSegmentationTask(
    backbone=backbone_and_head,  # 完整模型
    head=nn.Identity(),  # head 已包含在模型中
    learning_rate=1e-3,
    loss_configs=loss_configs,
    metric_configs=metric_configs
)

# 5. 使用 PyTorch Lightning Trainer
from pytorch_lightning import Trainer

trainer = Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1
)

# 6. 训练
trainer.fit(task, datamodule=your_datamodule)
```

### 示例 2: 使用 YAML 配置（推荐）

#### `configs/model/pointnet2_seg.yaml`
```yaml
class_path: pointsuite.tasks.semantic_segmentation.SemanticSegmentationTask
init_args:
  learning_rate: 0.001
  
  # Backbone 配置
  backbone:
    class_path: pointsuite.models.pointnet2.PointNet2SemanticSegmentation
    init_args:
      num_classes: 8
      in_channel: 3
      feature_dim: 0
      use_xyz: true
  
  # Head 配置（如果 backbone 已包含 head，使用 Identity）
  head:
    class_path: torch.nn.Identity
  
  # 损失函数配置
  loss_configs:
    - name: ce_loss
      class_path: pointsuite.losses.SemanticSegmentationLoss
      init_args:
        loss_type: cross_entropy
        ignore_index: -1
        label_smoothing: 0.1
      weight: 1.0
  
  # 指标配置
  metric_configs:
    - name: overall_acc
      class_path: pointsuite.metrics.OverallAccuracy
      init_args:
        ignore_index: -1
    
    - name: miou
      class_path: pointsuite.metrics.MeanIoU
      init_args:
        num_classes: 8
        ignore_index: -1
```

#### `configs/optimizer/adamw.yaml`
```yaml
class_path: torch.optim.AdamW
init_args:
  lr: ${model.init_args.learning_rate}  # 引用 model 中的 learning_rate
  weight_decay: 0.0001
  betas: [0.9, 0.999]
```

#### `configs/scheduler/cosine.yaml`
```yaml
class_path: torch.optim.lr_scheduler.CosineAnnealingLR
init_args:
  T_max: 100
  eta_min: 0.00001
```

#### 使用 LightningCLI 训练
```python
from pytorch_lightning.cli import LightningCLI
from pointsuite.data.point_datamodule import BinPklDataModule
from pointsuite.tasks.semantic_segmentation import SemanticSegmentationTask

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 添加自定义参数
        parser.link_arguments(
            "model.init_args.learning_rate",
            "optimizer.init_args.lr"
        )

if __name__ == "__main__":
    cli = MyLightningCLI(
        SemanticSegmentationTask,
        BinPklDataModule,
        save_config_overwrite=True
    )
```

命令行训练：
```bash
python train.py fit \
    --config configs/model/pointnet2_seg.yaml \
    --optimizer configs/optimizer/adamw.yaml \
    --lr_scheduler configs/scheduler/cosine.yaml \
    --data.data_root /path/to/data
```

### 示例 3: 多损失组合（实例分割）

```yaml
# configs/model/instance_seg.yaml
class_path: pointsuite.tasks.instance_segmentation.InstanceSegmentationTask
init_args:
  learning_rate: 0.001
  
  backbone:
    class_path: pointsuite.models.my_backbone.MyBackbone
  
  semantic_head:
    class_path: pointsuite.models.heads.SemanticHead
    init_args:
      num_classes: 8
  
  instance_head:
    class_path: pointsuite.models.heads.EmbeddingHead
    init_args:
      embedding_dim: 64
  
  # 多个损失函数
  loss_configs:
    - name: semantic_loss
      class_path: torch.nn.CrossEntropyLoss
      init_args:
        ignore_index: -1
      weight: 1.0
    
    - name: instance_loss
      class_path: pointsuite.losses.DiscriminativeLoss
      init_args:
        delta_v: 0.5
        delta_d: 1.5
        alpha: 1.0
        beta: 1.0
        gamma: 0.001
      weight: 1.0
  
  metric_configs:
    - name: sem_acc
      class_path: pointsuite.metrics.OverallAccuracy
    
    - name: sem_miou
      class_path: pointsuite.metrics.MeanIoU
      init_args:
        num_classes: 8
```

## 与旧框架的对比

### 旧框架（依赖注入）
```python
# 需要手动创建所有组件
optimizer_factory = lambda p: torch.optim.AdamW(p, lr=1e-3)
scheduler_factory = lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=100)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

task = SemanticSegmentationTask(
    backbone=backbone,
    head=head,
    loss_fn=loss_fn,
    optimizer_factory=optimizer_factory,
    scheduler_factory=scheduler_factory
)
```

### 新框架（配置驱动）
```python
# 从 YAML 配置自动创建
# 只需要一个配置文件即可

# 或者在代码中使用配置字典
loss_configs = [{"class_path": "...", "init_args": {...}}]
metric_configs = [{"class_path": "...", "init_args": {...}}]

task = SemanticSegmentationTask(
    backbone=backbone,
    head=head,
    learning_rate=1e-3,
    loss_configs=loss_configs,
    metric_configs=metric_configs
)
# optimizer 和 scheduler 由 LightningCLI 自动配置
```

## 自定义损失函数

```python
# pointsuite/losses/my_custom_loss.py
import torch.nn as nn

class MyCustomLoss(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def forward(self, preds, batch):
        """
        Args:
            preds: 模型输出（logits, dict, 等）
            batch: 批次数据字典
        
        Returns:
            loss: 标量损失值
        """
        # 从 batch 中获取标签
        target = batch['class']
        
        # 计算损失
        loss = ...
        
        return loss
```

在 YAML 中使用：
```yaml
loss_configs:
  - name: my_loss
    class_path: pointsuite.losses.my_custom_loss.MyCustomLoss
    init_args:
      param1: value1
      param2: value2
    weight: 1.0
```

## 自定义指标

```python
# pointsuite/metrics/my_metric.py
import torch.nn as nn

class MyMetric(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        # 使用 register_buffer 保存状态
        self.register_buffer('count', torch.tensor(0))
    
    def update(self, preds, batch):
        """更新指标状态"""
        # 计算并累积统计量
        self.count += 1
    
    def compute(self):
        """计算最终指标值"""
        return self.count.float()
    
    def reset(self):
        """重置状态"""
        self.count.zero_()
```

## 数据格式适配

我们的 collate_fn 产生的 batch 格式：
```python
batch = {
    'coord': torch.Tensor,   # [N, 3] 拼接后的坐标
    'feat': torch.Tensor,    # [N, C] 拼接后的特征
    'class': torch.Tensor,   # [N] 拼接后的标签
    'offset': torch.Tensor,  # [B] 累积偏移，如 [1000, 2500] 表示第一个样本1000点，第二个1500点
}
```

Task 框架会自动：
1. 从 `offset` 推断 `batch_size = len(offset)`
2. 将 batch 传递给 loss 和 metric
3. 处理不同 backbone 的输入需求

## 常见问题

### Q: 如何使用自定义的 backbone？
A: 只要你的 backbone 有 `forward()` 方法即可。如果需要特殊的输入格式，可以在 Task 的 `forward()` 中适配。

### Q: 如何添加更多的日志记录？
A: 在 Task 的 `training_step` 中使用 `self.log()` 或 `self.log_dict()`。

### Q: 如何处理不同的数据格式？
A: 覆盖 `_get_batch_size()` 方法，或在 loss/metric 中处理。

### Q: 如何使用旧的优化器工厂函数？
A: 可以直接实现 `configure_optimizers()` 覆盖默认行为：
```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

## 总结

新框架的优势：
- ✅ 配置驱动，易于实验
- ✅ 与 LightningCLI 无缝集成
- ✅ 自动处理指标计算和重置
- ✅ 多损失自动加权
- ✅ 代码更简洁，职责更清晰
- ✅ 适配我们项目的数据格式
