"""
示例配置：使用新的损失函数和指标

展示如何在配置文件中使用 models/losses/ 和 utils/metrics.py
"""

# =============================================================================
# 示例 0: 使用统一指标 SegmentationMetrics (推荐)
# =============================================================================
example_0_unified_metrics = """
model:
  class_path: pointsuite.tasks.SemanticSegmentationTask
  init_args:
    backbone:
      class_path: pointsuite.models.pointnet2.PointNet2SemanticSegmentation
      init_args:
        num_classes: 8
        in_channels: 3
    
    # 使用 CrossEntropyLoss
    losses:
      ce:
        class_path: pointsuite.models.losses.CrossEntropyLoss
        init_args:
          weight: [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
          label_smoothing: 0.1
          ignore_index: -1
    
    loss_weights:
      ce: 1.0
    
    # 使用统一指标 - 一次性计算所有指标，避免重复计算混淆矩阵
    metrics:
      all:
        class_path: pointsuite.utils.metrics.SegmentationMetrics
        init_args:
          num_classes: 8
          class_names: ['ground', 'vegetation', 'cars', 'trucks', 'powerlines', 'fences', 'poles', 'buildings']
          ignore_index: -1
    
    # 注意: SegmentationMetrics 的 compute() 返回字典，包含:
    # - overall_accuracy, mean_iou, mean_precision, mean_recall, mean_f1
    # - iou_per_class, precision_per_class, recall_per_class, f1_per_class

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001
    weight_decay: 0.01
"""

# =============================================================================
# 示例 1: 使用单个损失函数 + 分离指标
# =============================================================================
example_1_cross_entropy = """
model:
  class_path: pointsuite.tasks.SemanticSegmentationTask
  init_args:
    backbone:
      class_path: pointsuite.models.pointnet2.PointNet2SemanticSegmentation
      init_args:
        num_classes: 8
        in_channels: 3
    
    # 使用 CrossEntropyLoss
    losses:
      ce:
        class_path: pointsuite.models.losses.CrossEntropyLoss
        init_args:
          weight: [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # 类别权重
          label_smoothing: 0.1
          ignore_index: -1
    
    loss_weights:
      ce: 1.0
    
    # 使用 torchmetrics 指标
    metrics:
      oa:
        class_path: pointsuite.utils.metrics.OverallAccuracy
        init_args:
          ignore_index: -1
      
      miou:
        class_path: pointsuite.utils.metrics.MeanIoU
        init_args:
          num_classes: 8
          ignore_index: -1
      
      precision:
        class_path: pointsuite.utils.metrics.Precision
        init_args:
          num_classes: 8
          class_names: ['ground', 'vegetation', 'cars', 'trucks', 'powerlines', 'fences', 'poles', 'buildings']
          ignore_index: -1
      
      recall:
        class_path: pointsuite.utils.metrics.Recall
        init_args:
          num_classes: 8
          class_names: ['ground', 'vegetation', 'cars', 'trucks', 'powerlines', 'fences', 'poles', 'buildings']
          ignore_index: -1
      
      f1:
        class_path: pointsuite.utils.metrics.F1Score
        init_args:
          num_classes: 8
          class_names: ['ground', 'vegetation', 'cars', 'trucks', 'powerlines', 'fences', 'poles', 'buildings']
          ignore_index: -1

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001
    weight_decay: 0.01

lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max: 100
    eta_min: 0.00001
"""

# =============================================================================
# 示例 2: 使用多个损失函数组合
# =============================================================================
example_2_multi_loss = """
model:
  class_path: pointsuite.tasks.SemanticSegmentationTask
  init_args:
    backbone:
      class_path: pointsuite.models.pointnet2.PointNet2SemanticSegmentation
      init_args:
        num_classes: 8
        in_channels: 3
    
    # 组合多个损失函数
    losses:
      ce:
        class_path: pointsuite.models.losses.CrossEntropyLoss
        init_args:
          weight: [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
          ignore_index: -1
      
      focal:
        class_path: pointsuite.models.losses.FocalLoss
        init_args:
          gamma: 2.0
          alpha: [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
          ignore_index: -1
      
      lovasz:
        class_path: pointsuite.models.losses.LovaszLoss
        init_args:
          ignore_index: -1
          per_point: false
          classes: 'present'
    
    # 损失权重
    loss_weights:
      ce: 0.5      # CrossEntropy 权重
      focal: 0.3   # Focal Loss 权重
      lovasz: 0.2  # Lovasz Loss 权重
    
    # 指标
    metrics:
      oa:
        class_path: pointsuite.utils.metrics.OverallAccuracy
        init_args:
          ignore_index: -1
      
      miou:
        class_path: pointsuite.utils.metrics.MeanIoU
        init_args:
          num_classes: 8
          ignore_index: -1
      
      f1:
        class_path: pointsuite.utils.metrics.F1Score
        init_args:
          num_classes: 8
          ignore_index: -1

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001
    weight_decay: 0.01
"""

# =============================================================================
# 示例 3: 使用 Dice + CE 组合损失
# =============================================================================
example_3_dice_ce = """
model:
  class_path: pointsuite.tasks.SemanticSegmentationTask
  init_args:
    backbone:
      class_path: pointsuite.models.pointnet2.PointNet2SemanticSegmentation
      init_args:
        num_classes: 8
        in_channels: 3
    
    # 使用 DiceCELoss (内置组合)
    losses:
      dice_ce:
        class_path: pointsuite.models.losses.DiceCELoss
        init_args:
          dice_weight: 1.0
          ce_weight: 1.0
          smooth: 1.0
          ignore_index: -1
          class_weight: [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
          label_smoothing: 0.1
    
    loss_weights:
      dice_ce: 1.0
    
    metrics:
      oa:
        class_path: pointsuite.utils.metrics.OverallAccuracy
        init_args:
          ignore_index: -1
      
      miou:
        class_path: pointsuite.utils.metrics.MeanIoU
        init_args:
          num_classes: 8
          ignore_index: -1

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001
"""

# =============================================================================
# 示例 4: DDP 训练配置
# =============================================================================
example_4_ddp = """
# trainer.py
from lightning.pytorch.cli import LightningCLI
from pointsuite.data.point_datamodule import PointDataModule

cli = LightningCLI(
    datamodule_class=PointDataModule,
    save_config_callback=None,
    run=False
)

# 使用 DDP 启动训练
# torchrun --nproc_per_node=4 trainer.py fit --config config.yaml

# config.yaml 内容：
trainer:
  accelerator: gpu
  devices: 4
  strategy: ddp  # 使用 DDP 策略
  max_epochs: 100
  log_every_n_steps: 10
  
  # DDP 下 torchmetrics 会自动同步！

model:
  class_path: pointsuite.tasks.SemanticSegmentationTask
  init_args:
    backbone:
      class_path: pointsuite.models.pointnet2.PointNet2SemanticSegmentation
      init_args:
        num_classes: 8
        in_channels: 3
    
    losses:
      focal:
        class_path: pointsuite.models.losses.FocalLoss
        init_args:
          gamma: 2.0
          ignore_index: -1
    
    loss_weights:
      focal: 1.0
    
    # 所有指标自动支持 DDP 同步
    metrics:
      oa:
        class_path: pointsuite.utils.metrics.OverallAccuracy
        init_args:
          ignore_index: -1
      
      miou:
        class_path: pointsuite.utils.metrics.MeanIoU
        init_args:
          num_classes: 8
          ignore_index: -1

data:
  class_path: pointsuite.data.point_datamodule.PointDataModule
  init_args:
    train_dataset:
      class_path: pointsuite.data.datasets.BinPklDataset
      init_args:
        data_root: /path/to/dales/train
        # ... 其他参数
    
    batch_size: 16  # 每个 GPU 的 batch size
    num_workers: 4
"""

# =============================================================================
# Python 代码示例
# =============================================================================
python_example = """
import torch
from pointsuite.models.losses import CrossEntropyLoss, FocalLoss, DiceCELoss
from pointsuite.utils.metrics import (
    OverallAccuracy, MeanIoU, Precision, Recall, F1Score,
    SegmentationMetrics  # 推荐：统一计算所有指标
)

# 创建损失函数
ce_loss = CrossEntropyLoss(
    weight=torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    label_smoothing=0.1,
    ignore_index=-1
)

focal_loss = FocalLoss(
    gamma=2.0,
    alpha=torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    ignore_index=-1
)

dice_ce = DiceCELoss(
    dice_weight=1.0,
    ce_weight=1.0,
    ignore_index=-1
)

# 方法 1: 使用单独的指标（分别计算）
class_names = ['ground', 'vegetation', 'cars', 'trucks', 'powerlines', 'fences', 'poles', 'buildings']
oa_metric = OverallAccuracy(ignore_index=-1)
miou_metric = MeanIoU(num_classes=8, ignore_index=-1)
precision_metric = Precision(num_classes=8, class_names=class_names, ignore_index=-1)
recall_metric = Recall(num_classes=8, class_names=class_names, ignore_index=-1)
f1_metric = F1Score(num_classes=8, class_names=class_names, ignore_index=-1)

# 训练循环
for batch in train_loader:
    preds = model(batch)
    
    # 计算损失
    loss = 0.5 * ce_loss(preds, batch) + 0.5 * focal_loss(preds, batch)
    
    # 更新指标
    oa_metric.update(preds, batch['class'])
    miou_metric.update(preds, batch['class'])
    precision_metric.update(preds, batch['class'])
    recall_metric.update(preds, batch['class'])
    f1_metric.update(preds, batch['class'])
    
    # 反向传播
    loss.backward()
    optimizer.step()

# Epoch 结束时计算指标
oa = oa_metric.compute()
miou = miou_metric.compute()
precision = precision_metric.compute()
recall = recall_metric.compute()
f1 = f1_metric.compute()

print(f"OA: {oa:.4f}, mIoU: {miou:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# 获取详细的每类指标
f1_details = f1_metric.get_detailed_results()
for i, name in enumerate(f1_details['class_names']):
    print(f"{name}: F1={f1_details['f1_per_class'][i]:.4f}, "
          f"Prec={f1_details['precision_per_class'][i]:.4f}, "
          f"Rec={f1_details['recall_per_class'][i]:.4f}")

# 重置指标
oa_metric.reset()
miou_metric.reset()
precision_metric.reset()
recall_metric.reset()
f1_metric.reset()

# ============================================================================
# 方法 2: 使用统一指标（推荐，避免重复计算混淆矩阵）
# ============================================================================
seg_metric = SegmentationMetrics(
    num_classes=8,
    class_names=class_names,
    ignore_index=-1
)

# 训练循环
for batch in train_loader:
    preds = model(batch)
    
    # 计算损失
    loss = 0.5 * ce_loss(preds, batch) + 0.5 * focal_loss(preds, batch)
    
    # 只需更新一次！
    seg_metric.update(preds, batch['class'])
    
    # 反向传播
    loss.backward()
    optimizer.step()

# Epoch 结束时，一次性获取所有指标
all_metrics = seg_metric.compute()

print(f"OA: {all_metrics['overall_accuracy']:.4f}")
print(f"mIoU: {all_metrics['mean_iou']:.4f}")
print(f"Precision: {all_metrics['mean_precision']:.4f}")
print(f"Recall: {all_metrics['mean_recall']:.4f}")
print(f"F1: {all_metrics['mean_f1']:.4f}")

# 打印每个类别的详细指标
for i, name in enumerate(all_metrics['class_names']):
    iou = all_metrics['iou_per_class'][i]
    prec = all_metrics['precision_per_class'][i]
    rec = all_metrics['recall_per_class'][i]
    f1 = all_metrics['f1_per_class'][i]
    print(f"{name}: IoU={iou:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

# 重置指标
seg_metric.reset()
"""

if __name__ == "__main__":
    print("=" * 80)
    print("示例 0: 使用统一指标 SegmentationMetrics (推荐)")
    print("=" * 80)
    print(example_0_unified_metrics)
    
    print("\n" + "=" * 80)
    print("示例 1: 使用 CrossEntropyLoss + 分离指标")
    print("=" * 80)
    print(example_1_cross_entropy)
    
    print("\n" + "=" * 80)
    print("示例 2: 多损失函数组合")
    print("=" * 80)
    print(example_2_multi_loss)
    
    print("\n" + "=" * 80)
    print("示例 3: DiceCELoss")
    print("=" * 80)
    print(example_3_dice_ce)
    
    print("\n" + "=" * 80)
    print("示例 4: DDP 训练")
    print("=" * 80)
    print(example_4_ddp)
    
    print("\n" + "=" * 80)
    print("Python 代码示例")
    print("=" * 80)
    print(python_example)
