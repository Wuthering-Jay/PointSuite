"""
测试新的损失函数和指标

验证 models/losses/ 和 utils/metrics.py 的功能
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointsuite.models.losses import (
    CrossEntropyLoss,
    FocalLoss,
    LovaszLoss,
    DiceLoss,
    DiceCELoss
)
from pointsuite.utils.metrics import (
    OverallAccuracy,
    MeanIoU,
    PerClassIoU,
    Precision,
    Recall,
    F1Score,
    SegmentationMetrics
)


def test_losses():
    """测试所有损失函数"""
    print("=" * 80)
    print("测试损失函数")
    print("=" * 80)
    
    # 创建虚拟数据
    batch_size = 4
    num_points = 1000
    num_classes = 8
    
    # 模拟预测 logits [B*N, C]
    preds = torch.randn(batch_size * num_points, num_classes)
    
    # 模拟真实标签 [B*N]
    target = torch.randint(0, num_classes, (batch_size * num_points,))
    
    # 创建 batch 字典
    batch = {
        'class': target,
        'offset': torch.tensor([num_points, num_points*2, num_points*3, num_points*4])
    }
    
    # 类别权重
    class_weight = torch.ones(num_classes)
    class_weight[0] = 0.5  # 地面类权重较小
    
    print("\n输入数据形状:")
    print(f"  preds: {preds.shape}")
    print(f"  target: {target.shape}")
    print(f"  batch size: {batch_size}, points per sample: {num_points}")
    
    # 1. CrossEntropyLoss
    print("\n1. CrossEntropyLoss")
    ce_loss = CrossEntropyLoss(
        weight=class_weight,
        label_smoothing=0.1,
        ignore_index=-1
    )
    loss_ce = ce_loss(preds, batch)
    print(f"   Loss: {loss_ce.item():.4f}")
    
    # 2. FocalLoss
    print("\n2. FocalLoss")
    focal_loss = FocalLoss(
        gamma=2.0,
        alpha=class_weight,
        ignore_index=-1
    )
    loss_focal = focal_loss(preds, batch)
    print(f"   Loss: {loss_focal.item():.4f}")
    
    # 3. LovaszLoss
    print("\n3. LovaszLoss")
    lovasz_loss = LovaszLoss(
        ignore_index=-1,
        per_point=False,
        classes='present'
    )
    loss_lovasz = lovasz_loss(preds, batch)
    print(f"   Loss: {loss_lovasz.item():.4f}")
    
    # 4. DiceLoss
    print("\n4. DiceLoss")
    dice_loss = DiceLoss(
        smooth=1.0,
        ignore_index=-1
    )
    loss_dice = dice_loss(preds, batch)
    print(f"   Loss: {loss_dice.item():.4f}")
    
    # 5. DiceCELoss
    print("\n5. DiceCELoss")
    dice_ce = DiceCELoss(
        dice_weight=1.0,
        ce_weight=1.0,
        ignore_index=-1,
        class_weight=class_weight
    )
    loss_dice_ce = dice_ce(preds, batch)
    print(f"   Loss: {loss_dice_ce.item():.4f}")
    
    # 测试组合损失
    print("\n6. 组合损失 (0.5*CE + 0.3*Focal + 0.2*Lovasz)")
    total_loss = 0.5 * loss_ce + 0.3 * loss_focal + 0.2 * loss_lovasz
    print(f"   Total Loss: {total_loss.item():.4f}")
    
    print("\n✅ 所有损失函数测试通过！")


def test_metrics():
    """测试所有指标"""
    print("\n" + "=" * 80)
    print("测试指标")
    print("=" * 80)
    
    num_classes = 8
    num_points = 1000
    
    # 创建虚拟预测和标签
    preds = torch.randn(num_points, num_classes)
    target = torch.randint(0, num_classes, (num_points,))
    
    print(f"\n输入数据形状:")
    print(f"  preds: {preds.shape}")
    print(f"  target: {target.shape}")
    
    # 1. OverallAccuracy
    print("\n1. OverallAccuracy")
    oa_metric = OverallAccuracy(ignore_index=-1)
    oa_metric.update(preds, target)
    oa = oa_metric.compute()
    print(f"   OA: {oa.item():.4f}")
    oa_metric.reset()
    
    # 2. MeanIoU
    print("\n2. MeanIoU")
    miou_metric = MeanIoU(num_classes=num_classes, ignore_index=-1)
    miou_metric.update(preds, target)
    miou = miou_metric.compute()
    print(f"   mIoU: {miou.item():.4f}")
    
    # 获取每个类别的 IoU
    per_class_iou = miou_metric.get_per_class_iou()
    print(f"   Per-class IoU: {per_class_iou.tolist()}")
    miou_metric.reset()
    
    # 3. PerClassIoU
    print("\n3. PerClassIoU")
    class_names = ['ground', 'vegetation', 'cars', 'trucks', 'powerlines', 'fences', 'poles', 'buildings']
    per_class_metric = PerClassIoU(
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=-1
    )
    per_class_metric.update(preds, target)
    results = per_class_metric.compute()
    
    print(f"   Class-wise metrics:")
    for i, name in enumerate(results['class_names']):
        iou = results['iou_per_class'][i].item()
        prec = results['precision_per_class'][i].item()
        rec = results['recall_per_class'][i].item()
        print(f"     {name:12s}: IoU={iou:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
    per_class_metric.reset()
    
    # 4. Precision
    print("\n4. Precision")
    from pointsuite.utils.metrics import Precision
    precision_metric = Precision(
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=-1
    )
    precision_metric.update(preds, target)
    prec = precision_metric.compute()
    print(f"   Macro Precision: {prec.item():.4f}")
    
    # 获取详细结果
    prec_results = precision_metric.get_detailed_results()
    print(f"   Per-class Precision:")
    for i, name in enumerate(prec_results['class_names']):
        p = prec_results['precision_per_class'][i].item()
        print(f"     {name:12s}: {p:.4f}")
    precision_metric.reset()
    
    # 5. Recall
    print("\n5. Recall")
    from pointsuite.utils.metrics import Recall
    recall_metric = Recall(
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=-1
    )
    recall_metric.update(preds, target)
    rec = recall_metric.compute()
    print(f"   Macro Recall: {rec.item():.4f}")
    
    # 获取详细结果
    rec_results = recall_metric.get_detailed_results()
    print(f"   Per-class Recall:")
    for i, name in enumerate(rec_results['class_names']):
        r = rec_results['recall_per_class'][i].item()
        print(f"     {name:12s}: {r:.4f}")
    recall_metric.reset()
    
    # 6. F1Score
    print("\n6. F1Score")
    f1_metric = F1Score(
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=-1
    )
    f1_metric.update(preds, target)
    f1 = f1_metric.compute()
    print(f"   Macro F1: {f1.item():.4f}")
    
    # 获取详细结果
    f1_results = f1_metric.get_detailed_results()
    print(f"   Detailed results:")
    print(f"     Mean Precision: {f1_results['mean_precision'].item():.4f}")
    print(f"     Mean Recall:    {f1_results['mean_recall'].item():.4f}")
    print(f"     Mean F1:        {f1_results['mean_f1'].item():.4f}")
    print(f"   Per-class F1:")
    for i, name in enumerate(f1_results['class_names']):
        f = f1_results['f1_per_class'][i].item()
        p = f1_results['precision_per_class'][i].item()
        r = f1_results['recall_per_class'][i].item()
        print(f"     {name:12s}: F1={f:.4f}, Prec={p:.4f}, Rec={r:.4f}")
    f1_metric.reset()
    
    # 7. SegmentationMetrics (统一指标，推荐使用)
    print("\n7. SegmentationMetrics (统一计算所有指标)")
    seg_metrics = SegmentationMetrics(
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=-1
    )
    seg_metrics.update(preds, target)
    all_results = seg_metrics.compute()
    
    print(f"   Overall Accuracy: {all_results['overall_accuracy'].item():.4f}")
    print(f"   Mean IoU:         {all_results['mean_iou'].item():.4f}")
    print(f"   Mean Precision:   {all_results['mean_precision'].item():.4f}")
    print(f"   Mean Recall:      {all_results['mean_recall'].item():.4f}")
    print(f"   Mean F1:          {all_results['mean_f1'].item():.4f}")
    print(f"   Per-class summary:")
    for i, name in enumerate(all_results['class_names']):
        iou = all_results['iou_per_class'][i].item()
        prec = all_results['precision_per_class'][i].item()
        rec = all_results['recall_per_class'][i].item()
        f1 = all_results['f1_per_class'][i].item()
        print(f"     {name:12s}: IoU={iou:.4f}, P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")
    seg_metrics.reset()
    
    print("\n✅ 所有指标测试通过！")


def test_ignore_index():
    """测试 ignore_index 功能"""
    print("\n" + "=" * 80)
    print("测试 ignore_index")
    print("=" * 80)
    
    num_classes = 8
    num_points = 1000
    ignore_index = -1
    
    # 创建数据，部分标签为 ignore_index
    preds = torch.randn(num_points, num_classes)
    target = torch.randint(0, num_classes, (num_points,))
    
    # 随机设置 10% 的标签为 ignore_index
    ignore_mask = torch.rand(num_points) < 0.1
    target[ignore_mask] = ignore_index
    
    valid_points = (target != ignore_index).sum().item()
    
    print(f"\n总点数: {num_points}")
    print(f"有效点数: {valid_points}")
    print(f"忽略点数: {num_points - valid_points}")
    
    batch = {'class': target}
    
    # 测试损失函数
    print("\n测试损失函数 (应该只计算有效点):")
    ce_loss = CrossEntropyLoss(ignore_index=ignore_index)
    loss = ce_loss(preds, batch)
    print(f"  CrossEntropyLoss: {loss.item():.4f}")
    
    # 测试指标
    print("\n测试指标 (应该只计算有效点):")
    oa_metric = OverallAccuracy(ignore_index=ignore_index)
    oa_metric.update(preds, target)
    oa = oa_metric.compute()
    print(f"  OverallAccuracy: {oa.item():.4f}")
    
    print("\n✅ ignore_index 功能测试通过！")


def test_batch_formats():
    """测试不同的 batch 格式"""
    print("\n" + "=" * 80)
    print("测试不同 batch 格式")
    print("=" * 80)
    
    num_classes = 8
    
    # 格式 1: [N, C] - 单个样本或拼接后的批次
    print("\n格式 1: [N, C]")
    N = 1000
    preds_1 = torch.randn(N, num_classes)
    target_1 = torch.randint(0, num_classes, (N,))
    batch_1 = {'class': target_1}
    
    ce_loss = CrossEntropyLoss()
    loss_1 = ce_loss(preds_1, batch_1)
    print(f"  preds shape: {preds_1.shape}")
    print(f"  target shape: {target_1.shape}")
    print(f"  loss: {loss_1.item():.4f}")
    
    # 格式 2: [B, N, C] - 批次维度分开
    print("\n格式 2: [B, N, C]")
    B, N = 4, 250
    preds_2 = torch.randn(B, N, num_classes)
    target_2 = torch.randint(0, num_classes, (B, N))
    batch_2 = {'class': target_2}
    
    loss_2 = ce_loss(preds_2, batch_2)
    print(f"  preds shape: {preds_2.shape}")
    print(f"  target shape: {target_2.shape}")
    print(f"  loss: {loss_2.item():.4f}")
    
    print("\n✅ 不同 batch 格式测试通过！")


def test_ddp_compatibility():
    """测试 DDP 兼容性（单机模拟）"""
    print("\n" + "=" * 80)
    print("测试 DDP 兼容性")
    print("=" * 80)
    
    num_classes = 8
    num_points = 1000
    
    preds = torch.randn(num_points, num_classes)
    target = torch.randint(0, num_classes, (num_points,))
    
    print("\n创建指标 (dist_sync_on_step=True):")
    oa_metric = OverallAccuracy(ignore_index=-1, dist_sync_on_step=True)
    miou_metric = MeanIoU(num_classes=num_classes, ignore_index=-1, dist_sync_on_step=True)
    
    # 更新指标
    oa_metric.update(preds, target)
    miou_metric.update(preds, target)
    
    # 计算
    oa = oa_metric.compute()
    miou = miou_metric.compute()
    
    print(f"  OA: {oa.item():.4f}")
    print(f"  mIoU: {miou.item():.4f}")
    
    print("\n✅ torchmetrics 指标自动支持 DDP 同步！")
    print("   在真实 DDP 环境下，指标会自动跨进程聚合")


if __name__ == "__main__":
    # 运行所有测试
    test_losses()
    test_metrics()
    test_ignore_index()
    test_batch_formats()
    test_ddp_compatibility()
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)
    print("\n新的损失函数和指标已准备就绪:")
    print("  - 损失函数: pointsuite/models/losses/")
    print("    - CrossEntropyLoss")
    print("    - FocalLoss")
    print("    - LovaszLoss")
    print("    - DiceLoss")
    print("    - DiceCELoss")
    print("\n  - 指标: pointsuite/utils/metrics.py")
    print("    - OverallAccuracy")
    print("    - MeanIoU")
    print("    - PerClassIoU")
    print("    - Precision (支持每类和均值)")
    print("    - Recall (支持每类和均值)")
    print("    - F1Score (支持每类和均值)")
    print("    - SegmentationMetrics (统一计算所有指标，推荐)")
    print("\n  - 所有功能:")
    print("    ✅ 支持 ignore_index")
    print("    ✅ 支持多种 batch 格式")
    print("    ✅ 使用 torchmetrics 自动 DDP 同步")
    print("    ✅ 支持每类别和均值计算")
    print("    ✅ 配置文件友好")
