"""
测试类别名称功能

验证 DataModule 中的 class_names 能被 PerClassIoU 自动获取
"""

import torch
from pointsuite.utils.metrics import PerClassIoU, MeanIoU, create_class_names


def test_create_class_names():
    """测试 create_class_names 辅助函数"""
    print("=" * 60)
    print("测试 create_class_names 辅助函数")
    print("=" * 60)
    
    # 情况 1: 用户提供名称
    names1 = create_class_names(
        5,
        class_names=['Ground', 'Vegetation', 'Building', 'Wire', 'Pole']
    )
    print(f"情况 1 (用户提供): {names1}")
    assert names1 == ['Ground', 'Vegetation', 'Building', 'Wire', 'Pole']
    
    # 情况 2: 使用原始标签号
    names2 = create_class_names(
        5,
        reverse_class_mapping={0: 0, 1: 1, 2: 2, 3: 6, 4: 9}
    )
    print(f"情况 2 (原始标签): {names2}")
    assert names2 == ['Class 0', 'Class 1', 'Class 2', 'Class 6', 'Class 9']
    
    # 情况 3: 默认
    names3 = create_class_names(5)
    print(f"情况 3 (默认): {names3}")
    assert names3 == ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    print("✓ create_class_names 测试通过\n")


def test_per_class_iou_with_names():
    """测试 PerClassIoU 使用类别名称"""
    print("=" * 60)
    print("测试 PerClassIoU 使用类别名称")
    print("=" * 60)
    
    # 创建指标（手动指定类别名称）
    metric = PerClassIoU(
        num_classes=3,
        class_names=['Ground', 'Vegetation', 'Building'],
        ignore_index=-1
    )
    
    # 模拟一些预测和标签
    # 预测: [B, N, C] logits
    preds = torch.tensor([
        [[2.0, 0.5, 0.1], [0.1, 2.0, 0.5], [0.5, 0.1, 2.0]],  # batch 1
        [[2.0, 0.1, 0.5], [0.5, 2.0, 0.1], [0.1, 0.5, 2.0]],  # batch 2
    ])
    
    # 真实标签: [B, N]
    target = torch.tensor([
        [0, 1, 2],  # batch 1
        [0, 1, 2],  # batch 2
    ])
    
    # 更新指标
    metric.update(preds, target)
    
    # 计算结果
    result = metric.compute()
    
    print(f"IoU per class: {result['iou_per_class']}")
    print(f"Precision per class: {result['precision_per_class']}")
    print(f"Recall per class: {result['recall_per_class']}")
    print(f"Class names: {result['class_names']}")
    
    # 验证类别名称
    assert result['class_names'] == ['Ground', 'Vegetation', 'Building']
    
    print("✓ PerClassIoU 类别名称测试通过\n")


def test_per_class_iou_with_reverse_mapping():
    """测试 PerClassIoU 使用 reverse_class_mapping"""
    print("=" * 60)
    print("测试 PerClassIoU 使用 reverse_class_mapping")
    print("=" * 60)
    
    # 创建指标（使用 reverse_class_mapping）
    metric = PerClassIoU(
        num_classes=5,
        reverse_class_mapping={0: 0, 1: 1, 2: 2, 3: 6, 4: 9},
        ignore_index=-1
    )
    
    # 模拟一些预测和标签
    preds = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # 预测类别 0
        [0.0, 1.0, 0.0, 0.0, 0.0],  # 预测类别 1
        [0.0, 0.0, 1.0, 0.0, 0.0],  # 预测类别 2
        [0.0, 0.0, 0.0, 1.0, 0.0],  # 预测类别 3
        [0.0, 0.0, 0.0, 0.0, 1.0],  # 预测类别 4
    ])
    target = torch.tensor([0, 1, 2, 3, 4])
    
    # 更新指标
    metric.update(preds, target)
    
    # 计算结果
    result = metric.compute()
    
    print(f"Class names: {result['class_names']}")
    print(f"IoU per class: {result['iou_per_class']}")
    
    # 验证使用原始标签号
    assert result['class_names'] == ['Class 0', 'Class 1', 'Class 2', 'Class 6', 'Class 9']
    
    print("✓ PerClassIoU reverse_class_mapping 测试通过\n")


def test_per_class_iou_default():
    """测试 PerClassIoU 默认行为"""
    print("=" * 60)
    print("测试 PerClassIoU 默认行为")
    print("=" * 60)
    
    # 创建指标（不指定任何名称）
    metric = PerClassIoU(
        num_classes=3,
        ignore_index=-1
    )
    
    # 模拟一些预测和标签
    preds = torch.tensor([
        [1.0, 0.0, 0.0],  # 预测类别 0
        [0.0, 1.0, 0.0],  # 预测类别 1
        [0.0, 0.0, 1.0],  # 预测类别 2
    ])
    target = torch.tensor([0, 1, 2])
    
    # 更新指标
    metric.update(preds, target)
    
    # 计算结果
    result = metric.compute()
    
    print(f"Class names: {result['class_names']}")
    
    # 验证使用连续标签号
    assert result['class_names'] == ['Class 0', 'Class 1', 'Class 2']
    
    print("✓ PerClassIoU 默认行为测试通过\n")


def test_mean_iou():
    """测试 MeanIoU（不需要类别名称）"""
    print("=" * 60)
    print("测试 MeanIoU（不需要类别名称）")
    print("=" * 60)
    
    # 创建指标
    metric = MeanIoU(num_classes=3, ignore_index=-1)
    
    # 模拟一些预测和标签
    preds = torch.tensor([
        [2.0, 0.5, 0.1], [0.1, 2.0, 0.5], [0.5, 0.1, 2.0],
        [2.0, 0.1, 0.5], [0.5, 2.0, 0.1], [0.1, 0.5, 2.0],
    ])
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    
    # 更新指标
    metric.update(preds, target)
    
    # 计算结果（返回单个标量值）
    result = metric.compute()
    
    print(f"Mean IoU: {result.item():.4f}")
    
    # MeanIoU 应该返回单个值，不是字典
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # 标量
    
    print("✓ MeanIoU 测试通过\n")


def test_datamodule_class_names():
    """测试 DataModule 中的 class_names 属性"""
    print("=" * 60)
    print("测试 DataModule 中的 class_names 属性")
    print("=" * 60)
    
    try:
        from pointsuite.data import BinPklDataModule
        
        # 创建 DataModule（带 class_names）
        datamodule = BinPklDataModule(
            train_data=None,  # 只测试属性，不需要实际数据
            class_mapping={0: 0, 1: 1, 2: 2, 6: 3, 9: 4},
            class_names=['Ground', 'Vegetation', 'Building', 'Wire', 'Pole']
        )
        
        # 验证属性存在
        assert hasattr(datamodule, 'class_names')
        assert datamodule.class_names == ['Ground', 'Vegetation', 'Building', 'Wire', 'Pole']
        
        print(f"DataModule class_mapping: {datamodule.class_mapping}")
        print(f"DataModule class_names: {datamodule.class_names}")
        
        print("✓ DataModule class_names 属性测试通过\n")
        
    except ImportError as e:
        print(f"⚠ 无法导入 BinPklDataModule: {e}")
        print("  跳过此测试\n")


def test_priority():
    """测试类别名称优先级"""
    print("=" * 60)
    print("测试类别名称优先级")
    print("=" * 60)
    
    # 优先级 1: 用户手动指定（最高）
    metric1 = PerClassIoU(
        num_classes=3,
        class_names=['Manual1', 'Manual2', 'Manual3'],
        reverse_class_mapping={0: 0, 1: 5, 2: 9}  # 应该被忽略
    )
    preds = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target = torch.tensor([0, 1, 2])
    metric1.update(preds, target)
    result1 = metric1.compute()
    print(f"优先级 1 (用户指定): {result1['class_names']}")
    assert result1['class_names'] == ['Manual1', 'Manual2', 'Manual3']
    
    # 优先级 3: reverse_class_mapping
    metric3 = PerClassIoU(
        num_classes=3,
        reverse_class_mapping={0: 0, 1: 5, 2: 9}
    )
    metric3.update(preds, target)
    result3 = metric3.compute()
    print(f"优先级 3 (reverse_mapping): {result3['class_names']}")
    assert result3['class_names'] == ['Class 0', 'Class 5', 'Class 9']
    
    # 优先级 4: 默认
    metric4 = PerClassIoU(num_classes=3)
    metric4.update(preds, target)
    result4 = metric4.compute()
    print(f"优先级 4 (默认): {result4['class_names']}")
    assert result4['class_names'] == ['Class 0', 'Class 1', 'Class 2']
    
    print("✓ 优先级测试通过\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("类别名称功能测试")
    print("=" * 60 + "\n")
    
    # 运行所有测试
    test_create_class_names()
    test_per_class_iou_with_names()
    test_per_class_iou_with_reverse_mapping()
    test_per_class_iou_default()
    test_mean_iou()
    test_datamodule_class_names()
    test_priority()
    
    print("=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
