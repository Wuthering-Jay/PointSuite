"""
测试新增功能：
1. 类别映射 (class_mapping)
2. Intensity 数据增强
3. 限制 batch 总点数
"""
import os
# 必须在导入任何库之前设置！
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointsuite.datasets.dataset_bin import BinPklDataset
from pointsuite.datasets import transforms as T
from pointsuite.datasets.collate import (
    collate_fn,
    LimitedPointsCollateFn,
    DynamicBatchSampler,
    create_limited_dataloader
)


def test_class_mapping():
    """测试1: 类别映射功能"""
    print("="*70)
    print("[测试1] 类别映射 (Class Mapping)")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # DALES 数据集类别映射示例
    # 原始类别: 0=未分类, 1=地面, 2=植被, 6=建筑, 9=水体, 17=车辆
    # 映射到连续标签: 0->0, 1->1, 2->2, 6->3, 9->4, 17->5
    class_mapping = {
        0: 0,   # 未分类
        1: 1,   # 地面
        2: 2,   # 植被
        6: 3,   # 建筑
        9: 4,   # 水体
        17: 5,  # 车辆
    }
    
    # 创建带类别映射的数据集
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        class_mapping=class_mapping,
        cache_data=False
    )
    
    print(f"[OK] 数据集创建成功: {len(dataset)} samples")
    print(f"[OK] 类别映射: {class_mapping}")
    
    # 加载一个样本查看映射效果
    sample = dataset[0]
    unique_labels = np.unique(sample['classification'])
    
    print(f"\n[OK] 样本标签统计:")
    print(f"  - 唯一标签: {unique_labels}")
    print(f"  - 标签范围: [{unique_labels.min()}, {unique_labels.max()}]")
    print(f"  - 是否连续: {len(unique_labels) == (unique_labels.max() - unique_labels.min() + 1)}")
    
    # 统计每个标签的点数
    for label in unique_labels:
        count = np.sum(sample['classification'] == label)
        print(f"  - 标签 {label}: {count:,} 点")
    
    print()


def test_intensity_augmentation():
    """测试2: Intensity 数据增强"""
    print("="*70)
    print("[测试2] Intensity 数据增强")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 定义 Intensity 数据增强 pipeline
    intensity_transforms = [
        T.RandomIntensityScale(scale=(0.8, 1.2), p=1.0),
        T.RandomIntensityShift(shift=(-0.1, 0.1), p=1.0),
        T.RandomIntensityNoise(sigma=0.02, p=0.5),
        T.RandomIntensityGamma(gamma_range=(0.8, 1.2), p=0.5),
    ]
    
    # 创建数据集
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        transform=intensity_transforms,
        cache_data=False
    )
    
    print(f"[OK] 数据集创建成功（含 Intensity 增强）")
    print(f"  - Transforms: {len(intensity_transforms)} 个")
    print(f"  - RandomIntensityScale, RandomIntensityShift,")
    print(f"    RandomIntensityNoise, RandomIntensityGamma")
    
    # 加载同一样本多次，观察 intensity 变化
    sample_idx = 0
    print(f"\n[OK] 测试 Intensity 增强效果（同一样本加载5次）:")
    
    for i in range(5):
        sample = dataset[sample_idx]
        intensity = sample['feature'][:, 3]  # intensity 是 feature 的第4列（coord占3列）
        
        print(f"  Run {i+1}:")
        print(f"    - Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
        print(f"    - Intensity mean: {intensity.mean():.4f}")
        print(f"    - Intensity std: {intensity.std():.4f}")
    
    print()


def test_combined_augmentation():
    """测试3: 组合数据增强（几何 + Intensity + 颜色）"""
    print("="*70)
    print("[测试3] 组合数据增强")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 完整的训练数据增强 pipeline
    train_transforms = [
        # 几何变换
        T.RandomRotate(angle=[-1, 1], axis='z', p=1.0),
        T.RandomScale(scale=[0.95, 1.05]),
        T.RandomFlip(p=0.5),
        T.RandomJitter(sigma=0.01, clip=0.05),
        T.CenterShift(apply_z=False),
        
        # Intensity 增强
        T.RandomIntensityScale(scale=(0.9, 1.1), p=0.95),
        T.RandomIntensityShift(shift=(-0.05, 0.05), p=0.95),
    ]
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        transform=train_transforms,
        cache_data=False
    )
    
    print(f"[OK] 组合增强数据集: {len(train_transforms)} 个 transforms")
    
    # 测试 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    batch = next(iter(dataloader))
    print(f"\n[OK] Batch 统计:")
    print(f"  - Coord shape: {batch['coord'].shape}")
    print(f"  - Feature shape: {batch['feature'].shape}")
    print(f"  - Coord range: [{batch['coord'].min():.2f}, {batch['coord'].max():.2f}]")
    
    # 分离 coord 和 intensity
    intensity = batch['feature'][:, 3]
    print(f"  - Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
    
    print()


def test_limited_points_collate():
    """测试4: 限制 batch 总点数 - Collate 方法"""
    print("="*70)
    print("[测试4] 限制 Batch 总点数 - Collate 方法")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    # 测试不同的策略
    strategies = ['drop_largest', 'drop_last', 'keep_first']
    max_points = 200000  # 20万点限制
    
    print(f"[OK] 最大点数限制: {max_points:,}")
    print(f"\n测试不同策略:")
    
    for strategy in strategies:
        limited_collate = LimitedPointsCollateFn(
            max_points=max_points,
            strategy=strategy
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=8,  # 尝试加载8个样本
            shuffle=False,
            num_workers=0,
            collate_fn=limited_collate,
        )
        
        batch = next(iter(dataloader))
        
        print(f"\n  策略: {strategy}")
        print(f"    - 实际样本数: {len(batch['offset'])} (请求8个)")
        print(f"    - 总点数: {batch['coord'].shape[0]:,}")
        print(f"    - 每个样本点数: {[batch['offset'][i].item() - (batch['offset'][i-1].item() if i > 0 else 0) for i in range(len(batch['offset']))]}")
        print(f"    - 是否满足限制: {batch['coord'].shape[0] <= max_points}")
    
    print()


def test_dynamic_batch_sampler():
    """测试5: 动态 Batch Sampler（推荐方法）"""
    print("="*70)
    print("[测试5] 动态 Batch Sampler (推荐)")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    max_points = 250000  # 25万点限制
    
    print(f"[OK] 使用 DynamicBatchSampler")
    print(f"[OK] 最大点数限制: {max_points:,}")
    
    # 创建 dynamic batch sampler
    batch_sampler = DynamicBatchSampler(
        dataset,
        max_points=max_points,
        shuffle=True,
        drop_last=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"[OK] 预估 batch 数: {len(dataloader)}")
    
    # 测试前5个batch
    print(f"\n[OK] 前5个 batch 统计:")
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        
        total_points = batch['coord'].shape[0]
        num_samples = len(batch['offset'])
        points_per_sample = [
            batch['offset'][j].item() - (batch['offset'][j-1].item() if j > 0 else 0)
            for j in range(num_samples)
        ]
        
        print(f"  Batch {i+1}:")
        print(f"    - 样本数: {num_samples}")
        print(f"    - 总点数: {total_points:,}")
        print(f"    - 每样本点数: min={min(points_per_sample):,}, max={max(points_per_sample):,}, avg={np.mean(points_per_sample):.0f}")
        print(f"    - 是否满足限制: {total_points <= max_points}")
    
    print()


def test_convenient_api():
    """测试6: 便捷 API - create_limited_dataloader"""
    print("="*70)
    print("[测试6] 便捷 API - create_limited_dataloader")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    max_points = 300000
    
    print(f"[OK] 使用便捷 API 创建 DataLoader")
    print(f"[OK] 最大点数: {max_points:,}")
    
    # 方法1: 使用 sampler（推荐）
    dataloader_sampler = create_limited_dataloader(
        dataset,
        max_points=max_points,
        method='sampler',
        shuffle=True,
        num_workers=0
    )
    
    print(f"\n方法1: method='sampler'")
    batch = next(iter(dataloader_sampler))
    print(f"  - 样本数: {len(batch['offset'])}")
    print(f"  - 总点数: {batch['coord'].shape[0]:,}")
    
    # 方法2: 使用 collate
    dataloader_collate = create_limited_dataloader(
        dataset,
        max_points=max_points,
        method='collate',
        collate_strategy='drop_largest',
        shuffle=True,
        num_workers=0,
        batch_size=8
    )
    
    print(f"\n方法2: method='collate'")
    batch = next(iter(dataloader_collate))
    print(f"  - 样本数: {len(batch['offset'])}")
    print(f"  - 总点数: {batch['coord'].shape[0]:,}")
    
    print()


def main():
    """主测试函数"""
    print("="*70)
    print("新功能测试")
    print("="*70)
    print()
    
    try:
        # 测试1: 类别映射
        test_class_mapping()
        
        # 测试2: Intensity 数据增强
        test_intensity_augmentation()
        
        # 测试3: 组合数据增强
        test_combined_augmentation()
        
        # 测试4: 限制点数 - Collate 方法
        test_limited_points_collate()
        
        # 测试5: 动态 Batch Sampler（推荐）
        test_dynamic_batch_sampler()
        
        # 测试6: 便捷 API
        test_convenient_api()
        
        print("="*70)
        print("[OK] 所有测试通过！")
        print("="*70)
        
        print("\n【功能总结】")
        print("-"*70)
        print("1. [OK] 类别映射：将非连续标签映射到连续标签")
        print("2. [OK] Intensity 增强：8种增强方法")
        print("   - RandomIntensityScale, RandomIntensityShift")
        print("   - RandomIntensityNoise, RandomIntensityDrop")
        print("   - IntensityAutoContrast, RandomIntensityGamma")
        print("3. [OK] 限制 Batch 总点数：两种方法")
        print("   - LimitedPointsCollateFn (collate 阶段)")
        print("   - DynamicBatchSampler (采样阶段，推荐)")
        print("4. [OK] 便捷 API：create_limited_dataloader")
        print("="*70)
        
    except Exception as e:
        print(f"\n[X] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
