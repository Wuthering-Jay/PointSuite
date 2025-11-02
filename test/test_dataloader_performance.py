"""
DataLoader 性能测试：测试加载所有数据的速度表现

测试场景：
1. 不同 num_workers 的影响
2. cache_data 的影响
3. batch_size 的影响
4. 限制点数的性能开销
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointsuite.data.datasets.dataset_bin import BinPklDataset
from pointsuite.data import transforms as T
from pointsuite.data.datasets.collate import (
    collate_fn,
    DynamicBatchSampler,
    create_limited_dataloader
)


def test_basic_loading_speed():
    """测试1: 基础加载速度（不同 batch_size）"""
    print("="*70)
    print("[测试1] 基础加载速度 - 不同 batch_size")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    batch_sizes = [1, 2, 4, 8, 16]
    
    for batch_size in batch_sizes:
        dataset = BinPklDataset(
            data_root=data_root,
            split='train',
            assets=['coord', 'intensity', 'classification'],
            cache_data=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        
        # 测试加载速度
        start_time = time.time()
        total_samples = 0
        total_points = 0
        
        for i, batch in enumerate(dataloader):
            total_samples += len(batch['offset'])
            total_points += batch['coord'].shape[0]
            
            # 只测试前100个batch
            if i >= 100:
                break
        
        elapsed = time.time() - start_time
        
        print(f"\nBatch size: {batch_size}")
        print(f"  - 加载 {i+1} batches")
        print(f"  - 总样本数: {total_samples}")
        print(f"  - 总点数: {total_points:,}")
        print(f"  - 耗时: {elapsed:.2f}s")
        print(f"  - 速度: {total_samples/elapsed:.1f} samples/s, {total_points/elapsed:,.0f} points/s")
    
    print()


def test_num_workers_impact():
    """测试2: num_workers 的影响"""
    print("="*70)
    print("[测试2] num_workers 的影响")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    num_workers_list = [0, 2, 4]
    batch_size = 8
    
    for num_workers in num_workers_list:
        dataset = BinPklDataset(
            data_root=data_root,
            split='train',
            assets=['coord', 'intensity', 'classification'],
            cache_data=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=(num_workers > 0),
        )
        
        # 测试加载速度
        start_time = time.time()
        total_samples = 0
        total_points = 0
        
        for i, batch in enumerate(dataloader):
            total_samples += len(batch['offset'])
            total_points += batch['coord'].shape[0]
            
            if i >= 100:
                break
        
        elapsed = time.time() - start_time
        
        print(f"\nnum_workers: {num_workers}")
        print(f"  - 加载 {i+1} batches")
        print(f"  - 总样本数: {total_samples}")
        print(f"  - 总点数: {total_points:,}")
        print(f"  - 耗时: {elapsed:.2f}s")
        print(f"  - 速度: {total_samples/elapsed:.1f} samples/s, {total_points/elapsed:,.0f} points/s")
    
    print()


def test_cache_impact():
    """测试3: cache_data 的影响"""
    print("="*70)
    print("[测试3] cache_data 的影响")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    batch_size = 8
    
    for cache_enabled in [False, True]:
        dataset = BinPklDataset(
            data_root=data_root,
            split='train',
            assets=['coord', 'intensity', 'classification'],
            cache_data=cache_enabled
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        
        # 第一次遍历
        start_time = time.time()
        total_points = 0
        for i, batch in enumerate(dataloader):
            total_points += batch['coord'].shape[0]
            if i >= 100:
                break
        first_pass = time.time() - start_time
        
        # 第二次遍历（测试缓存效果）
        start_time = time.time()
        total_points_2 = 0
        for i, batch in enumerate(dataloader):
            total_points_2 += batch['coord'].shape[0]
            if i >= 100:
                break
        second_pass = time.time() - start_time
        
        print(f"\nCache enabled: {cache_enabled}")
        print(f"  - 第一次遍历: {first_pass:.2f}s, {total_points/first_pass:,.0f} points/s")
        print(f"  - 第二次遍历: {second_pass:.2f}s, {total_points_2/second_pass:,.0f} points/s")
        print(f"  - 加速比: {first_pass/second_pass:.2f}x")
    
    print()


def test_transforms_overhead():
    """测试4: 数据增强的开销"""
    print("="*70)
    print("[测试4] 数据增强的开销")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 完整的训练增强
    train_transforms = [
        T.RandomRotate(angle=[-1, 1], axis='z', p=1.0),
        T.RandomScale(scale=[0.95, 1.05]),
        T.RandomFlip(p=0.5),
        T.RandomJitter(sigma=0.01, clip=0.05),
        T.CenterShift(apply_z=False),
        T.RandomIntensityScale(scale=(0.9, 1.1), p=0.95),
        T.RandomIntensityShift(shift=(-0.05, 0.05), p=0.95),
        T.StandardNormalizeIntensity(),
    ]
    
    batch_size = 8
    
    for use_transforms in [False, True]:
        dataset = BinPklDataset(
            data_root=data_root,
            split='train',
            assets=['coord', 'intensity', 'classification'],
            transform=train_transforms if use_transforms else None,
            cache_data=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        
        start_time = time.time()
        total_points = 0
        for i, batch in enumerate(dataloader):
            total_points += batch['coord'].shape[0]
            if i >= 100:
                break
        elapsed = time.time() - start_time
        
        print(f"\nTransforms enabled: {use_transforms}")
        print(f"  - 耗时: {elapsed:.2f}s")
        print(f"  - 速度: {total_points/elapsed:,.0f} points/s")
    
    print()


def test_limited_points_overhead():
    """测试5: 限制点数的性能开销"""
    print("="*70)
    print("[测试5] 限制点数的性能开销")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    max_points = 300000
    batch_size = 8
    
    # 测试1: 无限制
    print("\n[方法1] 无限制")
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    start_time = time.time()
    total_batches = 0
    total_points = 0
    for i, batch in enumerate(dataloader):
        total_batches += 1
        total_points += batch['coord'].shape[0]
        if i >= 100:
            break
    elapsed_baseline = time.time() - start_time
    
    print(f"  - 耗时: {elapsed_baseline:.2f}s")
    print(f"  - Batches: {total_batches}")
    print(f"  - 速度: {total_points/elapsed_baseline:,.0f} points/s")
    
    # 测试2: DynamicBatchSampler
    print("\n[方法2] DynamicBatchSampler (推荐)")
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    batch_sampler = DynamicBatchSampler(
        dataset,
        max_points=max_points,
        shuffle=False,
        drop_last=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    start_time = time.time()
    total_batches = 0
    total_points = 0
    for i, batch in enumerate(dataloader):
        total_batches += 1
        total_points += batch['coord'].shape[0]
        if i >= 100:
            break
    elapsed_sampler = time.time() - start_time
    
    print(f"  - 耗时: {elapsed_sampler:.2f}s")
    print(f"  - Batches: {total_batches}")
    print(f"  - 速度: {total_points/elapsed_sampler:,.0f} points/s")
    print(f"  - 开销: {(elapsed_sampler/elapsed_baseline - 1)*100:.1f}%")
    
    print()


def test_full_epoch():
    """测试6: 完整 epoch 加载速度"""
    print("="*70)
    print("[测试6] 完整 Epoch 加载速度")
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
    
    print(f"\n数据集信息:")
    print(f"  - 总样本数: {len(dataset):,}")
    
    # 测试固定 batch_size
    batch_size = 8
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    print(f"\n开始加载完整数据集...")
    start_time = time.time()
    
    total_batches = 0
    total_samples = 0
    total_points = 0
    points_per_batch = []
    
    for i, batch in enumerate(dataloader):
        total_batches += 1
        num_samples = len(batch['offset'])
        num_points = batch['coord'].shape[0]
        
        total_samples += num_samples
        total_points += num_points
        points_per_batch.append(num_points)
        
        # 每1000个batch打印一次进度
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / len(dataloader) * 100
            print(f"  进度: {progress:.1f}% ({i+1}/{len(dataloader)} batches), "
                  f"耗时: {elapsed:.1f}s, "
                  f"速度: {total_points/elapsed:,.0f} points/s")
    
    total_time = time.time() - start_time
    
    print(f"\n[OK] 完整 Epoch 统计:")
    print(f"  - 总 batches: {total_batches:,}")
    print(f"  - 总样本数: {total_samples:,}")
    print(f"  - 总点数: {total_points:,}")
    print(f"  - 总耗时: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  - 平均速度: {total_samples/total_time:.1f} samples/s, {total_points/total_time:,.0f} points/s")
    print(f"  - 每 batch 点数: min={min(points_per_batch):,}, max={max(points_per_batch):,}, avg={np.mean(points_per_batch):,.0f}")
    
    print()


def test_intensity_normalization():
    """测试7: 测试新增的 Intensity 标准化功能"""
    print("="*70)
    print("[测试7] Intensity 标准化增强")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 测试不同的标准化方法
    normalization_methods = [
        ("无标准化", None),
        ("StandardNormalize", [T.StandardNormalizeIntensity()]),
        ("MinMaxNormalize", [T.MinMaxNormalizeIntensity(target_range=(0, 1))]),
        ("StandardNormalize + Scale", [
            T.StandardNormalizeIntensity(),
            T.RandomIntensityScale(scale=(0.9, 1.1), p=1.0)
        ]),
    ]
    
    sample_idx = 0
    
    print(f"\n测试不同标准化方法（样本 {sample_idx}）:\n")
    
    for name, transforms in normalization_methods:
        dataset = BinPklDataset(
            data_root=data_root,
            split='train',
            assets=['coord', 'intensity', 'classification'],
            transform=transforms,
            cache_data=False
        )
        
        sample = dataset[sample_idx]
        intensity = sample['feature'][:, 3]  # intensity 是 feature 的第4列
        
        print(f"{name}:")
        print(f"  - Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
        print(f"  - Intensity mean: {intensity.mean():.4f}")
        print(f"  - Intensity std: {intensity.std():.4f}")
        print()
    
    print()


def main():
    """主测试函数"""
    print("="*70)
    print("DataLoader 性能测试")
    print("="*70)
    print()
    
    try:
        # 测试1: 基础加载速度
        test_basic_loading_speed()
        
        # 测试2: num_workers 影响
        test_num_workers_impact()
        
        # 测试3: cache 影响
        test_cache_impact()
        
        # 测试4: 数据增强开销
        test_transforms_overhead()
        
        # 测试5: 限制点数开销
        test_limited_points_overhead()
        
        # 测试6: 完整 epoch
        print("\n" + "="*70)
        print("注意: 完整 Epoch 测试会加载所有数据，可能需要较长时间")
        print("="*70)
        response = input("是否执行完整 Epoch 测试? (y/n): ")
        if response.lower() == 'y':
            test_full_epoch()
        
        # 测试7: Intensity 标准化
        test_intensity_normalization()
        
        print("="*70)
        print("[OK] 性能测试完成！")
        print("="*70)
        
    except Exception as e:
        print(f"\n[X] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
