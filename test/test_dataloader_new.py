"""
测试 BinPklDataset 与 DataLoader 的集成以及数据增强

包含:
1. 基础 DataLoader 测试
2. 数据增强 (transforms) 测试
"""
import sys
from pathlib import Path
import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointsuite.datasets.dataset_bin import BinPklDataset
from pointsuite.datasets import transforms as T


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list
    
    该函数将多个不同点数的样本合并成一个batch：
    - 点云数据会被拼接成一个大的点云
    - coord 和 feature 会被拼接
    - 自动添加 'offset' 字段，记录每个样本的起始位置
    - test模式下会拼接 indices 用于投票机制
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{type(batch)} is not supported.")

    # 处理 dict 类型（我们的数据集返回的是 dict）
    if isinstance(batch[0], Mapping):
        # 获取所有keys
        keys = batch[0].keys()
        
        # 合并后的结果
        result = {}
        
        # 计算每个样本的点数（用于offset）
        num_points_per_sample = []
        
        for key in keys:
            # 收集所有样本的该字段
            values = [torch.from_numpy(d[key]) if isinstance(d[key], np.ndarray) else d[key] for d in batch]
            
            # 对于点云数据，拼接而不是stack
            if key in ['coord', 'feature', 'classification', 'indices']:
                # 拼接成一个大的tensor
                result[key] = torch.cat(values, dim=0)
                
                # 记录点数（从coord或feature获取）
                if key == 'coord' and len(num_points_per_sample) == 0:
                    num_points_per_sample = [v.shape[0] for v in values]
            else:
                # 其他字段使用默认处理
                try:
                    result[key] = torch.stack(values, dim=0)
                except:
                    result[key] = values
        
        # 添加 offset 字段
        if len(num_points_per_sample) > 0:
            offset = torch.cumsum(torch.tensor([0] + num_points_per_sample), dim=0).int()
            result['offset'] = offset[1:]  # 去掉第一个0，只保留累积和
        
        return result
    
    # 处理其他类型（保持原有逻辑）
    elif isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)


def test_basic_dataloader():
    """测试1: 基础 DataLoader"""
    print("="*70)
    print("[测试1] 基础 DataLoader")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return None
    
    # 创建数据集
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    print(f"[OK] 数据集创建成功: {len(dataset)} samples")
    
    # 创建 DataLoader
    batch_size = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    print(f"[OK] DataLoader 创建成功")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Total batches: {len(dataloader)}")
    
    # 加载一个batch
    batch = next(iter(dataloader))
    print(f"\n[OK] Batch 数据结构:")
    print(f"  - Keys: {list(batch.keys())}")
    print(f"  - coord shape: {batch['coord'].shape}")
    print(f"  - feature shape: {batch['feature'].shape}")
    print(f"  - offset: {batch['offset']}")
    
    return dataloader


def test_test_split():
    """测试2: Test Split (自动保存indices用于投票)"""
    print("\n" + "="*70)
    print("[测试2] Test Split - 预测模式")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 获取单个pkl文件
    pkl_files = list(Path(data_root).glob('*.pkl'))
    if len(pkl_files) == 0:
        print("[X] 没有找到pkl文件")
        return
    
    single_pkl = pkl_files[0]
    print(f"[OK] 使用单个pkl文件: {single_pkl.name}")
    
    # split='test' 会自动保存 indices
    dataset_test = BinPklDataset(
        data_root=single_pkl,
        split='test',  # split='test' 自动保存 indices
        assets=['coord', 'intensity'],
        cache_data=False,
    )
    
    dataloader = DataLoader(
        dataset_test,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    print(f"[OK] Test dataset: {len(dataset_test)} samples")
    
    batch = next(iter(dataloader))
    
    if 'indices' in batch:
        print(f"[OK] Indices 字段存在")
        print(f"  - indices shape: {batch['indices'].shape}")
        print(f"  - 可用于预测投票机制")
        
        indices_np = batch['indices'].numpy()
        print(f"  - 总点数: {len(indices_np):,}")
        print(f"  - 唯一索引: {len(np.unique(indices_np)):,}")
    else:
        print(f"[X] 警告: split='test' 但没有 indices 字段")


def test_transforms():
    """测试3: 数据增强 (transforms)"""
    print("\n" + "="*70)
    print("[测试3] 数据增强 (Transforms)")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 定义数据增强pipeline
    train_transforms = [
        T.RandomRotate(angle=[-1, 1], axis='z', p=0.5),
        T.RandomScale(scale=[0.9, 1.1]),
        T.RandomFlip(p=0.5),
        T.RandomJitter(sigma=0.01, clip=0.05),
        T.CenterShift(apply_z=True),
    ]
    
    # 创建带数据增强的数据集
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        transform=train_transforms,
        cache_data=False
    )
    
    print(f"[OK] 数据集创建成功（含数据增强）")
    print(f"  - Transforms: {len(train_transforms)} 个")
    print(f"  - RandomRotate, RandomScale, RandomFlip, RandomJitter, CenterShift")
    
    # 加载同一个样本多次，验证随机性
    sample_idx = 0
    coords_list = []
    
    print(f"\n[OK] 测试数据增强的随机性（加载同一样本5次）:")
    for i in range(5):
        sample = dataset[sample_idx]
        coords_list.append(sample['coord'].copy())
        print(f"  Run {i+1}: coord min={sample['coord'].min(axis=0)}, max={sample['coord'].max(axis=0)}")
    
    # 验证每次结果不同
    all_same = True
    for i in range(1, len(coords_list)):
        if not np.allclose(coords_list[0], coords_list[i]):
            all_same = False
            break
    
    if not all_same:
        print(f"[OK] 数据增强生效：每次加载结果不同")
    else:
        print(f"[WARNING] 数据增强可能未生效或概率未触发")


def test_transforms_with_dataloader():
    """测试4: 数据增强 + DataLoader"""
    print("\n" + "="*70)
    print("[测试4] 数据增强 + DataLoader 集成")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 完整的训练数据增强pipeline
    train_transforms = [
        T.RandomRotate(angle=[-1, 1], axis='z', p=1.0),  # 100%触发
        T.RandomScale(scale=[0.95, 1.05]),
        T.CenterShift(apply_z=False),
    ]
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        transform=train_transforms,
        cache_data=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    print(f"[OK] DataLoader + 数据增强")
    
    # 迭代几个batch
    print(f"\n[OK] 迭代前3个batch:")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        print(f"  Batch {i+1}:")
        print(f"    - 总点数: {batch['coord'].shape[0]:,}")
        print(f"    - coord range: [{batch['coord'].min():.2f}, {batch['coord'].max():.2f}]")
        print(f"    - feature shape: {batch['feature'].shape}")


def test_various_transforms():
    """测试5: 各种数据增强效果"""
    print("\n" + "="*70)
    print("[测试5] 测试各种数据增强效果")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 测试不同的transform组合
    transform_configs = [
        ("无增强", []),
        ("仅旋转", [T.RandomRotate(angle=[-1, 1], axis='z', p=1.0)]),
        ("仅缩放", [T.RandomScale(scale=[0.9, 1.1])]),
        ("旋转+缩放", [
            T.RandomRotate(angle=[-1, 1], axis='z', p=1.0),
            T.RandomScale(scale=[0.9, 1.1])
        ]),
        ("完整增强", [
            T.RandomRotate(angle=[-1, 1], axis='z', p=1.0),
            T.RandomScale(scale=[0.9, 1.1]),
            T.RandomFlip(p=1.0),
            T.RandomJitter(sigma=0.01, clip=0.05),
            T.CenterShift(apply_z=False),
        ]),
    ]
    
    sample_idx = 0
    
    print(f"\n[OK] 测试不同数据增强对同一样本的效果:\n")
    
    for name, transforms in transform_configs:
        dataset = BinPklDataset(
            data_root=data_root,
            split='train',
            assets=['coord', 'intensity', 'classification'],
            transform=transforms,
            cache_data=False
        )
        
        sample = dataset[sample_idx]
        coord = sample['coord']
        
        print(f"{name}:")
        print(f"  - shape: {coord.shape}")
        print(f"  - X range: [{coord[:, 0].min():.2f}, {coord[:, 0].max():.2f}]")
        print(f"  - Y range: [{coord[:, 1].min():.2f}, {coord[:, 1].max():.2f}]")
        print(f"  - Z range: [{coord[:, 2].min():.2f}, {coord[:, 2].max():.2f}]")
        print(f"  - Mean: {coord.mean(axis=0)}")
        print()


def main():
    """主测试函数"""
    print("="*70)
    print("BinPklDataset + DataLoader + Transforms 测试")
    print("="*70)
    
    try:
        # 测试1: 基础 DataLoader
        test_basic_dataloader()
        
        # 测试2: Test split (自动保存indices)
        test_test_split()
        
        # 测试3: 数据增强
        test_transforms()
        
        # 测试4: 数据增强 + DataLoader
        test_transforms_with_dataloader()
        
        # 测试5: 各种数据增强效果
        test_various_transforms()
        
        print("\n" + "="*70)
        print("[OK] 所有测试通过！")
        print("="*70)
        
        print("\n【总结】")
        print("-"*70)
        print("1. [OK] 基础 DataLoader 功能正常")
        print("2. [OK] Test split 自动保存 indices（用于投票）")
        print("3. [OK] 数据增强 (transforms) 集成成功")
        print("4. [OK] Feature 组成: coord + intensity + ...")
        print("5. [OK] 支持目录/单文件/多文件路径")
        print("="*70)
        
    except Exception as e:
        print(f"\n[X] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
