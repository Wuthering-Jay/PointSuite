"""
测试 Weighted Sampler 和 Dynamic Batch + Weighted Sampler 结合使用
"""

import sys
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointsuite.data.datasets.dataset_bin import BinPklDataset
from pointsuite.data.datamodule_bin import BinPklDataModule
from pointsuite.data.transforms import (
    CenterShift, AutoNormalizeHNorm, Collect, ToTensor
)


def print_section(title):
    """打印分隔符"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_weighted_sampler():
    """测试权重采样（仅）"""
    print_section("测试：Weighted Sampler（仅权重采样）")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    print("\n1. 创建数据集并生成样本权重...")
    
    # 创建数据集
    temp_dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'class'],
        transform=None,
        ignore_label=0
    )
    
    num_samples = len(temp_dataset)
    print(f"  - 数据集样本数: {num_samples}")
    
    # 生成随机权重
    np.random.seed(42)
    weights = np.random.exponential(scale=1.0, size=num_samples)
    weights = weights / weights.sum()
    
    print(f"\n2. 权重统计:")
    print(f"  - 最小权重: {weights.min():.6f}")
    print(f"  - 最大权重: {weights.max():.6f}")
    print(f"  - 平均权重: {weights.mean():.6f} (理论: {1/num_samples:.6f})")
    print(f"  - 权重标准差: {weights.std():.6f}")
    
    # 找出高权重样本
    high_weight_threshold = np.percentile(weights, 90)
    high_weight_indices = np.where(weights >= high_weight_threshold)[0]
    print(f"\n3. 高权重样本分析（前10%）:")
    print(f"  - 样本数: {len(high_weight_indices)}")
    print(f"  - 权重阈值: {high_weight_threshold:.6f}")
    print(f"  - 样本索引（前20个）: {high_weight_indices[:20].tolist()}")
    print(f"  - 权重值（前20个）: {[f'{weights[i]:.6f}' for i in high_weight_indices[:20]]}")
    
    # 创建带权重采样的 DataModule
    print("\n4. 创建带权重采样的 DataModule（固定 batch_size=2）...")
    
    train_transforms = [
        CenterShift(apply_z=True),
        AutoNormalizeHNorm(clip_range=None),
        Collect(
            keys=['coord', 'class'],
            offset_key={'offset': 'coord'},
            feat_keys={'feat': ['coord', 'h_norm']}
        ),
        ToTensor()
    ]
    
    datamodule = BinPklDataModule(
        data_root=data_root,
        batch_size=2,
        num_workers=0,
        assets=['coord', 'h_norm', 'class'],
        train_transforms=train_transforms,
        val_transforms=train_transforms,
        ignore_label=0,
        loop=5,
        use_dynamic_batch=False,
        train_sampler_weights=weights.tolist()
    )
    
    datamodule.setup()
    print("  ✓ DataModule 创建成功（仅权重采样，loop=5）")
    
    # 获取 dataloader
    print("\n5. 测试批次生成...")
    train_loader = datamodule.train_dataloader()
    
    print(f"  - DataLoader 批次数: {len(train_loader)}")
    print(f"  - 使用 WeightedRandomSampler")
    print(f"  - Loop: {datamodule.loop}")
    
    # 检查前10个批次
    n_batches = min(10, len(train_loader))
    batch_sizes = []
    total_points_list = []
    
    print(f"\n6. 前 {n_batches} 个批次的详情:")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        
        batch_size = len(batch['offset'])
        n_points = len(batch['coord'])
        
        batch_sizes.append(batch_size)
        total_points_list.append(n_points)
        
        print(f"  Batch {batch_idx}: batch_size={batch_size}, total_points={n_points:,}")
    
    print(f"\n✓ Weighted Sampler 测试通过")
    print(f"  - 批次大小固定: {batch_sizes[0]}")
    print(f"  - 平均总点数: {np.mean(total_points_list):,.0f}")
    print(f"  - 点数范围: [{min(total_points_list):,}, {max(total_points_list):,}]")
    
    return weights


def test_dynamic_batch_with_weighted_sampler(weights):
    """测试 Dynamic Batch + Weighted Sampler"""
    print_section("测试：Dynamic Batch + Weighted Sampler 结合")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    print("\n1. 使用相同的权重配置...")
    print(f"  - 权重数量: {len(weights)}")
    print(f"  - 最小权重: {weights.min():.6f}")
    print(f"  - 最大权重: {weights.max():.6f}")
    
    # 创建 Dynamic Batch + Weighted Sampler 的 DataModule
    print("\n2. 创建 Dynamic Batch + Weighted Sampler 的 DataModule...")
    
    train_transforms = [
        CenterShift(apply_z=True),
        AutoNormalizeHNorm(clip_range=None),
        Collect(
            keys=['coord', 'class'],
            offset_key={'offset': 'coord'},
            feat_keys={'feat': ['coord', 'h_norm']}
        ),
        ToTensor()
    ]
    
    datamodule = BinPklDataModule(
        data_root=data_root,
        batch_size=4,  # 会被忽略
        num_workers=0,
        assets=['coord', 'h_norm', 'class'],
        train_transforms=train_transforms,
        val_transforms=train_transforms,
        ignore_label=0,
        loop=5,
        use_dynamic_batch=True,
        max_points=150000,
        train_sampler_weights=weights.tolist()
    )
    
    datamodule.setup()
    print("  ✓ DataModule 创建成功（Dynamic Batch + Weighted Sampler，loop=5）")
    
    # 获取 dataloader
    print("\n3. 测试批次生成...")
    train_loader = datamodule.train_dataloader()
    
    print(f"  - DataLoader 批次数: {len(train_loader)}")
    print(f"  - 使用 DynamicBatchSampler + WeightedRandomSampler")
    print(f"  - Loop: {datamodule.loop}")
    print(f"  - Max points: {datamodule.max_points:,}")
    
    # 检查前10个批次
    n_batches = min(10, len(train_loader))
    batch_sizes = []
    total_points_list = []
    
    print(f"\n4. 前 {n_batches} 个批次的详情:")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        
        batch_size = len(batch['offset'])
        n_points = len(batch['coord'])
        
        batch_sizes.append(batch_size)
        total_points_list.append(n_points)
        
        print(f"  Batch {batch_idx}:")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Total points: {n_points:,}")
        print(f"    - Avg points/sample: {n_points/batch_size:,.0f}")
        
        assert n_points <= datamodule.max_points, \
            f"批次点数 {n_points} 超过限制 {datamodule.max_points}"
    
    print(f"\n5. 批次统计:")
    print(f"  - 平均 batch size: {np.mean(batch_sizes):.2f}")
    print(f"  - Batch size 范围: [{min(batch_sizes)}, {max(batch_sizes)}]")
    print(f"  - 平均总点数: {np.mean(total_points_list):,.0f}")
    print(f"  - 总点数范围: [{min(total_points_list):,}, {max(total_points_list):,}]")
    
    print(f"\n✓ Dynamic Batch + Weighted Sampler 测试通过")


def compare_results():
    """对比两种方式的差异"""
    print_section("对比分析")
    
    print("\n【方式1：仅 Weighted Sampler（固定 batch_size=2）】")
    print("  特点:")
    print("    - 每个批次固定包含 2 个样本")
    print("    - 根据权重随机采样样本")
    print("    - 总点数波动大（取决于采样到的样本大小）")
    print("    - 可能出现：小样本浪费内存，大样本超出限制")
    print("  ")
    print("  适用场景:")
    print("    ✓ 样本大小比较均匀")
    print("    ✓ 不关心总点数上限")
    print("    ✓ 需要固定的批次逻辑")
    
    print("\n【方式2：Dynamic Batch + Weighted Sampler】")
    print("  特点:")
    print("    - 批次大小动态调整（1-N个样本）")
    print("    - 根据权重随机采样样本")
    print("    - 总点数控制在 max_points 以内")
    print("    - 更好的内存利用率")
    print("  ")
    print("  适用场景:")
    print("    ✓ 样本大小差异大")
    print("    ✓ 需要严格控制GPU内存")
    print("    ✓ 既要处理类别不平衡又要控制内存")
    
    print("\n【Loop=5 的作用】")
    print("  - 两种方式都使用 loop=5")
    print("  - 在一个 epoch 中，数据集会被遍历 5 次")
    print("  - 结合权重采样：高权重样本更可能被多次采样")
    print("  - 总批次数 ≈ (样本数 × loop) / 平均批次大小")
    
    print("\n【关键区别】")
    print("  1. 批次大小:")
    print("     - 固定方式: batch_size 恒定")
    print("     - Dynamic: batch_size 动态（1到多个）")
    print("  ")
    print("  2. 内存使用:")
    print("     - 固定方式: 不可控，可能超出GPU内存")
    print("     - Dynamic: 严格控制在 max_points 以内")
    print("  ")
    print("  3. 计算效率:")
    print("     - 固定方式: 可能因批次过小而浪费GPU")
    print("     - Dynamic: 尽可能填满批次，提高GPU利用率")


def main():
    print("="*80)
    print("  Weighted Sampler 对比测试")
    print("="*80)
    
    try:
        # 测试1：仅权重采样
        weights = test_weighted_sampler()
        
        # 测试2：Dynamic Batch + 权重采样
        test_dynamic_batch_with_weighted_sampler(weights)
        
        # 对比分析
        compare_results()
        
        print_section("测试完成")
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        print_section("测试失败")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
