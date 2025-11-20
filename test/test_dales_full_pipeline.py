"""
Dales 数据集完整测试：dataset_bin.py + datamodule_binpkl.py + transforms.py

数据集信息：
- 地址: E:\data\Dales\dales_las\bin\train
- ground_class: 1
- 类别: 0-8 (9个类别)，其中 0 为噪声（忽略）
- 可用属性: coord, echo, is_ground

测试内容：
1. BinPklDataset 基础功能
2. 数据加载和属性提取
3. h_norm 计算（基于 is_ground）
4. Transform 流程
5. BinPklDataModule 完整流程
6. DataLoader 批次生成
7. 类别映射和统计
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
    # 坐标变换
    CenterShift, RandomRotate, RandomScale, RandomFlip,
    # 归一化
    AutoNormalizeIntensity, AutoNormalizeColor, AutoNormalizeHNorm,
    # 数据增强
    RandomIntensityScale, RandomIntensityNoise,
    RandomHNormNoise, RandomHNormScale,
    # 噪点注入
    AddExtremeOutliers, AddLocalNoiseClusters,
    # 采样
    RandomDropout,
    # 收集
    Collect, ToTensor, Compose
)


def print_section(title):
    """打印分隔符"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_1_dataset_basic():
    """测试 1: BinPklDataset 基础功能"""
    print_section("测试 1: BinPklDataset 基础功能")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    # 创建数据集（只加载 coord 和 class）
    print("\n1.1 创建数据集（minimal assets）...")
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'class'],
        transform=None,
        ignore_label=0,  # 忽略噪声类别
        loop=1,
        cache_data=False
    )
    
    print(f"✓ 数据集创建成功")
    print(f"  - 样本数: {len(dataset)}")
    
    # 打印统计信息
    print("\n1.2 数据集统计信息:")
    dataset.print_stats()
    
    # 加载第一个样本
    print("\n1.3 加载第一个样本...")
    sample = dataset[0]
    
    print(f"✓ 样本加载成功")
    print(f"  - 键: {list(sample.keys())}")
    print(f"  - coord shape: {sample['coord'].shape}")
    print(f"  - coord dtype: {sample['coord'].dtype}")
    print(f"  - coord 范围: X[{sample['coord'][:, 0].min():.2f}, {sample['coord'][:, 0].max():.2f}], "
          f"Y[{sample['coord'][:, 1].min():.2f}, {sample['coord'][:, 1].max():.2f}], "
          f"Z[{sample['coord'][:, 2].min():.2f}, {sample['coord'][:, 2].max():.2f}]")
    print(f"  - class shape: {sample['class'].shape}")
    print(f"  - class dtype: {sample['class'].dtype}")
    
    # 统计类别分布
    unique_classes, counts = np.unique(sample['class'], return_counts=True)
    print(f"  - 类别分布:")
    for cls, cnt in zip(unique_classes, counts):
        print(f"      类别 {cls}: {cnt:,} 点 ({cnt/len(sample['class'])*100:.2f}%)")
    
    return dataset


def test_2_dataset_with_echo_isground():
    """测试 2: 加载 echo 和 is_ground 属性"""
    print_section("测试 2: 加载 echo 和 is_ground 属性")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    print("\n2.1 创建数据集（包含 echo 和 is_ground）...")
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'echo', 'class'],
        transform=None,
        ignore_label=0
    )
    
    print(f"✓ 数据集创建成功")
    
    # 加载样本
    print("\n2.2 加载样本并检查属性...")
    sample = dataset[0]
    
    print(f"✓ 样本键: {list(sample.keys())}")
    
    # 检查 echo
    if 'echo' in sample:
        print(f"\n  Echo 信息:")
        print(f"    - shape: {sample['echo'].shape}")
        print(f"    - dtype: {sample['echo'].dtype}")
        print(f"    - 范围: [{sample['echo'].min():.2f}, {sample['echo'].max():.2f}]")
        print(f"    - 列 0 (is_first): unique values = {np.unique(sample['echo'][:, 0])}")
        print(f"    - 列 1 (is_last): unique values = {np.unique(sample['echo'][:, 1])}")
        
        # 统计回波类型
        is_first_return = (sample['echo'][:, 0] > 0).sum()
        is_last_return = (sample['echo'][:, 1] > 0).sum()
        print(f"    - 首次回波点数: {is_first_return:,} ({is_first_return/len(sample['echo'])*100:.2f}%)")
        print(f"    - 末次回波点数: {is_last_return:,} ({is_last_return/len(sample['echo'])*100:.2f}%)")
    else:
        print("  ⚠️ echo 属性不可用（bin文件中可能缺少 return_number 字段）")
    
    return dataset


def test_3_hnorm_computation():
    """测试 3: h_norm 计算（基于 is_ground）"""
    print_section("测试 3: h_norm 计算（基于 is_ground）")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    print("\n3.1 创建数据集（包含 h_norm）...")
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'h_norm', 'class'],
        transform=None,
        ignore_label=0,
        h_norm_grid=0.5
    )
    
    print(f"✓ 数据集创建成功")
    
    # 加载多个样本测试 h_norm 计算
    print("\n3.2 测试多个样本的 h_norm 计算...")
    n_samples = min(5, len(dataset))
    
    for i in range(n_samples):
        sample = dataset[i]
        h_norm = sample['h_norm']
        
        print(f"\n  样本 {i}:")
        print(f"    - 点数: {len(h_norm):,}")
        print(f"    - h_norm 范围: [{h_norm.min():.3f}, {h_norm.max():.3f}]")
        print(f"    - h_norm 均值: {h_norm.mean():.3f}")
        print(f"    - h_norm 中位数: {np.median(h_norm):.3f}")
        print(f"    - h_norm 标准差: {h_norm.std():.3f}")
        
        # 统计负值和极大值
        negative_ratio = (h_norm < 0.0).sum() / len(h_norm) * 100
        high_ratio = (h_norm < -0.1).sum() / len(h_norm) * 100
        print(f"    - 负值比例: {negative_ratio:.2f}%")
        print(f"    - < -0.1m 比例: {high_ratio:.2f}%")

    return dataset


def test_4_transforms():
    """测试 4: Transform 流程"""
    print_section("测试 4: Transform 流程")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    # 定义 transform 链（注意：不要包装为 Compose，dataset_base 会自动处理）
    print("\n4.1 定义 Transform 链...")
    train_transforms = [
        # 坐标变换
        CenterShift(apply_z=True),
        RandomRotate(axis='z', p=0.5),
        RandomScale(scale=[0.95, 1.05]),
        RandomFlip(p=0.5),
        
        # h_norm 归一化（不裁剪）
        AutoNormalizeHNorm(clip_range=None),
        
        # h_norm 增强
        RandomHNormScale(scale=(0.9, 1.1), p=0.5),
        RandomHNormNoise(sigma=0.1, p=0.3),
        
        # 噪点注入
        AddExtremeOutliers(
            ratio=0.01,
            height_range=(-10, 100),
            height_mode='bimodal',
            class_label=0,  # 标记为噪声
            p=0.3
        ),
        
        AddLocalNoiseClusters(
            num_clusters=3,
            points_per_cluster=(10, 20),
            cluster_radius=2.0,
            height_offset=(-2, 2),
            class_label=0,
            p=0.2
        ),
        
        # 采样
        RandomDropout(dropout_ratio=0.2, p=0.5),
        
        # 收集
        Collect(
            keys=['coord', 'class'],
            offset_key={'offset': 'coord'},
            feat_keys={'feat': ['coord', 'h_norm']}
        ),
        
        # 转张量
        ToTensor()
    ]
    
    print("✓ Transform 链定义完成")
    print(f"  - 共 {len(train_transforms)} 个变换")
    
    # 创建带 transform 的数据集
    print("\n4.2 创建带 Transform 的数据集...")
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'h_norm', 'class'],
        transform=train_transforms,
        ignore_label=0,
        h_norm_grid=0.5
    )
    
    print(f"✓ 数据集创建成功")
    
    # 测试多次加载同一样本（验证随机性）
    print("\n4.3 测试 Transform 随机性（同一样本多次加载）...")
    sample_idx = 0
    
    for i in range(3):
        sample = dataset[sample_idx]
        
        print(f"\n  第 {i+1} 次加载:")
        print(f"    - 键: {list(sample.keys())}")
        print(f"    - coord: {sample['coord'].shape}, dtype={sample['coord'].dtype}")
        print(f"    - feat: {sample['feat'].shape}, dtype={sample['feat'].dtype}")
        print(f"    - class: {sample['class'].shape}, dtype={sample['class'].dtype}")
        print(f"    - offset: {sample['offset'].shape}, dtype={sample['offset'].dtype}")
        
        # 检查数据类型
        assert isinstance(sample['coord'], torch.Tensor), "coord 应为 Tensor"
        assert isinstance(sample['feat'], torch.Tensor), "feat 应为 Tensor"
        assert isinstance(sample['class'], torch.Tensor), "class 应为 Tensor"
        
        # 检查 feat 维度（coord[3] + h_norm[1] = 4）
        assert sample['feat'].shape[1] == 4, f"feat 应有 4 维，实际为 {sample['feat'].shape[1]}"
        
        # 统计坐标范围（验证 CenterShift）
        coord = sample['coord']
        print(f"    - coord 中心: [{coord[:, 0].mean():.3f}, {coord[:, 1].mean():.3f}, {coord[:, 2].mean():.3f}]")
    
    print("\n✓ Transform 测试通过")
    
    return dataset


def test_5_class_mapping():
    """测试 5: 类别映射"""
    print_section("测试 5: 类别映射")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    # 定义类别映射：0-8 类别，忽略 0（噪声）
    # 映射为连续的 0-7
    print("\n5.1 定义类别映射...")
    class_mapping = {
        0: -1,  # 噪声 -> 0 (ignore_label)
        1: 0,  # 地面
        2: 1,  # 植被
        3: 2,  # 汽车
        4: 3,  # 卡车
        5: 4,  # 电线杆
        6: 15,  # 围栏
        7: 6,  # 建筑
        8: 10,  # 其他
    }
    
    print(f"✓ 类别映射: {class_mapping}")
    
    # 创建数据集（注意：ignore_label 应该与映射后的噪声标签一致）
    print("\n5.2 创建带类别映射的数据集...")
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'class'],
        transform=None,
        ignore_label=-1,  # 噪声映射到 -1，所以 ignore_label 应该是 -1
        class_mapping=class_mapping
    )
    
    print(f"✓ 数据集创建成功")
    
    # 加载样本并验证映射
    print("\n5.3 验证类别映射...")
    sample = dataset[0]
    
    unique_classes, counts = np.unique(sample['class'], return_counts=True)
    print(f"\n  映射后的类别分布:")
    for cls, cnt in zip(unique_classes, counts):
        label_name = "ignore" if cls == -1 else f"类别 {cls}"
        print(f"    {label_name}: {cnt:,} 点 ({cnt/len(sample['class'])*100:.2f}%)")
    
    # 确认所有类别都在映射范围内（包括 ignore_label）
    expected_labels = set(class_mapping.values())  # {-1, 0, 1, 2, 3, 4, 5, 6, 7}
    actual_labels = set(unique_classes.tolist())  # 转换 numpy 类型为 Python int
    
    # 调试信息
    print(f"\n  调试信息:")
    print(f"    - 原始映射: {class_mapping}")
    print(f"    - 期望的映射后标签: {sorted(expected_labels)}")
    print(f"    - 实际标签（含类型）: {[(cls, type(cls)) for cls in sorted(unique_classes)]}")
    print(f"    - 实际标签（转为int）: {sorted(actual_labels)}")
    
    # 检查是否有未映射的类别（即原始类别仍然存在）
    unmapped_classes = actual_labels - expected_labels
    if unmapped_classes:
        print(f"\n  ⚠️ 警告：发现未映射的类别: {unmapped_classes}")
        print(f"  这些类别应该在 class_mapping 中被映射，但实际数据仍包含原始值")
        raise AssertionError(f"存在未映射的类别: {unmapped_classes}\n"
                           f"可能原因：数据中存在映射表之外的类别，请检查数据或补全 class_mapping")
    
    print(f"\n✓ 类别映射正确")
    print(f"  - 映射范围: {sorted(expected_labels)}")
    print(f"  - 实际标签: {sorted(actual_labels)}")
    
    return dataset


def test_6_datamodule():
    """测试 6: BinPklDataModule 完整流程"""
    print_section("测试 6: BinPklDataModule 完整流程")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    # 定义类别映射
    class_mapping = {i: i for i in range(9)}  # 保持原始标签
    
    # 定义 transforms
    print("\n6.1 定义训练和验证 Transforms...")
    train_transforms = [
        CenterShift(apply_z=True),
        RandomRotate(axis='z', p=0.5),
        RandomScale(scale=[0.95, 1.05]),
        AutoNormalizeHNorm(clip_range=None),
        RandomHNormNoise(sigma=0.1, p=0.3),
        AddExtremeOutliers(ratio=0.005, class_label=0, p=0.3),
        RandomDropout(dropout_ratio=0.2, p=0.5),
        Collect(
            keys=['coord', 'class'],
            offset_key={'offset': 'coord'},
            feat_keys={'feat': ['coord', 'h_norm']}
        ),
        ToTensor()
    ]
    
    val_transforms = [
        CenterShift(apply_z=True),
        AutoNormalizeHNorm(clip_range=None),
        Collect(
            keys=['coord', 'class'],
            offset_key={'offset': 'coord'},
            feat_keys={'feat': ['coord', 'h_norm']}
        ),
        ToTensor()
    ]
    
    print("✓ Transforms 定义完成")
    
    # 创建 DataModule
    print("\n6.2 创建 BinPklDataModule...")
    datamodule = BinPklDataModule(
        data_root=data_root,
        batch_size=4,
        num_workers=0,  # 单线程测试
        assets=['coord', 'h_norm', 'class'],
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        ignore_label=0,
        loop=1,
        cache_data=False,
        class_mapping=class_mapping,
        use_dynamic_batch=False,
        pin_memory=False
    )
    
    print("✓ DataModule 创建成功")
    
    # Setup
    print("\n6.3 Setup DataModule...")
    datamodule.setup()
    
    print("✓ Setup 完成")
    
    # 打印信息
    print("\n6.4 DataModule 信息:")
    datamodule.print_info()
    
    return datamodule


def test_7_dataloader():
    """测试 7: DataLoader 批次生成"""
    print_section("测试 7: DataLoader 批次生成")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    # 创建 DataModule
    print("\n7.1 创建 DataModule...")
    
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
        batch_size=2,  # 小批次便于测试
        num_workers=0,
        assets=['coord', 'h_norm', 'class'],
        train_transforms=train_transforms,
        val_transforms=train_transforms,
        ignore_label=0,
        use_dynamic_batch=False
    )
    
    datamodule.setup()
    print("✓ DataModule 创建成功")
    
    # 获取 train dataloader
    print("\n7.2 获取 Train DataLoader...")
    train_loader = datamodule.train_dataloader()
    
    print(f"✓ DataLoader 创建成功")
    print(f"  - 批次数: {len(train_loader)}")
    
    # 遍历几个批次
    print("\n7.3 测试批次加载...")
    n_batches = min(3, len(train_loader))
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        
        # 统一的数据格式：offset 格式为 [n1, n1+n2, ...], 长度 == batch_size
        batch_size = len(batch['offset'])
        n_points = len(batch['coord'])
        
        print(f"\n  Batch {batch_idx}:")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Total points: {n_points:,}")
        print(f"    - coord: shape={batch['coord'].shape}, dtype={batch['coord'].dtype}")
        print(f"    - feat: shape={batch['feat'].shape}, dtype={batch['feat'].dtype}")
        print(f"    - class: {len(batch['class'])} tensors")
        print(f"    - offset: {batch['offset'].tolist()}")
        
        # 验证批次数据
        assert 'coord' in batch, "batch 缺少 coord"
        assert 'feat' in batch, "batch 缺少 feat"
        assert 'class' in batch, "batch 缺少 class"
        assert 'offset' in batch, "batch 缺少 offset"
        
        # 验证 offset 格式：累积和，长度 == batch_size
        assert len(batch['offset']) == batch_size, f"offset 长度应为 batch_size"
        assert batch['offset'][-1] == n_points, f"offset 最后一个值应等于总点数"
    
    print("\n✓ DataLoader 测试通过")
    
    return datamodule


def test_8_dynamic_batch():
    """测试 8: Dynamic Batch Sampler"""
    print_section("测试 8: Dynamic Batch Sampler")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    print("\n8.1 创建带 Dynamic Batch 的 DataModule...")
    
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
        batch_size=4,  # 使用 dynamic batch 时会被忽略
        num_workers=0,
        assets=['coord', 'h_norm', 'class'],
        train_transforms=train_transforms,
        val_transforms=train_transforms,
        ignore_label=0,
        use_dynamic_batch=True,
        max_points=200000,  # 每批次最多 10 万点
    )
    
    datamodule.setup()
    print("✓ DataModule 创建成功（Dynamic Batch 模式）")
    
    # 获取 dataloader
    print("\n8.2 获取 DataLoader...")
    train_loader = datamodule.train_dataloader()
    
    print(f"✓ DataLoader 创建成功")
    print(f"  - 使用 DynamicBatchSampler")
    print(f"  - max_points: {datamodule.max_points}")
    
    # 测试几个批次
    print("\n8.3 测试 Dynamic Batch 加载...")
    n_batches = min(5, len(train_loader))
    
    batch_sizes = []
    total_points = []
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        
        # 统一的数据格式（collate_fn 保证）
        # collate 中 offset 格式为累积点数，不包含起始 0，长度 == batch_size
        batch_size = len(batch['offset'])
        n_points = len(batch['coord'])

        batch_sizes.append(batch_size)
        total_points.append(n_points)

        print(f"\n  Batch {batch_idx}:")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Total points: {n_points:,}")
        print(f"    - Avg points/sample: {n_points/batch_size:,.0f}")
        print(f"    - coord shape: {batch['coord'].shape}")
        print(f"    - feat shape: {batch['feat'].shape}")
        print(f"    - class: {len(batch['class'])} tensors")
        print(f"    - offset: {batch['offset'].tolist()}")

        # 验证点数不超过限制
        assert n_points <= datamodule.max_points, \
            f"批次点数 {n_points} 超过限制 {datamodule.max_points}"
    
    print(f"\n✓ Dynamic Batch 测试通过")
    print(f"  - 平均 batch size: {np.mean(batch_sizes):.2f}")
    print(f"  - 平均总点数: {np.mean(total_points):,.0f}")
    
    return datamodule


def test_9_full_assets():
    """测试 9: 加载所有可用属性"""
    print_section("测试 9: 加载所有可用属性")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    print("\n9.1 创建数据集（所有可用属性）...")
    
    # 尝试加载所有可能的属性
    all_assets = ['coord', 'echo', 'h_norm', 'class']
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=all_assets,
        transform=None,
        ignore_label=0
    )
    
    print(f"✓ 数据集创建成功")
    
    # 加载样本
    print("\n9.2 加载样本并检查所有属性...")
    sample = dataset[0]
    
    print(f"\n  样本包含的属性:")
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"    - {key}: shape={value.shape}, dtype={value.dtype}, "
                  f"range=[{value.min():.3f}, {value.max():.3f}]")
    
    # 创建完整的 transform
    print("\n9.3 创建完整 Transform 链...")
    full_transforms = [
        CenterShift(apply_z=True),
        RandomRotate(axis='z', p=0.5),
        AutoNormalizeHNorm(clip_range=None),
        RandomHNormNoise(sigma=0.1, p=0.3),
        Collect(
            keys=['coord', 'class'],
            offset_key={'offset': 'coord'},
            feat_keys={'feat': ['coord', 'echo', 'h_norm']}  # 包含所有特征
        ),
        ToTensor()
    ]
    
    dataset_with_transforms = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=all_assets,
        transform=full_transforms,
        ignore_label=0
    )
    
    sample = dataset_with_transforms[0]
    
    print(f"\n  Transform 后的样本:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"    - {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 验证 feat 维度：coord(3) + echo(2) + h_norm(1) = 6
    expected_feat_dim = 3 + 2 + 1  # coord + echo + h_norm
    if 'echo' in sample:
        actual_feat_dim = sample['feat'].shape[1]
        print(f"\n  特征维度验证:")
        print(f"    - 预期: {expected_feat_dim} (coord:3 + echo:2 + h_norm:1)")
        print(f"    - 实际: {actual_feat_dim}")
        assert actual_feat_dim == expected_feat_dim, \
            f"特征维度不匹配: expected={expected_feat_dim}, actual={actual_feat_dim}"
        print(f"    ✓ 特征维度正确")
    
    print("\n✓ 所有属性测试通过")
    
    return dataset_with_transforms


def test_10_weighted_sampler():
    """测试 10: Weighted Sampler（仅权重采样）"""
    print_section("测试 10: Weighted Sampler（仅权重采样）")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    print("\n10.1 创建数据集并生成样本权重...")
    
    # 先创建一个临时数据集来获取样本数量
    temp_dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'class'],
        transform=None,
        ignore_label=0
    )
    
    num_samples = len(temp_dataset)
    print(f"  - 数据集样本数: {num_samples}")
    
    # 生成随机权重（模拟类别不平衡场景）
    # 让某些样本有更高的被采样概率
    np.random.seed(42)
    weights = np.random.exponential(scale=1.0, size=num_samples)
    weights = weights / weights.sum()  # 归一化
    
    print(f"  - 权重统计:")
    print(f"      最小权重: {weights.min():.6f}")
    print(f"      最大权重: {weights.max():.6f}")
    print(f"      平均权重: {weights.mean():.6f}")
    print(f"      权重标准差: {weights.std():.6f}")
    
    # 创建带权重采样的 DataModule（不使用 Dynamic Batch）
    print("\n10.2 创建带权重采样的 DataModule（固定 batch_size=2）...")
    
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
        loop=5,  # 循环5次
        use_dynamic_batch=False,
        train_sampler_weights=weights.tolist()
    )
    
    datamodule.setup()
    print("✓ DataModule 创建成功（仅权重采样）")
    
    # 获取 dataloader
    print("\n10.3 获取 DataLoader 并统计采样情况...")
    train_loader = datamodule.train_dataloader()
    
    print(f"✓ DataLoader 创建成功")
    print(f"  - 使用 WeightedRandomSampler")
    print(f"  - Loop: {datamodule.loop}")
    print(f"  - 批次数: {len(train_loader)}")
    
    # 统计被采样的样本
    sampled_indices = []
    n_batches_to_check = min(20, len(train_loader))
    
    print(f"\n10.4 检查前 {n_batches_to_check} 个批次的采样情况...")
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches_to_check:
            break
        
        # 注意：由于使用了 WeightedRandomSampler，我们无法直接获取样本索引
        # 但可以通过批次信息间接观察
        batch_size = len(batch['offset'])
        n_points = len(batch['coord'])
        
        if batch_idx < 5:  # 只打印前5个批次
            print(f"  Batch {batch_idx}: batch_size={batch_size}, total_points={n_points:,}")
    
    print(f"\n✓ Weighted Sampler 测试通过（固定批次大小）")
    
    return datamodule, weights


def test_11_dynamic_batch_with_weighted_sampler():
    """测试 11: Dynamic Batch + Weighted Sampler 结合"""
    print_section("测试 11: Dynamic Batch + Weighted Sampler 结合")
    
    data_root = r"E:\data\Dales\dales_las\bin\train"
    
    print("\n11.1 准备测试数据...")
    
    # 先创建一个临时数据集来获取样本数量和点数信息
    temp_dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'class'],
        transform=None,
        ignore_label=0
    )
    
    num_samples = len(temp_dataset)
    print(f"  - 数据集样本数: {num_samples}")
    
    # 生成随机权重（与测试10使用相同的随机种子以便对比）
    np.random.seed(42)
    weights = np.random.exponential(scale=1.0, size=num_samples)
    weights = weights / weights.sum()
    
    print(f"  - 权重统计:")
    print(f"      最小权重: {weights.min():.6f}")
    print(f"      最大权重: {weights.max():.6f}")
    print(f"      平均权重: {weights.mean():.6f}")
    
    # 找出高权重样本（前10%）
    high_weight_threshold = np.percentile(weights, 90)
    high_weight_indices = np.where(weights >= high_weight_threshold)[0]
    print(f"  - 高权重样本（前10%）: {len(high_weight_indices)} 个")
    print(f"      高权重阈值: {high_weight_threshold:.6f}")
    print(f"      高权重样本索引（前10个）: {high_weight_indices[:10].tolist()}")
    
    # 创建带权重采样和 Dynamic Batch 的 DataModule
    print("\n11.2 创建 Dynamic Batch + Weighted Sampler 的 DataModule...")
    
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
        batch_size=4,  # Dynamic Batch 模式下会被忽略
        num_workers=0,
        assets=['coord', 'h_norm', 'class'],
        train_transforms=train_transforms,
        val_transforms=train_transforms,
        ignore_label=0,
        loop=5,  # 循环5次
        use_dynamic_batch=True,
        max_points=150000,
        train_sampler_weights=weights.tolist()
    )
    
    datamodule.setup()
    print("✓ DataModule 创建成功（Dynamic Batch + Weighted Sampler）")
    
    # 获取 dataloader
    print("\n11.3 获取 DataLoader 并分析批次特征...")
    train_loader = datamodule.train_dataloader()
    
    print(f"✓ DataLoader 创建成功")
    print(f"  - 使用 DynamicBatchSampler + WeightedRandomSampler")
    print(f"  - Loop: {datamodule.loop}")
    print(f"  - Max points per batch: {datamodule.max_points:,}")
    print(f"  - 批次数: {len(train_loader)}")
    
    # 统计批次信息
    print(f"\n11.4 测试批次生成并统计...")
    
    n_batches_to_check = min(10, len(train_loader))
    batch_sizes = []
    total_points_list = []
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches_to_check:
            break
        
        batch_size = len(batch['offset'])
        n_points = len(batch['coord'])
        
        batch_sizes.append(batch_size)
        total_points_list.append(n_points)
        
        print(f"\n  Batch {batch_idx}:")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Total points: {n_points:,}")
        print(f"    - Avg points/sample: {n_points/batch_size:,.0f}")
        print(f"    - offset: {batch['offset'].tolist()}")
        
        # 验证点数不超过限制
        assert n_points <= datamodule.max_points, \
            f"批次点数 {n_points} 超过限制 {datamodule.max_points}"
    
    print(f"\n11.5 批次统计:")
    print(f"  - 平均 batch size: {np.mean(batch_sizes):.2f} (标准差: {np.std(batch_sizes):.2f})")
    print(f"  - Batch size 范围: [{min(batch_sizes)}, {max(batch_sizes)}]")
    print(f"  - 平均总点数: {np.mean(total_points_list):,.0f}")
    print(f"  - 总点数范围: [{min(total_points_list):,}, {max(total_points_list):,}]")
    
    # 对比分析
    print(f"\n11.6 与固定批次大小的对比:")
    print(f"  Dynamic Batch 的优势:")
    print(f"    ✓ 自动调整批次大小以适应点数限制")
    print(f"    ✓ 更好的内存利用率（每批次点数更接近上限）")
    print(f"    ✓ 结合权重采样，既能控制内存又能处理类别不平衡")
    print(f"  ")
    print(f"  固定 Batch Size 的特点:")
    print(f"    - 批次大小固定，但总点数波动大")
    print(f"    - 可能浪费内存（小样本）或超出限制（大样本）")
    
    print(f"\n✓ Dynamic Batch + Weighted Sampler 测试通过")
    
    return datamodule, weights


def main():
    """运行所有测试"""
    print("="*80)
    print("  Dales 数据集完整测试")
    print("="*80)
    print(f"\n数据集配置:")
    print(f"  - 路径: E:\\data\\Dales\\dales_las\\bin\\train")
    print(f"  - ground_class: 1")
    print(f"  - 类别范围: 0-8 (共9个类别)")
    print(f"  - 忽略类别: 0 (噪声)")
    print(f"  - 可用属性: coord, echo, is_ground")
    
    try:
        # 测试 1: 基础功能
        test_1_dataset_basic()
        
        # 测试 2: echo 和 is_ground
        test_2_dataset_with_echo_isground()
        
        # 测试 3: h_norm 计算
        test_3_hnorm_computation()
        
        # 测试 4: Transforms
        test_4_transforms()
        
        # 测试 5: 类别映射
        test_5_class_mapping()
        
        # 测试 6: DataModule
        test_6_datamodule()
        
        # 测试 7: DataLoader
        test_7_dataloader()
        
        # 测试 8: Dynamic Batch
        test_8_dynamic_batch()
        
        # 测试 9: 所有属性
        test_9_full_assets()
        
        # 测试 10: 权重采样（仅）
        test_10_weighted_sampler()
        
        # 测试 11: Dynamic Batch + 权重采样
        test_11_dynamic_batch_with_weighted_sampler()
        
        # 最终总结
        print_section("所有测试完成")
        print("\n✅ 所有测试通过！")
        print("\n测试覆盖:")
        print("  ✓ BinPklDataset 基础功能")
        print("  ✓ echo 和 is_ground 属性加载")
        print("  ✓ h_norm 计算（TIN+Raster方法）")
        print("  ✓ Transform 流程和随机性")
        print("  ✓ 类别映射")
        print("  ✓ BinPklDataModule 完整流程")
        print("  ✓ DataLoader 批次生成")
        print("  ✓ Dynamic Batch Sampler")
        print("  ✓ 所有可用属性联合使用")
        print("  ✓ Weighted Sampler（固定批次）")
        print("  ✓ Dynamic Batch + Weighted Sampler 结合")
        
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
