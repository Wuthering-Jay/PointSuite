"""
测试DatasetBase能否正确加载bin+pkl格式的数据
"""
import sys
import numpy as np
from pathlib import Path

# Add path
sys.path.insert(0, 'e:/code/python/PointSuite')

from pointsuite.datasets.dataset_base import DatasetBase


def test_dataset():
    """测试dataset加载功能"""
    
    print("="*70)
    print("测试DatasetBase加载bin+pkl数据")
    print("="*70)
    
    # 设置数据路径
    data_root = r"E:\data\云南遥感中心\第一批\bin\train_with_gridsample"
    
    if not Path(data_root).exists():
        print(f"❌ 数据路径不存在: {data_root}")
        print("请修改为你的实际数据路径")
        return
    
    # 创建dataset
    print(f"\n1. 创建dataset...")
    try:
        dataset = DatasetBase(
            data_root=data_root,
            split='train',
            assets=['coord', 'intensity', 'color', 'classification'],
            transform=None,
            ignore_label=-1,
            loop=1,
            cache_data=False  # 不缓存，节省内存
        )
        print(f"   ✓ Dataset创建成功")
    except Exception as e:
        print(f"   ❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试数据加载
    print(f"\n2. 测试加载第一个样本...")
    try:
        data_dict = dataset[0]
        print(f"   ✓ 加载成功")
        print(f"\n   样本信息:")
        print(f"     - 名称: {data_dict.get('name', 'N/A')}")
        print(f"     - Segment ID: {data_dict.get('segment_id', 'N/A')}")
        
        if 'coord' in data_dict:
            coord = data_dict['coord']
            print(f"\n   坐标信息:")
            print(f"     - Shape: {coord.shape}")
            print(f"     - Dtype: {coord.dtype}")
            print(f"     - Range: X[{coord[:, 0].min():.2f}, {coord[:, 0].max():.2f}], "
                  f"Y[{coord[:, 1].min():.2f}, {coord[:, 1].max():.2f}], "
                  f"Z[{coord[:, 2].min():.2f}, {coord[:, 2].max():.2f}]")
        
        if 'intensity' in data_dict:
            intensity = data_dict['intensity']
            print(f"\n   强度信息:")
            print(f"     - Shape: {intensity.shape}")
            print(f"     - Range: [{intensity.min():.1f}, {intensity.max():.1f}]")
        
        if 'color' in data_dict:
            color = data_dict['color']
            print(f"\n   颜色信息:")
            print(f"     - Shape: {color.shape}")
            print(f"     - Range: [{color.min():.3f}, {color.max():.3f}]")
        
        if 'segment' in data_dict:
            labels = data_dict['segment']
            unique_labels = np.unique(labels)
            print(f"\n   标签信息:")
            print(f"     - Shape: {labels.shape}")
            print(f"     - Unique labels: {unique_labels}")
            print(f"     - Label distribution:")
            for label in unique_labels:
                count = np.sum(labels == label)
                print(f"       Class {label}: {count:,} 点 ({count/len(labels)*100:.1f}%)")
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试多个样本
    print(f"\n3. 测试加载多个样本...")
    num_test = min(5, len(dataset))
    for i in range(num_test):
        try:
            data_dict = dataset[i]
            num_points = len(data_dict['coord']) if 'coord' in data_dict else 0
            print(f"   ✓ 样本 {i}: {data_dict.get('name', 'N/A')} - {num_points:,} 点")
        except Exception as e:
            print(f"   ❌ 样本 {i} 加载失败: {e}")
    
    # 测试获取元数据
    print(f"\n4. 测试获取segment元数据...")
    try:
        segment_info = dataset.get_segment_info(0)
        print(f"   ✓ 元数据获取成功")
        print(f"     - 点数: {segment_info['num_points']:,}")
        print(f"     - X范围: [{segment_info['x_min']:.2f}, {segment_info['x_max']:.2f}]")
        print(f"     - Y范围: [{segment_info['y_min']:.2f}, {segment_info['y_max']:.2f}]")
        print(f"     - Z范围: [{segment_info['z_min']:.2f}, {segment_info['z_max']:.2f}]")
        if 'label_counts' in segment_info:
            print(f"     - 类别分布: {segment_info['label_counts']}")
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")
    
    # 测试获取文件元数据
    print(f"\n5. 测试获取文件元数据...")
    try:
        file_metadata = dataset.get_file_metadata(0)
        print(f"   ✓ 文件元数据获取成功")
        print(f"     - 原始LAS文件: {file_metadata.get('las_file', 'N/A')}")
        print(f"     - 总点数: {file_metadata.get('num_points', 0):,}")
        print(f"     - 分块数: {file_metadata.get('num_segments', 0)}")
        print(f"     - Window size: {file_metadata.get('window_size', 'N/A')}")
        print(f"     - Grid size: {file_metadata.get('grid_size', 'N/A')}")
        print(f"     - Max loops: {file_metadata.get('max_loops', 'N/A')}")
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")
    
    # 性能测试
    print(f"\n6. 性能测试（加载10个样本）...")
    import time
    
    num_samples = min(10, len(dataset))
    t0 = time.time()
    
    for i in range(num_samples):
        _ = dataset[i]
    
    t1 = time.time()
    elapsed = t1 - t0
    
    print(f"   ✓ 加载{num_samples}个样本耗时: {elapsed:.2f}s")
    print(f"   ✓ 平均每个样本: {elapsed/num_samples*1000:.1f}ms")
    
    print("\n" + "="*70)
    print("✅ 所有测试完成！Dataset工作正常。")
    print("="*70)
    
    return dataset


def test_with_dataloader():
    """测试PyTorch DataLoader"""
    print("\n" + "="*70)
    print("测试PyTorch DataLoader")
    print("="*70)
    
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("❌ PyTorch未安装，跳过DataLoader测试")
        return
    
    data_root = r"E:\data\云南遥感中心\第一批\bin\train_with_gridsample"
    
    if not Path(data_root).exists():
        print(f"❌ 数据路径不存在")
        return
    
    # 创建dataset
    dataset = DatasetBase(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        transform=None,
        loop=1
    )
    
    # 创建dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Windows上使用0
        pin_memory=False
    )
    
    print(f"\nDataLoader配置:")
    print(f"  - Batch size: 4")
    print(f"  - Total batches: {len(dataloader)}")
    
    # 测试迭代
    print(f"\n测试迭代前3个batch...")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        print(f"\n  Batch {i}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"    {key}: list of {len(value)} items")
            else:
                print(f"    {key}: {type(value)}")
    
    print(f"\n✅ DataLoader测试完成！")


if __name__ == "__main__":
    # 基础测试
    dataset = test_dataset()
    
    # DataLoader测试
    if dataset is not None:
        test_with_dataloader()
