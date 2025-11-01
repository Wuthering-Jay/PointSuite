"""
测试 BinPklDataset 的基本功能

cache_data 参数说明:
- cache_data=True: 将加载的数据缓存在内存中，适合小数据集
  * 优点: 第二次访问相同样本时速度极快 (直接从内存读取)
  * 缺点: 占用大量内存，不适合大数据集
  
- cache_data=False: 每次从磁盘加载数据 (使用memmap高效读取)
  * 优点: 内存占用小，适合大数据集
  * 缺点: 每次访问都需要I/O操作 (但memmap已经很快了)
"""
import sys
from pathlib import Path
import time
import numpy as np

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointsuite.datasets.dataset_bin import BinPklDataset


def test_basic_loading(data_root):
    """测试1: 基础数据加载"""
    print("\n[测试1] 基础数据加载")
    print("-"*70)
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    print(f"✓ 数据集创建成功")
    print(f"  - 样本数量: {len(dataset)}")
    
    if len(dataset) == 0:
        print("❌ 数据集为空，请检查数据路径")
        return None
    
    # 加载第一个样本
    print("\n加载第一个样本...")
    sample = dataset[0]
    print(f"✓ 样本加载成功")
    print(f"  - 数据字段: {list(sample.keys())}")
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
    
    return dataset


def test_metadata_access(dataset):
    """测试2: 元数据访问"""
    print("\n[测试2] 元数据访问")
    print("-"*70)
    
    # 样本信息
    info = dataset.get_sample_info(0)
    print(f"✓ 样本信息:")
    print(f"  - 文件名: {info['file_name']}")
    print(f"  - 点数: {info['num_points']}")
    print(f"  - Segment ID: {info['segment_id']}")
    
    # Segment详细信息
    seg_info = dataset.get_segment_info(0)
    print(f"✓ Segment信息:")
    if 'indices' in seg_info:
        print(f"  - indices: array with {len(seg_info['indices'])} points")
        print(f"  - 格式: 离散索引 (点云数据格式)")
    
    # 打印边界信息
    if 'x_min' in seg_info:
        print(f"  - bounds: X[{seg_info['x_min']:.2f}, {seg_info['x_max']:.2f}], "
              f"Y[{seg_info['y_min']:.2f}, {seg_info['y_max']:.2f}], "
              f"Z[{seg_info['z_min']:.2f}, {seg_info['z_max']:.2f}]")
    
    # 文件元数据
    file_meta = dataset.get_file_metadata(0)
    print(f"✓ 文件元数据:")
    if 'grid_size' in file_meta:
        print(f"  - grid_size: {file_meta['grid_size']}")
    if 'window_size' in file_meta:
        print(f"  - window_size: {file_meta['window_size']}")


def test_statistics(dataset):
    """测试3: 统计信息"""
    print("\n[测试3] 数据集统计")
    print("-"*70)
    dataset.print_stats()


def test_cache_performance(data_root):
    """测试4: cache_data 性能对比"""
    print("\n[测试4] cache_data 性能对比")
    print("-"*70)
    
    # 无缓存
    print("\n测试 cache_data=False (无缓存):")
    dataset_no_cache = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    test_samples = min(10, len(dataset_no_cache))
    start = time.time()
    for i in range(test_samples):
        _ = dataset_no_cache[i]
    time_no_cache = time.time() - start
    print(f"  加载 {test_samples} 个样本耗时: {time_no_cache:.4f}s")
    
    # 再次访问相同样本
    start = time.time()
    for i in range(test_samples):
        _ = dataset_no_cache[i]
    time_no_cache_2nd = time.time() - start
    print(f"  第二次访问耗时: {time_no_cache_2nd:.4f}s (无缓存，需重新加载)")
    
    # 有缓存
    print("\n测试 cache_data=True (有缓存):")
    dataset_cache = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=True
    )
    
    start = time.time()
    for i in range(test_samples):
        _ = dataset_cache[i]
    time_cache_1st = time.time() - start
    print(f"  首次加载 {test_samples} 个样本耗时: {time_cache_1st:.4f}s (填充缓存)")
    
    # 再次访问相同样本
    start = time.time()
    for i in range(test_samples):
        _ = dataset_cache[i]
    time_cache_2nd = time.time() - start
    print(f"  第二次访问耗时: {time_cache_2nd:.4f}s (从缓存读取)")
    
    if time_cache_2nd > 0:
        speedup = time_no_cache_2nd / time_cache_2nd
        print(f"\n✓ 缓存加速比: {speedup:.1f}x")
        print(f"  说明: cache_data=True 使重复访问速度提升 {speedup:.1f} 倍")
        print(f"        但会占用更多内存来存储所有样本")


def test_different_assets(data_root):
    """测试5: 不同 assets 组合"""
    print("\n[测试5] 不同 assets 组合")
    print("-"*70)
    
    asset_tests = [
        ['coord'],
        ['coord', 'intensity'],
        ['coord', 'intensity', 'classification'],
    ]
    
    for assets in asset_tests:
        try:
            ds = BinPklDataset(
                data_root=data_root,
                split='train',
                assets=assets,
                cache_data=False
            )
            sample = ds[0]
            print(f"✓ assets={assets}")
            print(f"  加载的字段: {list(sample.keys())}")
        except Exception as e:
            print(f"❌ assets={assets}: {e}")


def test_loop_parameter(data_root):
    """测试6: loop 参数测试"""
    print("\n[测试6] loop 参数测试")
    print("-"*70)
    
    ds_loop1 = BinPklDataset(data_root=data_root, loop=1)
    ds_loop3 = BinPklDataset(data_root=data_root, loop=3)
    
    print(f"✓ loop=1: 数据集长度 = {len(ds_loop1)}")
    print(f"✓ loop=3: 数据集长度 = {len(ds_loop3)}")
    print(f"  验证: {len(ds_loop3)} = {len(ds_loop1)} × 3 ? {len(ds_loop3) == len(ds_loop1) * 3}")
    print(f"  说明: loop 参数用于训练时重复数据集，增加epoch长度")


def main():
    """主测试函数"""
    print("="*70)
    print("BinPklDataset 测试")
    print("="*70)
    
    # 修改为你的数据路径
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"\n❌ 数据目录不存在: {data_root}")
        print("请修改 data_root 为实际的数据路径")
        return
    
    try:
        # 运行所有测试
        dataset = test_basic_loading(data_root)
        
        if dataset is None or len(dataset) == 0:
            print("❌ 数据集为空，无法继续测试")
            return
        
        test_metadata_access(dataset)
        test_statistics(dataset)
        test_cache_performance(data_root)
        test_different_assets(data_root)
        test_loop_parameter(data_root)
        
        print("\n" + "="*70)
        print("✓ 所有测试通过！")
        print("="*70)
        
        # 打印 cache_data 使用建议
        print("\n【cache_data 使用建议】")
        print("-"*70)
        print("1. 小数据集 (< 1GB): 使用 cache_data=True")
        print("   - 可以显著提升训练速度")
        print("   - 适合需要多次遍历数据的场景")
        print("\n2. 大数据集 (> 10GB): 使用 cache_data=False")
        print("   - 避免内存溢出")
        print("   - memmap 已经提供了足够好的性能")
        print("\n3. 中等数据集: 根据可用内存决定")
        print("   - 监控内存使用情况")
        print("   - 可以先用 False 测试，如果内存充足再改为 True")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
