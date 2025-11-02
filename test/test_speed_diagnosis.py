"""
诊断数据集访问速度瓶颈

问题：一万多个样本需要数分钟访问，这明显不正常
可能原因：
1. 每次都从磁盘加载 pkl 元数据
2. 没有使用 cache_data
3. 频繁的 memmap 打开/关闭
4. 不必要的数据转换
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import time
import numpy as np
from collections import Counter

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointsuite.data.datasets.dataset_bin import BinPklDataset


def test_init_speed():
    """测试1: 初始化速度"""
    print("="*70)
    print("[测试1] 数据集初始化速度")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    start = time.time()
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    init_time = time.time() - start
    
    print(f"\n初始化时间: {init_time:.2f} 秒")
    print(f"样本数: {len(dataset):,}")
    print(f"平均每个样本: {init_time/len(dataset)*1000:.2f} ms")
    print()


def test_single_sample_access():
    """测试2: 单个样本访问速度"""
    print("="*70)
    print("[测试2] 单个样本访问速度")
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
    
    # 测试访问前 100 个样本
    print(f"\n测试访问前 100 个样本...")
    
    times = []
    for i in range(100):
        start = time.time()
        sample = dataset[i]
        elapsed = time.time() - start
        times.append(elapsed)
    
    print(f"\n前 100 个样本访问时间统计:")
    print(f"  - 总时间: {sum(times):.2f} 秒")
    print(f"  - 平均: {np.mean(times)*1000:.2f} ms")
    print(f"  - 中位数: {np.median(times)*1000:.2f} ms")
    print(f"  - 最小: {np.min(times)*1000:.2f} ms")
    print(f"  - 最大: {np.max(times)*1000:.2f} ms")
    print(f"  - 标准差: {np.std(times)*1000:.2f} ms")
    
    # 预估全部样本时间
    total_samples = len(dataset)
    estimated_time = np.mean(times) * total_samples
    print(f"\n预估遍历所有 {total_samples:,} 个样本需要: {estimated_time:.2f} 秒 ({estimated_time/60:.2f} 分钟)")
    print()


def test_repeated_access():
    """测试3: 重复访问同一样本"""
    print("="*70)
    print("[测试3] 重复访问同一样本（测试缓存效果）")
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
    
    print(f"\n重复访问样本 0，共 10 次...")
    
    times = []
    for i in range(10):
        start = time.time()
        sample = dataset[0]
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  第 {i+1} 次: {elapsed*1000:.2f} ms")
    
    print(f"\n重复访问时间分析:")
    print(f"  - 首次访问: {times[0]*1000:.2f} ms")
    print(f"  - 后续平均: {np.mean(times[1:])*1000:.2f} ms")
    print(f"  - 加速比: {times[0]/np.mean(times[1:]):.2f}x")
    
    if times[0] > np.mean(times[1:]) * 1.5:
        print(f"  - ✅ 有缓存机制（首次慢，后续快）")
    else:
        print(f"  - ⚠️ 无明显缓存效果（每次都从磁盘读）")
    print()


def test_with_cache():
    """测试4: 使用 cache_data=True"""
    print("="*70)
    print("[测试4] 启用 cache_data=True 的效果")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 测试少量样本
    print("\n创建数据集（cache_data=True，小数据集）...")
    
    start = time.time()
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=True
    )
    init_time = time.time() - start
    print(f"初始化时间: {init_time:.2f} 秒")
    
    # 首次遍历（会触发缓存）
    print(f"\n首次遍历前 100 个样本（触发缓存）...")
    start = time.time()
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
    first_pass = time.time() - start
    print(f"首次遍历时间: {first_pass:.2f} 秒")
    print(f"平均每样本: {first_pass/100*1000:.2f} ms")
    
    # 第二次遍历（从缓存读取）
    print(f"\n第二次遍历前 100 个样本（从缓存读取）...")
    start = time.time()
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
    second_pass = time.time() - start
    print(f"第二次遍历时间: {second_pass:.2f} 秒")
    print(f"平均每样本: {second_pass/100*1000:.2f} ms")
    
    # 加速比
    if second_pass > 0:
        speedup = first_pass / second_pass
        print(f"\n缓存加速比: {speedup:.2f}x")
        
        if speedup > 2:
            print(f"  - ✅ 缓存显著加速！")
        else:
            print(f"  - ⚠️ 缓存加速不明显")
    else:
        print(f"\n缓存加速比: ∞ (第二次几乎瞬时)")
        print(f"  - ✅ 缓存极度显著！")
    print()


def test_batch_sampler_speed():
    """测试5: DynamicBatchSampler 的性能瓶颈"""
    print("="*70)
    print("[测试5] DynamicBatchSampler 性能分析")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    from pointsuite.data.datasets.collate import DynamicBatchSampler
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    print(f"\n数据集大小: {len(dataset):,} 个样本")
    
    # 创建 batch sampler
    print("\n创建 DynamicBatchSampler...")
    start = time.time()
    batch_sampler = DynamicBatchSampler(
        dataset,
        max_points=300000,
        shuffle=False,  # 顺序访问，便于测试
        drop_last=False
    )
    sampler_init_time = time.time() - start
    print(f"Sampler 初始化时间: {sampler_init_time:.2f} 秒")
    
    # 遍历所有 batch indices
    print(f"\n遍历所有 batch indices（不加载数据）...")
    start = time.time()
    batch_count = 0
    sample_count = 0
    
    for batch_indices in batch_sampler:
        batch_count += 1
        sample_count += len(batch_indices)
        
        if batch_count % 500 == 0:
            elapsed = time.time() - start
            print(f"  已处理 {batch_count} batches, {sample_count:,} 样本, 用时 {elapsed:.2f}s")
    
    total_time = time.time() - start
    
    print(f"\n遍历统计:")
    print(f"  - 总 batches: {batch_count:,}")
    print(f"  - 总样本: {sample_count:,}")
    print(f"  - 总时间: {total_time:.2f} 秒")
    print(f"  - 每 batch 平均: {total_time/batch_count*1000:.2f} ms")
    print(f"  - 每样本平均: {total_time/sample_count*1000:.2f} ms")
    
    # 对比：直接遍历索引
    print(f"\n对比：直接遍历索引...")
    start = time.time()
    for i in range(len(dataset)):
        pass
    direct_time = time.time() - start
    print(f"直接遍历时间: {direct_time:.4f} 秒")
    
    overhead = (total_time - direct_time) / total_time * 100
    print(f"\nDynamicBatchSampler 额外开销: {overhead:.1f}%")
    print()


def profile_load_data():
    """测试6: 详细分析 _load_data 方法"""
    print("="*70)
    print("[测试6] 详细分析 _load_data 方法")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    import pickle
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    print(f"\n分析单个样本加载过程...")
    
    idx = 0
    sample_info = dataset.data_list[idx]
    
    # 步骤1: 获取路径
    start = time.time()
    bin_path = Path(sample_info['bin_path'])
    pkl_path = Path(sample_info['pkl_path'])
    segment_id = sample_info['segment_id']
    t1 = time.time() - start
    print(f"\n1. 获取路径: {t1*1000:.4f} ms")
    
    # 步骤2: 加载 pkl 元数据
    start = time.time()
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    t2 = time.time() - start
    print(f"2. 加载 pkl 元数据: {t2*1000:.2f} ms ⚠️")
    
    # 步骤3: 查找 segment info
    start = time.time()
    segment_info = None
    for seg in metadata['segments']:
        if seg['segment_id'] == segment_id:
            segment_info = seg
            break
    t3 = time.time() - start
    print(f"3. 查找 segment info: {t3*1000:.2f} ms")
    
    # 步骤4: 创建 memmap
    start = time.time()
    point_data = np.memmap(bin_path, dtype=metadata['dtype'], mode='r')
    t4 = time.time() - start
    print(f"4. 创建 memmap: {t4*1000:.2f} ms")
    
    # 步骤5: 索引数据
    start = time.time()
    indices = segment_info['indices']
    segment_points = point_data[indices]
    t5 = time.time() - start
    print(f"5. 索引数据: {t5*1000:.2f} ms")
    
    # 步骤6: 提取特征
    start = time.time()
    coord = np.stack([
        segment_points['X'],
        segment_points['Y'],
        segment_points['Z']
    ], axis=1).astype(np.float32)
    
    intensity = segment_points['intensity'].astype(np.float32)
    intensity = intensity / 65535.0
    intensity = intensity[:, np.newaxis]
    
    classification = segment_points['classification'].astype(np.int64)
    
    feature = np.concatenate([coord, intensity], axis=1)
    t6 = time.time() - start
    print(f"6. 提取特征: {t6*1000:.2f} ms")
    
    # 总结
    total = t1 + t2 + t3 + t4 + t5 + t6
    print(f"\n总计: {total*1000:.2f} ms")
    print(f"\n性能占比:")
    print(f"  - 获取路径: {t1/total*100:.1f}%")
    print(f"  - 加载 pkl: {t2/total*100:.1f}% ⚠️ 主要瓶颈")
    print(f"  - 查找 segment: {t3/total*100:.1f}%")
    print(f"  - 创建 memmap: {t4/total*100:.1f}%")
    print(f"  - 索引数据: {t5/total*100:.1f}%")
    print(f"  - 提取特征: {t6/total*100:.1f}%")
    
    print(f"\n💡 优化建议:")
    if t2 > total * 0.3:
        print(f"  - ⚠️ pkl 加载占 {t2/total*100:.1f}%，建议缓存 metadata！")
    if t5 > total * 0.2:
        print(f"  - ⚠️ 数据索引占 {t5/total*100:.1f}%，考虑预处理或优化索引方式")
    print()


def main():
    """主函数"""
    print("\n")
    print("="*70)
    print("数据集访问速度诊断")
    print("="*70)
    print()
    
    # 测试1: 初始化速度
    test_init_speed()
    
    # 测试2: 单样本访问
    test_single_sample_access()
    
    # 测试3: 重复访问
    test_repeated_access()
    
    # 测试4: cache_data
    test_with_cache()
    
    # 测试5: BatchSampler 性能
    test_batch_sampler_speed()
    
    # 测试6: 详细分析
    profile_load_data()
    
    print("="*70)
    print("[完成] 诊断完成")
    print("="*70)


if __name__ == '__main__':
    main()
