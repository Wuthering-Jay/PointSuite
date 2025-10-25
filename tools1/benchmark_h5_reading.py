"""
H5文件读取速度测试脚本

测试不同读取策略的性能：
1. 单线程顺序读取
2. 多进程并行读取
3. 预加载全部数据
4. 批量读取

运行方式：
    python benchmark_h5_reading.py <h5_file_path>
"""

import h5py
import numpy as np
import time
from pathlib import Path
from typing import List, Dict
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm


def read_single_segment(h5_path: str, segment_idx: int) -> Dict:
    """读取单个segment的数据"""
    with h5py.File(h5_path, 'r') as f:
        indices = f['segments'][f'segment_{segment_idx:04d}']['indices'][:]
        
        # 检查indices是否已排序，如果没有则需要排序
        if not np.all(indices[:-1] <= indices[1:]):
            # 获取排序顺序
            sort_order = np.argsort(indices)
            sorted_indices = indices[sort_order]
            
            # 使用排序后的indices读取数据
            xyz = np.stack([
                f['data']['x'][sorted_indices],
                f['data']['y'][sorted_indices],
                f['data']['z'][sorted_indices]
            ], axis=1)
            labels = f['data']['classification'][sorted_indices]
            
            # 恢复原始顺序
            unsort_order = np.argsort(sort_order)
            xyz = xyz[unsort_order]
            labels = labels[unsort_order]
        else:
            # indices已排序，直接读取
            xyz = np.stack([
                f['data']['x'][indices],
                f['data']['y'][indices],
                f['data']['z'][indices]
            ], axis=1)
            labels = f['data']['classification'][indices]
        
        data = {
            'xyz': xyz,
            'labels': labels
        }
    return data


def benchmark_sequential_read(h5_path: str, num_segments: int) -> float:
    """测试1: 单线程顺序读取"""
    print("\n=== 测试1: 单线程顺序读取 ===")
    
    start_time = time.time()
    total_points = 0
    
    for i in tqdm(range(num_segments), desc="顺序读取"):
        data = read_single_segment(h5_path, i)
        total_points += len(data['xyz'])
    
    elapsed = time.time() - start_time
    
    print(f"总时间: {elapsed:.2f}秒")
    print(f"总点数: {total_points:,}")
    print(f"速度: {num_segments/elapsed:.2f} segments/秒")
    print(f"速度: {total_points/elapsed:,.0f} 点/秒")
    
    return elapsed


def read_segment_worker(args):
    """多进程worker函数"""
    h5_path, idx = args
    return read_single_segment(h5_path, idx)


def benchmark_multiprocess_read(h5_path: str, num_segments: int, n_workers: int = 4) -> float:
    """测试2: 多进程并行读取"""
    print(f"\n=== 测试2: 多进程并行读取 (workers={n_workers}) ===")
    
    start_time = time.time()
    total_points = 0
    
    args_list = [(h5_path, i) for i in range(num_segments)]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(read_segment_worker, args_list),
            total=num_segments,
            desc="并行读取"
        ))
    
    for data in results:
        total_points += len(data['xyz'])
    
    elapsed = time.time() - start_time
    
    print(f"总时间: {elapsed:.2f}秒")
    print(f"总点数: {total_points:,}")
    print(f"速度: {num_segments/elapsed:.2f} segments/秒")
    print(f"速度: {total_points/elapsed:,.0f} 点/秒")
    print(f"加速比: {benchmark_sequential_read.__name__}的 {(num_segments/elapsed)/10:.1f}x")
    
    return elapsed


def benchmark_preload_all(h5_path: str, num_segments: int) -> float:
    """测试3: 预加载全部数据后访问"""
    print("\n=== 测试3: 预加载全部数据 ===")
    
    # 阶段1: 预加载
    print("阶段1: 加载全部数据到内存...")
    preload_start = time.time()
    
    with h5py.File(h5_path, 'r') as f:
        all_data = {
            'x': f['data']['x'][:],
            'y': f['data']['y'][:],
            'z': f['data']['z'][:],
            'labels': f['data']['classification'][:]
        }
        
        indices_list = []
        for i in range(num_segments):
            indices_list.append(f['segments'][f'segment_{i:04d}']['indices'][:])
    
    preload_time = time.time() - preload_start
    print(f"预加载时间: {preload_time:.2f}秒")
    
    # 阶段2: 访问数据
    print("阶段2: 从内存读取segments...")
    access_start = time.time()
    total_points = 0
    
    for i in tqdm(range(num_segments), desc="内存读取"):
        indices = indices_list[i]
        xyz = np.stack([
            all_data['x'][indices],
            all_data['y'][indices],
            all_data['z'][indices]
        ], axis=1)
        labels = all_data['labels'][indices]
        total_points += len(xyz)
    
    access_time = time.time() - access_start
    total_time = preload_time + access_time
    
    print(f"访问时间: {access_time:.2f}秒")
    print(f"总时间: {total_time:.2f}秒")
    print(f"总点数: {total_points:,}")
    print(f"速度 (包含预加载): {num_segments/total_time:.2f} segments/秒")
    print(f"速度 (仅访问): {num_segments/access_time:.2f} segments/秒")
    
    # 内存使用估算
    memory_mb = (all_data['x'].nbytes + all_data['y'].nbytes + 
                 all_data['z'].nbytes + all_data['labels'].nbytes) / (1024**2)
    print(f"内存占用: {memory_mb:.1f} MB")
    
    return total_time


def benchmark_batch_read(h5_path: str, num_segments: int, batch_size: int = 32) -> float:
    """测试4: 批量读取"""
    print(f"\n=== 测试4: 批量读取 (batch_size={batch_size}) ===")
    
    start_time = time.time()
    total_points = 0
    
    with h5py.File(h5_path, 'r') as f:
        num_batches = (num_segments + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="批量读取"):
            start_seg = batch_idx * batch_size
            end_seg = min(start_seg + batch_size, num_segments)
            
            # 批量读取
            for seg_idx in range(start_seg, end_seg):
                indices = f['segments'][f'segment_{seg_idx:04d}']['indices'][:]
                
                # 检查indices是否已排序
                if not np.all(indices[:-1] <= indices[1:]):
                    # 需要排序
                    sort_order = np.argsort(indices)
                    sorted_indices = indices[sort_order]
                    
                    xyz = np.stack([
                        f['data']['x'][sorted_indices],
                        f['data']['y'][sorted_indices],
                        f['data']['z'][sorted_indices]
                    ], axis=1)
                    labels = f['data']['classification'][sorted_indices]
                    
                    # 恢复原始顺序
                    unsort_order = np.argsort(sort_order)
                    xyz = xyz[unsort_order]
                    labels = labels[unsort_order]
                else:
                    # 已排序，直接读取
                    xyz = np.stack([
                        f['data']['x'][indices],
                        f['data']['y'][indices],
                        f['data']['z'][indices]
                    ], axis=1)
                    labels = f['data']['classification'][indices]
                
                total_points += len(xyz)
    
    elapsed = time.time() - start_time
    
    print(f"总时间: {elapsed:.2f}秒")
    print(f"总点数: {total_points:,}")
    print(f"速度: {num_segments/elapsed:.2f} segments/秒")
    print(f"速度: {total_points/elapsed:,.0f} 点/秒")
    
    return elapsed


def get_file_info(h5_path: str):
    """获取H5文件基本信息"""
    print("\n" + "="*60)
    print(f"H5文件信息: {Path(h5_path).name}")
    print("="*60)
    
    with h5py.File(h5_path, 'r') as f:
        num_segments = f['segments'].attrs['num_segments']
        total_points = len(f['data']['x'])
        
        # 获取数据集信息
        x_dataset = f['data']['x']
        compression = x_dataset.compression
        compression_opts = x_dataset.compression_opts
        chunks = x_dataset.chunks
        
        # 计算文件大小
        file_size_mb = Path(h5_path).stat().st_size / (1024**2)
        
        # 采样几个segment的大小
        segment_sizes = []
        for i in range(min(10, num_segments)):
            indices = f['segments'][f'segment_{i:04d}']['indices'][:]
            segment_sizes.append(len(indices))
        
        avg_seg_size = np.mean(segment_sizes)
        min_seg_size = np.min(segment_sizes)
        max_seg_size = np.max(segment_sizes)
        
        print(f"总点数: {total_points:,}")
        print(f"总segments: {num_segments}")
        print(f"平均segment大小: {avg_seg_size:.0f} 点")
        print(f"Segment大小范围: {min_seg_size} - {max_seg_size}")
        print(f"文件大小: {file_size_mb:.1f} MB")
        print(f"压缩方式: {compression}")
        if compression_opts:
            print(f"压缩级别: {compression_opts}")
        print(f"Chunk大小: {chunks}")
        
        return num_segments, total_points


def benchmark_random_access(h5_path: str, num_segments: int, num_samples: int = 100) -> float:
    """测试5: 随机访问性能"""
    print(f"\n=== 测试5: 随机访问 (采样{num_samples}个segments) ===")
    
    # 生成随机索引
    random_indices = np.random.choice(num_segments, size=min(num_samples, num_segments), replace=False)
    
    start_time = time.time()
    total_points = 0
    
    for idx in tqdm(random_indices, desc="随机访问"):
        data = read_single_segment(h5_path, int(idx))
        total_points += len(data['xyz'])
    
    elapsed = time.time() - start_time
    
    print(f"总时间: {elapsed:.2f}秒")
    print(f"采样segments: {len(random_indices)}")
    print(f"总点数: {total_points:,}")
    print(f"速度: {len(random_indices)/elapsed:.2f} segments/秒")
    print(f"速度: {total_points/elapsed:,.0f} 点/秒")
    
    return elapsed


def main():
    if len(sys.argv) < 2:
        print("用法: python benchmark_h5_reading.py <h5_file_path> [num_test_segments]")
        print("\n示例:")
        print("  python benchmark_h5_reading.py processed_02.h5")
        print("  python benchmark_h5_reading.py processed_02.h5 100  # 只测试前100个segments")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    
    if not Path(h5_path).exists():
        print(f"错误: 文件不存在: {h5_path}")
        sys.exit(1)
    
    # 获取文件信息
    num_segments, total_points = get_file_info(h5_path)
    
    # 如果指定了测试segment数量
    if len(sys.argv) >= 3:
        test_segments = min(int(sys.argv[2]), num_segments)
        print(f"\n*** 仅测试前 {test_segments} 个segments ***\n")
    else:
        test_segments = num_segments
    
    # 运行各项测试
    results = {}
    
    # 测试1: 顺序读取
    results['sequential'] = benchmark_sequential_read(h5_path, test_segments)
    
    # 测试2: 多进程并行读取（不同worker数量）
    cpu_count = multiprocessing.cpu_count()
    for n_workers in [2, 4, min(8, cpu_count)]:
        if n_workers <= cpu_count:
            results[f'multiprocess_{n_workers}'] = benchmark_multiprocess_read(
                h5_path, test_segments, n_workers
            )
    
    # 测试3: 预加载（仅对小文件测试）
    if total_points < 50_000_000:  # 小于5000万点
        results['preload'] = benchmark_preload_all(h5_path, test_segments)
    else:
        print("\n=== 测试3: 预加载全部数据 ===")
        print("跳过（文件太大，预加载会占用过多内存）")
    
    # 测试4: 批量读取
    results['batch'] = benchmark_batch_read(h5_path, test_segments, batch_size=32)
    
    # 测试5: 随机访问
    results['random'] = benchmark_random_access(h5_path, num_segments, num_samples=100)
    
    # 总结
    print("\n" + "="*60)
    print("性能测试总结")
    print("="*60)
    
    baseline = results['sequential']
    
    for test_name, elapsed in results.items():
        speedup = baseline / elapsed
        segments_per_sec = test_segments / elapsed
        print(f"{test_name:20s}: {elapsed:6.2f}秒 | {segments_per_sec:6.1f} seg/s | 加速比: {speedup:.2f}x")
    
    # 推荐
    print("\n推荐配置:")
    best_test = min(results.items(), key=lambda x: x[1])
    print(f"  最快方法: {best_test[0]} ({best_test[1]:.2f}秒)")
    
    if 'multiprocess_4' in results:
        print(f"  深度学习训练推荐: 使用DataLoader with num_workers=4-8")
        print(f"  预期性能: ~{test_segments/results['multiprocess_4']:.0f} segments/秒")


if __name__ == "__main__":
    main()
