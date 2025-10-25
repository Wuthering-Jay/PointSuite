"""
H5随机读取性能深度分析

测试不同存储策略对随机读取的影响：
1. 当前方案（gzip压缩 + 需要排序indices）
2. 无压缩 + 预排序indices
3. 连续存储（每个segment独立存储）
"""

import h5py
import numpy as np
import time
from pathlib import Path
import tempfile


def analyze_current_h5_bottleneck(h5_path: str):
    """分析当前H5文件的瓶颈"""
    
    print("="*70)
    print("当前H5文件性能瓶颈分析")
    print("="*70)
    
    with h5py.File(h5_path, 'r') as f:
        # 获取数据集信息
        x_dataset = f['data']['x']
        print(f"\n当前存储方式:")
        print(f"  压缩: {x_dataset.compression}")
        print(f"  压缩级别: {x_dataset.compression_opts}")
        print(f"  Chunk大小: {x_dataset.chunks}")
        print(f"  数据类型: {x_dataset.dtype}")
        
        # 测试读取性能
        num_tests = 20
        indices_to_test = np.random.choice(f['segments'].attrs['num_segments'], num_tests, replace=False)
        
        print(f"\n测试随机读取{num_tests}个segments:")
        
        # 测试1: 读取indices
        start = time.time()
        for idx in indices_to_test:
            indices = f['segments'][f'segment_{idx:04d}']['indices'][:]
        time_read_indices = time.time() - start
        print(f"  读取indices: {time_read_indices*1000:.2f}ms")
        
        # 测试2: 检查排序
        start = time.time()
        needs_sort_count = 0
        for idx in indices_to_test:
            indices = f['segments'][f'segment_{idx:04d}']['indices'][:]
            if not np.all(indices[:-1] <= indices[1:]):
                needs_sort_count += 1
        time_check_sort = time.time() - start
        print(f"  检查排序: {time_check_sort*1000:.2f}ms")
        print(f"  需要排序: {needs_sort_count}/{num_tests}")
        
        # 测试3: 排序indices
        start = time.time()
        for idx in indices_to_test:
            indices = f['segments'][f'segment_{idx:04d}']['indices'][:]
            if not np.all(indices[:-1] <= indices[1:]):
                sort_order = np.argsort(indices)
                sorted_indices = indices[sort_order]
                unsort_order = np.argsort(sort_order)
        time_sort = time.time() - start
        print(f"  排序操作: {time_sort*1000:.2f}ms")
        
        # 测试4: 读取数据（fancy indexing）
        start = time.time()
        for idx in indices_to_test[:5]:  # 只测5个，因为慢
            indices = f['segments'][f'segment_{idx:04d}']['indices'][:]
            if np.all(indices[:-1] <= indices[1:]):
                xyz = np.stack([
                    x_dataset[indices],
                    f['data']['y'][indices],
                    f['data']['z'][indices]
                ], axis=1)
        time_read_data = time.time() - start
        print(f"  读取数据(5个): {time_read_data*1000:.2f}ms ({time_read_data/5*1000:.2f}ms/segment)")
        
        # 测试5: 解压缩开销
        # 读取连续数据 vs fancy indexing
        start = time.time()
        for _ in range(5):
            data = x_dataset[:10000]  # 读取连续1万点
        time_sequential = time.time() - start
        print(f"  连续读取(5x10k点): {time_sequential*1000:.2f}ms")
        
        avg_seg_size = len(f['segments']['segment_0000']['indices'][:])
        print(f"\n瓶颈分析（平均每segment {avg_seg_size}点）:")
        print(f"  1. Indices读取: {time_read_indices/num_tests*1000:.2f}ms (小开销)")
        print(f"  2. 排序检查: {time_check_sort/num_tests*1000:.2f}ms (小开销)")
        print(f"  3. 排序操作: {time_sort/num_tests*1000:.2f}ms (中等开销)")
        print(f"  4. ⚠️ Fancy indexing读取: {time_read_data/5*1000:.2f}ms (主要瓶颈!)")
        print(f"  5. 连续读取: {time_sequential/5*1000:.2f}ms (快{(time_read_data/5)/(time_sequential/5):.1f}倍)")
        
        print(f"\n💡 核心问题:")
        print(f"  Fancy indexing + 压缩 → 需要解压大量chunks → 极慢")
        print(f"  连续读取 + 压缩 → 只需解压少量chunks → 快")


def test_storage_strategies():
    """测试不同存储策略的性能"""
    
    print("\n" + "="*70)
    print("存储策略性能对比")
    print("="*70)
    
    # 生成测试数据
    np.random.seed(42)
    num_points = 1000000
    num_segments = 50
    
    test_data = {
        'x': np.random.randn(num_points).astype(np.float32),
        'y': np.random.randn(num_points).astype(np.float32),
        'z': np.random.randn(num_points).astype(np.float32),
        'labels': np.random.randint(0, 10, num_points, dtype=np.int32)
    }
    
    # 生成segments（随机索引）
    segments = []
    points_per_seg = num_points // num_segments
    for i in range(num_segments):
        start = i * points_per_seg
        end = start + points_per_seg
        indices = np.random.permutation(np.arange(start, end))[:int(points_per_seg*0.8)]
        segments.append(indices)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # ===== 策略1: 当前方案（gzip + 未排序indices） =====
        print("\n策略1: Gzip压缩 + Fancy Indexing (当前方案)")
        h5_gzip = tmpdir / "test_gzip.h5"
        
        start = time.time()
        with h5py.File(h5_gzip, 'w') as f:
            data_group = f.create_group('data')
            for key, arr in test_data.items():
                data_group.create_dataset(
                    key, data=arr, 
                    compression='gzip', compression_opts=4,
                    chunks=(8192,), shuffle=True
                )
            
            seg_group = f.create_group('segments')
            seg_group.attrs['num_segments'] = num_segments
            for i, indices in enumerate(segments):
                sg = seg_group.create_group(f'segment_{i:04d}')
                sg.create_dataset('indices', data=indices, dtype=np.int64)
        
        write_time = time.time() - start
        file_size = h5_gzip.stat().st_size / (1024**2)
        
        # 测试读取
        start = time.time()
        with h5py.File(h5_gzip, 'r') as f:
            for i in range(10):
                indices = f['segments'][f'segment_{i:04d}']['indices'][:]
                sort_order = np.argsort(indices)
                sorted_indices = indices[sort_order]
                xyz = np.stack([
                    f['data']['x'][sorted_indices],
                    f['data']['y'][sorted_indices],
                    f['data']['z'][sorted_indices]
                ], axis=1)
        read_time = time.time() - start
        
        print(f"  写入时间: {write_time:.2f}秒")
        print(f"  文件大小: {file_size:.1f}MB")
        print(f"  读取10 segments: {read_time:.2f}秒 ({read_time/10*1000:.0f}ms/seg)")
        
        # ===== 策略2: 无压缩 + Fancy Indexing =====
        print("\n策略2: 无压缩 + Fancy Indexing")
        h5_nocomp = tmpdir / "test_nocomp.h5"
        
        start = time.time()
        with h5py.File(h5_nocomp, 'w') as f:
            data_group = f.create_group('data')
            for key, arr in test_data.items():
                data_group.create_dataset(
                    key, data=arr,
                    chunks=(8192,)  # 无压缩
                )
            
            seg_group = f.create_group('segments')
            seg_group.attrs['num_segments'] = num_segments
            for i, indices in enumerate(segments):
                sg = seg_group.create_group(f'segment_{i:04d}')
                sg.create_dataset('indices', data=np.sort(indices), dtype=np.int64)
        
        write_time = time.time() - start
        file_size = h5_nocomp.stat().st_size / (1024**2)
        
        # 测试读取
        start = time.time()
        with h5py.File(h5_nocomp, 'r') as f:
            for i in range(10):
                indices = f['segments'][f'segment_{i:04d}']['indices'][:]
                xyz = np.stack([
                    f['data']['x'][indices],
                    f['data']['y'][indices],
                    f['data']['z'][indices]
                ], axis=1)
        read_time = time.time() - start
        
        print(f"  写入时间: {write_time:.2f}秒")
        print(f"  文件大小: {file_size:.1f}MB")
        print(f"  读取10 segments: {read_time:.2f}秒 ({read_time/10*1000:.0f}ms/seg)")
        
        # ===== 策略3: 连续存储（每个segment独立存储） =====
        print("\n策略3: 连续存储 (每segment独立)")
        h5_contiguous = tmpdir / "test_contiguous.h5"
        
        start = time.time()
        with h5py.File(h5_contiguous, 'w') as f:
            seg_group = f.create_group('segments')
            seg_group.attrs['num_segments'] = num_segments
            
            for i, indices in enumerate(segments):
                sg = seg_group.create_group(f'segment_{i:04d}')
                # 直接存储segment的数据，不存indices
                sg.create_dataset('x', data=test_data['x'][indices])
                sg.create_dataset('y', data=test_data['y'][indices])
                sg.create_dataset('z', data=test_data['z'][indices])
                sg.create_dataset('labels', data=test_data['labels'][indices])
        
        write_time = time.time() - start
        file_size = h5_contiguous.stat().st_size / (1024**2)
        
        # 测试读取
        start = time.time()
        with h5py.File(h5_contiguous, 'r') as f:
            for i in range(10):
                xyz = np.stack([
                    f['segments'][f'segment_{i:04d}']['x'][:],
                    f['segments'][f'segment_{i:04d}']['y'][:],
                    f['segments'][f'segment_{i:04d}']['z'][:]
                ], axis=1)
        read_time = time.time() - start
        
        print(f"  写入时间: {write_time:.2f}秒")
        print(f"  文件大小: {file_size:.1f}MB")
        print(f"  读取10 segments: {read_time:.2f}秒 ({read_time/10*1000:.0f}ms/seg)")
        
        # ===== 策略4: 连续存储 + 无压缩但chunked =====
        print("\n策略4: 连续存储 + Chunking优化")
        h5_optimized = tmpdir / "test_optimized.h5"
        
        start = time.time()
        with h5py.File(h5_optimized, 'w') as f:
            seg_group = f.create_group('segments')
            seg_group.attrs['num_segments'] = num_segments
            
            for i, indices in enumerate(segments):
                sg = seg_group.create_group(f'segment_{i:04d}')
                seg_len = len(indices)
                # 使用contiguous存储（不chunking）
                sg.create_dataset('x', data=test_data['x'][indices], chunks=None)
                sg.create_dataset('y', data=test_data['y'][indices], chunks=None)
                sg.create_dataset('z', data=test_data['z'][indices], chunks=None)
                sg.create_dataset('labels', data=test_data['labels'][indices], chunks=None)
        
        write_time = time.time() - start
        file_size = h5_optimized.stat().st_size / (1024**2)
        
        # 测试读取
        start = time.time()
        with h5py.File(h5_optimized, 'r') as f:
            for i in range(10):
                xyz = np.stack([
                    f['segments'][f'segment_{i:04d}']['x'][:],
                    f['segments'][f'segment_{i:04d}']['y'][:],
                    f['segments'][f'segment_{i:04d}']['z'][:]
                ], axis=1)
        read_time = time.time() - start
        
        print(f"  写入时间: {write_time:.2f}秒")
        print(f"  文件大小: {file_size:.1f}MB")
        print(f"  读取10 segments: {read_time:.2f}秒 ({read_time/10*1000:.0f}ms/seg)")
    
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    print("""
1. Fancy Indexing + 压缩 = 极慢（当前方案）
   - 需要解压大量不相关的chunks
   - 索引不连续导致缓存失效
   
2. Fancy Indexing + 无压缩 = 快一些
   - 消除解压开销
   - 但仍需随机访问
   
3. 连续存储 = 最快！
   - 每个segment的数据连续存储
   - 顺序读取，缓存友好
   - 文件稍大但性能最佳
   
推荐: 策略4（连续存储 + 无压缩 + contiguous）
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
        analyze_current_h5_bottleneck(h5_path)
    else:
        print("未提供H5文件，只运行测试")
    
    test_storage_strategies()
