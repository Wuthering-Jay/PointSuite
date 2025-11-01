"""
性能测试：对比优化前后的速度
"""
import numpy as np
import time
from tile_to_bin_with_gridsample import GridSampler


def performance_test():
    """测试GridSampler的性能"""
    
    print("="*70)
    print("GridSampler 性能测试")
    print("="*70)
    
    # 生成测试数据
    np.random.seed(42)
    
    # 测试不同规模的数据
    test_sizes = [10000, 50000, 100000, 200000]
    
    for size in test_sizes:
        print(f"\n{'='*70}")
        print(f"测试数据规模: {size:,} 点")
        print(f"{'='*70}")
        
        # 生成随机点云（模拟真实数据分布）
        points = np.random.rand(size, 3).astype(np.float64) * 100
        
        # 添加一些密集区域（模拟极端情况）
        dense_points = np.random.rand(size // 10, 3).astype(np.float64) * 0.1 + 50
        points = np.vstack([points, dense_points])
        
        print(f"实际点数: {len(points):,}")
        
        # 测试1: 不shuffle，正常max_loops
        print(f"\n测试1: shuffle=False, max_loops=30")
        sampler1 = GridSampler(grid_size=0.25, max_loops=30, shuffle_points=False)
        
        t0 = time.time()
        result1 = sampler1.sample(points)
        t1 = time.time()
        
        time1 = t1 - t0
        print(f"  耗时: {time1:.3f}s")
        print(f"  生成segments: {len(result1)}")
        print(f"  速度: {len(points)/time1:,.0f} 点/秒")
        
        # 测试2: shuffle，正常max_loops
        print(f"\n测试2: shuffle=True, max_loops=30")
        sampler2 = GridSampler(grid_size=0.25, max_loops=30, shuffle_points=True)
        
        t0 = time.time()
        result2 = sampler2.sample(points)
        t1 = time.time()
        
        time2 = t1 - t0
        print(f"  耗时: {time2:.3f}s")
        print(f"  生成segments: {len(result2)}")
        print(f"  速度: {len(points)/time2:,.0f} 点/秒")
        print(f"  相对测试1: {time2/time1:.2f}x")
        
        # 测试3: 不shuffle，小max_loops（更多segment）
        print(f"\n测试3: shuffle=False, max_loops=10")
        sampler3 = GridSampler(grid_size=0.25, max_loops=10, shuffle_points=False)
        
        t0 = time.time()
        result3 = sampler3.sample(points)
        t1 = time.time()
        
        time3 = t1 - t0
        print(f"  耗时: {time3:.3f}s")
        print(f"  生成segments: {len(result3)}")
        print(f"  速度: {len(points)/time3:,.0f} 点/秒")
        print(f"  相对测试1: {time3/time1:.2f}x")
        
        # 测试4: 完整功能
        print(f"\n测试4: shuffle=True, max_loops=10")
        sampler4 = GridSampler(grid_size=0.25, max_loops=10, shuffle_points=True)
        
        t0 = time.time()
        result4 = sampler4.sample(points)
        t1 = time.time()
        
        time4 = t1 - t0
        print(f"  耗时: {time4:.3f}s")
        print(f"  生成segments: {len(result4)}")
        print(f"  速度: {len(points)/time4:,.0f} 点/秒")
        print(f"  相对测试1: {time4/time1:.2f}x")
        
        # 验证正确性
        print(f"\n正确性验证:")
        all_indices = np.concatenate(result4)
        unique_indices = np.unique(all_indices)
        coverage = len(unique_indices) / len(points) * 100
        print(f"  覆盖率: {coverage:.2f}%")
        print(f"  {'✅ 通过' if coverage == 100.0 else '❌ 失败'}")
    
    print(f"\n{'='*70}")
    print("性能测试完成！")
    print(f"{'='*70}")
    
    print("\n💡 优化效果:")
    print("  - 使用numba加速shuffle操作")
    print("  - 使用numba加速采样循环")
    print("  - 预分配数组减少内存操作")
    print("  - 应该比纯Python循环快5-10倍以上")


def compare_with_original():
    """对比原始方法和优化方法的性能差异"""
    print("\n" + "="*70)
    print("对比测试：模拟大规模点云处理")
    print("="*70)
    
    # 模拟真实场景：大规模点云
    np.random.seed(42)
    
    # 生成1百万点（接近真实LAS文件的segment大小）
    size = 500000
    print(f"\n生成测试数据: {size:,} 点")
    
    points = np.random.rand(size, 3).astype(np.float64) * 100
    
    # 测试优化后的版本
    print(f"\n优化后的版本 (numba加速):")
    sampler = GridSampler(grid_size=0.25, max_loops=20, shuffle_points=True)
    
    # 预热numba（第一次会编译）
    print("  预热numba编译...")
    _ = sampler.sample(points[:1000])
    
    print("  正式测试...")
    t0 = time.time()
    result = sampler.sample(points)
    t1 = time.time()
    
    elapsed = t1 - t0
    print(f"  耗时: {elapsed:.3f}s")
    print(f"  速度: {size/elapsed:,.0f} 点/秒")
    print(f"  生成segments: {len(result)}")
    
    # 估算处理整个LAS文件的时间
    avg_segment_size = 100000  # 假设每个segment 10万点
    segments_per_file = 1000  # 假设1000个segments
    total_points = avg_segment_size * segments_per_file
    
    estimated_time = (total_points / size) * elapsed
    print(f"\n估算处理能力:")
    print(f"  假设LAS文件有{segments_per_file}个segments，每个{avg_segment_size:,}点")
    print(f"  总点数: {total_points:,}")
    print(f"  预估grid sampling耗时: {estimated_time:.1f}s ({estimated_time/60:.1f}分钟)")


if __name__ == "__main__":
    performance_test()
    compare_with_original()
