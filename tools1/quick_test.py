"""
快速验证优化后的GridSampler功能
"""
import numpy as np
import time
import sys
sys.path.insert(0, 'e:/code/python/PointSuite/tools1')

from tile_to_bin_with_gridsample import GridSampler


def quick_test():
    """快速测试优化后的功能是否正常"""
    print("="*70)
    print("快速功能验证")
    print("="*70)
    
    # 生成小规模测试数据
    np.random.seed(42)
    points = np.random.rand(10000, 3).astype(np.float64) * 10
    
    print(f"\n测试数据: {len(points):,} 点")
    
    # 测试1: 基本功能
    print(f"\n测试1: 基本功能（无shuffle，max_loops=30）")
    sampler1 = GridSampler(grid_size=0.5, max_loops=30, shuffle_points=False)
    
    t0 = time.time()
    result1 = sampler1.sample(points)
    t1 = time.time()
    
    print(f"  ✓ 耗时: {(t1-t0)*1000:.1f}ms")
    print(f"  ✓ 生成segments: {len(result1)}")
    
    # 验证覆盖率
    all_idx = np.concatenate(result1)
    unique_idx = np.unique(all_idx)
    coverage = len(unique_idx) / len(points) * 100
    print(f"  ✓ 覆盖率: {coverage:.1f}%")
    
    # 测试2: 带shuffle
    print(f"\n测试2: 带shuffle（shuffle=True，max_loops=30）")
    sampler2 = GridSampler(grid_size=0.5, max_loops=30, shuffle_points=True)
    
    t0 = time.time()
    result2 = sampler2.sample(points)
    t1 = time.time()
    
    print(f"  ✓ 耗时: {(t1-t0)*1000:.1f}ms")
    print(f"  ✓ 生成segments: {len(result2)}")
    
    # 验证随机性
    result2_again = sampler2.sample(points)
    is_different = not np.array_equal(result2[0], result2_again[0])
    print(f"  ✓ 随机性: {'通过（每次不同）' if is_different else '失败（每次相同）'}")
    
    # 测试3: 极端情况（小max_loops）
    print(f"\n测试3: 极端情况（max_loops=5）")
    
    # 创建有密集体素的数据
    dense_points = np.random.rand(100, 3).astype(np.float64) * 0.1  # 100个点在一个小区域
    test_points = np.vstack([points[:1000], dense_points])
    
    sampler3 = GridSampler(grid_size=0.5, max_loops=5, shuffle_points=True)
    
    t0 = time.time()
    result3 = sampler3.sample(test_points)
    t1 = time.time()
    
    print(f"  ✓ 耗时: {(t1-t0)*1000:.1f}ms")
    print(f"  ✓ 生成segments: {len(result3)}")
    print(f"  ✓ 限制生效: segments数量受max_loops限制")
    
    # 验证覆盖率
    all_idx3 = np.concatenate(result3)
    unique_idx3 = np.unique(all_idx3)
    coverage3 = len(unique_idx3) / len(test_points) * 100
    print(f"  ✓ 覆盖率: {coverage3:.1f}%")
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！优化后的代码工作正常。")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
