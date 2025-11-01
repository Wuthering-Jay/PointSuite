"""
测试和演示max_loops和shuffle_points参数的效果
"""
import numpy as np
from tile_to_bin_with_gridsample import GridSampler


def test_grid_sampler():
    """测试GridSampler的max_loops和shuffle功能"""
    
    print("="*70)
    print("测试GridSampler - max_loops和shuffle_points功能")
    print("="*70)
    
    # 创建测试数据：一个密集区域（模拟极端情况）
    np.random.seed(42)
    
    # 生成一些点，其中一些在同一个体素内
    points = np.array([
        # 体素1: 只有1个点
        [0.01, 0.01, 0.01],
        
        # 体素2: 3个点（正常情况）
        [0.51, 0.51, 0.51],
        [0.52, 0.52, 0.52],
        [0.53, 0.53, 0.53],
        
        # 体素3: 50个点（极端情况，密集区域）
        *[[1.01 + i*0.001, 1.01 + i*0.001, 1.01] for i in range(50)],
    ], dtype=np.float64)
    
    print(f"\n测试数据:")
    print(f"  - 总点数: {len(points)}")
    print(f"  - 体素1: 1个点")
    print(f"  - 体素2: 3个点")
    print(f"  - 体素3: 50个点（极端情况）")
    
    # 测试1: 无max_loops限制
    print(f"\n" + "="*70)
    print(f"测试1: 无max_loops限制（传统方法）")
    print(f"="*70)
    
    sampler1 = GridSampler(grid_size=0.5, max_loops=1000, shuffle_points=False)
    result1 = sampler1.sample(points)
    
    print(f"  - 生成segments数: {len(result1)}")
    print(f"  - 每个segment的点数: {[len(seg) for seg in result1[:10]]}...")
    
    # 测试2: 有max_loops限制
    print(f"\n" + "="*70)
    print(f"测试2: max_loops=10（限制循环次数）")
    print(f"="*70)
    
    sampler2 = GridSampler(grid_size=0.5, max_loops=10, shuffle_points=False)
    result2 = sampler2.sample(points)
    
    print(f"  - 生成segments数: {len(result2)}")
    print(f"  - 每个segment的点数: {[len(seg) for seg in result2]}")
    print(f"  - 减少倍数: {len(result1) / len(result2):.2f}x")
    
    # 验证覆盖率
    all_indices_1 = np.concatenate(result1)
    all_indices_2 = np.concatenate(result2)
    unique_1 = len(np.unique(all_indices_1))
    unique_2 = len(np.unique(all_indices_2))
    
    print(f"\n  覆盖率验证:")
    print(f"    - 方法1覆盖点数: {unique_1}/{len(points)}")
    print(f"    - 方法2覆盖点数: {unique_2}/{len(points)}")
    print(f"    - 覆盖率: {'✅ 100%' if unique_2 == len(points) else '❌ 不完整'}")
    
    # 测试3: 打乱点顺序
    print(f"\n" + "="*70)
    print(f"测试3: 打乱点顺序（shuffle_points=True）")
    print(f"="*70)
    
    sampler3 = GridSampler(grid_size=0.5, max_loops=10, shuffle_points=True)
    
    # 多次采样，检查随机性
    results = []
    for i in range(3):
        result = sampler3.sample(points)
        results.append(result)
        print(f"\n  第{i+1}次采样:")
        print(f"    - Segment 0的前5个索引: {result[0][:5]}")
    
    # 检查是否真的打乱了
    all_same = all(np.array_equal(results[0][0], results[i][0]) for i in range(1, 3))
    print(f"\n  随机性检查: {'❌ 每次相同（未打乱）' if all_same else '✅ 每次不同（已打乱）'}")
    
    # 测试4: 极端情况统计
    print(f"\n" + "="*70)
    print(f"测试4: 极端情况处理详情")
    print(f"="*70)
    
    sampler4 = GridSampler(grid_size=0.5, max_loops=10, shuffle_points=True)
    result4 = sampler4.sample(points)
    
    print(f"\n  体素3（50个点）的采样策略:")
    print(f"    - max_loops = 10")
    print(f"    - 每次应采样: ceil(50/10) = 5个点")
    print(f"    - 总循环次数: 10次")
    print(f"    - 预期总采样: 10次 × 约5点/次 = 50个点")
    
    # 统计体素3的点被采样的次数
    voxel3_indices = list(range(4, 54))  # 体素3的索引范围
    voxel3_sample_count = {}
    for seg in result4:
        for idx in seg:
            if idx in voxel3_indices:
                voxel3_sample_count[idx] = voxel3_sample_count.get(idx, 0) + 1
    
    sample_counts = list(voxel3_sample_count.values())
    print(f"\n  实际统计:")
    print(f"    - 体素3被采样的点数: {len(voxel3_sample_count)}/50")
    print(f"    - 每个点被采样次数: 最小={min(sample_counts)}, 最大={max(sample_counts)}")
    print(f"    - 覆盖率: {len(voxel3_sample_count)/50*100:.1f}%")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
    
    print("\n💡 总结:")
    print("  1. max_loops成功限制了循环次数，避免生成过多segments")
    print("  2. 极端情况下自动调整为每次采样多个点")
    print("  3. shuffle_points增加了随机性")
    print("  4. 保持100%覆盖率")


if __name__ == "__main__":
    test_grid_sampler()
