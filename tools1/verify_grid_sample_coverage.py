import numpy as np
import pickle
from pathlib import Path
from typing import Union, Dict, List
from collections import Counter


def verify_segment_coverage(pkl_path: Union[str, Path], bin_path: Union[str, Path] = None):
    """
    验证分块结果中每个点的覆盖情况和重复采样率
    
    Args:
        pkl_path: pkl文件路径
        bin_path: bin文件路径（可选，用于获取总点数）
    """
    pkl_path = Path(pkl_path)
    
    print("="*70)
    print(f"验证文件: {pkl_path.name}")
    print("="*70)
    
    # 加载元数据
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    total_points = metadata['num_points']
    num_segments = metadata['num_segments']
    segments_info = metadata['segments']
    
    print(f"\n基本信息:")
    print(f"  - 总点数: {total_points:,}")
    print(f"  - 分块数: {num_segments:,}")
    print(f"  - Grid Sample: {metadata.get('grid_sample', False)}")
    if metadata.get('grid_sample'):
        print(f"  - Grid Size: {metadata.get('grid_size')}")
        print(f"  - Max Sample Loops: {metadata.get('max_sample_loops')}")
    print(f"  - Window Size: {metadata['window_size']}")
    print(f"  - Min Points: {metadata['min_points']}")
    print(f"  - Max Points: {metadata['max_points']}")
    print(f"  - Overlap: {metadata['overlap']}")
    
    # 统计每个点出现的次数
    print(f"\n{'='*70}")
    print(f"分析点覆盖情况...")
    print(f"{'='*70}")
    
    # 1. 收集所有分块中的所有点索引
    all_indices_in_segments = []
    segment_sizes = []
    
    for seg_info in segments_info:
        indices = seg_info['indices']
        segment_sizes.append(len(indices))
        all_indices_in_segments.extend(indices)
    
    all_indices_in_segments = np.array(all_indices_in_segments)
    
    # 2. 统计所有分块中点的总数（包含重复）
    total_point_instances_in_segments = len(all_indices_in_segments)
    
    # 3. 统计unique的点数
    unique_points_in_segments = np.unique(all_indices_in_segments)
    num_unique_points = len(unique_points_in_segments)
    
    # 4. 统计每个点出现的次数
    point_count = np.zeros(total_points, dtype=np.int32)
    for idx in all_indices_in_segments:
        point_count[idx] += 1
    
    # 5. 验证bin文件中的点数
    bin_path = pkl_path.with_suffix('.bin')
    if bin_path.exists():
        dtype = np.dtype(metadata['dtype'])
        bin_point_count = bin_path.stat().st_size // dtype.itemsize
    else:
        bin_path = None
        bin_point_count = None
    
    print(f"\n📊 点数统计:")
    print(f"  - Metadata中的总点数: {total_points:,}")
    if bin_path and bin_point_count:
        print(f"  - Bin文件中的点数: {bin_point_count:,}")
        if bin_point_count == total_points:
            print(f"    ✅ Bin文件与metadata匹配")
        else:
            print(f"    ⚠️  Bin文件与metadata不匹配 (差异: {abs(bin_point_count - total_points):,})")
    
    print(f"\n  - 所有分块中的点实例总数: {total_point_instances_in_segments:,}")
    print(f"  - 所有分块中的unique点数: {num_unique_points:,}")
    
    if num_unique_points == total_points:
        print(f"    ✅ Unique点数与总点数匹配")
    else:
        print(f"    ⚠️  Unique点数与总点数不匹配 (差异: {abs(num_unique_points - total_points):,})")
    
    # 6. 计算重复率
    repetition_count = total_point_instances_in_segments - num_unique_points
    if num_unique_points > 0:
        repetition_rate = (repetition_count / num_unique_points) * 100
    else:
        repetition_rate = 0
    
    print(f"\n📈 重复采样统计:")
    print(f"  - 重复的点实例数: {repetition_count:,}")
    print(f"  - 重复率: {repetition_rate:.2f}%")
    print(f"  - 平均每个点被采样: {total_point_instances_in_segments / num_unique_points:.2f} 次")
    
    # 7. 覆盖率统计
    uncovered_points = np.sum(point_count == 0)
    covered_once = np.sum(point_count == 1)
    covered_multiple = np.sum(point_count > 1)
    
    coverage_rate = (total_points - uncovered_points) / total_points * 100
    
    print(f"\n📊 点覆盖统计:")
    print(f"  - 未覆盖点数: {uncovered_points:,} ({uncovered_points/total_points*100:.2f}%)")
    print(f"  - 覆盖1次: {covered_once:,} ({covered_once/total_points*100:.2f}%)")
    print(f"  - 覆盖多次: {covered_multiple:,} ({covered_multiple/total_points*100:.2f}%)")
    print(f"  - 总覆盖率: {coverage_rate:.2f}%")
    
    if uncovered_points > 0:
        print(f"\n⚠️  警告: 有 {uncovered_points:,} 个点未被任何分块覆盖！")
        # 显示前10个未覆盖的点索引
        uncovered_indices = np.where(point_count == 0)[0]
        print(f"  未覆盖点索引示例: {uncovered_indices[:10]}")
    else:
        print(f"\n✅ 所有点都被覆盖！")
    
    # 8. 每个点出现次数的分布
    if covered_multiple > 0 or covered_once > 0:
        print(f"\n� 点出现次数分布:")
        max_coverage = point_count.max()
        print(f"  - 最大出现次数: {max_coverage}")
        
        # 统计每个出现次数的点数
        coverage_distribution = Counter(point_count[point_count > 0])
        print(f"  - 详细分布:")
        for times in sorted(coverage_distribution.keys()):
            count = coverage_distribution[times]
            percentage = count / num_unique_points * 100 if num_unique_points > 0 else 0
            print(f"    出现{times}次: {count:,} 点 ({percentage:.2f}%)")
    
    # 分块大小统计
    print(f"\n📦 分块大小统计:")
    segment_sizes = np.array(segment_sizes)
    print(f"  - 最小分块: {segment_sizes.min():,} 点")
    print(f"  - 最大分块: {segment_sizes.max():,} 点")
    print(f"  - 平均分块: {segment_sizes.mean():.0f} 点")
    print(f"  - 中位数: {np.median(segment_sizes):.0f} 点")
    print(f"  - 标准差: {segment_sizes.std():.0f} 点")
    
    # 分块大小分布
    print(f"\n  - 分块大小分布:")
    bins = [0, 1000, 5000, 10000, 50000, 100000, float('inf')]
    labels = ['<1K', '1K-5K', '5K-10K', '10K-50K', '50K-100K', '>100K']
    
    for i in range(len(bins)-1):
        count = np.sum((segment_sizes >= bins[i]) & (segment_sizes < bins[i+1]))
        if count > 0:
            percentage = count / len(segment_sizes) * 100
            print(f"    {labels[i]}: {count} 个分块 ({percentage:.1f}%)")
    
    print(f"\n{'='*70}")
    
    # 返回统计结果
    return {
        'total_points': total_points,
        'num_segments': num_segments,
        'total_point_instances': total_point_instances_in_segments,
        'num_unique_points': num_unique_points,
        'repetition_rate': repetition_rate,
        'uncovered_points': uncovered_points,
        'coverage_rate': coverage_rate,
        'covered_once': covered_once,
        'covered_multiple': covered_multiple,
        'max_coverage': point_count.max(),
        'segment_sizes': segment_sizes,
        'bin_point_count': bin_point_count,
    }


def compare_multiple_files(pkl_paths: List[Union[str, Path]]):
    """
    比较多个pkl文件的统计结果
    
    Args:
        pkl_paths: pkl文件路径列表
    """
    print("\n" + "="*70)
    print("批量验证多个文件")
    print("="*70)
    
    all_results = []
    
    for pkl_path in pkl_paths:
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            print(f"⚠️  文件不存在: {pkl_path}")
            continue
        
        try:
            result = verify_segment_coverage(pkl_path)
            result['filename'] = pkl_path.name
            all_results.append(result)
            print()
        except Exception as e:
            print(f"❌ 处理 {pkl_path.name} 时出错: {e}\n")
    
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("汇总对比")
        print("="*70)
        print(f"\n{'文件名':<30} {'总点数':>12} {'分块数':>8} {'覆盖率':>8} {'重复率':>8}")
        print("-"*70)
        
        for result in all_results:
            print(f"{result['filename']:<30} "
                  f"{result['total_points']:>12,} "
                  f"{result['num_segments']:>8,} "
                  f"{result['coverage_rate']:>7.1f}% "
                  f"{result['repetition_rate']:>7.1f}%")


def analyze_grid_sample_effect(pkl_with_grid: Union[str, Path], pkl_without_grid: Union[str, Path]):
    """
    对比有无grid sample的效果
    
    Args:
        pkl_with_grid: 带grid sample的pkl文件
        pkl_without_grid: 不带grid sample的pkl文件
    """
    print("\n" + "="*70)
    print("Grid Sample 效果对比")
    print("="*70)
    
    print("\n[1] 不带 Grid Sample:")
    result_without = verify_segment_coverage(pkl_without_grid)
    
    print("\n[2] 带 Grid Sample:")
    result_with = verify_segment_coverage(pkl_with_grid)
    
    print("\n" + "="*70)
    print("对比结果")
    print("="*70)
    
    print(f"\n分块数变化:")
    print(f"  不带 Grid Sample: {result_without['num_segments']:,} 个")
    print(f"  带 Grid Sample: {result_with['num_segments']:,} 个")
    print(f"  增加: {result_with['num_segments'] - result_without['num_segments']:,} 个 "
          f"({(result_with['num_segments'] / result_without['num_segments'] - 1) * 100:+.1f}%)")
    
    print(f"\n覆盖率:")
    print(f"  不带 Grid Sample: {result_without['coverage_rate']:.2f}%")
    print(f"  带 Grid Sample: {result_with['coverage_rate']:.2f}%")
    
    print(f"\n重复率:")
    print(f"  不带 Grid Sample: {result_without['avg_repetition_rate']:.2f}%")
    print(f"  带 Grid Sample: {result_with['avg_repetition_rate']:.2f}%")
    
    print(f"\n平均分块大小:")
    print(f"  不带 Grid Sample: {result_without['segment_sizes'].mean():.0f} 点")
    print(f"  带 Grid Sample: {result_with['segment_sizes'].mean():.0f} 点")


if __name__ == "__main__":
    # 示例1: 验证单个文件
    print("示例1: 验证单个pkl文件")
    
    pkl_file = Path(r"E:\data\云南遥感中心\第一批\bin\train") / "processed_02.pkl"
    
    if pkl_file.exists():
        result = verify_segment_coverage(pkl_file)
    else:
        print(f"⚠️  文件不存在: {pkl_file}")
        print("\n请修改脚本中的文件路径为实际路径")
    
    # 示例2: 批量验证多个文件
    print("\n\n" + "="*70)
    print("示例2: 批量验证多个文件")
    print("="*70)
    
    data_dir = Path(r"E:\data\云南遥感中心\第一批\bin\train")
    if data_dir.exists():
        pkl_files = list(data_dir.glob("*.pkl"))[:5]  # 只验证前5个
        if pkl_files:
            compare_multiple_files(pkl_files)
        else:
            print("⚠️  目录中没有找到pkl文件")
    
    # 示例3: 对比有无grid sample的效果（如果有两个版本的话）
    # print("\n\n" + "="*70)
    # print("示例3: Grid Sample 效果对比")
    # print("="*70)
    # 
    # pkl_with_grid = Path(r"E:\data\云南遥感中心\第一批\bin\train_with_grid") / "5080_54400.pkl"
    # pkl_without_grid = Path(r"E:\data\云南遥感中心\第一批\bin\train") / "5080_54400.pkl"
    # 
    # if pkl_with_grid.exists() and pkl_without_grid.exists():
    #     analyze_grid_sample_effect(pkl_with_grid, pkl_without_grid)
