"""
检查pkl文件中各个分块(segment)是否完全相同或有差异
"""
import pickle
import numpy as np
from pathlib import Path
from collections import Counter


def check_segment_uniqueness(pkl_path):
    """
    检查各个segment之间是否有完全相同的（无意义重复）
    
    Args:
        pkl_path: pkl文件路径
    """
    pkl_path = Path(pkl_path)
    
    print("="*70)
    print(f"检查文件: {pkl_path.name}")
    print("="*70)
    
    # 加载pkl文件
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    total_points = metadata['num_points']
    num_segments = metadata['num_segments']
    grid_size = metadata.get('grid_size', None)
    
    print(f"\n📊 基本信息:")
    print(f"  - 总点数: {total_points:,}")
    print(f"  - 分块数: {num_segments:,}")
    print(f"  - Grid Size: {grid_size if grid_size else 'N/A (未使用grid sampling)'}")
    
    # 收集所有分块的索引
    segments_list = []
    segment_sizes = []
    
    print(f"\n📦 分块详情:")
    for i, segment_info in enumerate(metadata['segments']):
        indices = segment_info['indices']
        segments_list.append(indices)
        segment_sizes.append(len(indices))
        
        if i < 5:  # 只显示前5个分块的详细信息
            print(f"  Segment {i}: {len(indices):,} 点 "
                  f"[索引范围: {indices.min()}-{indices.max()}]")
        elif i == 5:
            print(f"  ... (省略中间 {num_segments - 10} 个分块)")
        elif i >= num_segments - 5:
            print(f"  Segment {i}: {len(indices):,} 点 "
                  f"[索引范围: {indices.min()}-{indices.max()}]")
    
    print(f"\n📈 分块大小统计:")
    segment_sizes = np.array(segment_sizes)
    print(f"  - 最小: {segment_sizes.min():,} 点")
    print(f"  - 最大: {segment_sizes.max():,} 点")
    print(f"  - 平均: {segment_sizes.mean():.1f} 点")
    print(f"  - 中位数: {np.median(segment_sizes):.0f} 点")
    print(f"  - 标准差: {segment_sizes.std():.1f}")
    
    # ==================== 关键检查：segment之间是否完全相同 ====================
    print(f"\n🔍 Segment唯一性检查（检查是否有完全相同的segment）:")
    
    # 将每个segment转换为可哈希的形式（排序后的tuple）
    segment_hashes = []
    for i, seg in enumerate(segments_list):
        # 排序后转为tuple，方便比较
        sorted_seg = tuple(sorted(seg.tolist()))
        segment_hashes.append(sorted_seg)
    
    # 统计每个segment出现的次数
    hash_counter = Counter(segment_hashes)
    duplicate_segments = {h: count for h, count in hash_counter.items() if count > 1}
    
    print(f"  - 总segment数: {num_segments:,}")
    print(f"  - 唯一segment数: {len(hash_counter):,}")
    print(f"  - 完全相同的segment组数: {len(duplicate_segments)}")
    
    if len(duplicate_segments) == 0:
        print(f"  ✅ 所有segment都不相同，没有无意义的重复！")
    else:
        print(f"  ⚠️ 发现完全相同的segment！")
        print(f"\n  重复segment详情（前10组）:")
        
        # 找出哪些segment是重复的
        for idx, (seg_hash, count) in enumerate(list(duplicate_segments.items())[:10]):
            # 找到所有具有相同hash的segment索引
            duplicate_indices = [i for i, h in enumerate(segment_hashes) if h == seg_hash]
            seg_size = len(seg_hash)
            
            print(f"    组{idx+1}: {count}个相同segment (每个{seg_size:,}点)")
            print(f"      Segment IDs: {duplicate_indices}")
            
            # 显示前几个点索引
            first_few = list(seg_hash[:5])
            print(f"      前5个点索引: {first_few}")
    
    # ==================== 补充检查：segment之间的相似度 ====================
    print(f"\n🔬 Segment相似度分析（检查重叠程度）:")
    
    # 计算相邻segment之间的交集比例（采样检查，避免过慢）
    sample_size = min(50, num_segments - 1)
    if num_segments > 1:
        overlap_ratios = []
        for i in range(sample_size):
            seg1 = set(segments_list[i].tolist())
            seg2 = set(segments_list[i + 1].tolist())
            intersection = len(seg1 & seg2)
            union = len(seg1 | seg2)
            overlap_ratio = intersection / union if union > 0 else 0
            overlap_ratios.append(overlap_ratio)
        
        avg_overlap = np.mean(overlap_ratios)
        max_overlap = np.max(overlap_ratios)
        
        print(f"  - 采样检查: 前{sample_size}对相邻segment")
        print(f"  - 平均重叠率: {avg_overlap*100:.2f}%")
        print(f"  - 最大重叠率: {max_overlap*100:.2f}%")
        
        if avg_overlap < 0.1:
            print(f"  ✅ 相邻segment重叠很少，说明是不同的分块")
        elif avg_overlap > 0.8:
            print(f"  ⚠️ 相邻segment重叠很多，可能存在问题")
        else:
            print(f"  ℹ️ 相邻segment有一定重叠（可能是grid sampling导致）")
    
    # ==================== 原有的检查：点索引覆盖情况 ====================
    print(f"\n📊 点索引使用统计:")
    all_indices = []
    for seg in segments_list:
        all_indices.extend(seg.tolist())
    
    all_indices_array = np.array(all_indices)
    unique_indices = np.unique(all_indices_array)
    
    print(f"  - 所有索引总数: {len(all_indices_array):,}")
    print(f"  - 唯一索引数量: {len(unique_indices):,}")
    print(f"  - 重复使用次数: {len(all_indices_array) - len(unique_indices):,}")
    
    if len(all_indices_array) > len(unique_indices):
        # 统计每个点被使用的次数
        counter = Counter(all_indices)
        reuse_counts = list(counter.values())
        avg_reuse = np.mean(reuse_counts)
        max_reuse = np.max(reuse_counts)
        
        print(f"  - 平均每个点被使用: {avg_reuse:.2f} 次")
        print(f"  - 最多被使用: {max_reuse} 次")
        print(f"  ℹ️ 这是Grid Sampling Test模式的正常行为")
    
    # 检查覆盖率
    print(f"\n🎯 索引覆盖率检查:")
    expected_indices = set(range(total_points))
    actual_indices = set(all_indices)
    
    coverage = len(actual_indices) / total_points * 100
    print(f"  - 应该覆盖: {total_points:,} 个索引 (0-{total_points-1})")
    print(f"  - 实际覆盖: {len(actual_indices):,} 个索引")
    print(f"  - 覆盖率: {coverage:.2f}%")
    
    if actual_indices == expected_indices:
        print(f"  ✅ 完全覆盖，所有点都在某个分块中！")
    else:
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices
        
        if missing:
            print(f"  ⚠️ 有 {len(missing)} 个点未被任何分块包含")
            if len(missing) <= 10:
                print(f"    缺失索引: {sorted(list(missing))}")
            else:
                print(f"    缺失索引示例: {sorted(list(missing))[:10]} ...")
        
        if extra:
            print(f"  ⚠️ 有 {len(extra)} 个索引超出范围")
            if len(extra) <= 10:
                print(f"    超出索引: {sorted(list(extra))}")
            else:
                print(f"    超出索引示例: {sorted(list(extra))[:10]} ...")
    
    # 如果使用了grid sampling，检查采样率
    if grid_size:
        print(f"\n🔬 Grid Sampling 统计:")
        total_sampled_points = sum(segment_sizes)
        sampling_ratio = total_sampled_points / total_points
        print(f"  - 原始总点数: {total_points:,}")
        print(f"  - 采样后总点数: {total_sampled_points:,}")
        print(f"  - 采样倍率: {sampling_ratio:.2f}x")
        
        if sampling_ratio > 1:
            print(f"  ℹ️ 采样倍率>1表示test模式产生了多次采样")
        elif sampling_ratio < 1:
            print(f"  ℹ️ 采样倍率<1表示进行了下采样")
        else:
            print(f"  ℹ️ 采样倍率=1表示每个点只采样一次")
    
    print(f"\n" + "="*70)


def check_multiple_pkl_files(pkl_dir):
    """
    批量检查目录下所有pkl文件
    
    Args:
        pkl_dir: 包含pkl文件的目录
    """
    pkl_dir = Path(pkl_dir)
    pkl_files = list(pkl_dir.glob('*.pkl'))
    
    if not pkl_files:
        print(f"❌ 目录 {pkl_dir} 中没有找到pkl文件")
        return
    
    print(f"找到 {len(pkl_files)} 个pkl文件\n")
    
    for pkl_file in pkl_files:
        check_segment_uniqueness(pkl_file)
        print("\n")


if __name__ == "__main__":
    # 示例1: 检查单个pkl文件
    # pkl_file = r"E:\data\云南遥感中心\第一批\bin\train_with_gridsample\processed_02.pkl"
    
    # if Path(pkl_file).exists():
    #     check_segment_uniqueness(pkl_file)
    # else:
    #     print(f"文件不存在: {pkl_file}")
    #     print("\n请修改路径为你的实际pkl文件路径")
    
    # 示例2: 批量检查目录下所有pkl文件
    pkl_dir = r"E:\data\云南遥感中心\第一批\bin\train_with_gridsample"
    check_multiple_pkl_files(pkl_dir)
