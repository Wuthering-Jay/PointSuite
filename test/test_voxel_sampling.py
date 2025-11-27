"""
测试 tile_las1.py 生成的 pkl 文件索引覆盖度
以及 bin_to_las1.py 中的 voxel_modulo_sample 方法采样覆盖度

测试内容：
1. PKL 索引完整性测试：所有 segment 的 indices 是否覆盖了所有点，有无遗漏
2. 体素采样覆盖度测试：voxel_modulo_sample 能否覆盖所有点
3. 重复采样统计：多少点被重复采样，平均重复次数
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """ANSI 颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def format_number(num: int) -> str:
    """格式化大数字（千分位分隔）"""
    return f"{num:,}"


def format_percent(value: float) -> str:
    """格式化百分比"""
    return f"{value:.2f}%"


# ============================================================================
# 测试1: PKL 索引完整性测试
# ============================================================================

def test_pkl_index_coverage(pkl_path: str) -> Dict:
    """
    测试 PKL 文件中的索引是否覆盖了所有点
    
    Args:
        pkl_path: pkl 文件路径
        
    Returns:
        测试结果字典
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  📋 测试1: PKL 索引完整性测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    num_points = metadata['num_points']
    segments = metadata['segments']
    
    print(f"  {Colors.DIM}├─{Colors.RESET} 文件: {Colors.CYAN}{Path(pkl_path).name}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 总点数: {Colors.CYAN}{format_number(num_points)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} Segment 数量: {Colors.CYAN}{len(segments)}{Colors.RESET}")
    
    # 收集所有索引
    all_indices = []
    segment_sizes = []
    
    for seg_id, seg_info in enumerate(segments):
        indices = seg_info['indices']
        all_indices.extend(indices.tolist())
        segment_sizes.append(len(indices))
    
    all_indices = np.array(all_indices)
    
    # 统计分析
    total_indexed = len(all_indices)
    unique_indices = np.unique(all_indices)
    num_unique = len(unique_indices)
    
    # 检查是否覆盖所有点
    expected_indices = set(range(num_points))
    actual_indices = set(unique_indices)
    
    missing_indices = expected_indices - actual_indices
    extra_indices = actual_indices - expected_indices
    duplicate_count = total_indexed - num_unique
    
    # 计算索引出现次数
    index_counter = Counter(all_indices)
    max_repeat = max(index_counter.values()) if index_counter else 0
    
    # 输出结果
    print(f"\n  {Colors.BOLD}📊 索引统计:{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 总索引数: {Colors.CYAN}{format_number(total_indexed)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 唯一索引数: {Colors.CYAN}{format_number(num_unique)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 期望索引数: {Colors.CYAN}{format_number(num_points)}{Colors.RESET}")
    
    print(f"\n  {Colors.BOLD}🔍 覆盖度分析:{Colors.RESET}")
    coverage = num_unique / num_points * 100 if num_points > 0 else 0
    
    if len(missing_indices) == 0:
        print(f"  {Colors.DIM}├─{Colors.RESET} 覆盖率: {Colors.GREEN}100% ✓{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 遗漏点数: {Colors.GREEN}0 ✓{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}├─{Colors.RESET} 覆盖率: {Colors.RED}{format_percent(coverage)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 遗漏点数: {Colors.RED}{format_number(len(missing_indices))}{Colors.RESET}")
        if len(missing_indices) <= 10:
            print(f"  {Colors.DIM}│{Colors.RESET}   遗漏索引: {list(missing_indices)}")
    
    if len(extra_indices) > 0:
        print(f"  {Colors.DIM}├─{Colors.RESET} 超出范围索引: {Colors.RED}{format_number(len(extra_indices))}{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}├─{Colors.RESET} 超出范围索引: {Colors.GREEN}0 ✓{Colors.RESET}")
    
    print(f"\n  {Colors.BOLD}📈 重复统计 (跨 segment):{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 重复索引总数: {Colors.YELLOW}{format_number(duplicate_count)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 最大重复次数: {Colors.YELLOW}{max_repeat}{Colors.RESET}")
    
    # Segment 大小统计
    print(f"\n  {Colors.BOLD}📦 Segment 大小统计:{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 最小: {Colors.CYAN}{format_number(min(segment_sizes))}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 最大: {Colors.CYAN}{format_number(max(segment_sizes))}{Colors.RESET}")
    print(f"  {Colors.DIM}└─{Colors.RESET} 平均: {Colors.CYAN}{format_number(int(np.mean(segment_sizes)))}{Colors.RESET}")
    
    result = {
        'num_points': num_points,
        'num_segments': len(segments),
        'total_indexed': total_indexed,
        'num_unique': num_unique,
        'coverage': coverage,
        'missing_count': len(missing_indices),
        'extra_count': len(extra_indices),
        'duplicate_count': duplicate_count,
        'max_repeat': max_repeat,
        'passed': len(missing_indices) == 0 and len(extra_indices) == 0
    }
    
    return result


# ============================================================================
# 测试2: Voxel Modulo Sample 覆盖度测试
# ============================================================================

def voxel_modulo_sample_indices(segment_info: dict, 
                                 loop_idx: int,
                                 points_per_loop: int = 1) -> np.ndarray:
    """
    模拟 voxel_modulo_sample，返回采样的局部索引
    
    Args:
        segment_info: segment 元数据
        loop_idx: 当前采样轮次
        points_per_loop: 每轮每体素采样点数
        
    Returns:
        采样的局部索引数组
    """
    sort_idx = segment_info.get('sort_idx', None)
    voxel_counts = segment_info.get('voxel_counts', None)
    
    if sort_idx is None or voxel_counts is None:
        # 没有体素化信息，返回所有点
        return np.arange(len(segment_info['indices']))
    
    cumsum = np.cumsum(np.insert(voxel_counts, 0, 0))
    sampled_local_indices = []
    
    for voxel_idx in range(len(voxel_counts)):
        voxel_count = voxel_counts[voxel_idx]
        start_pos = cumsum[voxel_idx]
        
        for p in range(points_per_loop):
            logical_idx = loop_idx * points_per_loop + p
            local_idx = logical_idx % voxel_count
            sampled_local_indices.append(sort_idx[start_pos + local_idx])
    
    return np.array(sampled_local_indices, dtype=np.int32)


def test_voxel_sample_coverage(pkl_path: str, 
                                max_loops: Optional[int] = None,
                                segment_id: Optional[int] = None) -> Dict:
    """
    测试 voxel_modulo_sample 方法的采样覆盖度
    
    Args:
        pkl_path: pkl 文件路径
        max_loops: 最大采样轮次 (None 表示按最大体素点数)
        segment_id: 指定测试的 segment (None 表示测试所有)
        
    Returns:
        测试结果字典
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🔄 测试2: Voxel Modulo Sample 覆盖度测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    segments = metadata['segments']
    grid_size = metadata.get('grid_size', None)
    
    max_loops_str = str(max_loops) if max_loops is not None else "自动 (按最大体素点数)"
    print(f"  {Colors.DIM}├─{Colors.RESET} 文件: {Colors.CYAN}{Path(pkl_path).name}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} Grid Size: {Colors.CYAN}{grid_size or 'N/A'}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} Max Loops: {Colors.CYAN}{max_loops_str}{Colors.RESET}")
    
    # 确定要测试的 segments
    if segment_id is not None:
        test_segments = [segment_id]
    else:
        test_segments = list(range(len(segments)))
    
    print(f"  {Colors.DIM}└─{Colors.RESET} 测试 Segment 数: {Colors.CYAN}{len(test_segments)}{Colors.RESET}")
    
    # 汇总统计
    total_points = 0
    total_sampled = 0
    total_unique = 0
    all_repeat_counts = []
    
    segment_results = []
    
    for seg_id in test_segments:
        seg_info = segments[seg_id]
        indices = seg_info['indices']
        num_points = len(indices)
        total_points += num_points
        
        # 检查是否有体素化信息
        if 'sort_idx' not in seg_info or 'voxel_counts' not in seg_info:
            # 没有体素化信息，全量输出
            segment_results.append({
                'seg_id': seg_id,
                'num_points': num_points,
                'has_voxel': False,
                'coverage': 100.0,
                'unique_count': num_points,
                'sampled_count': num_points,
                'repeat_count': 0
            })
            total_sampled += num_points
            total_unique += num_points
            continue
        
        voxel_counts = seg_info['voxel_counts']
        max_voxel_count = int(voxel_counts.max()) if len(voxel_counts) > 0 else 1
        
        # 计算实际轮数和每轮采样点数
        if max_loops is None:
            actual_loops = max_voxel_count
            points_per_loop = 1
        elif max_voxel_count <= max_loops:
            actual_loops = max_voxel_count
            points_per_loop = 1
        else:
            actual_loops = max_loops
            points_per_loop = int(np.ceil(max_voxel_count / max_loops))
        
        # 收集所有采样的局部索引
        all_sampled = []
        for loop_idx in range(actual_loops):
            sampled = voxel_modulo_sample_indices(seg_info, loop_idx, points_per_loop)
            all_sampled.extend(sampled.tolist())
        
        all_sampled = np.array(all_sampled)
        unique_sampled = np.unique(all_sampled)
        
        # 统计每个点被采样的次数
        sample_counter = Counter(all_sampled)
        repeat_counts = list(sample_counter.values())
        all_repeat_counts.extend(repeat_counts)
        
        coverage = len(unique_sampled) / num_points * 100 if num_points > 0 else 0
        
        segment_results.append({
            'seg_id': seg_id,
            'num_points': num_points,
            'has_voxel': True,
            'num_voxels': len(voxel_counts),
            'max_voxel_count': max_voxel_count,
            'actual_loops': actual_loops,
            'points_per_loop': points_per_loop,
            'coverage': coverage,
            'unique_count': len(unique_sampled),
            'sampled_count': len(all_sampled),
            'repeat_count': len(all_sampled) - len(unique_sampled)
        })
        
        total_sampled += len(all_sampled)
        total_unique += len(unique_sampled)
    
    # 输出汇总结果
    print(f"\n  {Colors.BOLD}📊 汇总统计:{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 总点数: {Colors.CYAN}{format_number(total_points)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 总采样数: {Colors.CYAN}{format_number(total_sampled)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 唯一采样数: {Colors.CYAN}{format_number(total_unique)}{Colors.RESET}")
    
    overall_coverage = total_unique / total_points * 100 if total_points > 0 else 0
    if overall_coverage >= 99.99:
        print(f"  {Colors.DIM}├─{Colors.RESET} 总覆盖率: {Colors.GREEN}{format_percent(overall_coverage)} ✓{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}├─{Colors.RESET} 总覆盖率: {Colors.YELLOW}{format_percent(overall_coverage)}{Colors.RESET}")
    
    # 重复采样统计
    repeat_total = total_sampled - total_unique
    print(f"\n  {Colors.BOLD}🔁 重复采样统计:{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 重复采样总次数: {Colors.YELLOW}{format_number(repeat_total)}{Colors.RESET}")
    
    if all_repeat_counts:
        repeat_counter = Counter(all_repeat_counts)
        print(f"  {Colors.DIM}├─{Colors.RESET} 平均采样次数: {Colors.YELLOW}{np.mean(all_repeat_counts):.2f}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 最大采样次数: {Colors.YELLOW}{max(all_repeat_counts)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 最小采样次数: {Colors.YELLOW}{min(all_repeat_counts)}{Colors.RESET}")
        
        # 采样次数分布
        print(f"\n  {Colors.BOLD}📈 采样次数分布:{Colors.RESET}")
        sorted_counts = sorted(repeat_counter.items())
        for count, num_points in sorted_counts[:10]:  # 只显示前10个
            pct = num_points / len(all_repeat_counts) * 100
            bar_len = int(pct / 2)
            bar = '█' * bar_len
            print(f"  {Colors.DIM}│{Colors.RESET}   采样 {count} 次: {format_number(num_points)} 点 ({format_percent(pct)}) {Colors.CYAN}{bar}{Colors.RESET}")
        
        if len(sorted_counts) > 10:
            print(f"  {Colors.DIM}│{Colors.RESET}   ... 还有 {len(sorted_counts) - 10} 种采样次数")
    
    # 显示部分 segment 详情
    print(f"\n  {Colors.BOLD}📦 Segment 详情 (前5个):{Colors.RESET}")
    for res in segment_results[:5]:
        if res['has_voxel']:
            status = f"{Colors.GREEN}✓{Colors.RESET}" if res['coverage'] >= 99.99 else f"{Colors.YELLOW}!{Colors.RESET}"
            print(f"  {Colors.DIM}├─{Colors.RESET} Seg {res['seg_id']:4d}: "
                  f"{format_number(res['num_points']):>10} 点, "
                  f"{res['num_voxels']:>6} 体素, "
                  f"max={res['max_voxel_count']:>3}, "
                  f"loops={res['actual_loops']:>3}, "
                  f"ppl={res['points_per_loop']}, "
                  f"覆盖={format_percent(res['coverage']):>7} {status}")
        else:
            print(f"  {Colors.DIM}├─{Colors.RESET} Seg {res['seg_id']:4d}: "
                  f"{format_number(res['num_points']):>10} 点, "
                  f"无体素化, 全量输出")
    
    if len(segment_results) > 5:
        print(f"  {Colors.DIM}└─{Colors.RESET} ... 还有 {len(segment_results) - 5} 个 segment")
    
    result = {
        'total_points': total_points,
        'total_sampled': total_sampled,
        'total_unique': total_unique,
        'overall_coverage': overall_coverage,
        'repeat_total': repeat_total,
        'avg_sample_count': np.mean(all_repeat_counts) if all_repeat_counts else 1,
        'max_sample_count': max(all_repeat_counts) if all_repeat_counts else 1,
        'segment_results': segment_results,
        'passed': overall_coverage >= 99.99
    }
    
    return result


# ============================================================================
# 测试3: 验证采样是否遍历了所有点（严格测试）
# ============================================================================

def test_all_points_sampled(pkl_path: str, 
                             max_loops: Optional[int] = None,
                             segment_id: int = 0) -> Dict:
    """
    严格验证单个 segment 的采样是否遍历了所有点
    
    Args:
        pkl_path: pkl 文件路径
        max_loops: 最大采样轮次
        segment_id: 要测试的 segment ID
        
    Returns:
        测试结果字典
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🎯 测试3: 严格点覆盖验证 (Segment {segment_id}){Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    segments = metadata['segments']
    
    if segment_id >= len(segments):
        print(f"  {Colors.RED}❌ Segment {segment_id} 不存在 (共 {len(segments)} 个){Colors.RESET}")
        return {'passed': False, 'error': 'segment not found'}
    
    seg_info = segments[segment_id]
    indices = seg_info['indices']
    num_points = len(indices)
    
    print(f"  {Colors.DIM}├─{Colors.RESET} Segment 点数: {Colors.CYAN}{format_number(num_points)}{Colors.RESET}")
    
    if 'sort_idx' not in seg_info or 'voxel_counts' not in seg_info:
        print(f"  {Colors.DIM}└─{Colors.RESET} 无体素化信息，全量输出，覆盖 100%")
        return {'passed': True, 'coverage': 100.0}
    
    voxel_counts = seg_info['voxel_counts']
    sort_idx = seg_info['sort_idx']
    max_voxel_count = int(voxel_counts.max())
    
    print(f"  {Colors.DIM}├─{Colors.RESET} 体素数量: {Colors.CYAN}{len(voxel_counts)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 最大体素点数: {Colors.CYAN}{max_voxel_count}{Colors.RESET}")
    
    # 计算采样参数
    if max_loops is None:
        actual_loops = max_voxel_count
        points_per_loop = 1
    elif max_voxel_count <= max_loops:
        actual_loops = max_voxel_count
        points_per_loop = 1
    else:
        actual_loops = max_loops
        points_per_loop = int(np.ceil(max_voxel_count / max_loops))
    
    print(f"  {Colors.DIM}├─{Colors.RESET} 实际轮数: {Colors.CYAN}{actual_loops}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 每轮采样点数: {Colors.CYAN}{points_per_loop}{Colors.RESET}")
    
    # 模拟完整采样过程
    all_sampled = []
    for loop_idx in range(actual_loops):
        sampled = voxel_modulo_sample_indices(seg_info, loop_idx, points_per_loop)
        all_sampled.extend(sampled.tolist())
    
    all_sampled = np.array(all_sampled)
    unique_sampled = set(all_sampled)
    expected_points = set(range(num_points))
    
    # 检查覆盖
    missing = expected_points - unique_sampled
    extra = unique_sampled - expected_points
    
    print(f"\n  {Colors.BOLD}🔍 覆盖分析:{Colors.RESET}")
    
    if len(missing) == 0:
        print(f"  {Colors.DIM}├─{Colors.RESET} 覆盖状态: {Colors.GREEN}所有点都被采样 ✓{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}├─{Colors.RESET} 覆盖状态: {Colors.RED}有 {len(missing)} 个点未被采样{Colors.RESET}")
        if len(missing) <= 20:
            print(f"  {Colors.DIM}│{Colors.RESET}   未采样点: {sorted(missing)}")
    
    if len(extra) > 0:
        print(f"  {Colors.DIM}├─{Colors.RESET} 异常: {Colors.RED}有 {len(extra)} 个超出范围的索引{Colors.RESET}")
    
    # 详细分析每个体素的采样情况
    print(f"\n  {Colors.BOLD}📊 体素采样分析:{Colors.RESET}")
    
    cumsum = np.cumsum(np.insert(voxel_counts, 0, 0))
    
    voxel_stats = []
    for voxel_idx in range(min(5, len(voxel_counts))):  # 只分析前5个体素
        voxel_count = voxel_counts[voxel_idx]
        start_pos = cumsum[voxel_idx]
        end_pos = cumsum[voxel_idx + 1]
        
        # 该体素内的点索引
        voxel_point_indices = sort_idx[start_pos:end_pos]
        
        # 在所有采样中，该体素的点被采样的次数
        voxel_sampled = [idx for idx in all_sampled if idx in voxel_point_indices]
        voxel_unique = len(set(voxel_sampled))
        
        print(f"  {Colors.DIM}│{Colors.RESET}   体素 {voxel_idx}: {voxel_count} 点, "
              f"采样 {len(voxel_sampled)} 次, "
              f"唯一 {voxel_unique} 个, "
              f"覆盖 {format_percent(voxel_unique/voxel_count*100)}")
    
    if len(voxel_counts) > 5:
        print(f"  {Colors.DIM}│{Colors.RESET}   ... 还有 {len(voxel_counts) - 5} 个体素")
    
    coverage = len(unique_sampled) / num_points * 100
    
    result = {
        'num_points': num_points,
        'num_voxels': len(voxel_counts),
        'max_voxel_count': max_voxel_count,
        'actual_loops': actual_loops,
        'points_per_loop': points_per_loop,
        'total_sampled': len(all_sampled),
        'unique_sampled': len(unique_sampled),
        'missing_count': len(missing),
        'coverage': coverage,
        'passed': len(missing) == 0
    }
    
    return result


# ============================================================================
# 主测试入口
# ============================================================================

def run_all_tests(pkl_path: str, max_loops: Optional[int] = None):
    """
    运行所有测试
    
    Args:
        pkl_path: pkl 文件路径
        max_loops: 体素模式的最大采样轮次
    """
    print(f"\n{Colors.BOLD}{'#'*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}  🧪 Voxel Sampling 测试套件{Colors.RESET}")
    print(f"{Colors.BOLD}{'#'*70}{Colors.RESET}")
    print(f"  测试文件: {pkl_path}")
    print(f"  Max Loops: {max_loops if max_loops else '自动'}")
    
    results = {}
    
    # 测试1: PKL 索引完整性
    results['pkl_coverage'] = test_pkl_index_coverage(pkl_path)
    
    # 测试2: Voxel 采样覆盖度
    results['voxel_coverage'] = test_voxel_sample_coverage(pkl_path, max_loops)
    
    # 测试3: 严格点覆盖验证（第一个 segment）
    results['strict_coverage'] = test_all_points_sampled(pkl_path, max_loops, segment_id=0)
    
    # 汇总
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  📋 测试结果汇总{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    all_passed = True
    for name, result in results.items():
        passed = result.get('passed', False)
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        print(f"  {Colors.DIM}├─{Colors.RESET} {name}: {status}")
        all_passed = all_passed and passed
    
    print(f"\n  {Colors.BOLD}最终结果: ", end="")
    if all_passed:
        print(f"{Colors.GREEN}所有测试通过 ✓{Colors.RESET}")
    else:
        print(f"{Colors.RED}部分测试失败{Colors.RESET}")
    print()
    
    return results


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试 Voxel Sampling 覆盖度')
    parser.add_argument('--pkl', type=str, required=False,
                        help='PKL 文件路径')
    parser.add_argument('--max_loops', type=int, default=None,
                        help='最大采样轮次 (默认: 自动)')
    parser.add_argument('--segment', type=int, default=None,
                        help='指定测试的 segment ID')
    
    args = parser.parse_args()
    
    # 如果没有指定 pkl 文件，使用默认测试路径
    if args.pkl:
        pkl_path = args.pkl
    else:
        # 默认测试路径
        default_path = r"E:\data\DALES\dales_las\bin\train_logical\5080_54435.pkl"
        if Path(default_path).exists():
            pkl_path = default_path
        else:
            print(f"{Colors.RED}请指定 --pkl 参数{Colors.RESET}")
            print("用法: python test_voxel_sampling.py --pkl <pkl文件路径> [--max_loops N]")
            sys.exit(1)
    
    if not Path(pkl_path).exists():
        print(f"{Colors.RED}文件不存在: {pkl_path}{Colors.RESET}")
        sys.exit(1)
    
    # 运行测试
    if args.segment is not None:
        # 只测试指定的 segment
        test_all_points_sampled(pkl_path, args.max_loops, args.segment)
    else:
        # 运行所有测试
        run_all_tests(pkl_path, args.max_loops)
