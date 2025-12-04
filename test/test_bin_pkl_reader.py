"""
BIN + PKL 文件读取测试脚本
用于验证 tile_las1.py 生成的文件是否正确保存和组织
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

# ============================================================================
# 美化输出
# ============================================================================

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

def format_size(size_bytes: float) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def format_number(num: int) -> str:
    """格式化大数字（千分位分隔）"""
    return f"{num:,}"

# ============================================================================
# 读取函数
# ============================================================================

def read_bin_pkl(pkl_path: str, verbose: bool = True) -> dict:
    """
    读取 BIN + PKL 文件并验证数据完整性
    
    Args:
        pkl_path: PKL 文件路径
        verbose: 是否打印详细信息
        
    Returns:
        包含所有数据的字典
    """
    pkl_path = Path(pkl_path)
    bin_path = pkl_path.with_suffix('.bin')
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL 文件不存在: {pkl_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"BIN 文件不存在: {bin_path}")
    
    # 1. 读取 PKL 元数据
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # 2. 读取 BIN 数据
    dtype = metadata['dtype']
    struct_arr = np.fromfile(bin_path, dtype=dtype)
    
    if verbose:
        print(f"\n{Colors.BOLD}{'═'*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  📂 BIN + PKL 文件读取测试{Colors.RESET}")
        print(f"{Colors.BOLD}{'═'*70}{Colors.RESET}")
        
        print(f"\n  {Colors.BOLD}📄 文件信息{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} PKL: {pkl_path.name} ({format_size(pkl_path.stat().st_size)})")
        print(f"  {Colors.DIM}├─{Colors.RESET} BIN: {bin_path.name} ({format_size(bin_path.stat().st_size)})")
        print(f"  {Colors.DIM}└─{Colors.RESET} 原始LAS: {metadata.get('las_file', 'N/A')}")
        
        print(f"\n  {Colors.BOLD}📊 数据统计{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 总点数: {Colors.GREEN}{format_number(metadata['num_points'])}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 分块数: {Colors.GREEN}{metadata['num_segments']}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 字段数: {Colors.GREEN}{len(metadata['fields'])}{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 网格大小: {Colors.YELLOW}{metadata.get('grid_size', 'N/A')}m{Colors.RESET}")
        
        print(f"\n  {Colors.BOLD}📋 保存字段{Colors.RESET}")
        fields_str = ", ".join(metadata['fields'])
        print(f"  {Colors.DIM}└─{Colors.RESET} {Colors.CYAN}{fields_str}{Colors.RESET}")
        
        print(f"\n  {Colors.BOLD}🏷️ 类别分布{Colors.RESET}")
        if 'label_counts' in metadata and metadata['label_counts']:
            for label, count in sorted(metadata['label_counts'].items()):
                pct = count / metadata['num_points'] * 100
                bar_len = int(pct / 2)
                bar = '█' * bar_len + '░' * (50 - bar_len)
                print(f"  {Colors.DIM}├─{Colors.RESET} 类别 {label:2d}: {bar} {pct:5.1f}% ({format_number(count)})")
        else:
            print(f"  {Colors.DIM}└─{Colors.RESET} {Colors.YELLOW}无类别信息{Colors.RESET}")
    
    return {
        'metadata': metadata,
        'data': struct_arr,
        'bin_path': bin_path,
        'pkl_path': pkl_path
    }


def validate_segments(result: dict, verbose: bool = True) -> bool:
    """
    验证 segments 数据完整性
    
    Args:
        result: read_bin_pkl 返回的结果
        verbose: 是否打印详细信息
        
    Returns:
        是否验证通过
    """
    metadata = result['metadata']
    data = result['data']
    segments = metadata['segments']
    
    all_passed = True
    issues = []
    
    if verbose:
        print(f"\n  {Colors.BOLD}🔍 数据完整性验证{Colors.RESET}")
    
    # 1. 验证总点数
    total_points_in_segments = sum(seg['num_points'] for seg in segments)
    # 注意：overlap 模式下，总点数会大于原始点数（因为有重复）
    if not metadata.get('overlap', False):
        if total_points_in_segments != len(data):
            issues.append(f"分块点数总和 ({total_points_in_segments}) != 原始点数 ({len(data)})")
            all_passed = False
    
    if verbose:
        print(f"  {Colors.DIM}├─{Colors.RESET} 总点数验证: ", end="")
        if metadata.get('overlap', False):
            print(f"{Colors.YELLOW}跳过 (overlap模式){Colors.RESET}")
        else:
            print(f"{Colors.GREEN}[OK]{Colors.RESET}" if total_points_in_segments == len(data) else f"{Colors.RED}[FAIL]{Colors.RESET}")
    
    # 2. 验证索引范围
    invalid_indices = 0
    for seg in segments:
        indices = seg['indices']
        if len(indices) > 0:
            if indices.max() >= len(data) or indices.min() < 0:
                invalid_indices += 1
                all_passed = False
    
    if verbose:
        print(f"  {Colors.DIM}├─{Colors.RESET} 索引范围验证: ", end="")
        print(f"{Colors.GREEN}[OK]{Colors.RESET}" if invalid_indices == 0 else f"{Colors.RED}[FAIL] ({invalid_indices} 个无效){Colors.RESET}")
    
    # 3. 验证 sort_idx 和 voxel_counts
    voxel_mismatch = 0
    for seg in segments:
        sort_idx = seg['sort_idx']
        voxel_counts = seg['voxel_counts']
        if len(sort_idx) != seg['num_points']:
            voxel_mismatch += 1
            all_passed = False
        if voxel_counts.sum() != seg['num_points']:
            voxel_mismatch += 1
            all_passed = False
    
    if verbose:
        print(f"  {Colors.DIM}├─{Colors.RESET} 体素索引验证: ", end="")
        print(f"{Colors.GREEN}[OK]{Colors.RESET}" if voxel_mismatch == 0 else f"{Colors.RED}[FAIL] ({voxel_mismatch} 个不匹配){Colors.RESET}")
    
    # 4. 验证边界框
    bounds_valid = 0
    for seg in segments:
        bounds = seg['bounds']
        indices = seg['indices']
        if len(indices) > 0:
            x_vals = data['X'][indices]
            y_vals = data['Y'][indices]
            z_vals = data['Z'][indices]
            
            if (abs(x_vals.min() - bounds['x_min']) < 1e-6 and 
                abs(x_vals.max() - bounds['x_max']) < 1e-6 and
                abs(y_vals.min() - bounds['y_min']) < 1e-6 and
                abs(y_vals.max() - bounds['y_max']) < 1e-6 and
                abs(z_vals.min() - bounds['z_min']) < 1e-6 and
                abs(z_vals.max() - bounds['z_max']) < 1e-6):
                bounds_valid += 1
    
    if verbose:
        print(f"  {Colors.DIM}└─{Colors.RESET} 边界框验证: ", end="")
        print(f"{Colors.GREEN}[OK] ({bounds_valid}/{len(segments)} 通过){Colors.RESET}" if bounds_valid == len(segments) else f"{Colors.YELLOW}部分通过 ({bounds_valid}/{len(segments)}){Colors.RESET}")
    
    return all_passed


def show_segment_details(result: dict, segment_id: int = 0):
    """
    显示单个 segment 的详细信息
    
    Args:
        result: read_bin_pkl 返回的结果
        segment_id: 要显示的 segment ID
    """
    metadata = result['metadata']
    data = result['data']
    segments = metadata['segments']
    
    if segment_id >= len(segments):
        print(f"{Colors.RED}Error: segment_id {segment_id} 超出范围 (0-{len(segments)-1}){Colors.RESET}")
        return
    
    seg = segments[segment_id]
    indices = seg['indices']
    
    print(f"\n  {Colors.BOLD}📦 Segment #{segment_id} 详细信息{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 点数: {Colors.GREEN}{format_number(seg['num_points'])}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 体素数: {Colors.GREEN}{seg['num_voxels']}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 最大体素密度: {Colors.YELLOW}{seg['max_voxel_density']}{Colors.RESET}")
    
    bounds = seg['bounds']
    print(f"  {Colors.DIM}├─{Colors.RESET} 边界框:")
    print(f"  {Colors.DIM}│   ├─{Colors.RESET} X: [{bounds['x_min']:.2f}, {bounds['x_max']:.2f}]")
    print(f"  {Colors.DIM}│   ├─{Colors.RESET} Y: [{bounds['y_min']:.2f}, {bounds['y_max']:.2f}]")
    print(f"  {Colors.DIM}│   └─{Colors.RESET} Z: [{bounds['z_min']:.2f}, {bounds['z_max']:.2f}]")
    
    if 'label_counts' in seg and seg['label_counts']:
        print(f"  {Colors.DIM}├─{Colors.RESET} 类别分布:")
        for label, count in sorted(seg['label_counts'].items()):
            pct = count / seg['num_points'] * 100
            print(f"  {Colors.DIM}│   ├─{Colors.RESET} 类别 {label}: {pct:.1f}% ({format_number(count)})")
    
    # 显示前几个点的数据
    if len(indices) > 0:
        print(f"  {Colors.DIM}└─{Colors.RESET} 前5个点示例:")
        for i, idx in enumerate(indices[:5]):
            x, y, z = data['X'][idx], data['Y'][idx], data['Z'][idx]
            print(f"      [{i}] X={x:.4f}, Y={y:.4f}, Z={z:.4f}")


def test_data_access(result: dict, segment_id: int = 0):
    """
    测试数据访问流程（模拟 Dataset 的 __getitem__）
    
    Args:
        result: read_bin_pkl 返回的结果
        segment_id: 要测试的 segment ID
    """
    metadata = result['metadata']
    data = result['data']
    segments = metadata['segments']
    
    if segment_id >= len(segments):
        print(f"{Colors.RED}Error: segment_id {segment_id} 超出范围{Colors.RESET}")
        return
    
    seg = segments[segment_id]
    
    print(f"\n  {Colors.BOLD}🧪 数据访问测试 (Segment #{segment_id}){Colors.RESET}")
    
    # 模拟 Dataset.__getitem__ 的流程
    import time
    
    # 1. 获取索引
    t0 = time.time()
    indices = seg['indices']
    sort_idx = seg['sort_idx']
    voxel_counts = seg['voxel_counts']
    t1 = time.time()
    print(f"  {Colors.DIM}├─{Colors.RESET} 获取索引: {Colors.GREEN}{(t1-t0)*1000:.2f}ms{Colors.RESET}")
    
    # 2. 提取点云数据
    t0 = time.time()
    points = np.column_stack([
        data['X'][indices],
        data['Y'][indices],
        data['Z'][indices]
    ])
    t1 = time.time()
    print(f"  {Colors.DIM}├─{Colors.RESET} 提取坐标: {Colors.GREEN}{(t1-t0)*1000:.2f}ms{Colors.RESET} → shape={points.shape}")
    
    # 3. 应用排序
    t0 = time.time()
    sorted_points = points[sort_idx]
    t1 = time.time()
    print(f"  {Colors.DIM}├─{Colors.RESET} 应用排序: {Colors.GREEN}{(t1-t0)*1000:.2f}ms{Colors.RESET}")
    
    # 4. 归一化
    t0 = time.time()
    local_min = seg['local_min']
    normalized_points = sorted_points - local_min
    t1 = time.time()
    print(f"  {Colors.DIM}├─{Colors.RESET} 局部归一化: {Colors.GREEN}{(t1-t0)*1000:.2f}ms{Colors.RESET}")
    
    # 5. 计算体素中心偏移（用于 Grid Sampling）
    t0 = time.time()
    grid_size = metadata.get('grid_size', 0.5)
    voxel_indices = np.repeat(np.arange(len(voxel_counts)), voxel_counts)
    t1 = time.time()
    print(f"  {Colors.DIM}└─{Colors.RESET} 体素索引展开: {Colors.GREEN}{(t1-t0)*1000:.2f}ms{Colors.RESET} → {len(voxel_indices)} 点")
    
    print(f"\n  {Colors.GREEN}[OK] 数据访问测试通过{Colors.RESET}")


def run_full_test(data_dir: str):
    """
    运行完整测试
    
    Args:
        data_dir: 包含 bin+pkl 文件的目录
    """
    data_dir = Path(data_dir)
    pkl_files = sorted(data_dir.glob('*.pkl'))
    
    if not pkl_files:
        print(f"{Colors.RED}未找到 PKL 文件: {data_dir}{Colors.RESET}")
        return
    
    print(f"\n{Colors.BOLD}{'═'*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🧪 批量测试 - 共 {len(pkl_files)} 个文件{Colors.RESET}")
    print(f"{Colors.BOLD}{'═'*70}{Colors.RESET}")
    
    all_passed = True
    for pkl_file in pkl_files:
        try:
            result = read_bin_pkl(pkl_file, verbose=False)
            passed = validate_segments(result, verbose=False)
            
            status = f"{Colors.GREEN}[OK]{Colors.RESET}" if passed else f"{Colors.RED}[FAIL]{Colors.RESET}"
            print(f"  {status} {pkl_file.name}: {format_number(result['metadata']['num_points'])} 点, {result['metadata']['num_segments']} 块")
            
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  {Colors.RED}[FAIL] {pkl_file.name}: {e}{Colors.RESET}")
            all_passed = False
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    if all_passed:
        print(f"{Colors.BOLD}{Colors.GREEN}  [OK] 所有测试通过!{Colors.RESET}")
    else:
        print(f"{Colors.BOLD}{Colors.RED}  [FAIL] 部分测试失败{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BIN + PKL 文件读取测试')
    parser.add_argument('path', type=str, help='PKL 文件路径或包含 bin+pkl 的目录')
    parser.add_argument('--segment', '-s', type=int, default=0, help='要查看的 segment ID')
    parser.add_argument('--test-access', '-t', action='store_true', help='测试数据访问流程')
    parser.add_argument('--batch', '-b', action='store_true', help='批量测试目录中的所有文件')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if args.batch or path.is_dir():
        # 批量测试
        run_full_test(path if path.is_dir() else path.parent)
    else:
        # 单文件测试
        if not path.suffix == '.pkl':
            path = path.with_suffix('.pkl')
        
        result = read_bin_pkl(path, verbose=True)
        validate_segments(result, verbose=True)
        show_segment_details(result, segment_id=args.segment)
        
        if args.test_access:
            test_data_access(result, segment_id=args.segment)
    
    print()
