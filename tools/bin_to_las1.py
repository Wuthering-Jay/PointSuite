"""
bin+pkl 逻辑索引格式转 LAS 文件工具

对应 tile_las.py 的逻辑索引分块方式，将分块数据转换回 LAS 格式便于在专业软件中查看。

支持两种模式：
1. 全量模式 (full): 直接按 window size 分块转换，包含所有原始点
2. 网格采样模式 (grid): 利用 grid_size 网格化索引进行模运算采样
   - 支持 max_loops 限制总采样次数
   - 支持包含重复点的多轮采样（模拟训练时的数据增强效果）

用途：
- 检查分块效果是否正确
- 验证网格采样逻辑
- 在 CloudCompare 等软件中可视化查看
"""

import os
import numpy as np
import pickle
import laspy
import time
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# ============================================================================
# 美化输出辅助类 (复用 tile_las1.py 的风格)
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


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds/60:.2f}min"


def format_number(num: int) -> str:
    """格式化大数字（千分位分隔）"""
    return f"{num:,}"


# ============================================================================
# 核心转换函数
# ============================================================================

def create_las_from_segment(segment_data: np.ndarray, 
                            header_info: dict,
                            output_path: Union[str, Path],
                            verbose: bool = False):
    """
    根据 segment 数据创建 LAS 文件
    
    Args:
        segment_data: 结构化数组，包含所有点属性
        header_info: 原始 LAS 文件的头信息
        output_path: 输出 LAS 文件路径
        verbose: 是否输出详细信息
    """
    output_path = Path(output_path)
    
    # 创建 LAS 头
    header = laspy.LasHeader(
        point_format=header_info['point_format'],
        version=header_info['version']
    )
    
    # 设置坐标缩放和偏移
    header.x_scale = header_info['x_scale']
    header.y_scale = header_info['y_scale']
    header.z_scale = header_info['z_scale']
    header.x_offset = header_info['x_offset']
    header.y_offset = header_info['y_offset']
    header.z_offset = header_info['z_offset']
    
    # 设置其他头信息
    if 'system_identifier' in header_info:
        header.system_identifier = header_info['system_identifier']
    if 'generating_software' in header_info:
        header.generating_software = header_info['generating_software']
    
    # 恢复 VLRs (坐标系信息等)
    if 'vlrs' in header_info and header_info['vlrs']:
        for vlr_dict in header_info['vlrs']:
            try:
                vlr = laspy.VLR(
                    user_id=vlr_dict['user_id'],
                    record_id=vlr_dict['record_id'],
                    description=vlr_dict.get('description', ''),
                    record_data=vlr_dict.get('record_data', b'')
                )
                header.vlrs.append(vlr)
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ 无法恢复 VLR {vlr_dict.get('user_id', '?')}: {e}")
    
    # 创建 LAS 数据对象
    las = laspy.LasData(header)
    
    # 设置坐标（必须字段）
    las.x = segment_data['X']
    las.y = segment_data['Y']
    las.z = segment_data['Z']
    
    # 设置其他属性（如果存在）
    field_names = segment_data.dtype.names
    
    # 标准 LAS 字段映射
    standard_fields = {
        'intensity': 'intensity',
        'return_number': 'return_number',
        'number_of_returns': 'number_of_returns',
        'classification': 'classification',
        'scan_angle_rank': 'scan_angle_rank',
        'user_data': 'user_data',
        'point_source_id': 'point_source_id',
        'gps_time': 'gps_time',
        'red': 'red',
        'green': 'green',
        'blue': 'blue',
        'nir': 'nir',
        'edge_of_flight_line': 'edge_of_flight_line',
    }
    
    for field, las_attr in standard_fields.items():
        if field in field_names:
            try:
                setattr(las, las_attr, segment_data[field])
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ 无法设置字段 {field}: {e}")
    
    # 额外字段（通过 extra_bytes 写入）
    extra_fields = ['is_ground']  # tile_las1.py 可能生成的额外字段
    
    for field_name in extra_fields:
        if field_name in field_names:
            try:
                field_data = segment_data[field_name]
                extra_bytes = laspy.ExtraBytesParams(
                    name=field_name,
                    type=field_data.dtype
                )
                las.add_extra_dim(extra_bytes)
                setattr(las, field_name, field_data)
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ 无法添加额外字段 {field_name}: {e}")
    
    # 保存 LAS 文件
    las.write(output_path)


def grid_modulo_sample(segment_info: dict, 
                       mmap_data: np.ndarray,
                       loop_idx: int,
                       points_per_loop: int = 1) -> np.ndarray:
    """
    对 segment 进行网格模运算采样
    
    利用 tile_las.py 中预计算的 sort_idx 和 voxel_counts 进行高效采样。
    
    采样逻辑：
    - 正常情况 (grid_count <= num_loops): 每轮采样 1 个点，使用模运算循环选择
    - 极端情况 (grid_count > num_loops): 每轮采样多个点 (points_per_loop)，确保所有点都被采样
    
    Args:
        segment_info: segment 元数据，包含 indices, sort_idx, voxel_counts
        mmap_data: 内存映射的 bin 数据
        loop_idx: 当前采样轮次 (0-indexed)
        points_per_loop: 每轮从每个网格采样的点数 (用于极端情况)
        
    Returns:
        采样后的结构化数组
    """
    indices = segment_info['indices']
    sort_idx = segment_info.get('sort_idx', None)
    voxel_counts = segment_info.get('voxel_counts', None)
    
    # 如果没有网格化信息，返回全部数据
    if sort_idx is None or voxel_counts is None:
        return mmap_data[indices]
    
    # 网格模运算采样
    # sort_idx 是按网格 hash 排序后的局部索引
    # voxel_counts 是每个网格的点数
    
    # 计算每个网格的起始位置
    cumsum = np.cumsum(np.insert(voxel_counts, 0, 0))
    
    # 采样索引列表
    sampled_local_indices = []
    
    for grid_idx in range(len(voxel_counts)):
        grid_count = voxel_counts[grid_idx]
        start_pos = cumsum[grid_idx]
        
        # 根据 points_per_loop 计算本轮采样的点
        # 对于点数少的网格，使用模运算进行重复采样
        for p in range(points_per_loop):
            # 计算当前轮次要采的第 p 个点的逻辑位置
            logical_idx = loop_idx * points_per_loop + p
            # 模运算：循环采样（对于点数少的网格会重复采样）
            local_idx = logical_idx % grid_count
            sampled_local_indices.append(sort_idx[start_pos + local_idx])
    
    sampled_local_indices = np.array(sampled_local_indices, dtype=np.int32)
    
    # 转换为全局索引
    global_indices = indices[sampled_local_indices]
    
    return mmap_data[global_indices]


class BinToLasConverter:
    """
    bin+pkl 逻辑索引格式转 LAS 文件转换器
    
    支持两种模式：
    - full: 全量模式，输出所有原始点
    - grid: 网格采样模式，使用网格化索引进行模运算采样
    """
    
    def __init__(self,
                 input_dir: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 mode: str = 'full',
                 max_loops: Optional[int] = None,
                 segment_ids: Optional[List[int]] = None,
                 max_segments: Optional[int] = None,
                 n_workers: int = 4):
        """
        初始化转换器
        
        Args:
            input_dir: 包含 bin+pkl 文件的输入目录
            output_dir: 输出目录 (默认为 input_dir/las_output)
            mode: 转换模式
                - 'full': 全量模式，输出所有原始点
                - 'grid': 网格采样模式，使用网格化索引进行采样
            max_loops: 网格采样模式下的最大采样轮次
                - None: 按网格内最大点数进行采样（每轮采 1 个点）
                - 设置值: 如果网格最大点数 > max_loops，则每轮采多个点以确保在 max_loops 轮内采完
                - 如果 max_loops > 网格最大点数，则按实际最大点数采样
            segment_ids: 要转换的 segment ID 列表 (None 表示全部)
            max_segments: 最多转换多少个 segment (None 表示不限制)
            n_workers: 并行工作线程数
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / 'las_output'
        self.mode = mode
        self.max_loops = max_loops
        self.segment_ids = segment_ids
        self.max_segments = max_segments
        self.n_workers = n_workers
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        # 查找所有 bin+pkl 文件对
        self.file_pairs = self._find_file_pairs()
    
    def _find_file_pairs(self) -> List[Dict[str, Path]]:
        """查找所有 bin+pkl 文件对"""
        pairs = []
        
        for bin_path in sorted(self.input_dir.glob('*.bin')):
            pkl_path = bin_path.with_suffix('.pkl')
            if pkl_path.exists():
                pairs.append({
                    'bin': bin_path,
                    'pkl': pkl_path,
                    'name': bin_path.stem
                })
        
        return pairs
    
    def convert_all(self):
        """转换所有文件"""
        if not self.file_pairs:
            print(f"{Colors.RED}❌ 未找到有效的 bin+pkl 文件对{Colors.RESET}")
            return
        
        start_time = time.time()
        
        # 美化的标题输出
        print(f"\n{Colors.BOLD}{'═'*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  🔄 BIN+PKL → LAS 转换器 (Logical Index Mode){Colors.RESET}")
        print(f"{Colors.BOLD}{'═'*70}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 输入目录: {Colors.CYAN}{self.input_dir}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 输出目录: {Colors.CYAN}{self.output_dir}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 文件数量: {Colors.GREEN}{len(self.file_pairs)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 转换模式: {Colors.YELLOW}{self.mode}{Colors.RESET}")
        if self.mode == 'grid':
            max_loops_str = str(self.max_loops) if self.max_loops is not None else "自动 (按最大网格点数)"
            print(f"  {Colors.DIM}├─{Colors.RESET} 最大轮次: {Colors.YELLOW}{max_loops_str}{Colors.RESET}")
        if self.max_segments:
            print(f"  {Colors.DIM}├─{Colors.RESET} 每文件最大: {Colors.YELLOW}{self.max_segments} segments{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 并行线程: {Colors.GREEN}{self.n_workers}{Colors.RESET}")
        print(f"{Colors.BOLD}{'─'*70}{Colors.RESET}\n")
        
        # 处理每个文件
        for idx, pair in enumerate(self.file_pairs, 1):
            try:
                self._convert_file(pair, idx, len(self.file_pairs))
            except Exception as e:
                print(f"\n{Colors.RED}[ERROR] {pair['name']}: {e}{Colors.RESET}")
                import traceback
                traceback.print_exc()
        
        elapsed = time.time() - start_time
        
        # 美化的完成输出
        print(f"\n{Colors.BOLD}{'═'*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}  ✅ 转换完成!{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} ⏱️  总耗时: {Colors.CYAN}{format_time(elapsed)}{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 📄 平均每文件: {Colors.CYAN}{format_time(elapsed/len(self.file_pairs))}{Colors.RESET}")
        print(f"{Colors.BOLD}{'═'*70}{Colors.RESET}\n")
    
    def _convert_file(self, pair: Dict[str, Path], file_idx: int, total_files: int):
        """转换单个文件"""
        bin_path = pair['bin']
        pkl_path = pair['pkl']
        base_name = pair['name']
        
        print(f"{Colors.BOLD}{'─'*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}  📄 [{file_idx}/{total_files}] {base_name}{Colors.RESET}")
        print(f"{Colors.BOLD}{'─'*70}{Colors.RESET}")
        
        file_start = time.time()
        
        # 1. 加载元数据
        t0 = time.time()
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        t1 = time.time()
        
        header_info = metadata.get('header_info', {})
        segments_info = metadata['segments']
        total_segments = len(segments_info)
        grid_size = metadata.get('grid_size', None)
        
        print(f"  {Colors.DIM}├─{Colors.RESET} 📖 加载元数据: {Colors.GREEN}{format_time(t1-t0)}{Colors.RESET}")
        print(f"  {Colors.DIM}│{Colors.RESET}   - 总点数: {Colors.CYAN}{format_number(metadata['num_points'])}{Colors.RESET}")
        print(f"  {Colors.DIM}│{Colors.RESET}   - 总段数: {Colors.CYAN}{total_segments}{Colors.RESET}")
        print(f"  {Colors.DIM}│{Colors.RESET}   - Grid Size: {Colors.CYAN}{grid_size or 'N/A'}{Colors.RESET}")
        
        # 确定要处理的 segment IDs
        if self.segment_ids is not None:
            seg_ids = [i for i in self.segment_ids if i < total_segments]
        else:
            seg_ids = list(range(total_segments))
        
        if self.max_segments is not None:
            seg_ids = seg_ids[:self.max_segments]
        
        # 2. 使用 memmap 加载 bin 文件
        t0 = time.time()
        dtype = np.dtype(metadata['dtype'])
        mmap_data = np.memmap(bin_path, dtype=dtype, mode='r')
        t1 = time.time()
        print(f"  {Colors.DIM}├─{Colors.RESET} 🗂️  加载BIN: {Colors.GREEN}{format_time(t1-t0)}{Colors.RESET}")
        
        # 3. 创建输出子目录
        file_output_dir = self.output_dir / base_name
        if not file_output_dir.exists():
            file_output_dir.mkdir(parents=True)
        
        # 4. 转换 segments
        t0 = time.time()
        
        if self.mode == 'full':
            self._convert_full_mode(
                mmap_data, segments_info, seg_ids, 
                header_info, file_output_dir, base_name
            )
        elif self.mode == 'grid':
            self._convert_grid_mode(
                mmap_data, segments_info, seg_ids,
                header_info, file_output_dir, base_name,
                grid_size
            )
        else:
            raise ValueError(f"未知模式: {self.mode}")
        
        t1 = time.time()
        print(f"  {Colors.DIM}├─{Colors.RESET} 💾 保存LAS: {Colors.GREEN}{format_time(t1-t0)}{Colors.RESET}")
        
        # 总耗时
        total_time = time.time() - file_start
        print(f"  {Colors.DIM}└─{Colors.RESET} ⏱️  文件总耗时: {Colors.BOLD}{Colors.GREEN}{format_time(total_time)}{Colors.RESET}")
    
    def _convert_full_mode(self, 
                           mmap_data: np.ndarray,
                           segments_info: List[Dict],
                           seg_ids: List[int],
                           header_info: Dict,
                           output_dir: Path,
                           base_name: str):
        """
        全量模式转换：直接输出所有原始点
        """
        success_count = 0
        
        for seg_id in tqdm(seg_ids, desc="  转换segments", unit="seg",
                          bar_format='  {l_bar}{bar}| {n_fmt}/{total_fmt}'):
            try:
                segment_info = segments_info[seg_id]
                indices = segment_info['indices']
                
                # 提取全部数据
                segment_data = mmap_data[indices]
                
                # 生成输出文件名
                output_name = f"{base_name}_seg{seg_id:04d}.las"
                output_path = output_dir / output_name
                
                # 创建 LAS 文件
                create_las_from_segment(segment_data, header_info, output_path)
                
                success_count += 1
                
            except Exception as e:
                print(f"\n  ⚠️ Segment {seg_id} 转换失败: {e}")
        
        print(f"  {Colors.DIM}│{Colors.RESET}   → 成功: {Colors.GREEN}{success_count}/{len(seg_ids)}{Colors.RESET} segments")
    
    def _convert_grid_mode(self,
                            mmap_data: np.ndarray,
                            segments_info: List[Dict],
                            seg_ids: List[int],
                            header_info: Dict,
                            output_dir: Path,
                            base_name: str,
                            grid_size: Optional[float]):
        """
        网格采样模式转换：使用网格化索引进行模运算采样
        
        采样策略：
        - max_loops=None: 按网格内最大点数 max_count 进行 max_count 轮采样，每轮采 1 个点
        - max_loops 设置时:
          - 如果 max_count <= max_loops: 按 max_count 轮采样，每轮采 1 个点
          - 如果 max_count > max_loops: 按 max_loops 轮采样，每轮采 ceil(max_count/max_loops) 个点
        - 对于点数少于采样轮数的网格：使用模运算重复采样
        """
        if grid_size is None:
            print(f"  {Colors.YELLOW}⚠️  警告: 未找到 grid_size 信息，回退到全量模式{Colors.RESET}")
            self._convert_full_mode(mmap_data, segments_info, seg_ids, header_info, output_dir, base_name)
            return
        
        success_count = 0
        total_las_files = 0
        
        for seg_id in tqdm(seg_ids, desc="  转换segments", unit="seg",
                          bar_format='  {l_bar}{bar}| {n_fmt}/{total_fmt}'):
            try:
                segment_info = segments_info[seg_id]
                
                # 检查是否有体素化信息
                if 'sort_idx' not in segment_info or 'voxel_counts' not in segment_info:
                    # 没有体素化信息，输出全量
                    indices = segment_info['indices']
                    segment_data = mmap_data[indices]
                    
                    output_name = f"{base_name}_seg{seg_id:04d}_full.las"
                    output_path = output_dir / output_name
                    create_las_from_segment(segment_data, header_info, output_path)
                    total_las_files += 1
                else:
                    # 有体素化信息，进行多轮采样
                    voxel_counts = segment_info['voxel_counts']
                    max_voxel_count = int(voxel_counts.max()) if len(voxel_counts) > 0 else 1
                    
                    # 计算实际轮数和每轮采样点数
                    if self.max_loops is None:
                        # 未设置 max_loops：按最大体素点数采样，每轮采 1 个点
                        actual_loops = max_voxel_count
                        points_per_loop = 1
                    elif max_voxel_count <= self.max_loops:
                        # 最大点数 <= max_loops：按实际最大点数采样，每轮采 1 个点
                        actual_loops = max_voxel_count
                        points_per_loop = 1
                    else:
                        # 最大点数 > max_loops：限制轮数，每轮采多个点
                        actual_loops = self.max_loops
                        points_per_loop = int(np.ceil(max_voxel_count / self.max_loops))
                    
                    for loop_idx in range(actual_loops):
                        # 网格模运算采样（传入 points_per_loop）
                        segment_data = grid_modulo_sample(
                            segment_info, mmap_data, loop_idx, points_per_loop
                        )
                        
                        # 生成输出文件名 (包含 loop 索引)
                        output_name = f"{base_name}_seg{seg_id:04d}_loop{loop_idx:02d}.las"
                        output_path = output_dir / output_name
                        
                        create_las_from_segment(segment_data, header_info, output_path)
                        total_las_files += 1
                
                success_count += 1
                
            except Exception as e:
                print(f"\n  ⚠️ Segment {seg_id} 转换失败: {e}")
        
        print(f"  {Colors.DIM}│{Colors.RESET}   → 成功: {Colors.GREEN}{success_count}/{len(seg_ids)}{Colors.RESET} segments")
        print(f"  {Colors.DIM}│{Colors.RESET}   → 生成: {Colors.CYAN}{total_las_files}{Colors.RESET} LAS 文件")


def convert_bin_to_las(input_dir: Union[str, Path],
                       output_dir: Union[str, Path] = None,
                       mode: str = 'full',
                       max_loops: Optional[int] = None,
                       segment_ids: Optional[List[int]] = None,
                       max_segments: Optional[int] = None,
                       n_workers: int = 4):
    """
    便捷函数：将 bin+pkl 文件转换为 LAS 格式
    
    Args:
        input_dir: 包含 bin+pkl 文件的输入目录
        output_dir: 输出目录 (默认为 input_dir/las_output)
        mode: 转换模式
            - 'full': 全量模式，输出所有原始点
            - 'voxel': 体素模式，使用体素化索引进行采样
        max_loops: 体素模式下的最大采样轮次
            - None: 按体素内最大点数进行采样（每轮采 1 个点）
            - 设置值: 限制最大轮数，如果体素点数超过则每轮采多个点
        segment_ids: 要转换的 segment ID 列表 (None 表示全部)
        max_segments: 最多转换多少个 segment (None 表示不限制)
        n_workers: 并行工作线程数
    """
    converter = BinToLasConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        mode=mode,
        max_loops=max_loops,
        segment_ids=segment_ids,
        max_segments=max_segments,
        n_workers=n_workers
    )
    converter.convert_all()


def convert_single_file(bin_path: Union[str, Path],
                        pkl_path: Union[str, Path],
                        output_dir: Union[str, Path],
                        mode: str = 'full',
                        max_loops: Optional[int] = None,
                        segment_ids: Optional[List[int]] = None,
                        max_segments: Optional[int] = None):
    """
    转换单个 bin+pkl 文件对
    
    Args:
        bin_path: bin 文件路径
        pkl_path: pkl 文件路径
        output_dir: 输出目录
        mode: 转换模式 ('full' 或 'voxel')
        max_loops: 体素模式下的最大采样轮次 (None 表示按最大体素点数)
        segment_ids: 要转换的 segment ID 列表
        max_segments: 最多转换多少个 segment
    """
    bin_path = Path(bin_path)
    pkl_path = Path(pkl_path)
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # 创建临时目录结构以复用 Converter
    temp_input = bin_path.parent
    
    converter = BinToLasConverter(
        input_dir=temp_input,
        output_dir=output_dir,
        mode=mode,
        max_loops=max_loops,
        segment_ids=segment_ids,
        max_segments=max_segments
    )
    
    # 手动设置只处理这一个文件
    converter.file_pairs = [{
        'bin': bin_path,
        'pkl': pkl_path,
        'name': bin_path.stem
    }]
    
    converter.convert_all()


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # ==================== 使用示例 ====================
    
    # 示例1: 全量模式 - 转换整个目录
    # convert_bin_to_las(
    #     input_dir=r"E:\data\DALES\dales_las\bin\train_logical",
    #     output_dir=r"E:\data\DALES\dales_las\bin\train_logical\las_full",
    #     mode='full',
    #     max_segments=5  # 每个 bin 文件最多转换 5 个 segment
    # )
    
    # 示例2: 体素模式 - 使用体素化采样
    # convert_bin_to_las(
    #     input_dir=r"E:\data\DALES\dales_las\bin\train_logical",
    #     output_dir=r"E:\data\DALES\dales_las\bin\train_logical\las_voxel",
    #     mode='voxel',
    #     max_loops=5,  # 每个 segment 生成 5 个采样版本
    #     max_segments=3  # 每个 bin 文件最多转换 3 个 segment
    # )
    
    # 示例3: 转换单个文件
    bin_file = r"E:\data\DALES\dales_las\bin\train_logical\5080_54435.bin"
    pkl_file = r"E:\data\DALES\dales_las\bin\train_logical\5080_54435.pkl"
    output_dir = r"E:\data\DALES\dales_las\bin\train_logical\las_test"
    
    if Path(bin_file).exists() and Path(pkl_file).exists():
        # 全量模式测试
        print("\n" + "="*70)
        print("测试: 全量模式")
        print("="*70)
        convert_single_file(
            bin_path=bin_file,
            pkl_path=pkl_file,
            output_dir=output_dir + "_full",
            mode='full',
        )
        
        # # 体素模式测试
        # print("\n" + "="*70)
        # print("测试: 体素模式")
        # print("="*70)
        # convert_single_file(
        #     bin_path=bin_file,
        #     pkl_path=pkl_file,
        #     output_dir=output_dir + "_voxel",
        #     mode='voxel',
        #     # max_loops=5,
        # )
    else:
        print(f"测试文件不存在，请修改路径后运行")
        print(f"  bin: {bin_file}")
        print(f"  pkl: {pkl_file}")
