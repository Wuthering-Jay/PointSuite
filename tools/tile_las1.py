import os
import numpy as np
import laspy
import pickle
import time
import multiprocessing
import math
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from numba import jit, prange
from sklearn.neighbors import KDTree

# ============================================================================
# 美化输出辅助类
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
# Numba 加速函数 (哈希与坐标计算)
# ============================================================================

@jit(nopython=True, parallel=True)
def compute_grid_coord_numba(coord, grid_size):
    """计算网格坐标"""
    n = coord.shape[0]
    grid_coord = np.empty_like(coord, dtype=np.int64)
    for i in prange(n):
        for j in range(3):
            grid_coord[i, j] = np.floor(coord[i, j] / grid_size)
    return grid_coord

@jit(nopython=True, parallel=True)
def ravel_hash_vec_numba(arr, arr_min, arr_max):
    """计算空间哈希值"""
    n = arr.shape[0]
    d = arr.shape[1]
    keys = np.zeros(n, dtype=np.uint64)
    
    # 归一化并转换为 uint64
    arr_normalized = np.empty_like(arr, dtype=np.uint64)
    for i in prange(n):
        for j in range(d):
            arr_normalized[i, j] = np.uint64(arr[i, j] - arr_min[j])
    
    # 计算每一维度的跨度
    arr_max_plus_one = np.empty(d, dtype=np.uint64)
    for j in range(d):
        arr_max_plus_one[j] = np.uint64(arr_max[j] - arr_min[j] + 1)
    
    # Fortran style flatten
    for i in prange(n):
        key = np.uint64(0)
        for j in range(d - 1):
            key += arr_normalized[i, j]
            key *= arr_max_plus_one[j + 1]
        key += arr_normalized[i, d - 1]
        keys[i] = key
    
    return keys

# ============================================================================
# 核心处理类
# ============================================================================

class LASProcessorLogicalIndex:
    def __init__(self,
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 window_size: Tuple[float, float] = (50.0, 50.0),
                 overlap: bool = False,
                 grid_size: float = 0.5,      # 仅用于生成逻辑索引，不进行物理降采样
                 min_points: int = 1000,
                 max_points: int = 5000,      # 通常不再需要强制切分，因为我们有完美的batch控制
                 ground_class: Optional[int] = 2):
        
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.window_size = window_size
        self.overlap = overlap
        # overlap_ratio = 0.5 if overlap else 0.0
        self.grid_size = grid_size
        self.min_points = min_points
        self.max_points = max_points
        self.ground_class = ground_class
        
        # 计算步长 (Stride)
        # self.stride = (
        #     window_size[0] * (1 - overlap_ratio),
        #     window_size[1] * (1 - overlap_ratio)
        # )
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        self.las_files = self._find_las_files()

    def _find_las_files(self) -> List[Path]:
        """
        查找输入路径下的所有 LAS/LAZ 文件
        """
        if self.input_path.is_file():
            return [self.input_path]
        elif self.input_path.is_dir():
            return sorted(list(self.input_path.glob('*.las')) + list(self.input_path.glob('*.laz')))
        else:
            raise ValueError(f"Invalid path: {self.input_path}")

    def process_all_files(self, n_workers=None):
        """
        处理所有 LAS/LAZ 文件
        """

        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        start_time = time.time()

        # 美化的标题输出
        print(f"\n{Colors.BOLD}{'═'*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  🚀 LAS 逻辑索引分块处理器 (Logical Index Tiling){Colors.RESET}")
        print(f"{Colors.BOLD}{'═'*70}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 总文件数: {Colors.GREEN}{len(self.las_files)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} CPU 核心: {Colors.GREEN}{n_workers}{Colors.RESET}")
        grid_size_str = f"{self.grid_size}m" if self.grid_size is not None else "跳过体素化"
        print(f"  {Colors.DIM}├─{Colors.RESET} 网格大小: {Colors.YELLOW}{grid_size_str}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 窗口大小: {Colors.YELLOW}{self.window_size}m{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 重叠模式: {Colors.GREEN if self.overlap else Colors.DIM}{'是' if self.overlap else '否'}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 点数范围: {Colors.YELLOW}{self.min_points} ~ {self.max_points or '无限制'}{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 地面类别: {Colors.YELLOW}{self.ground_class or '未指定'}{Colors.RESET}")
        print(f"{Colors.BOLD}{'─'*70}{Colors.RESET}\n")
        
        # 顺序处理每个文件，但文件内部并行处理segments
        for idx, las_file in enumerate(self.las_files, 1):
            try:
                self.process_file(las_file, n_workers=n_workers, file_idx=idx, total_files=len(self.las_files))
            except Exception as e:
                print(f"\n{Colors.RED}[ERROR] {las_file.name}: {e}{Colors.RESET}")
                import traceback
                traceback.print_exc()

        elapsed = time.time() - start_time
        
        # 美化的完成输出
        print(f"\n{Colors.BOLD}{'═'*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}  ✅ 处理完成!{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} ⏱️  总耗时: {Colors.CYAN}{format_time(elapsed)}{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 📄 平均每文件: {Colors.CYAN}{format_time(elapsed/len(self.las_files))}{Colors.RESET}")
        print(f"{Colors.BOLD}{'═'*70}{Colors.RESET}\n")

    def process_file(self, las_file: Path, n_workers=None, file_idx=1, total_files=1):

        print(f"{Colors.BOLD}{'─'*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}  📄 [{file_idx}/{total_files}] {las_file.name}{Colors.RESET}")
        print(f"{Colors.BOLD}{'─'*70}{Colors.RESET}")
        file_start = time.time()

        # 1. 读取数据
        t0 = time.time()
        with laspy.open(las_file) as fh:
            las_data = fh.read()
        t1 = time.time()
        print(f"  {Colors.DIM}├─{Colors.RESET} 📖 读取LAS: {Colors.GREEN}{format_time(t1-t0)}{Colors.RESET} → {Colors.CYAN}{format_number(len(las_data.points))}{Colors.RESET} 点")
            
        # 获取坐标 (laspy 默认返回 float64)
        t0 = time.time()
        points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
        t1 = time.time()
        
        # 2. 滑动窗口切块 (获取索引列表)
        t0 = time.time()
        result = self.segment_point_cloud(points, n_workers=n_workers)
        segments_indices, seg1_count, seg2_count = result
        t1 = time.time()
        
        # 显示分块信息
        if self.overlap and seg1_count is not None:
            print(f"  {Colors.DIM}├─{Colors.RESET} 🔲 分块处理: {Colors.GREEN}{format_time(t1-t0)}{Colors.RESET} → {Colors.CYAN}{len(segments_indices)}{Colors.RESET} 块 ({seg1_count} + {seg2_count})")
        else:
            print(f"  {Colors.DIM}├─{Colors.RESET} 🔲 分块处理: {Colors.GREEN}{format_time(t1-t0)}{Colors.RESET} → {Colors.CYAN}{len(segments_indices)}{Colors.RESET} 块")
        
        # 3. 处理并保存
        t0 = time.time()
        self._save_bin_pkl(las_file, las_data, segments_indices)
        t1 = time.time()
        
        # 总耗时
        total_time = time.time() - file_start
        print(f"  {Colors.DIM}└─{Colors.RESET} ⏱️  文件总耗时: {Colors.BOLD}{Colors.GREEN}{format_time(total_time)}{Colors.RESET}")

    def segment_point_cloud(self, points: np.ndarray, n_workers: int = 4) -> List[np.ndarray]:
        """
        Segment point cloud into tiles based on window size.
        
        Args:
            points: Point cloud array (N, 3)
            n_workers: Number of parallel workers
            
        Returns:
            List of segment indices
        """
        import time
        
        if not self.overlap:
            # 正常模式：单次网格分割
            t0 = time.time()
            segments = self._grid_segmentation(points, offset_x=0, offset_y=0, n_workers=n_workers, show_details=False)
            return segments, None, None
        else:
            # Overlap模式：两次网格分割（偏移半个窗口）
            x_size, y_size = self.window_size
            
            # 第一次分割：正常网格
            t0 = time.time()
            segments1 = self._grid_segmentation(points, offset_x=0, offset_y=0, n_workers=n_workers, show_details=False)
            
            # 第二次分割：偏移半个窗口
            segments2 = self._grid_segmentation(points, offset_x=x_size/2, offset_y=y_size/2, n_workers=n_workers, show_details=False)
            
            # 合并两次分割结果
            all_segments = segments1 + segments2
            
            return all_segments, len(segments1), len(segments2)
        
    def _grid_segmentation(self, points: np.ndarray, offset_x: float = 0, offset_y: float = 0, n_workers: int = 4, show_details: bool = False) -> List[np.ndarray]:
        """
        Perform grid-based segmentation with optional offset.
        
        Args:
            points: Point cloud array (N, 3)
            offset_x: X offset for grid origin
            offset_y: Y offset for grid origin
            n_workers: Number of parallel workers
            show_details: Whether to print detailed progress
            
        Returns:
            List of segment indices
        """
        import time
        
       # 1. 窗口分组 (优化版：使用 argsort 代替 where 循环)
        t0 = time.time()
        x_size, y_size = self.window_size
        
        # 计算原点
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        origin_x = min_x - offset_x
        origin_y = min_y - offset_y
        
        # 计算窗口索引
        # 优化：直接计算 long 型索引，不计算 num_windows，避免溢出风险
        x_bins = ((points[:, 0] - origin_x) / x_size).astype(np.int64)
        y_bins = ((points[:, 1] - origin_y) / y_size).astype(np.int64)
        
        # 使用 cantor pairing 或类似的 hash 方式组合二维索引，或者简单的字符串组合（慢）
        # 这里为了速度，假设 y_bins 范围不会太大，使用大数乘法组合
        # 假设 y 方向不会超过 1,000,000 个 grid
        y_multiplier = 1000000
        window_ids = x_bins * y_multiplier + y_bins
        
        # 🚀 核心优化：使用 argsort 一次性分组，避免 N 次全量扫描
        sort_idx = np.argsort(window_ids)
        sorted_window_ids = window_ids[sort_idx]
        
        # 找到切分点
        unique_ids, split_indices = np.unique(sorted_window_ids, return_index=True)
        # split_indices[0] 是 0，我们需要的切分点是 split_indices[1:]
        # np.split 会返回列表
        segments = np.split(sort_idx, split_indices[1:])
        
        # 2. Min阈值处理（优先处理，合并边界上点少的无效窗口）
        if self.min_points is not None:
            before_count = len(segments)
            segments = self.apply_min_threshold(points, segments, min_threshold=self.min_points)
        
        # 3. Max阈值处理（最后处理）
        if self.max_points is not None:
            before_count = len(segments)
            segments = self.apply_max_threshold(points, segments, n_workers=n_workers)
            
        return segments
    
    def apply_max_threshold(self, points: np.ndarray, segments: List[np.ndarray], n_workers: int = 4) -> List[np.ndarray]:
        """
        Apply max_points threshold to segments, subdividing large segments.
        
        Args:
            points: Point cloud array
            segments: List of segment indices
            n_workers: Number of parallel workers
            
        Returns:
            List of processed segment indices
        """
        large_segment_indices = [i for i, segment in enumerate(segments) if len(segment) > self.max_points]
        
        if not large_segment_indices:
            return segments
        
        result_segments = [segment for i, segment in enumerate(segments) if i not in large_segment_indices]
        large_segments = [segments[i] for i in large_segment_indices]
        
        def process_segment(segment):
            if len(segment) <= self.max_points:
                return [segment]
            
            segment_points = points[segment]
            ranges = np.ptp(segment_points[:, :2], axis=0)
            split_dim = np.argmax(ranges[:2])
            sorted_indices = np.argsort(segment_points[:, split_dim])
            
            mid = len(sorted_indices) // 2
            left_half = segment[sorted_indices[:mid]]
            right_half = segment[sorted_indices[mid:]]
            
            result = []
            result.extend(process_segment(left_half))
            result.extend(process_segment(right_half))
            return result
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 使用传入的n_workers参数
        max_workers = min(n_workers, len(large_segments)) if len(large_segments) > 0 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_segment, segment) for segment in large_segments]
            for future in as_completed(futures):
                result_segments.extend(future.result())
        
        return result_segments
    
    def _process_single_segment(self, args) -> dict:
        """
        处理单个 segment 的体素化（用于并行处理）
        """
        i, indices, lx, ly, lz, l_class, bin_name, pkl_name = args
        
        # 1. 提取当前块坐标 (Float64)
        seg_points = np.column_stack((lx[indices], ly[indices], lz[indices]))
        
        # 2. 局部坐标归一化
        local_min = seg_points.min(0)
        local_points = (seg_points - local_min).astype(np.float64)
        
        # 3. 体素化处理 (如果 grid_size 为 None 则跳过)
        if self.grid_size is not None:
            # 计算 Grid Hash (使用纯 NumPy 向量化，避免 Numba JIT 开销)
            grid_coord = np.floor(local_points / self.grid_size).astype(np.int64)
            
            # 确保非负且紧凑
            if len(grid_coord) > 0:
                grid_min = grid_coord.min(0)
                grid_coord -= grid_min
                arr_max = grid_coord.max(0)
                
                # 向量化 ravel hash (Fortran style)
                multipliers = np.cumprod(np.concatenate([[1], arr_max[1:] + 1])).astype(np.uint64)
                keys = (grid_coord.astype(np.uint64) * multipliers).sum(axis=1)
            else:
                keys = np.zeros(0, dtype=np.uint64)
            
            # 生成逻辑排序索引
            sort_ptr = np.argsort(keys, kind='mergesort').astype(np.int32)
            keys_sorted = keys[sort_ptr]
            
            # 计算体素统计
            _, voxel_counts = np.unique(keys_sorted, return_counts=True)
        else:
            # 跳过体素化：保持原始顺序
            sort_ptr = np.arange(len(indices), dtype=np.int32)
            voxel_counts = np.array([len(indices)], dtype=np.int64)  # 单个"体素"包含所有点
        
        # 4. 统计类别分布
        label_counts = {}
        unique_labels = []
        if l_class is not None:
            seg_labels = l_class[indices]
            unique_labels, u_counts = np.unique(seg_labels, return_counts=True)
            label_counts = {int(k): int(v) for k, v in zip(unique_labels, u_counts)}
        
        # 5. 计算边界框
        bounds = {
            'x_min': float(seg_points[:, 0].min()),
            'x_max': float(seg_points[:, 0].max()),
            'y_min': float(seg_points[:, 1].min()),
            'y_max': float(seg_points[:, 1].max()),
            'z_min': float(seg_points[:, 2].min()),
            'z_max': float(seg_points[:, 2].max())
        }

        return {
            'segment_id': i,
            'indices': indices,
            'num_points': len(indices),
            'sort_idx': sort_ptr,
            'voxel_counts': voxel_counts,
            'num_voxels': len(voxel_counts),
            'max_voxel_density': voxel_counts.max() if len(voxel_counts) > 0 else 0,
            'local_min': local_min,
            'label_counts': label_counts,
            'unique_labels': unique_labels,
            'bounds': bounds,
            'bin_path': bin_name,
            'pkl_path': pkl_name
        }

    def _process_segments_parallel(self, segments, lx, ly, lz, l_class, bin_path, pkl_path, n_workers=None):
        """
        并行处理所有 segments 的体素化
        """
        from concurrent.futures import ThreadPoolExecutor
        
        if n_workers is None:
            n_workers = min(8, max(1, multiprocessing.cpu_count() - 1))
        
        bin_name = str(bin_path.name)
        pkl_name = str(pkl_path.name)
        
        # 准备参数
        args_list = [
            (i, indices, lx, ly, lz, l_class, bin_name, pkl_name)
            for i, indices in enumerate(segments)
        ]
        
        # 使用线程池并行处理
        segments_info = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(self._process_single_segment, args_list))
            segments_info = sorted(results, key=lambda x: x['segment_id'])
        
        return segments_info

    def apply_min_threshold(self, points: np.ndarray, segments: List[np.ndarray], 
                           min_threshold: Optional[int] = None) -> List[np.ndarray]:
        """
        Apply min_points threshold using KD-Tree.
        
        Args:
            points: Point cloud array
            segments: List of segment indices
            min_threshold: Minimum points threshold (if None, use self.min_points)
        
        Returns:
            List of processed segment indices
        """
        if len(segments) <= 1:
            return segments
        
        # 使用传入的min_threshold，如果没有则使用self.min_points
        effective_min = min_threshold if min_threshold is not None else self.min_points
        
        centroids = np.array([np.mean(points[segment][:, :2], axis=0) for segment in segments])
        small_segments = [i for i, segment in enumerate(segments) if len(segment) < effective_min]
        
        if not small_segments:
            return segments
        
        valid_indices = [i for i in range(len(segments)) if i not in small_segments]
        if not valid_indices:
            return segments
        
        valid_centroids = centroids[valid_indices]
        kdtree = KDTree(valid_centroids)
        
        small_segments.sort(key=lambda i: len(segments[i]))
        
        for small_idx in small_segments:
            if small_idx >= len(segments):
                continue
            
            _, nearest_idx = kdtree.query([centroids[small_idx]], k=1)
            nearest_idx = valid_indices[nearest_idx[0][0]]
            
            if nearest_idx != small_idx and nearest_idx < len(segments):
                segments[nearest_idx] = np.concatenate([segments[nearest_idx], segments[small_idx]])
                segments[small_idx] = np.array([], dtype=int)
        
        return [segment for segment in segments if len(segment) > 0]


    def _save_bin_pkl(self, las_file, las_data, segments):
        base_name = las_file.stem
        bin_path = self.output_dir / f"{base_name}.bin"
        pkl_path = self.output_dir / f"{base_name}.pkl"
        
        print(f"  {Colors.DIM}├─{Colors.RESET} 💾 保存文件...")
        t0 = time.time()

        # --- A. 收集字段 ---
        
        # 1. 核心字段: 强制 float64 保证精度
        core_fields = ['X', 'Y', 'Z']
        dtype_list = [('X', np.float64), ('Y', np.float64), ('Z', np.float64)]
        
        # 初始数据字典 (laspy.x 已经是 float64)
        data_dict = {
            'X': np.array(las_data.x, dtype=np.float64), 
            'Y': np.array(las_data.y, dtype=np.float64), 
            'Z': np.array(las_data.z, dtype=np.float64)
        }
        
        # 2. 扩展的可选字段列表
        optional_fields = [
            'intensity', 'return_number', 'number_of_returns', 
            'classification', 'scan_angle_rank', 'user_data', 
            'point_source_id', 'gps_time', 
            'red', 'green', 'blue', 'nir', 'edge_of_flight_line'
        ]
        
        # 3. 动态收集存在的字段
        # 注意: laspy 属性是小写的，但我们保存的 key 用标准名(通常大写或驼峰，但这里保持小写属性名对应的原始名)
        # 为了兼容性，除了XYZ，其他字段我们使用小写或 laspy 属性名
        has_classification = False
        fields_to_save = list(core_fields) # 先加入核心字段
        
        for field in optional_fields:
            # laspy 属性通常是小写的
            field_lower = field.lower()
            
            if hasattr(las_data, field_lower):
                arr = getattr(las_data, field_lower)
                # 使用字段原名作为 key (如 'red', 'intensity')
                # 注意：XYZ 我们用大写，其他通常用小写
                # 这里我们统一使用 field (列表中的名字) 作为 key
                data_dict[field] = arr
                dtype_list.append((field, arr.dtype))
                fields_to_save.append(field)
                
                if field_lower == 'classification':
                    has_classification = True
        
        # 4. 兜底处理：如果没有 classification，补全为 0
        if not has_classification:
            print(f"  {Colors.DIM}│{Colors.RESET}  {Colors.YELLOW}⚠️  无 classification 字段，补全为 0{Colors.RESET}")
            data_dict['classification'] = np.zeros(len(las_data.points), dtype=np.uint8)
            dtype_list.append(('classification', np.uint8))
            fields_to_save.append('classification')
            
        # 5. 🔥 生成 is_ground 字段 🔥
        # 基于 classification 生成，避免 Dataset 重复计算
        if self.ground_class is not None:
            # 确保使用刚才（可能补全的）classification 数据
            cls_data = data_dict['classification']
            is_ground = (cls_data == self.ground_class).astype(np.uint8)
            
            data_dict['is_ground'] = is_ground
            dtype_list.append(('is_ground', np.uint8))
            fields_to_save.append('is_ground')

        # 6. 创建结构化数组并保存
        struct_arr = np.zeros(len(las_data.points), dtype=dtype_list)
        for field in fields_to_save:
            struct_arr[field] = data_dict[field]
            
        struct_arr.tofile(bin_path)
        
        t1 = time.time()
        bin_size = bin_path.stat().st_size
        print(f"  {Colors.DIM}│{Colors.RESET}  📁 BIN: {Colors.GREEN}{format_time(t1-t0)}{Colors.RESET} → {Colors.CYAN}{format_size(bin_size)}{Colors.RESET}")

        # --- B. 生成 PKL (逻辑索引元数据 & 关键头文件信息) ---
        t0 = time.time()
        
        # 预取坐标以加速循环 (引用上面已经转好的 float64 数组)
        lx, ly, lz = data_dict['X'], data_dict['Y'], data_dict['Z']
        l_class = data_dict.get('classification', None)
        
        # 🚀 并行处理所有 segments 的体素化
        segments_info = self._process_segments_parallel(
            segments, lx, ly, lz, l_class, bin_path, pkl_path
        )
            
        # 8. 🔥 收集完整的 LAS Header 和 VLRs 信息 🔥
        # 收集完整的LAS头文件信息
        header_info = {
            'version': f"{las_data.header.version.major}.{las_data.header.version.minor}",
            'point_format': las_data.header.point_format.id,
            'point_count': las_data.header.point_count,
            'x_scale': las_data.header.x_scale,
            'y_scale': las_data.header.y_scale,
            'z_scale': las_data.header.z_scale,
            'x_offset': las_data.header.x_offset,
            'y_offset': las_data.header.y_offset,
            'z_offset': las_data.header.z_offset,
            'x_min': las_data.header.x_min,
            'x_max': las_data.header.x_max,
            'y_min': las_data.header.y_min,
            'y_max': las_data.header.y_max,
            'z_min': las_data.header.z_min,
            'z_max': las_data.header.z_max,
        }
        
        # 保存其他头文件属性
        if hasattr(las_data.header, 'system_identifier'):
            header_info['system_identifier'] = las_data.header.system_identifier
        if hasattr(las_data.header, 'generating_software'):
            header_info['generating_software'] = las_data.header.generating_software
        if hasattr(las_data.header, 'creation_date'):
            header_info['creation_date'] = str(las_data.header.creation_date)
        if hasattr(las_data.header, 'global_encoding'):
            try:
                # global_encoding可能是对象，需要转换
                ge = las_data.header.global_encoding
                if hasattr(ge, 'value'):
                    header_info['global_encoding'] = int(ge.value)
                else:
                    header_info['global_encoding'] = int(ge)
            except:
                pass
        
        # 保存坐标系信息（VLRs - Variable Length Records）
        vlrs_info = []
        if hasattr(las_data.header, 'vlrs'):
            for vlr in las_data.header.vlrs:
                vlr_dict = {
                    'user_id': vlr.user_id,
                    'record_id': vlr.record_id,
                    'description': vlr.description,
                }
                # 保存VLR数据（二进制）
                if hasattr(vlr, 'record_data'):
                    vlr_dict['record_data'] = bytes(vlr.record_data)
                vlrs_info.append(vlr_dict)
        header_info['vlrs'] = vlrs_info
        
        # 保存CRS信息（如果有）
        if hasattr(las_data, 'crs'):
            try:
                header_info['crs'] = str(las_data.crs)
            except:
                header_info['crs'] = None
        
        # 9. 统计全局类别分布 (Global Label Counts)
        global_label_counts = {}
        if 'classification' in data_dict:
            unique_labels, u_counts = np.unique(data_dict['classification'], return_counts=True)
            global_label_counts = {int(k): int(v) for k, v in zip(unique_labels, u_counts)}

        # 保存 PKL
        metadata = {
            'las_file': las_file.name,
            'num_points': len(las_data.points),
            'num_segments': len(segments_info),
            'fields': fields_to_save,
            'dtype': dtype_list,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'min_points': self.min_points,
            'max_points': self.max_points,
            'segments': segments_info,
            'grid_size': self.grid_size, # 记录生成索引时的 grid size
            'header_info': header_info,  # 🔥 补回头文件信息
            'label_counts': global_label_counts # 🔥 补回全局类别统计
        }
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        t1 = time.time()
        pkl_size = pkl_path.stat().st_size
        print(f"  {Colors.DIM}│{Colors.RESET}  📦 PKL: {Colors.GREEN}{format_time(t1-t0)}{Colors.RESET} → {Colors.CYAN}{format_size(pkl_size)}{Colors.RESET} ({len(segments_info)} 块)")
        print(f"  {Colors.DIM}│{Colors.RESET}  📋 字段: {Colors.CYAN}{', '.join(fields_to_save)}{Colors.RESET}")

if __name__ == "__main__":
    # 示例用法
    processor = LASProcessorLogicalIndex(
        input_path=r"E:\data\DALES\dales_las\train",
        output_dir=r"E:\data\DALES\dales_las\bin\train_logical",
        window_size=(50.0, 50.0),
        overlap=False, 
        grid_size=None,     # 统一 Grid Size
        min_points=5000,
        max_points=None,
        ground_class=None
    )
    processor.process_all_files()