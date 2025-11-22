import os

# 限制线程数，避免小任务的线程调度开销（维持之前的优化）
os.environ['NUMBA_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'

import numpy as np
import laspy
import pickle
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
from sklearn.neighbors import KDTree
from tqdm import tqdm
from collections import defaultdict
from numba import jit, prange


@jit(nopython=True, parallel=True)
def ravel_hash_vec_numba(arr, arr_min, arr_max):
    """
    Ravel hash function accelerated with numba.
    
    Args:
        arr: Input coordinates array (N, 3)
        arr_min: Minimum coordinates for each dimension
        arr_max: Maximum coordinates for each dimension
        
    Returns:
        Hash keys for each point
    """
    n = arr.shape[0]
    d = arr.shape[1]
    keys = np.zeros(n, dtype=np.uint64)
    
    # Normalize coordinates
    arr_normalized = np.empty_like(arr, dtype=np.uint64)
    for i in prange(n):
        for j in range(d):
            arr_normalized[i, j] = np.uint64(arr[i, j] - arr_min[j])
    
    # Calculate max + 1 for each dimension
    arr_max_plus_one = np.empty(d, dtype=np.uint64)
    for j in range(d):
        arr_max_plus_one[j] = np.uint64(arr_max[j] - arr_min[j] + 1)
    
    # Fortran style indexing
    for i in prange(n):
        key = np.uint64(0)
        for j in range(d - 1):
            key += arr_normalized[i, j]
            key *= arr_max_plus_one[j + 1]
        key += arr_normalized[i, d - 1]
        keys[i] = key
    
    return keys


@jit(nopython=True, parallel=True)
def compute_grid_coord_numba(coord, grid_size):
    """
    Compute grid coordinates accelerated with numba.
    
    Args:
        coord: Point coordinates (N, 3)
        grid_size: Grid size for sampling
        
    Returns:
        grid_coord: Grid coordinates (N, 3)
        scaled_coord: Scaled coordinates (N, 3)
    """
    n = coord.shape[0]
    scaled_coord = coord / grid_size
    grid_coord = np.floor(scaled_coord).astype(np.int64)
    return grid_coord, scaled_coord


@jit(nopython=True)
def shuffle_within_voxels_numba(idx_sort, cumsum_counts, count):
    """
    Shuffle points within each voxel using numba.
    
    Args:
        idx_sort: Sorted indices
        cumsum_counts: Cumulative sum of voxel counts
        count: Number of points in each voxel
        
    Returns:
        Shuffled idx_sort array
    """
    idx_sort_shuffled = idx_sort.copy()
    
    for i in range(len(count)):
        start_idx = cumsum_counts[i]
        end_idx = cumsum_counts[i + 1]
        
        # Fisher-Yates shuffle algorithm
        for j in range(end_idx - start_idx - 1, 0, -1):
            k = np.random.randint(0, j + 1)
            # Swap
            temp = idx_sort_shuffled[start_idx + j]
            idx_sort_shuffled[start_idx + j] = idx_sort_shuffled[start_idx + k]
            idx_sort_shuffled[start_idx + k] = temp
    
    return idx_sort_shuffled


@jit(nopython=True)
def sample_voxels_numba(idx_sort, cumsum_counts, count, num_loops, max_loops, points_per_loop):
    """
    Sample points from voxels using numba acceleration.
    
    Args:
        idx_sort: Sorted (and possibly shuffled) indices
        cumsum_counts: Cumulative sum of voxel counts
        count: Number of points in each voxel
        num_loops: Number of sampling loops
        max_loops: Maximum loops threshold
        points_per_loop: Points to sample per loop in extreme cases
        
    Returns:
        List of sampled index arrays
    """
    num_voxels = len(count)
    
    # Pre-allocate result arrays
    result_list = []
    
    for loop_idx in range(num_loops):
        # Estimate size for this loop
        estimated_size = 0
        for voxel_idx in range(num_voxels):
            voxel_count = count[voxel_idx]
            if voxel_count <= max_loops:
                estimated_size += 1
            else:
                sample_start = loop_idx * points_per_loop
                sample_end = min(sample_start + points_per_loop, voxel_count)
                if sample_start < voxel_count:
                    estimated_size += (sample_end - sample_start)
        
        # Allocate array for this loop
        idx_part = np.empty(estimated_size, dtype=np.int64)
        current_pos = 0
        
        for voxel_idx in range(num_voxels):
            voxel_count = count[voxel_idx]
            start_idx = cumsum_counts[voxel_idx]
            
            if voxel_count <= max_loops:
                # Normal case: sample one point
                local_idx = loop_idx % voxel_count
                idx_part[current_pos] = idx_sort[start_idx + local_idx]
                current_pos += 1
            else:
                # Extreme case: sample multiple points
                sample_start = loop_idx * points_per_loop
                sample_end = min(sample_start + points_per_loop, voxel_count)
                
                if sample_start < voxel_count:
                    for local_idx in range(sample_start, sample_end):
                        idx_part[current_pos] = idx_sort[start_idx + local_idx]
                        current_pos += 1
        
        # Only keep the filled portion
        if current_pos > 0:
            result_list.append(idx_part[:current_pos])
    
    return result_list


class GridSampler:
    """
    Grid sampling for point clouds using ravel hash with numba acceleration.
    Only returns point indices in test mode.
    """
    
    def __init__(self, grid_size=0.05, max_loops=30, shuffle_points=True):
        """
        Initialize grid sampler.
        
        Args:
            grid_size: Size of the grid cell for sampling
            max_loops: Maximum number of sampling iterations (to avoid extreme cases)
            shuffle_points: Whether to shuffle points within each voxel for randomness
        """
        self.grid_size = grid_size
        self.max_loops = max_loops
        self.shuffle_points = shuffle_points
    
    def sample(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Perform grid sampling on point cloud (test mode).
        Returns list of index arrays for each sampling iteration.
        
        Args:
            points: Point cloud array (N, 3) containing xyz coordinates
            
        Returns:
            List of index arrays, each corresponding to one sampling iteration
        """
        # 1. Compute grid coordinates using numba
        grid_coord, scaled_coord = compute_grid_coord_numba(
            points.astype(np.float64), 
            np.float64(self.grid_size)
        )
        
        # 2. Normalize grid coordinates
        min_coord = grid_coord.min(0)
        grid_coord = grid_coord - min_coord
        
        # 3. Compute hash using numba
        arr_min = np.zeros(3, dtype=np.int64)
        arr_max = grid_coord.max(0)
        key = ravel_hash_vec_numba(grid_coord, arr_min, arr_max)
        
        # 4. Sort by hash key
        idx_sort = np.argsort(key, kind='mergesort')
        key_sort = key[idx_sort]
        
        # 5. Get unique keys and counts
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        
        # 6. Prepare cumsum for voxel boundaries
        cumsum_counts = np.cumsum(np.insert(count, 0, 0))
        
        # 7. Shuffle points within each voxel for randomness (using numba)
        if self.shuffle_points:
            idx_sort = shuffle_within_voxels_numba(idx_sort, cumsum_counts, count)
        
        # 8. Test mode with max_loops control
        max_count = count.max()
        
        # 计算实际的循环次数和每次采样数
        if max_count <= self.max_loops:
            # 正常情况：每次采1个点
            num_loops = max_count
            points_per_loop = 1
        else:
            # 极端情况：限制循环次数，每次采多个点
            num_loops = self.max_loops
            points_per_loop = int(np.ceil(max_count / self.max_loops))
        
        # 9. Sample using numba-accelerated function
        data_part_list = sample_voxels_numba(
            idx_sort, cumsum_counts, count, 
            num_loops, self.max_loops, points_per_loop
        )
        
        return data_part_list


class LASProcessorToBinWithGridSample:
    def __init__(self,
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 window_size: Tuple[float, float] = (50.0, 50.0),
                 min_points: Optional[int] = 1000,
                 max_points: Optional[int] = 5000,
                 overlap: bool = False,
                 grid_size: Optional[float] = None,
                 max_loops: int = 30,
                 shuffle_points: bool = True,
                 ground_class: Optional[int] = 2):
        """
        Initialize LAS point cloud processor with grid sampling.
        
        Args:
            input_path: Path to LAS file or directory containing LAS files
            output_dir: Directory to save processed files (default: same as input)
            window_size: (x_size, y_size) for rectangular windows (in units of the LAS file)
            min_points: Minimum points threshold for a valid segment (None to skip)
            max_points: Maximum points threshold before further segmentation (None to skip)
            overlap: Whether to use overlap mode (offset grid by half window size)
            grid_size: Grid size for grid sampling (None to skip grid sampling)
            max_loops: Maximum number of sampling iterations for grid sampling
            shuffle_points: Whether to shuffle points within each voxel for randomness
            ground_class: Classification value for ground points (default: 2, None to skip is_ground generation)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.window_size = window_size
        self.min_points = min_points
        self.max_points = max_points
        self.overlap = overlap
        self.grid_size = grid_size
        self.max_loops = max_loops
        self.shuffle_points = shuffle_points
        self.ground_class = ground_class
        
        # Initialize grid sampler if grid_size is specified
        self.grid_sampler = GridSampler(grid_size, max_loops, shuffle_points) if grid_size is not None else None
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        self.las_files = self._find_las_files()
    
    def _find_las_files(self) -> List[Path]:
        """Find all LAS files in the input path."""
        if self.input_path.is_file() and self.input_path.suffix.lower() in ['.las', '.laz']:
            return [self.input_path]
        elif self.input_path.is_dir():
            return list(self.input_path.glob('*.las')) + list(self.input_path.glob('*.laz'))
        else:
            raise ValueError(f"Input path {self.input_path} is not a valid LAS file or directory")
    
    def process_all_files(self, n_workers: int = None):
        """
        Process all discovered LAS files.
        并行处理在单个LAS文件内部进行，而不是跨文件并行。
        
        Args:
            n_workers: Number of parallel workers for segment processing (None = auto)
        """
        import time
        import multiprocessing
        
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)
        
        start_time = time.time()
        
        print("="*70)
        print(f"Starting LAS to BIN/PKL conversion with Grid Sampling")
        print("="*70)
        print(f"Total files: {len(self.las_files)}")
        print(f"Window size: {self.window_size}")
        print(f"Min points: {self.min_points}")
        print(f"Max points: {self.max_points}")
        if self.grid_sampler:
            print(f"Grid sampling: ✅ Enabled")
            print(f"  - Grid size: {self.grid_size}")
            print(f"  - Max loops: {self.max_loops}")
            print(f"  - Shuffle points: {'✅ Yes' if self.shuffle_points else '❌ No'}")
        else:
            print(f"Grid sampling: ❌ Disabled")
        print(f"Overlap mode: {'✅ Enabled' if self.overlap else '❌ Disabled'}")
        if self.ground_class is not None:
            print(f"Ground classification: {self.ground_class} → is_ground field")
        else:
            print(f"Ground classification: ❌ Disabled")
        print(f"Parallel workers: {n_workers} (per file)")
        print("-"*70)
        
        # 顺序处理每个文件，但文件内部并行处理segments
        for las_file in tqdm(self.las_files, desc="Processing files", unit="file",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            try:
                self.process_file(las_file, n_workers=n_workers)
            except Exception as e:
                print(f"\n[ERROR] {las_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print(f"Conversion completed successfully!")
        print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")
        print(f"Average: {elapsed_time/len(self.las_files):.2f}s per file")
        print("="*70)
    
    def process_file(self, las_file: Union[str, Path], n_workers: int = 4):
        """
        Process a single LAS file and save to bin+pkl format.
        
        Args:
            las_file: Path to LAS file
            n_workers: Number of parallel workers for segment processing
        """
        import time
        las_file = Path(las_file)
        
        file_start = time.time()
        print(f"\n{'='*70}")
        print(f"📄 Processing: {las_file.name}")
        print(f"{'='*70}")
        
        # 1. 读取LAS文件
        t0 = time.time()
        with laspy.open(las_file) as fh:
            las_data = fh.read()
        t1 = time.time()
        print(f"  ✓ 读取LAS文件: {t1-t0:.2f}s ({len(las_data.points):,} 点)")
        
        # 2. 准备点云数据
        t0 = time.time()
        point_data = np.vstack((
            las_data.x, 
            las_data.y, 
            las_data.z
        )).transpose()
        t1 = time.time()
        print(f"  ✓ 准备点云数据: {t1-t0:.2f}s")
        
        # 3. 分割处理（这里会有详细子阶段输出）
        t0 = time.time()
        segments = self.segment_point_cloud(point_data, n_workers=n_workers)
        t1 = time.time()
        print(f"  ✓ 总分割时间: {t1-t0:.2f}s → {len(segments)} segments")
        
        # 4. 保存文件
        t0 = time.time()
        self.save_segments_as_bin_pkl(las_file, las_data, segments)
        t1 = time.time()
        print(f"  ✓ 保存文件: {t1-t0:.2f}s")
        
        file_total = time.time() - file_start
        print(f"  🎯 总计: {file_total:.2f}s")
        print(f"{'='*70}")
    
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
        
        print(f"  📦 开始分割 ({n_workers} workers)...")
        
        if not self.overlap:
            # 正常模式：单次网格分割
            t0 = time.time()
            segments = self._grid_segmentation(points, offset_x=0, offset_y=0, n_workers=n_workers)
            t1 = time.time()
            print(f"     单次网格分割: {t1-t0:.2f}s")
            return segments
        else:
            # Overlap模式：两次网格分割（偏移半个窗口）
            x_size, y_size = self.window_size
            
            # 第一次分割：正常网格
            t0 = time.time()
            segments1 = self._grid_segmentation(points, offset_x=0, offset_y=0, n_workers=n_workers)
            t1 = time.time()
            print(f"     第1次网格分割: {t1-t0:.2f}s → {len(segments1)} segments")
            
            # 第二次分割：偏移半个窗口
            t0 = time.time()
            segments2 = self._grid_segmentation(points, offset_x=x_size/2, offset_y=y_size/2, n_workers=n_workers)
            t1 = time.time()
            print(f"     第2次网格分割: {t1-t0:.2f}s → {len(segments2)} segments")
            
            # 合并两次分割结果
            all_segments = segments1 + segments2
            
            return all_segments
    
    def _grid_segmentation(self, points: np.ndarray, offset_x: float = 0, offset_y: float = 0, n_workers: int = 4) -> List[np.ndarray]:
        """
        Perform grid-based segmentation with optional offset.
        
        Args:
            points: Point cloud array (N, 3)
            offset_x: X offset for grid origin
            offset_y: Y offset for grid origin
            n_workers: Number of parallel workers
            
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
        
        # 过滤空 segment (np.unique 保证了 unique_ids 对应存在的 segments，通常不需要过滤，但 split 会产生第一个空如果索引0有值)
        # np.unique return_index 返回的是每个唯一值第一次出现的索引
        # 实际 segments 应该是 [split_indices[i]:split_indices[i+1]]
        
        t1 = time.time()
        print(f"       - 窗口分组: {t1-t0:.3f}s → {len(segments)} 窗口")
        
        # 2. Min阈值处理（优先处理，合并边界上点少的无效窗口）
        if self.min_points is not None:
            t0 = time.time()
            before_count = len(segments)
            segments = self.apply_min_threshold(points, segments, min_threshold=self.min_points)
            t1 = time.time()
            print(f"       - Min阈值处理: {t1-t0:.3f}s ({before_count} → {len(segments)} segments)")
        
        # 3. Grid Sampling处理（在Min和Max之间）
        if self.grid_sampler is not None:
            t0 = time.time()
            before_count = len(segments)
            total_points_before = sum(len(seg) for seg in segments)
            segments = self.apply_grid_sampling(points, segments)
            total_points_after = sum(len(seg) for seg in segments)
            t1 = time.time()
            print(f"       - Grid采样处理: {t1-t0:.3f}s ({before_count} → {len(segments)} segments, "
                  f"{total_points_before:,} → {total_points_after:,} points)")
        
        # 4. Max阈值处理（最后处理）
        if self.max_points is not None:
            t0 = time.time()
            before_count = len(segments)
            segments = self.apply_max_threshold(points, segments, n_workers=n_workers)
            t1 = time.time()
            print(f"       - Max阈值处理: {t1-t0:.3f}s ({before_count} → {len(segments)} segments)")
            
        return segments
    
    def apply_grid_sampling(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply grid sampling to each segment.
        
        Args:
            points: Point cloud array (N, 3)
            segments: List of segment indices
            
        Returns:
            List of sampled segment indices (expanded due to multiple sampling iterations)
        """
        sampled_segments = []
        
        for segment in segments:
            segment_points = points[segment]
            
            # Perform grid sampling (returns list of index arrays)
            sampled_indices_list = self.grid_sampler.sample(segment_points)
            
            # Convert local indices to global indices and add to result
            for local_indices in sampled_indices_list:
                global_indices = segment[local_indices]
                sampled_segments.append(global_indices)
        
        return sampled_segments
    
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
    
    def save_segments_as_bin_pkl(self, las_file: Path, las_data: laspy.LasData, segments: List[np.ndarray]):
        """
        Save segmented point clouds to bin+pkl format.
        
        Args:
            las_file: Original LAS file path
            las_data: Original LAS data
            segments: List of index arrays for segments
        """
        import time
        
        base_name = las_file.stem
        
        # 准备保存所有点云数据到一个bin文件
        bin_path = self.output_dir / f"{base_name}.bin"
        pkl_path = self.output_dir / f"{base_name}.pkl"
        
        print(f"  💾 保存到 bin+pkl...")
        
        # 1. 收集字段
        t0 = time.time()
        
        # 只保存真正有意义数据的字段
        # 必须保存的核心字段
        core_fields = ['X', 'Y', 'Z']
        
        # 可选但常用的字段（需要检查是否存在）
        optional_fields = ['intensity', 'return_number', 'number_of_returns', 
                          'classification', 'scan_angle_rank', 'user_data', 
                          'point_source_id', 'gps_time', 
                          'red', 'green', 'blue', 'nir',
                          'edge_of_flight_line']
        
        # 构建字段列表：只保存实际存在且有数据的字段
        fields_to_save = []
        dtype_list = []
        data_dict = {}
        
        # 保存核心字段（必须有）
        for field in core_fields:
            field_lower = field.lower()
            if hasattr(las_data, field_lower):
                data = getattr(las_data, field_lower)
                fields_to_save.append(field)
                data_dict[field] = data
                dtype_list.append((field, data.dtype))
        
        # 保存可选字段（只有存在时才保存）
        has_classification = False
        for field in optional_fields:
            field_lower = field.lower()
            if hasattr(las_data, field_lower):
                data = getattr(las_data, field_lower)
                fields_to_save.append(field)
                data_dict[field] = data
                dtype_list.append((field, data.dtype))
                if field_lower == 'classification':
                    has_classification = True
        
        # 如果没有classification，添加默认值0（这是唯一添加默认值的字段）
        if not has_classification:
            fields_to_save.append('classification')
            data_dict['classification'] = np.zeros(len(las_data.points), dtype=np.uint8)
            dtype_list.append(('classification', np.uint8))
        
        # 生成 is_ground 字段（基于 classification）
        if self.ground_class is not None and has_classification:
            is_ground = (las_data.classification == self.ground_class).astype(np.uint8)
            fields_to_save.append('is_ground')
            data_dict['is_ground'] = is_ground
            dtype_list.append(('is_ground', np.uint8))
        
        # 创建结构化数组
        structured_array = np.zeros(len(las_data.points), dtype=dtype_list)
        for field in fields_to_save:
            structured_array[field] = data_dict[field]
        
        t1 = time.time()
        print(f"     - 收集字段: {t1-t0:.3f}s ({len(fields_to_save)} 个字段： {fields_to_save})")
        
        # 2. 保存为bin文件
        t0 = time.time()
        structured_array.tofile(bin_path)
        t1 = time.time()
        bin_size_mb = bin_path.stat().st_size / (1024**2)
        print(f"     - 写入bin: {t1-t0:.3f}s ({bin_size_mb:.1f} MB)")
        
        # 准备pkl文件的元数据
        metadata = {
            'las_file': las_file.name,
            'num_points': len(las_data.points),
            'num_segments': len(segments),
            'fields': fields_to_save,
            'dtype': dtype_list,
            'window_size': self.window_size,
            'min_points': self.min_points,
            'max_points': self.max_points,
            'overlap': self.overlap,
            'grid_size': self.grid_size,
            'max_loops': self.max_loops if self.grid_size else None,
            'shuffle_points': self.shuffle_points if self.grid_size else None,
        }
        
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
        
        metadata['header_info'] = header_info
        
        # 统计整个文件的类别分布
        if has_classification:
            unique_labels, counts = np.unique(las_data.classification, return_counts=True)
            label_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        else:
            label_counts = {0: len(las_data.points)}
        metadata['label_counts'] = label_counts
        
        t1 = time.time()
        print(f"     - 准备metadata: {t1-t0:.3f}s")
        
        # 3. 收集每个分块的信息
        t0 = time.time()
        segments_info = []
        
        # 优化：预先获取 numpy 数组，避免在循环中反复访问 las_data 属性（可能触发 getter 开销）
        # 注意：使用 points 数组（如果之前已经有了）或者从 las_data 提取
        # 这里直接使用 las_data 的数组引用
        lx, ly, lz = las_data.x, las_data.y, las_data.z
        
        for i, segment_indices in enumerate(segments):
            segment_info = {
                'segment_id': i,
                'indices': segment_indices,
                'num_points': len(segment_indices),
                'bin_file': base_name,
                'bin_path': str(bin_path),
                'pkl_path': str(pkl_path),
            }
            
            # 优化：提取当前 segment 的坐标子集，只做一次切片
            seg_x = lx[segment_indices]
            seg_y = ly[segment_indices]
            seg_z = lz[segment_indices]
            
            # 计算边界（使用子集计算，快得多）
            segment_info['x_min'] = float(np.min(seg_x))
            segment_info['x_max'] = float(np.max(seg_x))
            segment_info['y_min'] = float(np.min(seg_y))
            segment_info['y_max'] = float(np.max(seg_y))
            segment_info['z_min'] = float(np.min(seg_z))
            segment_info['z_max'] = float(np.max(seg_z))
            
            segments_info.append(segment_info)
        
        metadata['segments'] = segments_info
        
        t1 = time.time()
        print(f"     - 收集segments信息: {t1-t0:.3f}s ({len(segments)} segments)")
        
        # 4. 保存pkl文件
        t0 = time.time()
        with open(pkl_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        t1 = time.time()
        pkl_size_mb = pkl_path.stat().st_size / (1024**2)
        print(f"     - 写入pkl: {t1-t0:.3f}s ({pkl_size_mb:.1f} MB)")


def process_las_files_to_bin_with_gridsample(input_path, output_dir=None, window_size=(50.0, 50.0), 
                                              min_points=None, max_points=None,
                                              overlap=False, grid_size=None,
                                              max_loops=30, shuffle_points=True,
                                              ground_class=2, n_workers=None):
    """
    Process LAS files with grid sampling and save to bin+pkl format.
    并行处理在单个LAS文件内部进行（处理segments），而不是跨文件并行。
    
    Args:
        input_path: Path to LAS file or directory containing LAS files
        output_dir: Directory to save processed files (default: same as input)
        window_size: (x_size, y_size) for rectangular windows
        min_points: Minimum points threshold for a valid segment
        max_points: Maximum points threshold before further segmentation
        overlap: Whether to use overlap mode (offset grid by half window size)
        grid_size: Grid size for grid sampling (None to skip grid sampling)
        max_loops: Maximum number of sampling iterations (to avoid extreme cases)
        shuffle_points: Whether to shuffle points within each voxel for randomness
        ground_class: Classification value for ground points (default: 2, None to skip is_ground generation)
        n_workers: Number of parallel workers for segment processing (None = auto, uses CPU count - 1)
    """
    processor = LASProcessorToBinWithGridSample(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        overlap=overlap,
        grid_size=grid_size,
        max_loops=max_loops,
        shuffle_points=shuffle_points,
        ground_class=ground_class
    )
    processor.process_all_files(n_workers=n_workers)


# 提供一个辅助函数用于加载数据
def load_segment_from_bin(bin_path: Union[str, Path], 
                          pkl_path: Union[str, Path], 
                          segment_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用np.memmap从bin文件中加载指定分块的数据。
    
    Args:
        bin_path: bin文件路径
        pkl_path: pkl文件路径
        segment_id: 要加载的分块ID
        
    Returns:
        (segment_data, segment_info): 分块的点云数据和元数据
    """
    bin_path = Path(bin_path)
    pkl_path = Path(pkl_path)
    
    # 加载元数据
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # 获取分块信息
    segment_info = metadata['segments'][segment_id]
    indices = segment_info['indices']
    
    # 使用memmap加载数据
    dtype = np.dtype(metadata['dtype'])
    mmap_data = np.memmap(bin_path, dtype=dtype, mode='r')
    
    # 读取指定分块的数据
    segment_data = mmap_data[indices]
    
    return segment_data, segment_info


def load_all_segments_info(pkl_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    加载所有分块的元数据信息（不加载实际点云数据）。
    
    Args:
        pkl_path: pkl文件路径
        
    Returns:
        所有分块的元数据列表
    """
    pkl_path = Path(pkl_path)
    
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return metadata['segments']


if __name__ == "__main__":
    # 示例：处理LAS文件（带Grid Sampling）
    input_path = r"E:\data\DALES\dales_las\test"
    output_dir = r"E:\data\DALES\dales_las\bin\test"
    window_size = (50.0, 50.0)
    min_points = 4096 * 5
    max_points = 4096 * 16 * 4
    overlap = False
    grid_size = None  # 🔥 设置grid size启用grid sampling
    max_loops = 10  # 🔥 grid size开启时的最大采样循环次数（避免极端情况）
    shuffle_points = True  # 🔥 打乱体素内点顺序（提高随机性）
    max_workers = 8  # 自动检测CPU核心数
    ground_class = None  # 🔥 地面点的classification值（None则不生成is_ground字段）
    
    # 处理文件（并行处理在单个LAS文件内部进行）
    process_las_files_to_bin_with_gridsample(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        overlap=overlap,
        grid_size=grid_size,  # 🔥 设置grid_size启用grid sampling（None则跳过）
        max_loops=max_loops,  # 🔥 最大循环次数（当体素内点>max_loops时，每次采样多个点）
        shuffle_points=shuffle_points,  # 🔥 是否打乱体素内点顺序
        ground_class=ground_class,  # 🔥 地面点classification值（2是LAS标准，None则不生成is_ground）
        n_workers=max_workers  # 🔥 并行worker数（None=自动，每个文件内部并行处理segments）
    )
    
    # 示例：如何加载数据
    # print("\n" + "="*50)
    # print("示例：如何加载分块数据")
    # print("="*50)
    
    # bin_file = Path(output_dir) / "5080_54400.bin"
    # pkl_file = Path(output_dir) / "5080_54400.pkl"
    
    # if bin_file.exists() and pkl_file.exists():
    #     # 加载所有分块信息
    #     all_segments = load_all_segments_info(pkl_file)
    #     print(f"\n总共有 {len(all_segments)} 个分块")
        
    #     # 加载第一个分块的数据
    #     if len(all_segments) > 0:
    #         segment_data, segment_info = load_segment_from_bin(bin_file, pkl_file, 0)
    #         print(f"\n第一个分块信息:")
    #         print(f"  - 点数: {segment_info['num_points']}")
    #         print(f"  - 类别: {segment_info['unique_labels']}")
    #         print(f"  - 类别分布: {segment_info['label_counts']}")
    #         print(f"\n点云数据shape: {segment_data.shape}")
    #         print(f"可用字段: {segment_data.dtype.names}")
    #         print(f"\n前5个点的xyz坐标:")
    #         # 字段名是大写的 X, Y, Z
    #         for i in range(min(5, len(segment_data))):
    #             print(f"  Point {i}: X={segment_data['X'][i]:.2f}, Y={segment_data['Y'][i]:.2f}, Z={segment_data['Z'][i]:.2f}, class={segment_data['classification'][i]}")
