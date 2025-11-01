import numpy as np
import laspy
import pickle
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
from sklearn.neighbors import KDTree
from tqdm import tqdm
from collections import defaultdict


class LASProcessorToBin:
    def __init__(self,
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 window_size: Tuple[float, float] = (50.0, 50.0),
                 min_points: Optional[int] = 1000,
                 max_points: Optional[int] = 5000,
                 overlap: bool = False):
        """
        Initialize LAS point cloud processor that saves to bin+pkl format.
        
        Args:
            input_path: Path to LAS file or directory containing LAS files
            output_dir: Directory to save processed files (default: same as input)
            window_size: (x_size, y_size) for rectangular windows (in units of the LAS file)
            min_points: Minimum points threshold for a valid segment (None to skip)
            max_points: Maximum points threshold before further segmentation (None to skip)
            overlap: Whether to use overlap mode (offset grid by half window size)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.window_size = window_size
        self.min_points = min_points
        self.max_points = max_points
        self.overlap = overlap
        
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
        print(f"Starting LAS to BIN/PKL conversion")
        print("="*70)
        print(f"Total files: {len(self.las_files)}")
        print(f"Window size: {self.window_size}")
        print(f"Min points: {self.min_points}")
        print(f"Max points: {self.max_points}")
        print(f"Overlap mode: {'✅ Enabled' if self.overlap else '❌ Disabled'}")
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
        
        # 1. 窗口分组
        t0 = time.time()
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
        x_size, y_size = self.window_size
        
        # 应用偏移
        origin_x = min_x - offset_x
        origin_y = min_y - offset_y
        
        num_windows_x = max(1, int(np.ceil((max_x - origin_x) / x_size)))
        num_windows_y = max(1, int(np.ceil((max_y - origin_y) / y_size)))
        
        # 计算每个点所属的窗口索引
        x_bins = np.clip(((points[:, 0] - origin_x) / x_size).astype(int), 0, num_windows_x - 1)
        y_bins = np.clip(((points[:, 1] - origin_y) / y_size).astype(int), 0, num_windows_y - 1)
        
        # 组合窗口索引
        window_ids = x_bins * num_windows_y + y_bins
        
        # 分组
        unique_ids, indices = np.unique(window_ids, return_inverse=True)
        segments = [np.where(indices == i)[0] for i in range(len(unique_ids))]
        t1 = time.time()
        print(f"       - 窗口分组: {t1-t0:.3f}s → {len(segments)} 窗口")
        
        # 2. Min阈值处理（优先处理，合并边界上点少的无效窗口）
        if self.min_points is not None:
            t0 = time.time()
            before_count = len(segments)
            segments = self.apply_min_threshold(points, segments, min_threshold=self.min_points)
            t1 = time.time()
            print(f"       - Min阈值处理: {t1-t0:.3f}s ({before_count} → {len(segments)} segments)")
        
        # 3. Max阈值处理（最后处理）
        if self.max_points is not None:
            t0 = time.time()
            before_count = len(segments)
            segments = self.apply_max_threshold(points, segments, n_workers=n_workers)
            t1 = time.time()
            print(f"       - Max阈值处理: {t1-t0:.3f}s ({before_count} → {len(segments)} segments)")
            
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
                          'red', 'green', 'blue', 'nir']
        
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
        
        # 创建结构化数组
        structured_array = np.zeros(len(las_data.points), dtype=dtype_list)
        for field in fields_to_save:
            structured_array[field] = data_dict[field]
        
        t1 = time.time()
        print(f"     - 收集字段: {t1-t0:.3f}s ({len(fields_to_save)} 字段)")
        
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
        for i, segment_indices in enumerate(segments):
            segment_info = {
                'segment_id': i,
                'indices': segment_indices,  # 在bin文件中的索引
                'num_points': len(segment_indices),
            }
            
            # 统计该分块中的类别信息
            if has_classification:
                segment_labels = las_data.classification[segment_indices]
                unique_segment_labels, segment_counts = np.unique(segment_labels, return_counts=True)
                segment_label_counts = {int(label): int(count) for label, count in zip(unique_segment_labels, segment_counts)}
                segment_info['unique_labels'] = [int(label) for label in unique_segment_labels]
                segment_info['label_counts'] = segment_label_counts
            else:
                segment_info['unique_labels'] = [0]
                segment_info['label_counts'] = {0: len(segment_indices)}
            
            # 计算分块的边界信息 - 直接从原始las_data获取
            segment_info['x_min'] = float(np.min(las_data.x[segment_indices]))
            segment_info['x_max'] = float(np.max(las_data.x[segment_indices]))
            segment_info['y_min'] = float(np.min(las_data.y[segment_indices]))
            segment_info['y_max'] = float(np.max(las_data.y[segment_indices]))
            segment_info['z_min'] = float(np.min(las_data.z[segment_indices]))
            segment_info['z_max'] = float(np.max(las_data.z[segment_indices]))
            
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


def process_las_files_to_bin(input_path, output_dir=None, window_size=(50.0, 50.0), 
                              min_points=None, max_points=None,
                              overlap=False,
                              n_workers=None):
    """
    Process LAS files and save to bin+pkl format.
    并行处理在单个LAS文件内部进行（处理segments），而不是跨文件并行。
    
    Args:
        input_path: Path to LAS file or directory containing LAS files
        output_dir: Directory to save processed files (default: same as input)
        window_size: (x_size, y_size) for rectangular windows
        min_points: Minimum points threshold for a valid segment
        max_points: Maximum points threshold before further segmentation
        overlap: Whether to use overlap mode (offset grid by half window size)
        n_workers: Number of parallel workers for segment processing (None = auto, uses CPU count - 1)
    """
    processor = LASProcessorToBin(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        overlap=overlap
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
    # 示例：处理LAS文件
    input_path = r"E:\data\云南遥感中心\第一批\train"
    output_dir = r"E:\data\云南遥感中心\第一批\bin\train"
    window_size = (150.0, 150.0)
    min_points = 4096 * 2
    max_points = 4096 * 16 * 2
    overlap = False
    use_parallel = True
    max_workers = None  # 自动检测CPU核心数
    
    # 处理文件（并行处理在单个LAS文件内部进行）
    process_las_files_to_bin(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        overlap=overlap,    # 🔥 设置为True启用overlap模式（segment数量约翻倍）
        n_workers=max_workers # 🔥 并行worker数（None=自动，每个文件内部并行处理segments）
    )
    
    # 示例：如何加载数据
    print("\n" + "="*50)
    print("示例：如何加载分块数据")
    print("="*50)
    
    bin_file = Path(output_dir) / "5080_54400.bin"
    pkl_file = Path(output_dir) / "5080_54400.pkl"
    
    if bin_file.exists() and pkl_file.exists():
        # 加载所有分块信息
        all_segments = load_all_segments_info(pkl_file)
        print(f"\n总共有 {len(all_segments)} 个分块")
        
        # 加载第一个分块的数据
        if len(all_segments) > 0:
            segment_data, segment_info = load_segment_from_bin(bin_file, pkl_file, 0)
            print(f"\n第一个分块信息:")
            print(f"  - 点数: {segment_info['num_points']}")
            print(f"  - 类别: {segment_info['unique_labels']}")
            print(f"  - 类别分布: {segment_info['label_counts']}")
            print(f"\n点云数据shape: {segment_data.shape}")
            print(f"可用字段: {segment_data.dtype.names}")
            print(f"\n前5个点的xyz坐标:")
            # 字段名是大写的 X, Y, Z
            for i in range(min(5, len(segment_data))):
                print(f"  Point {i}: X={segment_data['X'][i]:.2f}, Y={segment_data['Y'][i]:.2f}, Z={segment_data['Z'][i]:.2f}, class={segment_data['classification'][i]}")
