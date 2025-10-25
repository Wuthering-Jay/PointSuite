"""
将Las/LAZ切片并保存为H5格式
"""
import laspy
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple, List, Union
from tqdm import tqdm
from sklearn.neighbors import KDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time


class H5Processor:
    """超快H5处理器"""
    
    def __init__(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        window_size: Tuple[float, float] = (150., 150.),
        min_points: Optional[int] = 4096 * 2,
        max_points: Optional[int] = 4096 * 16 * 2,
        overlap: bool = False,
        n_workers: Optional[int] = None
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.min_points = min_points
        self.max_points = max_points
        self.overlap = overlap
        self.n_workers = n_workers or multiprocessing.cpu_count()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.las_files = self._find_las_files()
    
    def _find_las_files(self) -> List[Path]:
        """查找所有LAS文件"""
        if self.input_path.is_file() and self.input_path.suffix.lower() in ['.las', '.laz']:
            return [self.input_path]
        elif self.input_path.is_dir():
            return list(self.input_path.glob('*.las')) + list(self.input_path.glob('*.laz'))
        else:
            raise ValueError(f"Invalid input path: {self.input_path}")
    
    def run(self):
        """处理所有文件"""
        print(f"\n找到{len(self.las_files)}个LAS/LAZ文件")
        
        if len(self.las_files) == 1:
            # 单文件直接处理
            print(f"\n处理: {self.las_files[0].name}")
            self.process_file(self.las_files[0])
        else:
            # 多文件并行处理
            print(f"\n使用{self.n_workers}个进程并行处理...")
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_file = {
                    executor.submit(
                        self._process_file_worker,
                        las_file,
                        self.output_dir,
                        self.window_size,
                        self.min_points,
                        self.max_points,
                        self.overlap
                    ): las_file
                    for las_file in self.las_files
                }
                
                for future in future_to_file:
                    las_file = future_to_file[future]
                    try:
                        num_segs, file_size, elapsed = future.result()
                        print(f"  ✅ {las_file.name}: {num_segs} segments, "
                              f"{file_size:.1f}MB, {elapsed:.2f}秒")
                    except Exception as e:
                        print(f"  ❌ {las_file.name}: 错误 - {e}")
    
    @staticmethod
    def _process_file_worker(
        las_file: Path,
        output_dir: Path,
        window_size: Tuple[float, float],
        min_points: Optional[int],
        max_points: Optional[int],
        overlap: bool
    ) -> Tuple[int, float, float]:
        """
        Worker函数 - 由多进程调用
        
        Returns:
            (num_segments, file_size_mb, elapsed_seconds)
        """
        processor = H5Processor.__new__(H5Processor)
        processor.output_dir = output_dir
        processor.window_size = window_size
        processor.min_points = min_points
        processor.max_points = max_points
        processor.overlap = overlap
        
        return processor.process_file(las_file)
    
    def process_file(self, las_file: Path) -> Tuple[int, float, float]:
        """
        处理单个LAS文件
        
        Returns:
            (num_segments, file_size_mb, elapsed_seconds)
        """
        start_time = time.time()
        
        # 读取LAS
        with laspy.open(las_file) as fh:
            las_data = fh.read()
        
        # 提取点云数据
        points = np.vstack([las_data.x, las_data.y, las_data.z]).T
        
        # 分割
        segments = self._segment_point_cloud(las_data, points)
        
        # 写入H5
        output_path = self.output_dir / f"{las_file.stem}.h5"
        self._save_to_h5(output_path, las_data, segments)
        
        # 统计
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        elapsed = time.time() - start_time
        
        return len(segments), file_size_mb, elapsed
    
    def _segment_point_cloud(
        self,
        las_data: laspy.LasData,
        points: np.ndarray
    ) -> List[np.ndarray]:
        """
        点云分割
        
        overlap模式：
        - 先按正常网格分割一次，应用阈值处理
        - 再按偏移半个网格的位置分割一次，独立应用阈值处理
        - 最后合并两次分割的结果，实现重叠分块
        
        关键：两次分割的阈值处理是独立的，避免合并时的混乱
        """
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
        x_size, y_size = self.window_size
        
        # 第一次分割：正常网格
        segments = self._grid_segmentation(points, min_x, min_y, max_x, max_y, 
                                          x_size, y_size, offset_x=0, offset_y=0)
        
        # 对第一次分割应用阈值处理
        if self.max_points is not None:
            segments = self._apply_max_threshold(points, segments)
        if self.min_points is not None:
            segments = self._apply_min_threshold(points, segments)
        
        # Overlap模式：添加偏移网格分割（独立处理）
        if self.overlap:
            offset_segments = self._grid_segmentation(
                points, min_x, min_y, max_x, max_y,
                x_size, y_size, 
                offset_x=x_size / 2,  # 偏移半个网格
                offset_y=y_size / 2
            )
            
            # 对偏移分割独立应用阈值处理
            if self.max_points is not None:
                offset_segments = self._apply_max_threshold(points, offset_segments)
            if self.min_points is not None:
                offset_segments = self._apply_min_threshold(points, offset_segments)
            
            # 合并两次分割的结果
            segments.extend(offset_segments)
        
        return segments
    
    def _grid_segmentation(
        self,
        points: np.ndarray,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        x_size: float,
        y_size: float,
        offset_x: float = 0,
        offset_y: float = 0
    ) -> List[np.ndarray]:
        """
        网格分割（支持偏移）
        
        Args:
            points: 点云数据
            min_x, min_y, max_x, max_y: 点云范围
            x_size, y_size: 网格大小
            offset_x, offset_y: 网格偏移量（用于overlap模式）
        
        Returns:
            segments列表
        """
        # 应用偏移
        min_x_offset = min_x + offset_x
        min_y_offset = min_y + offset_y
        
        # 计算网格数量
        num_windows_x = max(1, int(np.ceil((max_x - min_x_offset) / x_size)))
        num_windows_y = max(1, int(np.ceil((max_y - min_y_offset) / y_size)))
        
        # 计算每个点所属的网格
        x_bins = ((points[:, 0] - min_x_offset) / x_size).astype(int)
        y_bins = ((points[:, 1] - min_y_offset) / y_size).astype(int)
        
        # 裁剪到有效范围
        x_bins = np.clip(x_bins, 0, num_windows_x - 1)
        y_bins = np.clip(y_bins, 0, num_windows_y - 1)
        
        # 组合窗口ID
        window_ids = x_bins * num_windows_y + y_bins
        
        # 分组
        unique_ids, indices = np.unique(window_ids, return_inverse=True)
        segments = [np.where(indices == i)[0] for i in range(len(unique_ids))]
        
        # 过滤空segment
        segments = [seg for seg in segments if len(seg) > 0]
        
        return segments
    
    def _apply_max_threshold(
        self,
        points: np.ndarray,
        segments: List[np.ndarray]
    ) -> List[np.ndarray]:
        """递归二分法处理大segment"""
        large_indices = [i for i, seg in enumerate(segments) if len(seg) > self.max_points]
        
        if not large_indices:
            return segments
        
        result = [seg for i, seg in enumerate(segments) if i not in large_indices]
        large_segs = [segments[i] for i in large_indices]
        
        def subdivide(segment):
            """递归二分"""
            if len(segment) <= self.max_points:
                return [segment]
            
            seg_points = points[segment]
            ranges = np.ptp(seg_points[:, :2], axis=0)
            split_dim = np.argmax(ranges)
            sorted_indices = np.argsort(seg_points[:, split_dim])
            
            mid = len(sorted_indices) // 2
            left = segment[sorted_indices[:mid]]
            right = segment[sorted_indices[mid:]]
            
            return subdivide(left) + subdivide(right)
        
        # 线程并行（CPU密集型任务）
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(8, len(large_segs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(subdivide, seg) for seg in large_segs]
            for future in futures:
                result.extend(future.result())
        
        return result
    
    def _apply_min_threshold(
        self,
        points: np.ndarray,
        segments: List[np.ndarray]
    ) -> List[np.ndarray]:
        """KD-Tree合并小segment"""
        if len(segments) <= 1:
            return segments
        
        centroids = np.array([np.mean(points[seg][:, :2], axis=0) for seg in segments])
        small_indices = [i for i, seg in enumerate(segments) if len(seg) < self.min_points]
        
        if not small_indices:
            return segments
        
        valid_indices = [i for i in range(len(segments)) if i not in small_indices]
        if not valid_indices:
            return segments
        
        kdtree = KDTree(centroids[valid_indices])
        small_indices.sort(key=lambda i: len(segments[i]))
        
        for small_idx in small_indices:
            if small_idx >= len(segments):
                continue
            
            _, nearest = kdtree.query([centroids[small_idx]], k=1)
            nearest_idx = valid_indices[nearest[0][0]]
            
            if nearest_idx != small_idx and nearest_idx < len(segments):
                segments[nearest_idx] = np.concatenate([
                    segments[nearest_idx],
                    segments[small_idx]
                ])
                segments[small_idx] = np.array([], dtype=int)
        
        return [seg for seg in segments if len(seg) > 0]
    
    def _save_to_h5(
        self,
        output_path: Path,
        las_data: laspy.LasData,
        segments: List[np.ndarray]
    ):
        """
        保存为快速H5格式
        
        关键：
        - 每个segment单独存储所有LAS字段
        - chunks=None → contiguous layout
        - 无压缩 → 最快读取
        """
        with h5py.File(output_path, 'w') as f:
            # Header
            header = f.create_group('header')
            header.attrs['num_points'] = len(las_data.points)
            header.attrs['point_format'] = las_data.header.point_format.id
            header.attrs['version_major'] = las_data.header.version.major
            header.attrs['version_minor'] = las_data.header.version.minor
            header.attrs['x_scale'] = las_data.header.x_scale
            header.attrs['y_scale'] = las_data.header.y_scale
            header.attrs['z_scale'] = las_data.header.z_scale
            header.attrs['x_offset'] = las_data.header.x_offset
            header.attrs['y_offset'] = las_data.header.y_offset
            header.attrs['z_offset'] = las_data.header.z_offset
            
            # CRS信息
            if hasattr(las_data, 'crs') and las_data.crs is not None:
                header.attrs['crs'] = str(las_data.crs)
            
            # 确定可用字段
            available_fields = []
            for field in las_data.point_format.dimension_names:
                # 跳过原始整数坐标(X,Y,Z)，我们保存浮点坐标(x,y,z)
                if field in ['X', 'Y', 'Z']:
                    continue
                if hasattr(las_data, field):
                    available_fields.append(field)
            
            # 确保包含xyz坐标
            for coord in ['x', 'y', 'z']:
                if coord not in available_fields:
                    available_fields.insert(0, coord)
            
            header.attrs['available_fields'] = ','.join(available_fields)
            
            # Segments
            segs_group = f.create_group('segments')
            segs_group.attrs['num_segments'] = len(segments)
            
            # 写入每个segment
            for i, indices in enumerate(segments):
                seg_group = segs_group.create_group(f'segment_{i:04d}')
                
                # 保存所有可用字段（contiguous, no compression）
                for field in available_fields:
                    field_data = getattr(las_data, field)[indices]
                    
                    # 确定数据类型
                    if field in ['x', 'y', 'z']:
                        dtype = np.float64
                    elif field == 'classification':
                        dtype = np.int32
                    elif field in ['intensity', 'red', 'green', 'blue', 'point_source_id']:
                        dtype = np.uint16
                    elif field in ['return_number', 'number_of_returns', 'user_data']:
                        dtype = np.uint8
                    elif field == 'scan_angle_rank':
                        dtype = np.int8
                    elif field == 'gps_time':
                        dtype = np.float64
                    else:
                        dtype = field_data.dtype
                    
                    seg_group.create_dataset(
                        field,
                        data=field_data,
                        dtype=dtype,
                        chunks=None  # Contiguous!
                    )
                
                # Metadata
                seg_group.attrs['num_points'] = len(indices)


def process_las_to_h5(
    input_path: str,
    output_dir: str,
    window_size: Tuple[float, float] = (150., 150.),
    min_points: Optional[int] = 4096 * 2,
    max_points: Optional[int] = 4096 * 16 * 2,
    overlap: bool = False,
    n_workers: Optional[int] = None
):
    """
    快速处理LAS到H5
    
    Args:
        input_path: LAS文件或目录
        output_dir: 输出目录
        window_size: 窗口大小
        min_points: 最小点数
        max_points: 最大点数
        overlap: 是否启用重叠模式（会生成2倍的segments）
        n_workers: 并行进程数
    """
    processor = H5Processor(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        overlap=overlap,
        n_workers=n_workers
    )
    processor.run()


if __name__ == "__main__":
    # 配置
    input_path = r"E:\data\DALES\dales_las\train" # 输入LAS/LAZ目录或文件
    output_dir = r"E:\data\DALES\dales_las\h5\train" # 输出H5目录
    window_size = (50., 50.) # 窗口大小
    min_points = 4096 * 2 # 最小点数
    max_points = 4096 * 16 * 2 # 最大点数
    overlap = True  # 是否启用重叠分块（True会生成约2倍segments）
    n_workers = 4  # 并行进程数
    
    print("="*70)
    print("Las/LAZ 切片并保存为快速H5格式")
    print("="*70)
    print(f"输入: {input_path}")
    print(f"输出: {output_dir}")
    print(f"窗口大小: {window_size}")
    print(f"点数范围: {min_points} - {max_points}")
    print(f"重叠模式: {'✅ 开启' if overlap else '❌ 关闭'}")
    print(f"并行进程: {n_workers}")
    print("="*70)
    
    start = time.time()
    process_las_to_h5(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        overlap=overlap,
        n_workers=n_workers
    )
    elapsed = time.time() - start
    
    print("="*70)
    print(f"✅ 完成！总耗时: {elapsed:.2f}秒")
    print("="*70)
