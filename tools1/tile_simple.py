import numpy as np
import laspy
from pathlib import Path
from typing import Union, List, Tuple, Optional
from sklearn.neighbors import KDTree
from tqdm import tqdm

class LASProcessor:
    def __init__(self,
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 window_size: Tuple[float, float] = (50.0, 50.0),
                 min_points: Optional[int] = 1000,
                 max_points: Optional[int] = 5000):
        """
        Initialize LAS point cloud processor (simplified version without label processing).
        
        Args:
            input_path: Path to LAS file or directory containing LAS files
            output_dir: Directory to save processed files (default: same as input)
            window_size: (x_size, y_size) for rectangular windows (in units of the LAS file)
            min_points: Minimum points threshold for a valid segment (None to skip)
            max_points: Maximum points threshold before further segmentation (None to skip)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.window_size = window_size
        self.min_points = min_points
        self.max_points = max_points
        
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
    
    def process_all_files(self):
        """Process all discovered LAS files."""
        for las_file in tqdm(self.las_files, desc="Processing files", unit="file", position=0):
            print(f"Processing {las_file}...")
            self.process_file(las_file)
    
    def process_file(self, las_file: Union[str, Path]):
        """Process a single LAS file."""
        las_file = Path(las_file)
        
        with laspy.open(las_file) as fh:
            las_data = fh.read()
        
        # Get all points
        point_data = np.vstack((
            las_data.x, 
            las_data.y, 
            las_data.z
        )).transpose()
        
        segments = self.segment_point_cloud(point_data)
        
        self.save_segments_as_las(las_file, las_data, segments)
            
    def segment_point_cloud(self, points: np.ndarray) -> List[np.ndarray]:
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
        x_size, y_size = self.window_size
        
        num_windows_x = max(1, int(np.ceil((max_x - min_x) / x_size)))
        num_windows_y = max(1, int(np.ceil((max_y - min_y) / y_size)))
        
        # 计算每个点所属的窗口索引
        x_bins = np.clip(((points[:, 0] - min_x) / x_size).astype(int), 0, num_windows_x - 1)
        y_bins = np.clip(((points[:, 1] - min_y) / y_size).astype(int), 0, num_windows_y - 1)
        
        # 组合窗口索引
        window_ids = x_bins * num_windows_y + y_bins
        
        # 分组
        unique_ids, indices = np.unique(window_ids, return_inverse=True)
        segments = [np.where(indices == i)[0] for i in range(len(unique_ids))]
        
        # 后续阈值处理
        if self.max_points is not None:
            segments = self.apply_max_threshold(points, segments)
        if self.min_points is not None:
            segments = self.apply_min_threshold(points, segments)
            
        return segments
    
    def apply_max_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply max_points threshold to segments, subdividing large segments with parallel processing.
        
        Args:
            points: Nx3 array of point coordinates
            segments: List of index arrays representing segments
            
        Returns:
            Processed segments meeting the max threshold
        """
        # Quickly identify segments that need subdivision
        large_segment_indices = [i for i, segment in enumerate(segments) if len(segment) > self.max_points]
        
        if not large_segment_indices:
            return segments
        
        print(f"Subdividing {len(large_segment_indices)} segments with more than {self.max_points} points...")
        
        # Process large segments only, keeping small segments as is
        result_segments = [segment for i, segment in enumerate(segments) if i not in large_segment_indices]
        large_segments = [segments[i] for i in large_segment_indices]
        
        # Function to recursively subdivide a segment
        def process_segment(segment):
            if len(segment) <= self.max_points:
                return [segment]
            
            # Use existing subdivision logic
            segment_points = points[segment]
            ranges = np.ptp(segment_points[:, :2], axis=0)
            split_dim = np.argmax(ranges[:2])
            sorted_indices = np.argsort(segment_points[:, split_dim])
            
            # Split into two halves
            mid = len(sorted_indices) // 2
            left_half = segment[sorted_indices[:mid]]
            right_half = segment[sorted_indices[mid:]]
            
            # Recursively process both halves
            result = []
            result.extend(process_segment(left_half))
            result.extend(process_segment(right_half))
            return result
        
        # Process large segments with progress bar
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing
        
        max_workers = max(1, min(multiprocessing.cpu_count(), len(large_segments)))
        with tqdm(total=len(large_segments), desc="Subdividing large segments", unit="segment") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_segment, segment) for segment in large_segments]
                for future in as_completed(futures):
                    # Add the subdivided segments to our result
                    result_segments.extend(future.result())
                    pbar.update(1)
        
        print(f"Subdivision complete. {len(result_segments)} segments after processing.")
        return result_segments
    
    def apply_min_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply min_points threshold using KD-Tree for faster nearest segment finding.
        """
        if len(segments) <= 1:
            return segments
        
        print(f"Merging segments with fewer than {self.min_points} points using KD-Tree approach...")
        
        # Calculate all segment centroids upfront
        centroids = np.array([np.mean(points[segment][:, :2], axis=0) for segment in segments])
        
        # Identify small segments
        small_segments = [i for i, segment in enumerate(segments) if len(segment) < self.min_points]
        
        if not small_segments:
            return segments
        
        # Build KD-Tree from centroids of non-small segments
        valid_indices = [i for i in range(len(segments)) if i not in small_segments]
        if not valid_indices:  # No valid segments to merge with
            return segments
        
        valid_centroids = centroids[valid_indices]
        kdtree = KDTree(valid_centroids)
        
        # Process small segments in ascending size order (merge smallest first)
        small_segments.sort(key=lambda i: len(segments[i]))
        
        with tqdm(total=len(small_segments), desc="Merging small segments", unit="segment") as pbar:
            for small_idx in small_segments:
                if small_idx >= len(segments):  # Skip if segment was already removed
                    pbar.update(1)
                    continue
                
                # Find nearest non-small segment
                _, nearest_idx = kdtree.query([centroids[small_idx]], k=1)
                nearest_idx = valid_indices[nearest_idx[0][0]]
                
                if nearest_idx != small_idx and nearest_idx < len(segments):
                    # Merge small segment into nearest valid segment
                    segments[nearest_idx] = np.concatenate([segments[nearest_idx], segments[small_idx]])
                    # Mark segment for removal
                    segments[small_idx] = np.array([], dtype=int)
                
                pbar.update(1)
        
        # Remove empty segments
        return [segment for segment in segments if len(segment) > 0]
    
    def save_segments_as_las(self, las_file: Path, las_data: laspy.LasData, segments: List[np.ndarray]):
        """
        Save segmented point clouds to separate LAS files.
        
        Args:
            las_file: Original LAS file path
            las_data: Original LAS data
            segments: List of index arrays for segments
        """
        base_name = las_file.stem
        
        print(f"Saving {len(segments)} segments as LAS files...")
        for i, segment_indices in tqdm(enumerate(segments), total=len(segments), desc="Saving LAS segments", unit="file", position=0):
            # Create a copy of the header
            header = laspy.LasHeader(point_format=las_data.header.point_format, 
                                     version=las_data.header.version)
            
            # Copy scale and offset values from original header to maintain coordinate reference
            header.x_scale = las_data.header.x_scale
            header.y_scale = las_data.header.y_scale
            header.z_scale = las_data.header.z_scale
            header.x_offset = las_data.header.x_offset
            header.y_offset = las_data.header.y_offset
            header.z_offset = las_data.header.z_offset
            
            # Create a new LAS data with the correct point count
            new_las = laspy.LasData(header)
            
            # Create points array with the correct size
            new_las.points = laspy.ScaleAwarePointRecord.zeros(
                len(segment_indices),
                header=header
            )
            
            # Copy points from this segment
            for dimension in las_data.point_format.dimension_names:
                setattr(new_las, dimension, getattr(las_data, dimension)[segment_indices])
            
            # Also copy the header spatial reference system if it exists
            if hasattr(las_data.header, 'vlrs'):
                for vlr in las_data.header.vlrs:
                    new_las.header.vlrs.append(vlr)
                    
            # Copy CRS information if available
            if hasattr(las_data, 'crs'):
                new_las.crs = las_data.crs
                
            # Save to file
            output_path = self.output_dir / f"{base_name}_segment_{i:04d}.las"
            new_las.write(output_path)


def process_las_files(input_path, output_dir=None, window_size=(50.0, 50.0), 
                      min_points=None, max_points=None):
    """
    Process LAS files with simplified version (no label filtering/remapping/weighting).
    
    Args:
        input_path: Path to LAS file or directory containing LAS files
        output_dir: Directory to save processed files (default: same as input)
        window_size: (x_size, y_size) for rectangular windows
        min_points: Minimum points threshold for a valid segment
        max_points: Maximum points threshold before further segmentation
    """
    processor = LASProcessor(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points
    )
    processor.process_all_files()
    
    
if __name__ == "__main__":
    input_path = r"E:\data\Dales\dales_las\test"
    output_dir = r"E:\data\Dales\dales_las\tile_simple\test"
    window_size = (50.0, 50.0)
    min_points = 4096 * 2
    max_points = 4096 * 16 * 2
    
    process_las_files(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points
    )
