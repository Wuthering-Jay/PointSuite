import json
import numpy as np
import laspy
import h5py
from pathlib import Path
from typing import Union, List, Tuple, Optional
from sklearn.neighbors import KDTree
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

class LASToH5Processor:
    def __init__(self,
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 window_size: Tuple[float, float] = (50.0, 50.0),
                 min_points: Optional[int] = 1000,
                 max_points: Optional[int] = 5000,
                 n_workers: Optional[int] = None):
        """
        Initialize LAS to H5 processor.
        
        Args:
            input_path: Path to LAS file or directory containing LAS files
            output_dir: Directory to save H5 files (default: same as input)
            window_size: (x_size, y_size) for rectangular windows (in units of the LAS file)
            min_points: Minimum points threshold for a valid segment (None to skip)
            max_points: Maximum points threshold before further segmentation (None to skip)
            n_workers: Number of parallel workers for processing multiple files (default: CPU count)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.window_size = window_size
        self.min_points = min_points
        self.max_points = max_points
        self.n_workers = n_workers or multiprocessing.cpu_count()
        
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
        """Process all discovered LAS files with parallel processing."""
        if len(self.las_files) == 1:
            # Single file, process directly
            print(f"\nProcessing {self.las_files[0]}...")
            self.process_file(self.las_files[0])
        else:
            # Multiple files, use parallel processing
            print(f"\nProcessing {len(self.las_files)} files with {self.n_workers} workers...")
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(self._process_file_worker, las_file, self.output_dir, 
                                  self.window_size, self.min_points, self.max_points): las_file 
                    for las_file in self.las_files
                }
                
                # Process results as they complete
                for future in tqdm(as_completed(future_to_file), total=len(self.las_files), 
                                 desc="Processing files", unit="file"):
                    las_file = future_to_file[future]
                    try:
                        num_segments = future.result()
                        print(f"\n  {las_file.name}: {num_segments} segments")
                    except Exception as e:
                        print(f"\n  Error processing {las_file.name}: {e}")
    
    @staticmethod
    def _process_file_worker(las_file: Path, output_dir: Path, window_size: Tuple[float, float],
                            min_points: Optional[int], max_points: Optional[int]) -> int:
        """
        Worker function for parallel processing of LAS files.
        
        Returns:
            Number of segments created
        """
        # Create a temporary processor instance for this worker
        processor = LASToH5Processor.__new__(LASToH5Processor)
        processor.output_dir = output_dir
        processor.window_size = window_size
        processor.min_points = min_points
        processor.max_points = max_points
        
        return processor.process_file(las_file)
    
    def process_file(self, las_file: Union[str, Path]) -> int:
        """
        Process a single LAS file and save to H5.
        
        Returns:
            Number of segments created
        """
        las_file = Path(las_file)
        
        with laspy.open(las_file) as fh:
            las_data = fh.read()
        
        # Extract all point data
        point_data = np.vstack((
            las_data.x, 
            las_data.y, 
            las_data.z
        )).transpose()
        
        # Perform segmentation
        segments = self.segment_point_cloud(point_data)
        
        # Get classification data (default to 0 if not present)
        if hasattr(las_data, 'classification'):
            classification = np.array(las_data.classification, dtype=np.int32)
        else:
            classification = np.zeros(len(las_data.points), dtype=np.int32)
        
        # Calculate label statistics
        unique_labels, label_counts = np.unique(classification, return_counts=True)
        label_stats = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
        
        # Save to H5 file
        output_path = self.output_dir / f"{las_file.stem}.h5"
        self.save_to_h5(output_path, las_data, segments, label_stats)
        
        return len(segments)
    
    def segment_point_cloud(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Segment the point cloud into rectangular windows and apply thresholds.
        
        Args:
            points: Nx3 array of point coordinates (x, y, z)
            
        Returns:
            List of index arrays, each representing points in a segment
        """
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
        x_size, y_size = self.window_size
        
        num_windows_x = max(1, int(np.ceil((max_x - min_x) / x_size)))
        num_windows_y = max(1, int(np.ceil((max_y - min_y) / y_size)))
        
        # Calculate window index for each point
        x_bins = np.clip(((points[:, 0] - min_x) / x_size).astype(int), 0, num_windows_x - 1)
        y_bins = np.clip(((points[:, 1] - min_y) / y_size).astype(int), 0, num_windows_y - 1)
        
        # Combine window indices
        window_ids = x_bins * num_windows_y + y_bins
        
        # Group points by window
        unique_ids, indices = np.unique(window_ids, return_inverse=True)
        segments = [np.where(indices == i)[0] for i in range(len(unique_ids))]
        
        # Apply thresholds
        if self.max_points is not None:
            segments = self.apply_max_threshold(points, segments)
        if self.min_points is not None:
            segments = self.apply_min_threshold(points, segments)
            
        return segments
    
    def apply_max_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply max_points threshold to segments, subdividing large segments.
        
        Args:
            points: Nx3 array of point coordinates
            segments: List of index arrays representing segments
            
        Returns:
            Processed segments meeting the max threshold
        """
        large_segment_indices = [i for i, segment in enumerate(segments) if len(segment) > self.max_points]
        
        if not large_segment_indices:
            return segments
        
        print(f"  Subdividing {len(large_segment_indices)} segments with more than {self.max_points} points...")
        
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
        import multiprocessing
        
        max_workers = max(1, min(multiprocessing.cpu_count(), len(large_segments)))
        with tqdm(total=len(large_segments), desc="  Subdividing", unit="segment", leave=False) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_segment, segment) for segment in large_segments]
                for future in as_completed(futures):
                    result_segments.extend(future.result())
                    pbar.update(1)
        
        return result_segments
    
    def apply_min_threshold(self, points: np.ndarray, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply min_points threshold using KD-Tree for faster nearest segment finding.
        """
        if len(segments) <= 1:
            return segments
        
        print(f"  Merging segments with fewer than {self.min_points} points...")
        
        centroids = np.array([np.mean(points[segment][:, :2], axis=0) for segment in segments])
        small_segments = [i for i, segment in enumerate(segments) if len(segment) < self.min_points]
        
        if not small_segments:
            return segments
        
        valid_indices = [i for i in range(len(segments)) if i not in small_segments]
        if not valid_indices:
            return segments
        
        valid_centroids = centroids[valid_indices]
        kdtree = KDTree(valid_centroids)
        
        small_segments.sort(key=lambda i: len(segments[i]))
        
        with tqdm(total=len(small_segments), desc="  Merging", unit="segment", leave=False) as pbar:
            for small_idx in small_segments:
                if small_idx >= len(segments):
                    pbar.update(1)
                    continue
                
                _, nearest_idx = kdtree.query([centroids[small_idx]], k=1)
                nearest_idx = valid_indices[nearest_idx[0][0]]
                
                if nearest_idx != small_idx and nearest_idx < len(segments):
                    segments[nearest_idx] = np.concatenate([segments[nearest_idx], segments[small_idx]])
                    segments[small_idx] = np.array([], dtype=int)
                
                pbar.update(1)
        
        return [segment for segment in segments if len(segment) > 0]
    
    def save_to_h5(self, output_path: Path, las_data: laspy.LasData, 
                   segments: List[np.ndarray], label_stats: dict):
        """
        Save LAS data and segments to H5 file.
        
        Args:
            output_path: Path to output H5 file
            las_data: Original LAS data
            segments: List of segment indices
            label_stats: Dictionary of label counts
        """
        with h5py.File(output_path, 'w') as h5f:
            # Save header information
            header_group = h5f.create_group('header')
            header_group.attrs['file_name'] = str(las_data.header.file_source_id)
            header_group.attrs['point_format'] = las_data.header.point_format.id
            header_group.attrs['version_major'] = las_data.header.version.major
            header_group.attrs['version_minor'] = las_data.header.version.minor
            header_group.attrs['point_count'] = len(las_data.points)
            
            # Save scale and offset
            header_group.attrs['x_scale'] = las_data.header.x_scale
            header_group.attrs['y_scale'] = las_data.header.y_scale
            header_group.attrs['z_scale'] = las_data.header.z_scale
            header_group.attrs['x_offset'] = las_data.header.x_offset
            header_group.attrs['y_offset'] = las_data.header.y_offset
            header_group.attrs['z_offset'] = las_data.header.z_offset
            
            # Save bounding box
            header_group.attrs['x_min'] = las_data.header.x_min
            header_group.attrs['y_min'] = las_data.header.y_min
            header_group.attrs['z_min'] = las_data.header.z_min
            header_group.attrs['x_max'] = las_data.header.x_max
            header_group.attrs['y_max'] = las_data.header.y_max
            header_group.attrs['z_max'] = las_data.header.z_max
            
            # Save CRS information if available
            if hasattr(las_data, 'crs') and las_data.crs is not None:
                header_group.attrs['crs'] = str(las_data.crs)
            
            # Save VLRs if available
            if hasattr(las_data.header, 'vlrs') and len(las_data.header.vlrs) > 0:
                vlr_group = header_group.create_group('vlrs')
                for idx, vlr in enumerate(las_data.header.vlrs):
                    vlr_subgroup = vlr_group.create_group(f'vlr_{idx}')
                    vlr_subgroup.attrs['user_id'] = vlr.user_id
                    vlr_subgroup.attrs['record_id'] = vlr.record_id
                    vlr_subgroup.attrs['description'] = vlr.description
            
            # Save all point data fields
            data_group = h5f.create_group('data')
            
            # Compression parameters optimized for fast reading
            # Using gzip (built-in, reliable) with chunking for efficient random access
            # Chunk size optimized for typical segment sizes (4K-16K points)
            num_points = len(las_data.points)
            chunk_size = min(8192, num_points)  # 8K points per chunk
            
            # Compression options: gzip level 4 (good balance of speed and compression)
            comp_opts = {
                'compression': 'gzip',
                'compression_opts': 4,
                'chunks': (chunk_size,),
                'shuffle': True  # Improves compression ratio for numerical data
            }
            
            # Always save XYZ
            data_group.create_dataset('x', data=las_data.x, dtype=np.float64, **comp_opts)
            data_group.create_dataset('y', data=las_data.y, dtype=np.float64, **comp_opts)
            data_group.create_dataset('z', data=las_data.z, dtype=np.float64, **comp_opts)
            
            # Save classification (or default zeros)
            if hasattr(las_data, 'classification'):
                data_group.create_dataset('classification', data=np.array(las_data.classification, dtype=np.int32), **comp_opts)
            else:
                data_group.create_dataset('classification', data=np.zeros(len(las_data.points), dtype=np.int32), **comp_opts)
            
            # Save other available fields
            available_fields = []
            
            # Intensity
            if hasattr(las_data, 'intensity'):
                data_group.create_dataset('intensity', data=las_data.intensity, dtype=np.uint16, **comp_opts)
                available_fields.append('intensity')
            
            # Return number and number of returns
            if hasattr(las_data, 'return_number'):
                data_group.create_dataset('return_number', data=las_data.return_number, dtype=np.uint8, **comp_opts)
                available_fields.append('return_number')
            
            if hasattr(las_data, 'number_of_returns'):
                data_group.create_dataset('number_of_returns', data=las_data.number_of_returns, dtype=np.uint8, **comp_opts)
                available_fields.append('number_of_returns')
            
            # Color (RGB)
            if hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue'):
                data_group.create_dataset('red', data=las_data.red, dtype=np.uint16, **comp_opts)
                data_group.create_dataset('green', data=las_data.green, dtype=np.uint16, **comp_opts)
                data_group.create_dataset('blue', data=las_data.blue, dtype=np.uint16, **comp_opts)
                available_fields.extend(['red', 'green', 'blue'])
            
            # Scan angle
            if hasattr(las_data, 'scan_angle_rank') or hasattr(las_data, 'scan_angle'):
                if hasattr(las_data, 'scan_angle'):
                    data_group.create_dataset('scan_angle', data=las_data.scan_angle, **comp_opts)
                    available_fields.append('scan_angle')
                else:
                    data_group.create_dataset('scan_angle_rank', data=las_data.scan_angle_rank, dtype=np.int8, **comp_opts)
                    available_fields.append('scan_angle_rank')
            
            # User data
            if hasattr(las_data, 'user_data'):
                data_group.create_dataset('user_data', data=las_data.user_data, dtype=np.uint8, **comp_opts)
                available_fields.append('user_data')
            
            # Point source ID
            if hasattr(las_data, 'point_source_id'):
                data_group.create_dataset('point_source_id', data=las_data.point_source_id, dtype=np.uint16, **comp_opts)
                available_fields.append('point_source_id')
            
            # GPS time
            if hasattr(las_data, 'gps_time'):
                data_group.create_dataset('gps_time', data=las_data.gps_time, dtype=np.float64, **comp_opts)
                available_fields.append('gps_time')
            
            # Extra dimensions (custom fields)
            for extra_dim_name in las_data.point_format.extra_dimension_names:
                extra_data = getattr(las_data, extra_dim_name)
                data_group.create_dataset(extra_dim_name, data=extra_data, **comp_opts)
                available_fields.append(extra_dim_name)
            
            # Store list of available fields
            data_group.attrs['available_fields'] = json.dumps(available_fields)
            
            # Save label statistics
            stats_group = h5f.create_group('label_statistics')
            for label, count in label_stats.items():
                stats_group.attrs[f'label_{label}'] = count
            
            # Save segment information
            segments_group = h5f.create_group('segments')
            segments_group.attrs['num_segments'] = len(segments)
            
            # Compression for segment indices (smaller chunks for faster access)
            seg_comp_opts = {
                'compression': 'gzip',
                'compression_opts': 4,
                'shuffle': True
            }
            
            # Save each segment's indices and unique labels
            for i, segment_indices in enumerate(segments):
                seg_group = segments_group.create_group(f'segment_{i:04d}')
                # Sort indices for efficient H5 reading (required by h5py fancy indexing)
                sorted_segment_indices = np.sort(segment_indices)
                # Use chunking for indices based on segment size
                seg_chunk = min(4096, len(sorted_segment_indices))
                seg_group.create_dataset('indices', data=sorted_segment_indices, dtype=np.int64, 
                                       chunks=(seg_chunk,), **seg_comp_opts)
                
                # Get classification for this segment (using original unsorted indices)
                if hasattr(las_data, 'classification'):
                    segment_labels = las_data.classification[segment_indices]
                else:
                    segment_labels = np.zeros(len(segment_indices), dtype=np.int32)
                
                unique_labels = np.unique(segment_labels)
                seg_group.create_dataset('unique_labels', data=unique_labels)
                seg_group.attrs['num_points'] = len(segment_indices)


def process_las_to_h5(input_path, output_dir=None, window_size=(50.0, 50.0), 
                      min_points=None, max_points=None, n_workers=None):
    """
    Process LAS files and save as H5 format.
    
    Args:
        input_path: Path to LAS file or directory containing LAS files
        output_dir: Directory to save H5 files (default: same as input)
        window_size: (x_size, y_size) for rectangular windows
        min_points: Minimum points threshold for a valid segment
        max_points: Maximum points threshold before further segmentation
        n_workers: Number of parallel workers (default: CPU count)
    """
    processor = LASToH5Processor(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        n_workers=n_workers
    )
    processor.process_all_files()


if __name__ == "__main__":
    
    # Example: Convert LAS to H5 with parallel processing
    input_path = r"E:\data\云南遥感中心\第一批\train"
    output_dir = r"E:\data\云南遥感中心\第一批\h5\train"
    window_size = (150., 150.)
    min_points = 4096 * 2
    max_points = 4096 * 16 * 2
    
    process_las_to_h5(
        input_path=input_path,
        output_dir=output_dir,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        n_workers=8
    )
