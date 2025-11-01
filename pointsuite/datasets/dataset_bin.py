"""
Dataset for loading bin+pkl format point cloud data.

This module implements the dataset class for our custom bin+pkl data format,
where point cloud data is stored in binary files (.bin) with metadata in pickle files (.pkl).
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional

from .dataset_base import DatasetBase


class BinPklDataset(DatasetBase):
    """
    Dataset class specifically for bin+pkl format point cloud data.
    
    This dataset loads pre-processed point cloud segments stored in binary format (.bin)
    with metadata stored in pickle format (.pkl).
    
    Data structure:
    - .bin file: Contains all point data in structured numpy array format
    - .pkl file: Contains metadata including:
        - segment information (indices, bounds, label counts)
        - original LAS file header
        - processing parameters (window_size, grid_size, etc.)
    
    Each segment becomes one training sample.
    """
    
    def __init__(
        self,
        data_root,
        split='train',
        assets=None,
        transform=None,
        ignore_label=-1,
        loop=1,
        cache_data=False,
    ):
        """
        Initialize BinPklDataset.
        
        Args:
            data_root: Root directory containing bin+pkl files, or a single pkl file path,
                      or a list of pkl file paths
            split: Dataset split ('train', 'val', 'test')
                  - train/val: 不存储点索引
                  - test: 存储点索引用于预测投票机制
            assets: List of data attributes to load (default: ['coord', 'intensity', 'classification'])
            transform: Data transforms to apply
            ignore_label: Label to ignore in training
            loop: Number of times to loop through dataset (for training)
            cache_data: Whether to cache loaded data in memory.
                       - If True: All loaded samples are cached in memory for faster repeated access.
                                 Suitable for small datasets that fit in RAM.
                       - If False: Data is loaded from disk each time (using memmap for efficiency).
                                  Suitable for large datasets.
        """
        # Set default assets if not specified
        if assets is None:
            assets = ['coord', 'intensity', 'classification']
        
        # Call parent init
        super().__init__(
            data_root=data_root,
            split=split,
            assets=assets,
            transform=transform,
            ignore_label=ignore_label,
            loop=loop,
            cache_data=cache_data
        )
    
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """
        Load list of all data samples.
        
        Returns:
            List of dicts containing sample information
        """
        data_list = []
        
        # Handle different data_root types
        pkl_files = []
        
        if isinstance(self.data_root, (list, tuple)):
            # List of pkl file paths
            pkl_files = [Path(p) for p in self.data_root]
            print(f"Loading from {len(pkl_files)} specified pkl files")
        elif self.data_root.is_file() and self.data_root.suffix == '.pkl':
            # Single pkl file
            pkl_files = [self.data_root]
            print(f"Loading from single pkl file: {self.data_root.name}")
        else:
            # Directory containing pkl files
            pkl_files = sorted(self.data_root.glob('*.pkl'))
            if len(pkl_files) == 0:
                raise ValueError(f"No pkl files found in {self.data_root}")
            print(f"Found {len(pkl_files)} pkl files in directory")
        
        # Load metadata from each pkl file
        total_segments = 0
        
        for pkl_path in pkl_files:
            if not pkl_path.exists():
                print(f"Warning: {pkl_path} not found, skipping")
                continue
                
            bin_path = pkl_path.with_suffix('.bin')
            
            if not bin_path.exists():
                print(f"Warning: {bin_path.name} not found, skipping {pkl_path.name}")
                continue
            
            # Load pkl metadata
            with open(pkl_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Add each segment as a separate data sample
            for segment_info in metadata['segments']:
                total_segments += 1
                
                data_list.append({
                    'bin_path': str(bin_path),
                    'pkl_path': str(pkl_path),
                    'segment_id': segment_info['segment_id'],
                    'num_points': segment_info['num_points'],
                    'file_name': bin_path.stem,
                    'bounds': {
                        'x_min': segment_info.get('x_min', 0),
                        'x_max': segment_info.get('x_max', 0),
                        'y_min': segment_info.get('y_min', 0),
                        'y_max': segment_info.get('y_max', 0),
                        'z_min': segment_info.get('z_min', 0),
                        'z_max': segment_info.get('z_max', 0),
                    }
                })
        
        print(f"Loaded {total_segments} segments from {len(pkl_files)} files")
        
        return data_list
    
    def _load_data(self, idx: int) -> Dict[str, Any]:
        """
        Load a specific data sample.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Dict containing loaded data (coord, intensity, classification, etc.)
        """
        sample_info = self.data_list[idx]
        
        # Get paths
        bin_path = Path(sample_info['bin_path'])
        segment_id = sample_info['segment_id']
        
        # Load pkl metadata to get segment info
        pkl_path = Path(sample_info['pkl_path'])
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Find the segment info
        segment_info = None
        for seg in metadata['segments']:
            if seg['segment_id'] == segment_id:
                segment_info = seg
                break
        
        if segment_info is None:
            raise ValueError(f"Segment {segment_id} not found in {pkl_path}")
        
        # Load point data from bin file using memmap
        point_data = np.memmap(bin_path, dtype=metadata['dtype'], mode='r')
        
        # Extract segment points using discrete indices
        # Point cloud data always uses discrete indices (non-continuous)
        if 'indices' not in segment_info:
            raise ValueError(f"Segment info must contain 'indices' field")
        
        indices = segment_info['indices']
        segment_points = point_data[indices]
        
        # Extract requested assets
        data = {}
        features = []  # Will store [coord, intensity, color, ...] in order
        
        # Always extract coord first
        coord = np.stack([
            segment_points['X'],
            segment_points['Y'],
            segment_points['Z']
        ], axis=1).astype(np.float32)
        data['coord'] = coord
        features.append(coord)  # coord is always first in feature
        
        # Extract other features according to assets order
        for asset in self.assets:
            if asset == 'coord':
                continue  # Already handled
                
            elif asset == 'intensity':
                # Normalize intensity to [0, 1]
                intensity = segment_points['intensity'].astype(np.float32)
                if intensity.max() > 0:
                    intensity = intensity / 65535.0  # Assuming 16-bit intensity
                intensity = intensity[:, np.newaxis]  # [N, 1]
                features.append(intensity)
                
            elif asset == 'color' and all(c in segment_points.dtype.names for c in ['red', 'green', 'blue']):
                # Extract and normalize RGB colors
                color = np.stack([
                    segment_points['red'],
                    segment_points['green'],
                    segment_points['blue']
                ], axis=1).astype(np.float32)
                if color.max() > 0:
                    color = color / 65535.0  # Assuming 16-bit color
                features.append(color)  # [N, 3]
                
            elif asset == 'classification':
                # Store separately as target, not in feature
                classification = segment_points['classification'].astype(np.int64)
                data['classification'] = classification
                
            elif asset == 'return_number' and 'return_number' in segment_points.dtype.names:
                return_num = segment_points['return_number'].astype(np.float32)[:, np.newaxis]
                features.append(return_num)
                
            elif asset == 'number_of_returns' and 'number_of_returns' in segment_points.dtype.names:
                num_returns = segment_points['number_of_returns'].astype(np.float32)[:, np.newaxis]
                features.append(num_returns)
        
        # Concatenate all features: [coord, intensity, color, ...]
        data['feature'] = np.concatenate(features, axis=1).astype(np.float32)
        
        # In test split, store point indices for voting mechanism
        if self.split == 'test':
            data['indices'] = indices.copy()  # Store original point indices
        
        return data
    
    def get_segment_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a specific segment.
        
        Args:
            idx: Index of the segment
            
        Returns:
            Dict containing segment metadata
        """
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data_list)})")
        
        sample_info = self.data_list[idx]
        pkl_path = Path(sample_info['pkl_path'])
        
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Find the segment info
        segment_id = sample_info['segment_id']
        for seg in metadata['segments']:
            if seg['segment_id'] == segment_id:
                return seg
        
        raise ValueError(f"Segment {segment_id} not found")
    
    def get_file_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for the file containing a specific segment.
        
        Args:
            idx: Index of the segment
            
        Returns:
            Dict containing file-level metadata
        """
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data_list)})")
        
        sample_info = self.data_list[idx]
        pkl_path = Path(sample_info['pkl_path'])
        
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Return metadata excluding the segments list (which can be large)
        file_metadata = {k: v for k, v in metadata.items() if k != 'segments'}
        return file_metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dict containing dataset statistics
        """
        if len(self.data_list) == 0:
            return {}
        
        # Collect statistics
        num_points_list = [s['num_points'] for s in self.data_list]
        
        stats = {
            'num_samples': len(self.data_list),
            'num_points': {
                'total': sum(num_points_list),
                'mean': np.mean(num_points_list),
                'median': np.median(num_points_list),
                'min': np.min(num_points_list),
                'max': np.max(num_points_list),
                'std': np.std(num_points_list),
            }
        }
        
        # Get label distribution from first file
        if len(self.data_list) > 0:
            pkl_path = Path(self.data_list[0]['pkl_path'])
            with open(pkl_path, 'rb') as f:
                metadata = pickle.load(f)
            
            if 'label_counts' in metadata:
                stats['label_distribution'] = metadata['label_counts']
        
        return stats
    
    def print_stats(self):
        """Print dataset statistics."""
        stats = self.get_stats()
        
        print("="*70)
        print("Dataset Statistics")
        print("="*70)
        print(f"Split: {self.split}")
        print(f"Samples: {stats['num_samples']:,}")
        print(f"\nPoints per sample:")
        print(f"  - Total: {stats['num_points']['total']:,}")
        print(f"  - Mean: {stats['num_points']['mean']:,.1f}")
        print(f"  - Median: {stats['num_points']['median']:,.0f}")
        print(f"  - Min: {stats['num_points']['min']:,}")
        print(f"  - Max: {stats['num_points']['max']:,}")
        print(f"  - Std: {stats['num_points']['std']:,.1f}")
        
        if 'label_distribution' in stats:
            print(f"\nLabel distribution (overall):")
            for label, count in sorted(stats['label_distribution'].items()):
                print(f"  Class {label}: {count:,}")
        
        print("="*70)


def create_dataset(
    data_root,
    split='train',
    assets=None,
    transform=None,
    ignore_label=-1,
    loop=1,
    cache_data=False,
    **kwargs
):
    """
    Factory function to create BinPklDataset.
    
    Args:
        data_root: Root directory, single pkl file, or list of pkl files
        split: Dataset split ('train', 'val', 'test')
        assets: List of data attributes to load
        transform: Data transforms
        ignore_label: Label to ignore
        loop: Dataset loop factor
        cache_data: Whether to cache data
        **kwargs: Additional arguments
        
    Returns:
        BinPklDataset instance
    """
    return BinPklDataset(
        data_root=data_root,
        split=split,
        assets=assets,
        transform=transform,
        ignore_label=ignore_label,
        loop=loop,
        cache_data=cache_data,
    )
