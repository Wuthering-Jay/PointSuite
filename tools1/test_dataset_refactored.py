"""
Test script for the refactored dataset architecture.

This script tests:
1. BinPklDataset initialization
2. Data loading with memmap
3. Asset extraction (coord, intensity, classification, etc.)
4. Filtering (min_points, max_points)
5. Dataset statistics
6. Metadata access methods
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pointsuite.datasets.dataset_bin import BinPklDataset
import numpy as np


def test_basic_loading():
    """Test basic dataset loading and iteration."""
    print("="*70)
    print("Test 1: Basic Dataset Loading")
    print("="*70)
    
    # Change this to your actual data directory
    data_root = r"E:\data\point_cloud\test_output"
    
    if not Path(data_root).exists():
        print(f"Data directory not found: {data_root}")
        print("Please update the data_root path in the script.")
        return None
    
    # Create dataset
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        min_points=100,
        max_points=None,
        cache_data=False
    )
    
    print(f"\nDataset created successfully!")
    print(f"Number of samples: {len(dataset)}")
    
    if len(dataset) > 0:
        print(f"\n--- Loading first sample ---")
        sample = dataset[0]
        
        print(f"Keys in sample: {list(sample.keys())}")
        
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                      f"min={value.min():.4f}, max={value.max():.4f}")
            else:
                print(f"  {key}: {value}")
    
    return dataset


def test_filtering():
    """Test dataset filtering by point count."""
    print("\n" + "="*70)
    print("Test 2: Dataset Filtering")
    print("="*70)
    
    data_root = r"E:\data\point_cloud\test_output"
    
    if not Path(data_root).exists():
        print(f"Data directory not found: {data_root}")
        return
    
    # Test different filter settings
    filters = [
        (None, None, "No filtering"),
        (500, None, "Min 500 points"),
        (None, 2000, "Max 2000 points"),
        (500, 2000, "500-2000 points"),
    ]
    
    for min_pts, max_pts, desc in filters:
        dataset = BinPklDataset(
            data_root=data_root,
            split='train',
            assets=['coord'],
            min_points=min_pts,
            max_points=max_pts,
            cache_data=False
        )
        print(f"\n{desc}: {len(dataset)} samples")


def test_metadata_access():
    """Test metadata access methods."""
    print("\n" + "="*70)
    print("Test 3: Metadata Access")
    print("="*70)
    
    data_root = r"E:\data\point_cloud\test_output"
    
    if not Path(data_root).exists():
        print(f"Data directory not found: {data_root}")
        return
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord'],
        cache_data=False
    )
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    # Test get_sample_info (from base class)
    print("\n--- Sample Info (from base class) ---")
    info = dataset.get_sample_info(0)
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Test get_segment_info (from BinPklDataset)
    print("\n--- Segment Info (from BinPklDataset) ---")
    seg_info = dataset.get_segment_info(0)
    for key, value in seg_info.items():
        print(f"{key}: {value}")
    
    # Test get_file_metadata (from BinPklDataset)
    print("\n--- File Metadata (from BinPklDataset) ---")
    file_meta = dataset.get_file_metadata(0)
    for key, value in file_meta.items():
        if isinstance(value, dict) and len(value) > 5:
            print(f"{key}: {type(value)} with {len(value)} items")
        else:
            print(f"{key}: {value}")


def test_statistics():
    """Test dataset statistics."""
    print("\n" + "="*70)
    print("Test 4: Dataset Statistics")
    print("="*70)
    
    data_root = r"E:\data\point_cloud\test_output"
    
    if not Path(data_root).exists():
        print(f"Data directory not found: {data_root}")
        return
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    # Print statistics using built-in method
    dataset.print_stats()


def test_caching():
    """Test data caching functionality."""
    print("\n" + "="*70)
    print("Test 5: Data Caching")
    print("="*70)
    
    data_root = r"E:\data\point_cloud\test_output"
    
    if not Path(data_root).exists():
        print(f"Data directory not found: {data_root}")
        return
    
    import time
    
    # Test without caching
    print("\n--- Without caching ---")
    dataset_no_cache = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False
    )
    
    if len(dataset_no_cache) > 0:
        start = time.time()
        for i in range(min(10, len(dataset_no_cache))):
            _ = dataset_no_cache[i]
        time_no_cache = time.time() - start
        print(f"Time to load 10 samples: {time_no_cache:.4f}s")
    
    # Test with caching
    print("\n--- With caching ---")
    dataset_cache = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=True
    )
    
    if len(dataset_cache) > 0:
        # First pass (populate cache)
        start = time.time()
        for i in range(min(10, len(dataset_cache))):
            _ = dataset_cache[i]
        time_first = time.time() - start
        print(f"Time to load 10 samples (first pass): {time_first:.4f}s")
        
        # Second pass (from cache)
        start = time.time()
        for i in range(min(10, len(dataset_cache))):
            _ = dataset_cache[i]
        time_cached = time.time() - start
        print(f"Time to load 10 samples (from cache): {time_cached:.4f}s")
        print(f"Speedup: {time_first/time_cached:.2f}x")


def test_all_assets():
    """Test loading different asset combinations."""
    print("\n" + "="*70)
    print("Test 6: Different Asset Combinations")
    print("="*70)
    
    data_root = r"E:\data\point_cloud\test_output"
    
    if not Path(data_root).exists():
        print(f"Data directory not found: {data_root}")
        return
    
    asset_combinations = [
        ['coord'],
        ['coord', 'intensity'],
        ['coord', 'classification'],
        ['coord', 'intensity', 'classification'],
        ['coord', 'intensity', 'classification', 'color'],
    ]
    
    for assets in asset_combinations:
        try:
            dataset = BinPklDataset(
                data_root=data_root,
                split='train',
                assets=assets,
                cache_data=False
            )
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"\nAssets {assets}:")
                print(f"  Loaded keys: {list(sample.keys())}")
            else:
                print(f"\nAssets {assets}: Dataset is empty")
        except Exception as e:
            print(f"\nAssets {assets}: Error - {e}")


def main():
    """Run all tests."""
    print("Testing Refactored Dataset Architecture")
    print("="*70)
    
    try:
        # Test 1: Basic loading
        dataset = test_basic_loading()
        
        if dataset is not None and len(dataset) > 0:
            # Test 2: Filtering
            test_filtering()
            
            # Test 3: Metadata access
            test_metadata_access()
            
            # Test 4: Statistics
            test_statistics()
            
            # Test 5: Caching
            test_caching()
            
            # Test 6: Different assets
            test_all_assets()
        
        print("\n" + "="*70)
        print("All tests completed!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
