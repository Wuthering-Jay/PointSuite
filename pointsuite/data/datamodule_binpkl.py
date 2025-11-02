"""
BinPkl format specific DataModule

This module provides a PyTorch Lightning DataModule specifically
for the BinPkl dataset format.
"""

from typing import Optional, List, Dict
from .datamodule_base import DataModuleBase
from .datasets.dataset_bin import BinPklDataset


class BinPklDataModule(DataModuleBase):
    """
    PyTorch Lightning DataModule for BinPkl format point cloud datasets.
    
    This DataModule is specifically designed for the bin+pkl data format,
    where point cloud data is stored in binary files (.bin) with metadata
    in pickle files (.pkl).
    
    Features:
    - Automatic setup of train/val/test datasets
    - Support for DynamicBatchSampler for memory control
    - Support for WeightedRandomSampler for class imbalance
    - Configurable assets (coord, intensity, color, classification, etc.)
    - Class label mapping support
    - Data caching and looping options
    
    Example:
        >>> # Basic usage
        >>> datamodule = BinPklDataModule(
        ...     data_root='path/to/data',
        ...     train_files=['train.pkl'],
        ...     val_files=['val.pkl'],
        ...     batch_size=8,
        ...     num_workers=4
        ... )
        >>> datamodule.setup()
        >>> 
        >>> # With DynamicBatchSampler and weighted sampling
        >>> datamodule = BinPklDataModule(
        ...     data_root='path/to/data',
        ...     use_dynamic_batch=True,
        ...     max_points=500000,
        ...     train_sampler_weights=weights,
        ...     assets=['coord', 'intensity', 'classification']
        ... )
        >>> 
        >>> # Use with Trainer
        >>> trainer = pl.Trainer()
        >>> trainer.fit(model, datamodule)
    """
    
    def __init__(
        self,
        data_root: str,
        train_files: Optional[List[str]] = None,
        val_files: Optional[List[str]] = None,
        test_files: Optional[List[str]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        assets: Optional[List[str]] = None,
        train_transforms: Optional[List] = None,
        val_transforms: Optional[List] = None,
        test_transforms: Optional[List] = None,
        ignore_label: int = -1,
        loop: int = 1,
        cache_data: bool = False,
        class_mapping: Optional[Dict[int, int]] = None,
        use_dynamic_batch: bool = False,
        max_points: int = 500000,
        train_sampler_weights: Optional[List[float]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = 2,
        **kwargs
    ):
        """
        Initialize BinPklDataModule.
        
        Args:
            data_root: Root directory containing the data files
            train_files: List of training pkl file names. If None, auto-discover from data_root
            val_files: List of validation pkl file names. If None, auto-discover from data_root
            test_files: List of test pkl file names. If None, auto-discover from data_root
            batch_size: Batch size for DataLoader (not used when use_dynamic_batch=True)
            num_workers: Number of workers for data loading
            assets: List of data attributes to load (e.g., ['coord', 'intensity', 'classification'])
                   If None, uses default: ['coord', 'intensity', 'classification']
            train_transforms: List of transforms for training data
            val_transforms: List of transforms for validation data
            test_transforms: List of transforms for test data
            ignore_label: Label to ignore in training/evaluation
            loop: Number of times to loop through training dataset (for data augmentation)
            cache_data: Whether to cache loaded data in memory
            class_mapping: Dict mapping original class labels to continuous labels
                          Example: {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
            use_dynamic_batch: Whether to use DynamicBatchSampler (recommended for memory control)
                              If True, batch_size parameter is ignored
            max_points: Maximum points per batch (only used with use_dynamic_batch=True)
            train_sampler_weights: Optional weights for WeightedRandomSampler (training only)
                                  If provided, will create a WeightedRandomSampler for training
                                  Can be used with use_dynamic_batch=True
            pin_memory: Whether to use pinned memory in DataLoader (faster GPU transfer)
            persistent_workers: Keep workers alive between epochs (faster but uses more memory)
            prefetch_factor: Number of batches to prefetch per worker
            **kwargs: Additional arguments passed to BinPklDataset
        """
        # Store BinPklDataset specific parameters
        self.assets = assets or ['coord', 'intensity', 'classification']
        self.ignore_label = ignore_label
        self.loop = loop
        self.cache_data = cache_data
        self.class_mapping = class_mapping
        
        # Call parent constructor
        super().__init__(
            data_root=data_root,
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            batch_size=batch_size,
            num_workers=num_workers,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            use_dynamic_batch=use_dynamic_batch,
            max_points=max_points,
            train_sampler_weights=train_sampler_weights,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            **kwargs
        )
    
    def _create_dataset(self, data_paths, split: str, transforms):
        """
        Create a BinPklDataset instance for the given split.
        
        Args:
            data_paths: Path(s) to the pkl files
            split: Dataset split ('train', 'val', 'test')
            transforms: List of transforms to apply
            
        Returns:
            BinPklDataset instance
        """
        return BinPklDataset(
            data_root=data_paths,
            split=split,
            assets=self.assets,
            transform=transforms,
            ignore_label=self.ignore_label,
            loop=self.loop if split == 'train' else 1,  # Only loop for training
            cache_data=self.cache_data,
            class_mapping=self.class_mapping,
            **self.kwargs
        )
    
    def get_dataset_info(self, split: str = 'train') -> Dict:
        """
        Get information about a dataset split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            Dict containing dataset information including BinPkl-specific info
        """
        info = super().get_dataset_info(split)
        
        # Add BinPkl-specific information
        info['dataset_type'] = 'BinPklDataset'
        info['assets'] = self.assets
        info['ignore_label'] = self.ignore_label
        info['cache_data'] = self.cache_data
        info['class_mapping'] = self.class_mapping
        
        return info
    
    def print_info(self):
        """Print information about all initialized datasets with BinPkl-specific details."""
        print("=" * 60)
        print("BinPklDataModule Information")
        print("=" * 60)
        print(f"Data root: {self.data_root}")
        print(f"Dataset type: BinPklDataset")
        print(f"Assets: {self.assets}")
        print(f"Ignore label: {self.ignore_label}")
        print(f"Loop (train): {self.loop}")
        print(f"Cache data: {self.cache_data}")
        if self.class_mapping:
            print(f"Class mapping: {self.class_mapping}")
        print(f"Use dynamic batch: {self.use_dynamic_batch}")
        if self.use_dynamic_batch:
            print(f"Max points per batch: {self.max_points}")
            print(f"Weighted sampling: {'Yes' if self.train_sampler_weights is not None else 'No'}")
        else:
            print(f"Batch size: {self.batch_size}")
        print(f"Num workers: {self.num_workers}")
        print(f"Collate function: {type(self.collate_fn).__name__}")
        print("-" * 60)
        
        for split in ['train', 'val', 'test']:
            try:
                info = super().get_dataset_info(split)
                print(f"{split.upper()} dataset:")
                print(f"  - Samples: {info.get('num_samples', 'N/A')}")
                print(f"  - Total length (with loop): {info['total_length']}")
                print(f"  - Loop: {info.get('loop', 1)}")
                print(f"  - Cache: {info.get('cache_enabled', False)}")
            except ValueError:
                print(f"{split.upper()} dataset: Not initialized")
        
        print("=" * 60)


# Backward compatibility: alias for the old name
PointDataModule = BinPklDataModule
