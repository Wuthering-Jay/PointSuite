"""
Base DataModule for Point Cloud Data

This module provides an abstract base class for PyTorch Lightning DataModules
that can be extended for different dataset formats.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional, List, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

from .datasets.collate import collate_fn, DynamicBatchSampler


class DataModuleBase(pl.LightningDataModule, ABC):
    """
    Abstract base class for point cloud data modules.
    
    This class provides common functionality for data loading, including:
    - Setup of train/val/test datasets
    - Creation of DataLoaders with DynamicBatchSampler support
    - Support for WeightedRandomSampler for handling class imbalance
    - Memory-efficient data loading with configurable workers
    
    Subclasses must implement:
    - _create_dataset(): Create dataset instances for each split
    
    Example:
        >>> class MyDataModule(DataModuleBase):
        ...     def _create_dataset(self, data_paths, split, transforms):
        ...         return MyDataset(data_paths, split=split, transform=transforms)
        ...
        >>> datamodule = MyDataModule(
        ...     data_root='path/to/data',
        ...     train_files=['train.pkl'],
        ...     batch_size=8
        ... )
    """
    
    def __init__(
        self,
        data_root: str,
        train_files: Optional[List[str]] = None,
        val_files: Optional[List[str]] = None,
        test_files: Optional[List[str]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transforms: Optional[List] = None,
        val_transforms: Optional[List] = None,
        test_transforms: Optional[List] = None,
        use_dynamic_batch: bool = False,
        max_points: int = 500000,
        train_sampler_weights: Optional[List[float]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = 2,
        **kwargs
    ):
        """
        Initialize DataModuleBase.
        
        Args:
            data_root: Root directory containing the data files
            train_files: List of training file names. If None, auto-discover from data_root
            val_files: List of validation file names. If None, auto-discover from data_root
            test_files: List of test file names. If None, auto-discover from data_root
            batch_size: Batch size for DataLoader (not used when use_dynamic_batch=True)
            num_workers: Number of workers for data loading
            train_transforms: List of transforms for training data
            val_transforms: List of transforms for validation data
            test_transforms: List of transforms for test data
            use_dynamic_batch: Whether to use DynamicBatchSampler (recommended for memory control)
                              If True, batch_size parameter is ignored
            max_points: Maximum points per batch (only used with use_dynamic_batch=True)
            train_sampler_weights: Optional weights for WeightedRandomSampler (training only)
                                  If provided, will create a WeightedRandomSampler for training
                                  Can be used with use_dynamic_batch=True
            pin_memory: Whether to use pinned memory in DataLoader (faster GPU transfer)
            persistent_workers: Keep workers alive between epochs (faster but uses more memory)
            prefetch_factor: Number of batches to prefetch per worker
            **kwargs: Additional arguments passed to subclass and dataset
        """
        super().__init__()
        
        # Save hyperparameters (excluding transforms and weights to avoid serialization issues)
        self.save_hyperparameters(ignore=['train_transforms', 'val_transforms', 'test_transforms', 'train_sampler_weights'])
        
        # Store core parameters
        self.data_root = Path(data_root)
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Store transforms
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        
        # Store sampling parameters
        self.use_dynamic_batch = use_dynamic_batch
        self.max_points = max_points
        self.train_sampler_weights = train_sampler_weights
        
        # Store DataLoader parameters
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        
        # Store additional kwargs for subclasses
        self.kwargs = kwargs
        
        # Collate function (always use basic collate_fn with DynamicBatchSampler)
        self.collate_fn = collate_fn
        
        # Dataset placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Validation
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")
    
    @abstractmethod
    def _create_dataset(self, data_paths, split: str, transforms):
        """
        Create a dataset instance for the given split.
        
        This method must be implemented by subclasses to create
        the appropriate dataset type.
        
        Args:
            data_paths: Path(s) to the data files (can be Path, list of Paths, or list of strings)
            split: Dataset split ('train', 'val', 'test')
            transforms: List of transforms to apply
            
        Returns:
            Dataset instance
        """
        raise NotImplementedError("Subclass must implement _create_dataset()")
    
    def prepare_data(self):
        """
        Download, tokenize, etc. (single process on 1 GPU/TPU).
        
        This is called only on 1 GPU in distributed training.
        Use this for data preparation steps that should only be done once.
        """
        # In most cases, we assume data is already prepared
        # Subclasses can override this method if needed
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage (fit, validate, test, predict).
        
        This is called on every GPU in distributed training.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict', or None for all)
        """
        # Setup training dataset
        if stage == 'fit' or stage is None:
            if self.train_files is not None:
                # Use specified files
                train_paths = [self.data_root / f for f in self.train_files]
            else:
                # Auto-discover train files
                train_paths = self.data_root
            
            self.train_dataset = self._create_dataset(
                data_paths=train_paths,
                split='train',
                transforms=self.train_transforms
            )
        
        # Setup validation dataset
        if stage == 'fit' or stage == 'validate' or stage is None:
            if self.val_files is not None:
                # Use specified files
                val_paths = [self.data_root / f for f in self.val_files]
            else:
                # Auto-discover val files
                val_paths = self.data_root
            
            self.val_dataset = self._create_dataset(
                data_paths=val_paths,
                split='val',
                transforms=self.val_transforms
            )
        
        # Setup test dataset
        if stage == 'test' or stage == 'predict' or stage is None:
            if self.test_files is not None:
                # Use specified files
                test_paths = [self.data_root / f for f in self.test_files]
            else:
                # Auto-discover test files
                test_paths = self.data_root
            
            self.test_dataset = self._create_dataset(
                data_paths=test_paths,
                split='test',
                transforms=self.test_transforms
            )
    
    def _create_dataloader(
        self,
        dataset,
        shuffle: bool = False,
        drop_last: bool = False,
        use_sampler_weights: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader with appropriate settings.
        
        Args:
            dataset: Dataset instance
            shuffle: Whether to shuffle data (ignored if using batch_sampler)
            drop_last: Whether to drop last incomplete batch
            use_sampler_weights: Whether to use weighted sampling (only for training)
            
        Returns:
            DataLoader instance
        """
        if self.use_dynamic_batch:
            # Create base sampler if weights are provided and requested
            base_sampler = None
            if use_sampler_weights and self.train_sampler_weights is not None:
                base_sampler = WeightedRandomSampler(
                    weights=self.train_sampler_weights,
                    num_samples=len(dataset),
                    replacement=True  # Use replacement to support oversampling
                )
            
            # Create DynamicBatchSampler
            batch_sampler = DynamicBatchSampler(
                dataset=dataset,
                max_points=self.max_points,
                shuffle=(shuffle and base_sampler is None),  # Only shuffle if no base_sampler
                drop_last=drop_last,
                sampler=base_sampler
            )
            
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            )
        else:
            # Use standard fixed batch_size
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                drop_last=drop_last,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        return self._create_dataloader(
            dataset=self.train_dataset,
            shuffle=True,
            drop_last=True,
            use_sampler_weights=True  # Enable weighted sampling for training
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        return self._create_dataloader(
            dataset=self.val_dataset,
            shuffle=False,
            drop_last=False,
            use_sampler_weights=False  # No weighted sampling for validation
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        return self._create_dataloader(
            dataset=self.test_dataset,
            shuffle=False,
            drop_last=False,
            use_sampler_weights=False  # No weighted sampling for test
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create and return the prediction DataLoader (same as test)."""
        return self.test_dataloader()
    
    def teardown(self, stage: Optional[str] = None):
        """
        Clean up after training/testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        # Clean up datasets to free memory
        if stage == 'fit':
            self.train_dataset = None
            self.val_dataset = None
        elif stage == 'test':
            self.test_dataset = None
    
    def on_exception(self, exception: BaseException):
        """
        Called when an exception is raised during training/testing.
        
        Args:
            exception: The exception that was raised
        """
        # Clean up resources
        self.teardown()
    
    # Utility methods
    
    def get_dataset_info(self, split: str = 'train') -> Dict[str, Any]:
        """
        Get information about a dataset split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            Dict containing dataset information
        """
        if split == 'train' and self.train_dataset is not None:
            dataset = self.train_dataset
        elif split == 'val' and self.val_dataset is not None:
            dataset = self.val_dataset
        elif split == 'test' and self.test_dataset is not None:
            dataset = self.test_dataset
        else:
            raise ValueError(f"Dataset for split '{split}' not initialized. Call setup() first.")
        
        # Get basic info
        info = {
            'split': split,
            'total_length': len(dataset),
        }
        
        # Add dataset-specific info if available
        if hasattr(dataset, 'data_list'):
            info['num_samples'] = len(dataset.data_list)
        if hasattr(dataset, 'loop'):
            info['loop'] = dataset.loop
        if hasattr(dataset, 'cache_data'):
            info['cache_enabled'] = dataset.cache_data
        if hasattr(dataset, 'assets'):
            info['assets'] = dataset.assets
        if hasattr(dataset, 'class_mapping'):
            info['class_mapping'] = dataset.class_mapping
        
        return info
    
    def print_info(self):
        """Print information about all initialized datasets."""
        print("=" * 60)
        print(f"{self.__class__.__name__} Information")
        print("=" * 60)
        print(f"Data root: {self.data_root}")
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
                info = self.get_dataset_info(split)
                print(f"{split.upper()} dataset:")
                for key, value in info.items():
                    if key != 'split':
                        print(f"  - {key}: {value}")
            except ValueError:
                print(f"{split.upper()} dataset: Not initialized")
        
        print("=" * 60)
