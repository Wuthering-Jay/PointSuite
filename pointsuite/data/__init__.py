"""
PointSuite Data Module

This package provides PyTorch Lightning compatible DataModules and Datasets
for point cloud data loading and processing.

Main components:
- DataModuleBase: Abstract base class for data modules
- BinPklDataModule: DataModule for bin+pkl format datasets  
- PointDataModule: Alias for BinPklDataModule (backward compatibility)
- BinPklDataset: Dataset for bin+pkl format
- Various transforms and collate functions
"""

# Base classes
from .datamodule_base import DataModuleBase
from .datamodule_binpkl import BinPklDataModule, PointDataModule

# Datasets
from .datasets.dataset_base import DatasetBase
from .datasets.dataset_bin import BinPklDataset

# Collate functions
from .datasets.collate import (
    collate_fn,
    LimitedPointsCollateFn,
    DynamicBatchSampler,
    create_limited_dataloader
)

__all__ = [
    # DataModules
    'DataModuleBase',
    'BinPklDataModule',
    'PointDataModule',  # Backward compatibility
    
    # Datasets
    'DatasetBase',
    'BinPklDataset',
    
    # Collate functions
    'collate_fn',
    'LimitedPointsCollateFn',
    'DynamicBatchSampler',
    'create_limited_dataloader',
]
