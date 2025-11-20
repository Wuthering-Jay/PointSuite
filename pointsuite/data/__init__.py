"""
PointSuite 数据模块

本包提供与 PyTorch Lightning 兼容的 DataModule 和 Dataset，
用于点云数据加载和处理

主要组件：
- DataModuleBase: 数据模块的抽象基类
- BinPklDataModule: bin+pkl 格式数据集的 DataModule
- PointDataModule: BinPklDataModule 的别名（向后兼容）
- BinPklDataset: bin+pkl 格式的数据集
- 各种变换和合并函数
"""

# 基类
from .datamodule_base import DataModuleBase
from .datamodule_bin import BinPklDataModule

# 向后兼容别名
PointDataModule = BinPklDataModule

# 数据集
from .datasets.dataset_base import DatasetBase
from .datasets.dataset_bin import BinPklDataset

# 合并函数
from .datasets.collate import (
    collate_fn,
    # LimitedPointsCollateFn,
    DynamicBatchSampler,
    # create_limited_dataloader
)

__all__ = [
    # 数据模块
    'DataModuleBase',
    'BinPklDataModule',
    'PointDataModule',  # 向后兼容
    
    # 数据集
    'DatasetBase',
    'BinPklDataset',
    
    # 合并函数
    'collate_fn',
    'LimitedPointsCollateFn',
    'DynamicBatchSampler',
    # 'create_limited_dataloader',
]
