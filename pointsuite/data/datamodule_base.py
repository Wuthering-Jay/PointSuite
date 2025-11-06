"""
点云数据的基础 DataModule

本模块提供了一个抽象基类，用于 PyTorch Lightning DataModule，
可以扩展以支持不同的数据集格式。
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional, List, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

from .datasets.collate import collate_fn, DynamicBatchSampler


class DataModuleBase(pl.LightningDataModule, ABC):
    """
    点云数据模块的抽象基类
    
    本类提供数据加载的通用功能，包括：
    - 设置训练/验证/测试数据集
    - 创建支持 DynamicBatchSampler 的 DataLoader
    - 支持 WeightedRandomSampler 以处理类别不平衡
    - 配置可调节的工作进程以实现内存高效的数据加载
    
    子类必须实现：
    - _create_dataset(): 为每个数据集划分创建数据集实例
    
    示例：
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
        初始化 DataModuleBase
        
        参数：
            data_root: 包含数据文件的根目录
            train_files: 训练文件名列表。如果为 None，则从 data_root 自动发现
            val_files: 验证文件名列表。如果为 None，则从 data_root 自动发现
            test_files: 测试文件名列表。如果为 None，则从 data_root 自动发现
            batch_size: DataLoader 的批次大小（当 use_dynamic_batch=True 时不使用）
            num_workers: 数据加载的工作进程数
            train_transforms: 训练数据的变换列表
            val_transforms: 验证数据的变换列表
            test_transforms: 测试数据的变换列表
            use_dynamic_batch: 是否使用 DynamicBatchSampler（推荐用于内存控制）
                              如果为 True，batch_size 参数将被忽略
            max_points: 每个批次的最大点数（仅在 use_dynamic_batch=True 时使用）
            train_sampler_weights: WeightedRandomSampler 的可选权重（仅用于训练）
                                  如果提供，将为训练创建 WeightedRandomSampler
                                  可与 use_dynamic_batch=True 一起使用
            pin_memory: 是否在 DataLoader 中使用固定内存（更快的 GPU 传输）
            persistent_workers: 在 epoch 之间保持工作进程活动（更快但使用更多内存）
            prefetch_factor: 每个工作进程预取的批次数
            **kwargs: 传递给子类和数据集的其他参数
        """
        super().__init__()
        
        # 保存超参数（排除 transforms 和 weights 以避免序列化问题）
        self.save_hyperparameters(ignore=['train_transforms', 'val_transforms', 'test_transforms', 'train_sampler_weights'])
        
        # 存储核心参数
        self.data_root = Path(data_root)
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 存储数据变换
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        
        # 存储采样参数
        self.use_dynamic_batch = use_dynamic_batch
        self.max_points = max_points
        self.train_sampler_weights = train_sampler_weights
        
        # 存储 DataLoader 参数
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        
        # 存储子类的额外参数
        self.kwargs = kwargs
        
        # 合并函数（始终使用基本的 collate_fn 配合 DynamicBatchSampler）
        self.collate_fn = collate_fn
        
        # 数据集占位符
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 验证
        if not self.data_root.exists():
            raise ValueError(f"数据根目录不存在: {self.data_root}")
    
    @abstractmethod
    def _create_dataset(self, data_paths, split: str, transforms):
        """
        为给定的数据集划分创建数据集实例
        
        子类必须实现此方法以创建适当的数据集类型
        
        参数：
            data_paths: 数据文件的路径（可以是 Path、Path 列表或字符串列表）
            split: 数据集划分（'train'、'val'、'test'）
            transforms: 要应用的变换列表
            
        返回：
            数据集实例
        """
        raise NotImplementedError("子类必须实现 _create_dataset() 方法")
    
    def prepare_data(self):
        """
        下载、分词等数据准备工作（在 1 个 GPU/TPU 上的单进程中执行）
        
        在分布式训练中，此方法仅在 1 个 GPU 上调用
        用于只需执行一次的数据准备步骤
        """
        # 在大多数情况下，我们假设数据已经准备好了
        # 如果需要，子类可以覆盖此方法
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        为每个阶段设置数据集（fit、validate、test、predict）
        
        在分布式训练中，此方法在每个 GPU 上调用
        
        参数：
            stage: 当前阶段（'fit'、'validate'、'test'、'predict'，或 None 表示所有阶段）
        """
        # 设置训练数据集
        if stage == 'fit' or stage is None:
            if self.train_files is not None:
                # 使用指定的文件
                train_paths = [self.data_root / f for f in self.train_files]
            else:
                # 自动发现训练文件
                train_paths = self.data_root
            
            self.train_dataset = self._create_dataset(
                data_paths=train_paths,
                split='train',
                transforms=self.train_transforms
            )
        
        # 设置验证数据集
        if stage == 'fit' or stage == 'validate' or stage is None:
            if self.val_files is not None:
                # 使用指定的文件
                val_paths = [self.data_root / f for f in self.val_files]
            else:
                # 自动发现验证文件
                val_paths = self.data_root
            
            self.val_dataset = self._create_dataset(
                data_paths=val_paths,
                split='val',
                transforms=self.val_transforms
            )
        
        # 设置测试数据集
        if stage == 'test' or stage == 'predict' or stage is None:
            if self.test_files is not None:
                # 使用指定的文件
                test_paths = [self.data_root / f for f in self.test_files]
            else:
                # 自动发现测试文件
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
        创建具有适当设置的 DataLoader
        
        参数：
            dataset: 数据集实例
            shuffle: 是否打乱数据（如果使用 batch_sampler 则忽略）
            drop_last: 是否丢弃最后一个不完整的批次
            use_sampler_weights: 是否使用加权采样（仅用于训练）
            
        返回：
            DataLoader 实例
        """
        if self.use_dynamic_batch:
            # 如果提供了权重并且被请求，则创建基础采样器
            base_sampler = None
            if use_sampler_weights and self.train_sampler_weights is not None:
                base_sampler = WeightedRandomSampler(
                    weights=self.train_sampler_weights,
                    num_samples=len(dataset),
                    replacement=True  # 使用有放回采样以支持过采样
                )
            
            # 创建 DynamicBatchSampler
            batch_sampler = DynamicBatchSampler(
                dataset=dataset,
                max_points=self.max_points,
                shuffle=(shuffle and base_sampler is None),  # 仅在没有 base_sampler 时打乱
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
            # 使用标准的固定 batch_size
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
        """创建并返回训练 DataLoader"""
        return self._create_dataloader(
            dataset=self.train_dataset,
            shuffle=True,
            drop_last=True,
            use_sampler_weights=True  # 为训练启用加权采样
        )
    
    def val_dataloader(self) -> DataLoader:
        """创建并返回验证 DataLoader"""
        return self._create_dataloader(
            dataset=self.val_dataset,
            shuffle=False,
            drop_last=False,
            use_sampler_weights=False  # 验证不使用加权采样
        )
    
    def test_dataloader(self) -> DataLoader:
        """创建并返回测试 DataLoader"""
        return self._create_dataloader(
            dataset=self.test_dataset,
            shuffle=False,
            drop_last=False,
            use_sampler_weights=False  # 测试不使用加权采样
        )
    
    def predict_dataloader(self) -> DataLoader:
        """创建并返回预测 DataLoader（与测试相同）"""
        return self.test_dataloader()
    
    def teardown(self, stage: Optional[str] = None):
        """
        训练/测试后清理资源
        
        参数：
            stage: 当前阶段（'fit'、'validate'、'test'、'predict'）
        """
        # 清理数据集以释放内存
        if stage == 'fit':
            self.train_dataset = None
            self.val_dataset = None
        elif stage == 'test':
            self.test_dataset = None
    
    def on_exception(self, exception: BaseException):
        """
        在训练/测试期间引发异常时调用
        
        参数：
            exception: 引发的异常
        """
        # 清理资源
        self.teardown()
    
    # 工具方法
    
    def get_dataset_info(self, split: str = 'train') -> Dict[str, Any]:
        """
        获取数据集划分的信息
        
        参数：
            split: 数据集划分（'train'、'val'、'test'）
            
        返回：
            包含数据集信息的字典
        """
        if split == 'train' and self.train_dataset is not None:
            dataset = self.train_dataset
        elif split == 'val' and self.val_dataset is not None:
            dataset = self.val_dataset
        elif split == 'test' and self.test_dataset is not None:
            dataset = self.test_dataset
        else:
            raise ValueError(f"划分 '{split}' 的数据集未初始化。请先调用 setup()")
        
        # 获取基本信息
        info = {
            'split': split,
            'total_length': len(dataset),
        }
        
        # 如果可用，添加数据集特定的信息
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
        """打印所有已初始化数据集的信息"""
        print("=" * 60)
        print(f"{self.__class__.__name__} 信息")
        print("=" * 60)
        print(f"数据根目录: {self.data_root}")
        print(f"使用动态批次: {self.use_dynamic_batch}")
        if self.use_dynamic_batch:
            print(f"每批次最大点数: {self.max_points}")
            print(f"加权采样: {'是' if self.train_sampler_weights is not None else '否'}")
        else:
            print(f"批次大小: {self.batch_size}")
        print(f"工作进程数: {self.num_workers}")
        print(f"合并函数: {self.collate_fn.__name__ if hasattr(self.collate_fn, '__name__') else type(self.collate_fn).__name__}")
        print("-" * 60)
        
        for split in ['train', 'val', 'test']:
            try:
                info = self.get_dataset_info(split)
                print(f"{split.upper()} 数据集:")
                for key, value in info.items():
                    if key != 'split':
                        print(f"  - {key}: {value}")
            except ValueError:
                print(f"{split.upper()} 数据集: 未初始化")
        
        print("=" * 60)
