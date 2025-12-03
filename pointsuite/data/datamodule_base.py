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
        train_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        predict_data: Optional[Any] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transforms: Optional[List] = None,
        val_transforms: Optional[List] = None,
        test_transforms: Optional[List] = None,
        predict_transforms: Optional[List] = None,
        train_loop: int = 1,
        val_loop: int = 1,
        test_loop: int = 1,
        predict_loop: int = 1,
        use_dynamic_batch: bool = False,
        max_points: int = 500000,
        use_dynamic_batch_inference: bool = False,
        max_points_inference: Optional[int] = None,
        use_weighted_sampler: bool = False,
        train_sampler_weights: Optional[List[float]] = None,
        class_weights: Optional[Any] = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = 2,
        **kwargs
    ):
        """
        初始化 DataModuleBase
        
        参数：
            train_data: 训练数据路径，可以是:
                       - 字符串：单个文件路径或包含多个文件的目录
                       - 列表：文件路径列表（支持跨目录）
                       - None：不使用训练数据
            val_data: 验证数据路径（格式同 train_data）
            test_data: 测试数据路径（格式同 train_data）
            predict_data: 预测数据路径（格式同 train_data），如果为 None 则使用 test_data
            batch_size: DataLoader 的批次大小（当 use_dynamic_batch=True 时不使用）
            num_workers: 数据加载的工作进程数
            train_transforms: 训练数据的变换列表
            val_transforms: 验证数据的变换列表
            test_transforms: 测试数据的变换列表
            predict_transforms: 预测数据的变换列表，如果为 None 则使用 test_transforms
            train_loop: 训练数据集循环次数（数据增强）
            val_loop: 验证数据集循环次数（Test-Time Augmentation）
            test_loop: 测试数据集循环次数（Test-Time Augmentation）
            predict_loop: 预测数据集循环次数（Test-Time Augmentation）
            use_dynamic_batch: 是否在训练阶段使用 DynamicBatchSampler（推荐用于内存控制）
                              如果为 True，batch_size 参数将被忽略
            max_points: 训练阶段每个批次的最大点数（仅在 use_dynamic_batch=True 时使用）
            use_dynamic_batch_inference: 是否在推理阶段（val/test/predict）使用 DynamicBatchSampler
                                        默认为 False（与训练阶段独立）
                                        推荐：大场景推理时设置为 True 以避免 OOM
                                        注意：与 TTA (loop > 1) 一起使用时，点数基于 transform 前的值预计算
                                              如果 transform 大幅增加点数（如密集采样），请谨慎使用
            max_points_inference: 推理阶段每个批次的最大点数
                                 如果为 None，则使用 max_points 的值
                                 推荐：根据 GPU 内存设置（推理时通常可以比训练时更大）
            use_weighted_sampler: 是否使用 WeightedRandomSampler（独立于 use_dynamic_batch）
                                 如果为 True 且提供了 train_sampler_weights，将启用加权采样
            train_sampler_weights: WeightedRandomSampler 的权重列表（仅用于训练）
                                  长度必须等于 train_dataset 的实际长度（考虑 loop）
                                  ⚠️ 不会保存到超参数中（数组太长）
            pin_memory: 是否在 DataLoader 中使用固定内存（更快的 GPU 传输）
            persistent_workers: 在 epoch 之间保持工作进程活动（更快但使用更多内存）
            prefetch_factor: 每个工作进程预取的批次数
            **kwargs: 传递给子类和数据集的其他参数
        """
        super().__init__()
        
        # 保存超参数（排除 transforms 和 weights 以避免序列化问题）
        self.save_hyperparameters(ignore=[
            'train_transforms', 'val_transforms', 'test_transforms', 'predict_transforms',
            # 'train_sampler_weights'  # weights 数组太长，不适合保存
        ])
        
        # 存储数据路径（灵活支持文件/目录/列表）
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.predict_data = predict_data if predict_data is not None else test_data
        
        # 存储基本参数
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 存储数据变换
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.predict_transforms = predict_transforms if predict_transforms is not None else test_transforms
        
        # 存储循环参数（支持 Test-Time Augmentation）
        self.train_loop = train_loop
        self.val_loop = val_loop
        self.test_loop = test_loop
        self.predict_loop = predict_loop
        
        # 存储采样参数
        self.use_dynamic_batch = use_dynamic_batch
        self.max_points = max_points
        # 推理阶段的动态 batch 设置（与训练阶段独立）
        self.use_dynamic_batch_inference = use_dynamic_batch_inference
        self.max_points_inference = max_points_inference if max_points_inference is not None else max_points
        self.use_weighted_sampler = use_weighted_sampler
        self.train_sampler_weights = train_sampler_weights
        self.class_weights = class_weights
        
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
        self.predict_dataset = None
        
        # 验证：至少提供一个数据源
        if all(x is None for x in [train_data, val_data, test_data, predict_data]):
            raise ValueError("必须至少提供一个数据源（train_data/val_data/test_data/predict_data）")
    
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
        # 🔍 调试：打印 setup 被调用的 stage
        print(f"[DEBUG] DataModuleBase.setup(stage='{stage}')")
        
        # 设置训练数据集
        if (stage == 'fit' or stage is None) and self.train_data is not None:
            self.train_dataset = self._create_dataset(
                data_paths=self.train_data,
                split='train',
                transforms=self.train_transforms
            )
            
            # 如果启用加权采样但未提供权重，则自动计算
            if self.use_weighted_sampler and self.train_sampler_weights is None:
                self.train_sampler_weights = self._compute_sample_weights(self.train_dataset)
            
            # 如果启用加权采样但未提供权重，则自动计算
            if self.use_weighted_sampler and self.train_sampler_weights is None:
                self.train_sampler_weights = self._compute_sample_weights(self.train_dataset)
        
        # 设置验证数据集
        if (stage == 'fit' or stage == 'validate' or stage is None) and self.val_data is not None:
            self.val_dataset = self._create_dataset(
                data_paths=self.val_data,
                split='val',
                transforms=self.val_transforms
            )
        
        # 设置测试数据集
        if (stage == 'test' or stage is None) and self.test_data is not None:
            self.test_dataset = self._create_dataset(
                data_paths=self.test_data,
                split='test',
                transforms=self.test_transforms
            )
        
        # 设置预测数据集（独立于测试）
        if (stage == 'predict' or stage is None) and self.predict_data is not None:
            print(f"[DEBUG] Creating predict_dataset from predict_data={self.predict_data}")
            self.predict_dataset = self._create_dataset(
                data_paths=self.predict_data,
                split='predict',
                transforms=self.predict_transforms
            )
            print(f"[DEBUG] predict_dataset created with {len(self.predict_dataset)} samples")
    
    def _compute_sample_weights(self, dataset):
        """
        自动计算训练样本权重
        
        参数：
            dataset: 数据集实例
            
        返回：
            样本权重列表（考虑 train_loop）
        """
        import torch
        import numpy as np
        
        # 转换 class_weights 为字典
        if self.class_weights is None:
            # 从数据集自动计算类别权重
            print("自动从数据集计算类别权重...")
            class_weights_dict = dataset.compute_class_weights(
                method='sqrt_inverse',  # sqrt_inverse 比 log_inverse 差异更大
                smooth=1.0,
                normalize=True
            )
            
            if class_weights_dict is None:
                print("警告: 数据集不支持自动类别权重计算，使用均匀权重")
                return None
            
            print(f"计算的类别权重: {class_weights_dict}")
        elif isinstance(self.class_weights, torch.Tensor):
            class_weights_dict = {i: float(w) for i, w in enumerate(self.class_weights)}
        elif isinstance(self.class_weights, dict):
            class_weights_dict = self.class_weights
        else:
            print(f"警告: class_weights 类型不支持: {type(self.class_weights)}")
            return None
        
        # 获取基础样本权重（不考虑 loop）
        base_weights = dataset.get_sample_weights(class_weights_dict)
        
        if base_weights is None:
            print("警告: 数据集不支持样本权重计算")
            return None
        
        # 如果 train_loop > 1，重复权重
        if self.train_loop > 1:
            weights = np.tile(base_weights, self.train_loop)
        else:
            weights = base_weights
        
        print(f"计算样本权重:")
        print(f"  - 基础样本数: {len(base_weights)}")
        print(f"  - Train loop: {self.train_loop}")
        print(f"  - 最终样本数: {len(weights)}")
        print(f"  - 权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  - 权重均值: {weights.mean():.4f}")
        
        return weights.tolist()
    
    def _create_dataloader(
        self,
        dataset,
        shuffle: bool = False,
        drop_last: bool = False,
        use_sampler_weights: bool = False,
        use_dynamic_batch: Optional[bool] = None,
        max_points: Optional[int] = None
    ) -> DataLoader:
        """
        创建具有适当设置的 DataLoader
        
        参数：
            dataset: 数据集实例
            shuffle: 是否打乱数据（如果使用 sampler 则忽略）
            drop_last: 是否丢弃最后一个不完整的批次
            use_sampler_weights: 是否使用加权采样（仅用于训练）
            use_dynamic_batch: 是否使用动态 batch（如果为 None 则使用实例默认值）
            max_points: 最大点数（如果为 None 则使用实例默认值）
            
        返回：
            DataLoader 实例
        """
        # 使用传入的参数或实例默认值
        _use_dynamic_batch = use_dynamic_batch if use_dynamic_batch is not None else self.use_dynamic_batch
        _max_points = max_points if max_points is not None else self.max_points
        # 创建基础采样器（仅用于训练）
        # val/test/predict 必须访问所有样本，不使用 sampler
        base_sampler = None
        if use_sampler_weights and self.use_weighted_sampler and self.train_sampler_weights is not None:
            # 验证 weights 长度与 dataset 长度匹配
            if len(self.train_sampler_weights) != len(dataset):
                raise ValueError(
                    f"train_sampler_weights 长度 ({len(self.train_sampler_weights)}) "
                    f"与 dataset 长度 ({len(dataset)}) 不匹配。\n"
                    f"提示：如果使用 train_loop > 1，weights 需要重复 train_loop 次。\n"
                    f"例如：weights = original_weights * train_loop"
                )
            
            base_sampler = WeightedRandomSampler(
                weights=self.train_sampler_weights,
                num_samples=len(dataset),
                replacement=True  # 使用有放回采样以支持过采样
            )
        
        if _use_dynamic_batch:
            # 使用 DynamicBatchSampler（可以与 base_sampler 结合）
            # 注意：shuffle 控制是否打乱，同时也决定 batch 数的计算方式
            # - shuffle=True: 保守估计 batch 数，用于训练
            # - shuffle=False: 精确计算 batch 数，用于 test/predict
            batch_sampler = DynamicBatchSampler(
                dataset=dataset,
                max_points=_max_points,
                shuffle=(shuffle and base_sampler is None),  # 仅在没有 base_sampler 时打乱
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
            # 注意：如果有 base_sampler，则不能同时使用 shuffle
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=base_sampler,
                shuffle=(shuffle and base_sampler is None),  # sampler 和 shuffle 互斥
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
            use_sampler_weights=True,  # 为训练启用加权采样
            use_dynamic_batch=self.use_dynamic_batch,
            max_points=self.max_points
        )
    
    def val_dataloader(self) -> DataLoader:
        """创建并返回验证 DataLoader（必须访问所有样本）"""
        if self.val_dataset is None:
            return None
        return self._create_dataloader(
            dataset=self.val_dataset,
            shuffle=False,
            drop_last=False,
            use_sampler_weights=False,  # 验证不使用加权采样，必须访问所有样本
            use_dynamic_batch=self.use_dynamic_batch_inference,
            max_points=self.max_points_inference
        )
    
    def test_dataloader(self) -> DataLoader:
        """创建并返回测试 DataLoader（必须访问所有样本）"""
        return self._create_dataloader(
            dataset=self.test_dataset,
            shuffle=False,
            drop_last=False,
            use_sampler_weights=False,  # 测试不使用加权采样，必须访问所有样本
            use_dynamic_batch=self.use_dynamic_batch_inference,
            max_points=self.max_points_inference
        )
    
    def predict_dataloader(self) -> DataLoader:
        """创建并返回预测 DataLoader（必须访问所有样本）"""
        # 如果有独立的 predict_dataset，使用它；否则回退到 test_dataset
        dataset = self.predict_dataset if self.predict_dataset is not None else self.test_dataset
        return self._create_dataloader(
            dataset=dataset,
            shuffle=False,
            drop_last=False,
            use_sampler_weights=False,  # 预测不使用加权采样，必须访问所有样本
            use_dynamic_batch=self.use_dynamic_batch_inference,
            max_points=self.max_points_inference
        )
    
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
        elif stage == 'predict':
            self.predict_dataset = None
    
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
            split: 数据集划分（'train'、'val'、'test'、'predict'）
            
        返回：
            包含数据集信息的字典
        """
        if split == 'train' and self.train_dataset is not None:
            dataset = self.train_dataset
        elif split == 'val' and self.val_dataset is not None:
            dataset = self.val_dataset
        elif split == 'test' and self.test_dataset is not None:
            dataset = self.test_dataset
        elif split == 'predict' and self.predict_dataset is not None:
            dataset = self.predict_dataset
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
        print(f"训练数据: {self.train_data}")
        print(f"验证数据: {self.val_data}")
        print(f"测试数据: {self.test_data}")
        print(f"预测数据: {self.predict_data}")
        print(f"使用动态批次: {self.use_dynamic_batch}")
        if self.use_dynamic_batch:
            print(f"每批次最大点数: {self.max_points}")
        else:
            print(f"批次大小: {self.batch_size}")
        print(f"使用加权采样: {self.use_weighted_sampler}")
        if self.use_weighted_sampler:
            print(f"权重已提供: {'是' if self.train_sampler_weights is not None else '否'}")
            if self.train_sampler_weights is not None:
                print(f"权重数量: {len(self.train_sampler_weights)}")
        print(f"工作进程数: {self.num_workers}")
        print(f"合并函数: {self.collate_fn.__name__ if hasattr(self.collate_fn, '__name__') else type(self.collate_fn).__name__}")
        print("-" * 60)
        
        for split in ['train', 'val', 'test', 'predict']:
            try:
                info = self.get_dataset_info(split)
                print(f"{split.upper()} 数据集:")
                for key, value in info.items():
                    if key != 'split':
                        print(f"  - {key}: {value}")
            except ValueError:
                print(f"{split.upper()} 数据集: 未初始化")
        
        print("=" * 60)
