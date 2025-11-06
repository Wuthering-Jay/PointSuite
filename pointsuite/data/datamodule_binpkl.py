"""
BinPkl 格式专用 DataModule

本模块提供了一个专门用于 BinPkl 数据集格式的 PyTorch Lightning DataModule
"""

from typing import Optional, List, Dict
from .datamodule_base import DataModuleBase
from .datasets.dataset_bin import BinPklDataset


class BinPklDataModule(DataModuleBase):
    """
    BinPkl 格式点云数据集的 PyTorch Lightning DataModule
    
    此 DataModule 专门为 bin+pkl 数据格式设计，
    其中点云数据存储在二进制文件（.bin）中，元数据存储在 pickle 文件（.pkl）中
    
    特性：
    - 自动设置训练/验证/测试数据集
    - 支持 DynamicBatchSampler 进行内存控制
    - 支持 WeightedRandomSampler 处理类别不平衡
    - 可配置资产（坐标、强度、颜色、分类等）
    - 支持类别标签映射
    - 数据缓存和循环选项
    
    示例：
        >>> # 基本用法
        >>> datamodule = BinPklDataModule(
        ...     data_root='path/to/data',
        ...     train_files=['train.pkl'],
        ...     val_files=['val.pkl'],
        ...     batch_size=8,
        ...     num_workers=4
        ... )
        >>> datamodule.setup()
        >>> 
        >>> # 使用 DynamicBatchSampler 和加权采样
        >>> datamodule = BinPklDataModule(
        ...     data_root='path/to/data',
        ...     use_dynamic_batch=True,
        ...     max_points=500000,
        ...     train_sampler_weights=weights,
        ...     assets=['coord', 'intensity', 'classification']
        ... )
        >>> 
        >>> # 与 Trainer 一起使用
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
        h_norm_grid: Optional[float] = 1.0,
        use_dynamic_batch: bool = False,
        max_points: int = 500000,
        train_sampler_weights: Optional[List[float]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = 2,
        **kwargs
    ):
        """
        初始化 BinPklDataModule
        
        参数：
            data_root: 包含数据文件的根目录
            train_files: 训练 pkl 文件名列表。如果为 None，则从 data_root 自动发现
            val_files: 验证 pkl 文件名列表。如果为 None，则从 data_root 自动发现
            test_files: 测试 pkl 文件名列表。如果为 None，则从 data_root 自动发现
            batch_size: DataLoader 的批次大小（当 use_dynamic_batch=True 时不使用）
            num_workers: 数据加载的工作进程数
            assets: 要加载的数据属性列表（例如 ['coord', 'intensity', 'classification']）
                   如果为 None，则使用默认值：['coord', 'intensity', 'classification']
            train_transforms: 训练数据的变换列表
            val_transforms: 验证数据的变换列表
            test_transforms: 测试数据的变换列表
            ignore_label: 在训练/评估中忽略的标签
            loop: 遍历训练数据集的次数（用于数据增强）
            cache_data: 是否在内存中缓存加载的数据
            class_mapping: 将原始类别标签映射到连续标签的字典
                          示例：{0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
            h_norm_grid: 计算归一化高程时使用的栅格分辨率（米）
            use_dynamic_batch: 是否使用 DynamicBatchSampler（推荐用于内存控制）
                              如果为 True，batch_size 参数将被忽略
            max_points: 每个批次的最大点数（仅在 use_dynamic_batch=True 时使用）
            train_sampler_weights: WeightedRandomSampler 的可选权重（仅用于训练）
                                  如果提供，将为训练创建 WeightedRandomSampler
                                  可与 use_dynamic_batch=True 一起使用
            pin_memory: 是否在 DataLoader 中使用固定内存（更快的 GPU 传输）
            persistent_workers: 在 epoch 之间保持工作进程活动（更快但使用更多内存）
            prefetch_factor: 每个工作进程预取的批次数
            **kwargs: 传递给 BinPklDataset 的其他参数
        """
        # 存储 BinPklDataset 特定参数
        self.assets = assets or ['coord', 'intensity', 'classification']
        self.ignore_label = ignore_label
        self.loop = loop
        self.cache_data = cache_data
        self.class_mapping = class_mapping
        self.h_norm_grid = h_norm_grid
        
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
        为给定的数据集划分创建 BinPklDataset 实例
        
        参数：
            data_paths: pkl 文件的路径
            split: 数据集划分（'train'、'val'、'test'）
            transforms: 要应用的变换列表
            
        返回：
            BinPklDataset 实例
        """
        return BinPklDataset(
            data_root=data_paths,
            split=split,
            assets=self.assets,
            transform=transforms,
            ignore_label=self.ignore_label,
            loop=self.loop if split == 'train' else 1,  # 仅对训练进行循环
            cache_data=self.cache_data,
            class_mapping=self.class_mapping,
            h_norm_grid=self.h_norm_grid,
            **self.kwargs
        )
    
    def get_dataset_info(self, split: str = 'train') -> Dict:
        """
        获取数据集划分的信息
        
        参数：
            split: 数据集划分（'train'、'val'、'test'）
            
        返回：
            包含数据集信息的字典，包括 BinPkl 特定信息
        """
        info = super().get_dataset_info(split)
        
        # 添加 BinPkl 特定信息
        info['dataset_type'] = 'BinPklDataset'
        info['assets'] = self.assets
        info['ignore_label'] = self.ignore_label
        info['cache_data'] = self.cache_data
        info['class_mapping'] = self.class_mapping
        
        return info
    
    def print_info(self):
        """打印所有已初始化数据集的信息，包括 BinPkl 特定详情"""
        print("=" * 60)
        print("BinPklDataModule 信息")
        print("=" * 60)
        print(f"数据根目录: {self.data_root}")
        print(f"数据集类型: BinPklDataset")
        print(f"资产: {self.assets}")
        print(f"忽略标签: {self.ignore_label}")
        print(f"循环次数（训练）: {self.loop}")
        print(f"缓存数据: {self.cache_data}")
        if self.class_mapping:
            print(f"类别映射: {self.class_mapping}")
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
                info = super().get_dataset_info(split)
                print(f"{split.upper()} 数据集:")
                print(f"  - 样本数: {info.get('num_samples', '不适用')}")
                print(f"  - 总长度（含循环）: {info['total_length']}")
                print(f"  - 循环: {info.get('loop', 1)}")
                print(f"  - 缓存: {info.get('cache_enabled', False)}")
            except ValueError:
                print(f"{split.upper()} 数据集: 未初始化")
        
        print("=" * 60)


# 向后兼容：旧名称的别名
PointDataModule = BinPklDataModule
