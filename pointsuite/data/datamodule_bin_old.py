"""
BinPkl 格式专用 DataModule

本模块提供了一个专门用于 BinPkl 数据集格式的 PyTorch Lightning DataModule
"""

from typing import Optional, List, Dict, Any
from .datamodule_base import DataModuleBase
from .datasets.dataset_bin import BinPklDataset


class BinPklDataModule(DataModuleBase):
    """
    BinPkl 格式点云数据集的 PyTorch Lightning DataModule
    
    此 DataModule 专门为 bin+pkl 数据格式设计，
    其中点云数据存储在二进制文件（.bin）中，元数据存储在 pickle 文件（.pkl）中
    
    特性：
    - 灵活的数据路径配置（支持跨目录文件）
    - 独立的训练/验证/测试/预测数据集配置
    - 支持 DynamicBatchSampler 进行内存控制
    - 支持 WeightedRandomSampler 处理类别不平衡（独立于 DynamicBatchSampler）
    - 可配置资产（坐标、强度、颜色、分类等）
    - 支持类别标签映射
    - 数据缓存和循环选项
    
    示例：
        >>> # 基本用法：跨目录配置
        >>> datamodule = BinPklDataModule(
        ...     train_data='data/train',  # 或文件列表
        ...     val_data='data/val',
        ...     test_data='data/test',
        ...     batch_size=8,
        ...     num_workers=4
        ... )
        >>> 
        >>> # 使用 DynamicBatchSampler 和独立的加权采样
        >>> datamodule = BinPklDataModule(
        ...     train_data=['data/train/file1.pkl', 'data/other/file2.pkl'],
        ...     val_data='data/val',
        ...     use_dynamic_batch=True,
        ...     max_points=500000,
        ...     use_weighted_sampler=True,
        ...     train_sampler_weights=weights,  # 长度 = len(train_dataset) * loop
        ...     assets=['coord', 'intensity', 'classification']
        ... )
        >>> 
        >>> # 独立的预测数据集
        >>> datamodule = BinPklDataModule(
        ...     train_data='data/train',
        ...     val_data='data/val',
        ...     test_data='data/test',
        ...     predict_data='data/new_scenes',  # 不同于 test
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
        assets: Optional[List[str]] = None,
        train_transforms: Optional[List] = None,
        val_transforms: Optional[List] = None,
        test_transforms: Optional[List] = None,
        predict_transforms: Optional[List] = None,
        ignore_label: int = -1,
        train_loop: int = 1,
        val_loop: int = 1,
        test_loop: int = 1,
        predict_loop: int = 1,
        cache_data: bool = False,
        class_mapping: Optional[Dict[int, int]] = None,
        class_names: Optional[List[str]] = None,
        h_norm_grid: Optional[float] = 1.0,
        use_dynamic_batch: bool = False,
        max_points: int = 500000,
        use_dynamic_batch_inference: bool = False,
        max_points_inference: Optional[int] = None,
        use_weighted_sampler: bool = False,
        train_sampler_weights: Optional[List[float]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = 2,
        **kwargs
    ):
        """
        初始化 BinPklDataModule
        
        参数：
            train_data: 训练数据路径，可以是:
                       - 字符串：单个 pkl 文件或包含 pkl 文件的目录
                       - 列表：pkl 文件路径列表（支持跨目录）
                       - None：不使用训练数据
            val_data: 验证数据路径（格式同 train_data）
            test_data: 测试数据路径（格式同 train_data）
            predict_data: 预测数据路径（格式同 train_data），如果为 None 则使用 test_data
            batch_size: DataLoader 的批次大小（当 use_dynamic_batch=True 时不使用）
            num_workers: 数据加载的工作进程数
            assets: 要加载的数据属性列表（例如 ['coord', 'intensity', 'classification']）
                   如果为 None，则使用默认值：['coord', 'intensity', 'classification']
            train_transforms: 训练数据的变换列表
            val_transforms: 验证数据的变换列表
            test_transforms: 测试数据的变换列表
            predict_transforms: 预测数据的变换列表，如果为 None 则使用 test_transforms
            ignore_label: 在训练/评估中忽略的标签
            train_loop: 训练数据集循环次数（数据增强），默认 1
                       ⚠️ 如果使用 train_sampler_weights，weights 长度必须 = 样本数 * train_loop
            val_loop: 验证数据集循环次数（Test-Time Augmentation），默认 1
                     设置 > 1 启用 TTA，通过多次增强投票提高精度
            test_loop: 测试数据集循环次数（Test-Time Augmentation），默认 1
            predict_loop: 预测数据集循环次数（Test-Time Augmentation），默认 1
            cache_data: 是否在内存中缓存加载的数据
            class_mapping: 将原始类别标签映射到连续标签的字典
                          示例：{0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
                          不在 mapping 中的类别会被设为 ignore_label
            class_names: 类别名称列表（按映射后的连续标签顺序）
                        示例：['Ground', 'Vegetation', 'Building', 'Car', 'Wire']
                        如果为 None，指标计算时会使用 "Class n" 格式
                        如果有 class_mapping，会使用原始标签号（如 "Class 6"）
            h_norm_grid: 计算归一化高程时使用的栅格分辨率（米）
            use_dynamic_batch: 是否在训练阶段使用 DynamicBatchSampler（推荐用于内存控制）
            max_points: 训练阶段每个批次的最大点数（仅在 use_dynamic_batch=True 时使用）
            use_dynamic_batch_inference: 是否在推理阶段（val/test/predict）使用 DynamicBatchSampler
                                        默认为 False（与训练阶段独立）
                                        推荐：大场景推理时设置为 True 以避免 OOM
                                        注意：与 TTA (loop > 1) 一起使用时，点数基于 transform 前的值预计算
                                              如果 transform 大幅增加点数（如密集采样），请谨慎使用
            max_points_inference: 推理阶段每个批次的最大点数
                                 如果为 None，则使用 max_points 的值
                                 推荐：推理时通常可以比训练时更大
            use_weighted_sampler: 是否使用 WeightedRandomSampler（独立于 use_dynamic_batch）
            train_sampler_weights: WeightedRandomSampler 的权重列表（仅用于训练）
                                  长度必须等于 len(train_dataset)，考虑 loop
                                  示例：如果 loop=2，weights = original_weights * 2
                                  ⚠️ 不会保存到超参数中（数组太长）
            pin_memory: 是否在 DataLoader 中使用固定内存
            persistent_workers: 在 epoch 之间保持工作进程活动
            prefetch_factor: 每个工作进程预取的批次数
            **kwargs: 传递给 BinPklDataset 的其他参数
        """
        # 存储 BinPklDataset 特定参数
        self.assets = assets or ['coord', 'intensity', 'classification']
        self.ignore_label = ignore_label
        self.cache_data = cache_data
        self.class_mapping = class_mapping
        self.class_names = class_names
        self.h_norm_grid = h_norm_grid
        
        # Call parent constructor
        super().__init__(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            predict_data=predict_data,
            batch_size=batch_size,
            num_workers=num_workers,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            predict_transforms=predict_transforms,
            train_loop=train_loop,
            val_loop=val_loop,
            test_loop=test_loop,
            predict_loop=predict_loop,
            use_dynamic_batch=use_dynamic_batch,
            max_points=max_points,
            use_dynamic_batch_inference=use_dynamic_batch_inference,
            max_points_inference=max_points_inference,
            use_weighted_sampler=use_weighted_sampler,
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
            split: 数据集划分（'train'、'val'、'test'、'predict'）
            transforms: 要应用的变换列表
            
        返回：
            BinPklDataset 实例
        """
        # 根据 split 选择对应的 loop
        loop_map = {
            'train': self.train_loop,
            'val': self.val_loop,
            'test': self.test_loop,
            'predict': self.predict_loop,
        }
        loop = loop_map.get(split, 1)
        
        return BinPklDataset(
            data_root=data_paths,
            split=split,
            assets=self.assets,
            transform=transforms,
            ignore_label=self.ignore_label,
            loop=loop,
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
        print(f"训练数据: {self.train_data}")
        print(f"验证数据: {self.val_data}")
        print(f"测试数据: {self.test_data}")
        print(f"预测数据: {self.predict_data}")
        print(f"数据集类型: BinPklDataset")
        print(f"资产: {self.assets}")
        print(f"忽略标签: {self.ignore_label}")
        print(f"循环配置:")
        print(f"  - 训练: {self.train_loop}")
        print(f"  - 验证: {self.val_loop} {'(TTA 启用)' if self.val_loop > 1 else ''}")
        print(f"  - 测试: {self.test_loop} {'(TTA 启用)' if self.test_loop > 1 else ''}")
        print(f"  - 预测: {self.predict_loop} {'(TTA 启用)' if self.predict_loop > 1 else ''}")
        print(f"缓存数据: {self.cache_data}")
        if self.class_mapping:
            print(f"类别映射: {self.class_mapping}")
        if self.class_names:
            print(f"类别名称: {self.class_names}")
        print(f"使用动态批次: {self.use_dynamic_batch}")
        if self.use_dynamic_batch:
            print(f"每批次最大点数: {self.max_points}")
        else:
            print(f"批次大小: {self.batch_size}")
        print(f"使用加权采样: {self.use_weighted_sampler}")
        if self.use_weighted_sampler and self.train_sampler_weights is not None:
            print(f"权重数量: {len(self.train_sampler_weights)}")
        print(f"工作进程数: {self.num_workers}")
        print(f"合并函数: {self.collate_fn.__name__ if hasattr(self.collate_fn, '__name__') else type(self.collate_fn).__name__}")
        print("-" * 60)
        
        for split in ['train', 'val', 'test', 'predict']:
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
