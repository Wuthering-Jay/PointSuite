"""
BinPkl 格式专用 DataModule (对应 tile_las.py 生成的逻辑索引格式)

本模块提供了一个专门用于 BinPkl 数据集格式的 PyTorch Lightning DataModule
支持体素模式和全量模式，以及动态批处理
"""

from typing import Optional, List, Dict, Any, Union
from .datamodule_base import DataModuleBase
from .datasets.dataset_bin import BinPklDataset
from ..utils.mapping import ClassMappingInput


class BinPklDataModule(DataModuleBase):
    """
    BinPkl 格式点云数据集的 PyTorch Lightning DataModule
    
    此 DataModule 专门为 tile_las.py 生成的 bin+pkl 数据格式设计，
    支持体素化采样和全量模式
    
    特性：
    - 支持两种模式：full (全量) 和 voxel (体素采样)
    - 体素模式下 train/val 随机采样，test/predict 模运算全覆盖
    - 支持 DynamicBatchSampler 进行内存控制
    - 支持类别标签映射
    - 数据缓存和循环选项
    
    示例：
        >>> # 基本用法
        >>> datamodule = BinPklDataModule(
        ...     train_data='data/train',
        ...     val_data='data/val',
        ...     test_data='data/test',
        ...     mode='voxel',
        ...     batch_size=8,
        ...     num_workers=4
        ... )
        >>> 
        >>> # 使用 DynamicBatchSampler
        >>> datamodule = BinPklDataModule(
        ...     train_data='data/train',
        ...     val_data='data/val',
        ...     mode='voxel',
        ...     max_loops=10,  # 限制 test/predict 的采样轮数
        ...     use_dynamic_batch=True,
        ...     max_points=100000,  # 体素模式下是体素数限制
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
        class_mapping: ClassMappingInput = None,
        class_names: Optional[List[str]] = None,
        h_norm_grid: Optional[float] = 1.0,
        mode: str = 'grid',
        max_loops: Optional[int] = None,
        use_dynamic_batch: bool = False,
        max_points: int = 100000,
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
            train_data: 训练数据路径
            val_data: 验证数据路径
            test_data: 测试数据路径
            predict_data: 预测数据路径
            batch_size: DataLoader 的批次大小
            num_workers: 数据加载的工作进程数
            assets: 要加载的数据属性列表
            train_transforms: 训练数据的变换列表
            val_transforms: 验证数据的变换列表
            test_transforms: 测试数据的变换列表
            predict_transforms: 预测数据的变换列表
            ignore_label: 在训练/评估中忽略的标签
            train_loop: 训练数据集循环次数
            val_loop: 验证数据集循环次数
            test_loop: 测试数据集循环次数
            predict_loop: 预测数据集循环次数
            class_mapping: 类别标签映射配置，支持以下格式：
                - None: 不做映射，使用原始标签
                - Dict[int, int]: 显式映射 {原始ID: 新ID}
                - List[int]: 原始类别ID列表，自动映射为 [0, 1, 2, ...]
                示例：{1: 0, 2: 1, 6: 2} 或 [1, 2, 6]（两者等价）
            class_names: 类别名称列表
            h_norm_grid: 计算归一化高程的栅格分辨率
            mode: 采样模式 ('full' 或 'grid')
                - 'full': 全量模式，加载所有原始点
                - 'grid': 网格采样模式，基于网格化索引采样
            max_loops: 网格采样模式下 test/predict 的最大采样轮次
                - None: 按网格内最大点数采样
                - 设置值: 限制最大轮数
            use_dynamic_batch: 是否使用 DynamicBatchSampler
            max_points: 每个批次的最大点数
                       (网格模式下指采样后的点数)
            use_dynamic_batch_inference: 推理阶段是否使用 DynamicBatchSampler
            max_points_inference: 推理阶段每个批次的最大点数
            use_weighted_sampler: 是否使用加权采样
            train_sampler_weights: 加权采样权重
            pin_memory: 是否使用固定内存
            persistent_workers: 是否保持工作进程
            prefetch_factor: 预取因子
        """
        # 存储 BinPklDataset 特定参数
        self.assets = assets or ['coord', 'intensity', 'classification']
        self.ignore_label = ignore_label
        self.class_mapping = class_mapping
        self.class_names = class_names
        self.h_norm_grid = h_norm_grid
        self.mode = mode
        self.max_loops = max_loops
        
        # 调用父类构造函数
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
            split: 数据集划分
            transforms: 要应用的变换列表
            
        返回：
            BinPklDataset 实例
        """
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
            class_mapping=self.class_mapping,
            h_norm_grid=self.h_norm_grid,
            mode=self.mode,
            max_loops=self.max_loops,
            **self.kwargs
        )
    
    def get_dataset_info(self, split: str = 'train') -> Dict:
        """
        获取数据集划分的信息
        
        参数：
            split: 数据集划分
            
        返回：
            包含数据集信息的字典
        """
        info = super().get_dataset_info(split)
        
        # 添加 BinPkl 特定信息
        info['dataset_type'] = 'BinPklDataset'
        info['assets'] = self.assets
        info['ignore_label'] = self.ignore_label
        info['class_mapping'] = self.class_mapping
        info['mode'] = self.mode
        info['max_loops'] = self.max_loops
        
        return info
    
    def print_info(self) -> None:
        """
        打印所有已初始化数据集的信息
        
        输出包括数据路径、数据集配置、循环配置和数据集统计信息。
        """
        from ..utils.logger import (
            print_header,
            print_section,
            log_info,
            log_debug,
        )
        
        print_header("BinPklDataModule 信息")
        
        # 数据路径
        print_section("数据路径")
        log_info(f"训练数据: {self.train_data or 'N/A'}")
        log_info(f"验证数据: {self.val_data or 'N/A'}")
        log_info(f"测试数据: {self.test_data or 'N/A'}")
        log_info(f"预测数据: {self.predict_data or 'N/A'}")
        
        # 数据集配置
        print_section("数据集配置")
        log_info(f"数据集类型: BinPklDataset")
        log_info(f"属性字段: {', '.join(self.assets)}")
        log_info(f"忽略标签: {self.ignore_label}")
        log_info(f"采样模式: {self.mode}")
        log_info(f"最大轮次: {self.max_loops or '自动'}")
        
        # 循环配置
        print_section("循环配置")
        log_info(f"训练循环: {self.train_loop}")
        log_info(f"验证循环: {self.val_loop}")
        log_info(f"测试循环: {self.test_loop}")
        log_info(f"预测循环: {self.predict_loop}")
        
        if self.class_mapping:
            log_info(f"类别映射: {self.class_mapping}")
        if self.class_names:
            log_info(f"类别名称: {', '.join(self.class_names)}")
        
        # 加载配置
        print_section("加载配置")
        log_info(f"动态批次: {'是' if self.use_dynamic_batch else '否'}")
        log_info(f"批次大小/最大点数: {self.max_points if self.use_dynamic_batch else self.batch_size}")
        log_info(f"加权采样: {'是' if self.use_weighted_sampler else '否'}")
        log_info(f"工作进程: {self.num_workers}")
        
        # 数据集统计
        print_section("数据集统计")
        for split in ['train', 'val', 'test', 'predict']:
            try:
                info = super().get_dataset_info(split)
                dataset = getattr(self, f'{split}_dataset', None)
                samples = info.get('num_samples', 'N/A')
                total = info['total_length']
                mode = dataset.mode if dataset and hasattr(dataset, 'mode') else 'N/A'
                log_info(f"{split.upper():8}: {samples} 样本, 总长度 {total}, 模式 {mode}")
            except ValueError:
                log_debug(f"{split.upper():8}: 未初始化")
    
    def get_sample_num_points(self, split: str = 'train') -> List[int]:
        """
        获取指定划分的每个样本点数列表
        用于 DynamicBatchSampler
        
        参数：
            split: 数据集划分
            
        返回：
            每个样本的点数列表
        """
        dataset = getattr(self, f'{split}_dataset', None)
        if dataset is None:
            raise ValueError(f"{split} 数据集未初始化")
        
        if hasattr(dataset, 'get_sample_num_points'):
            return dataset.get_sample_num_points()
        else:
            # 回退到基本方法
            return [s['num_points'] for s in dataset.data_list]
