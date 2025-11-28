"""
BinPkl1 格式专用 DataModule (对应 tile_las1.py 生成的逻辑索引格式)

本模块提供了一个专门用于 BinPkl1 数据集格式的 PyTorch Lightning DataModule
支持体素模式和全量模式，以及动态批处理
"""

from typing import Optional, List, Dict, Any
from .datamodule_base import DataModuleBase
from .datasets.dataset_bin1 import BinPklDataset1


class BinPklDataModule1(DataModuleBase):
    """
    BinPkl1 格式点云数据集的 PyTorch Lightning DataModule
    
    此 DataModule 专门为 tile_las1.py 生成的 bin+pkl 数据格式设计，
    支持体素化采样和全量模式
    
    特性：
    - 支持两种模式：full (全量) 和 voxel (体素采样)
    - 体素模式下 train/val 随机采样，test/predict 模运算全覆盖
    - 支持 DynamicBatchSampler 进行内存控制
    - 支持类别标签映射
    - 数据缓存和循环选项
    
    示例：
        >>> # 基本用法
        >>> datamodule = BinPklDataModule1(
        ...     train_data='data/train',
        ...     val_data='data/val',
        ...     test_data='data/test',
        ...     mode='voxel',
        ...     batch_size=8,
        ...     num_workers=4
        ... )
        >>> 
        >>> # 使用 DynamicBatchSampler
        >>> datamodule = BinPklDataModule1(
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
        cache_data: bool = False,
        class_mapping: Optional[Dict[int, int]] = None,
        class_names: Optional[List[str]] = None,
        h_norm_grid: Optional[float] = 1.0,
        mode: str = 'voxel',
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
        初始化 BinPklDataModule1
        
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
            cache_data: 是否在内存中缓存加载的数据
            class_mapping: 类别标签映射字典
            class_names: 类别名称列表
            h_norm_grid: 计算归一化高程的栅格分辨率
            mode: 采样模式 ('full' 或 'voxel')
                - 'full': 全量模式，加载所有原始点
                - 'voxel': 体素模式，基于体素化索引采样
            max_loops: 体素模式下 test/predict 的最大采样轮次
                - None: 按体素内最大点数采样
                - 设置值: 限制最大轮数
            use_dynamic_batch: 是否使用 DynamicBatchSampler
            max_points: 每个批次的最大点数
                       (体素模式下指采样后的点数)
            use_dynamic_batch_inference: 推理阶段是否使用 DynamicBatchSampler
            max_points_inference: 推理阶段每个批次的最大点数
            use_weighted_sampler: 是否使用加权采样
            train_sampler_weights: 加权采样权重
            pin_memory: 是否使用固定内存
            persistent_workers: 是否保持工作进程
            prefetch_factor: 预取因子
        """
        # 存储 BinPklDataset1 特定参数
        self.assets = assets or ['coord', 'intensity', 'classification']
        self.ignore_label = ignore_label
        self.cache_data = cache_data
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
        为给定的数据集划分创建 BinPklDataset1 实例
        
        参数：
            data_paths: pkl 文件的路径
            split: 数据集划分
            transforms: 要应用的变换列表
            
        返回：
            BinPklDataset1 实例
        """
        loop_map = {
            'train': self.train_loop,
            'val': self.val_loop,
            'test': self.test_loop,
            'predict': self.predict_loop,
        }
        loop = loop_map.get(split, 1)
        
        return BinPklDataset1(
            data_root=data_paths,
            split=split,
            assets=self.assets,
            transform=transforms,
            ignore_label=self.ignore_label,
            loop=loop,
            cache_data=self.cache_data,
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
        
        # 添加 BinPkl1 特定信息
        info['dataset_type'] = 'BinPklDataset1'
        info['assets'] = self.assets
        info['ignore_label'] = self.ignore_label
        info['cache_data'] = self.cache_data
        info['class_mapping'] = self.class_mapping
        info['mode'] = self.mode
        info['max_loops'] = self.max_loops
        
        return info
    
    def print_info(self):
        """打印所有已初始化数据集的信息"""
        print("=" * 70)
        print("BinPklDataModule1 信息")
        print("=" * 70)
        print(f"训练数据: {self.train_data}")
        print(f"验证数据: {self.val_data}")
        print(f"测试数据: {self.test_data}")
        print(f"预测数据: {self.predict_data}")
        print(f"数据集类型: BinPklDataset1")
        print(f"资产: {self.assets}")
        print(f"忽略标签: {self.ignore_label}")
        print(f"采样模式: {self.mode}")
        if self.mode == 'voxel':
            print(f"最大采样轮次 (test/predict): {self.max_loops or '自动'}")
        print(f"循环配置:")
        print(f"  - 训练: {self.train_loop}")
        print(f"  - 验证: {self.val_loop}")
        print(f"  - 测试: {self.test_loop}")
        print(f"  - 预测: {self.predict_loop}")
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
        print(f"工作进程数: {self.num_workers}")
        print("-" * 70)
        
        for split in ['train', 'val', 'test', 'predict']:
            try:
                info = super().get_dataset_info(split)
                dataset = getattr(self, f'{split}_dataset', None)
                print(f"{split.upper()} 数据集:")
                print(f"  - 样本数: {info.get('num_samples', '不适用')}")
                print(f"  - 总长度（含循环）: {info['total_length']}")
                if dataset is not None and hasattr(dataset, 'mode'):
                    print(f"  - 模式: {dataset.mode}")
            except ValueError:
                print(f"{split.upper()} 数据集: 未初始化")
        
        print("=" * 70)
    
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
