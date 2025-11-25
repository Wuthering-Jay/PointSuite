import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from collections.abc import Sequence
from ..transforms import Compose


class DatasetBase(Dataset, ABC):
    """
    点云数据的抽象基础数据集类
    
    这是所有数据集实现的基础
    子类应实现抽象方法来处理特定的数据格式
    """

    VALID_ASSETS = [
        "coord",  # XYZ 坐标（必需）
        "color",  # RGB 颜色
        "normal",  # 法向量
        "intensity",  # 强度
        "echo",  # 回波信息
        "h_norm",  # 归一化高程
        "class",  # 分类标签
    ]

    def __init__(
            self,
            data_root,
            split: str = 'train',
            assets: Optional[List[str]] = None,
            transform: Optional[List] = None,
            ignore_label: int = -1,
            loop: int = 1,
            cache_data: bool = False,
            class_mapping: Optional[Dict[int, int]] = None,
            **kwargs
    ):
        """
        初始化基础数据集
        
        参数：
            data_root: 根目录、单个文件路径或文件路径列表
            split: 数据集划分（'train'、'val'、'test'）
            assets: 要加载的数据属性列表（None 表示使用默认值）
            transform: 要应用的数据变换
            ignore_label: 在训练/评估中忽略的标签
            loop: 遍历数据集的次数（用于训练增强）
            cache_data: 是否在内存中缓存加载的数据，数据集较小时适用
            class_mapping: 将原始类别标签映射到连续标签的字典
                          示例：{0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
                          如果为 None，则不应用映射
            **kwargs: 子类的其他参数
        """
        super().__init__()
        
        # 处理不同的 data_root 类型
        if isinstance(data_root, (list, tuple)):
            # 路径列表
            self.data_root = data_root
        else:
            # 单个路径
            self.data_root = Path(data_root)
        
        self.split = split
        self.assets = assets if assets is not None else self.VALID_ASSETS.copy()
        self.transform = Compose(transform) if transform is not None else None
        self.ignore_label = ignore_label
        self.loop = loop  # 支持所有 split 的 loop（Test-Time Augmentation）
        self.cache_data = cache_data
        self.class_mapping = class_mapping
        
        # 如果启用则缓存数据
        self.data_cache = {} if cache_data else None
        
        # 验证数据根目录（对于列表类型跳过验证，由子类处理）
        if not isinstance(self.data_root, (list, tuple)) and not self.data_root.exists():
            raise ValueError(f"数据根目录不存在: {self.data_root}")
        
        # 加载数据列表（由子类实现）
        self.data_list = self._load_data_list()
        
        # 打印初始化信息
        self._print_init_info()
    
    def _print_init_info(self):
        """打印数据集初始化信息"""
        print(f"==> {self.__class__.__name__} ({self.split}) 已初始化:")
        print(f"    - 数据根目录: {self.data_root}")
        print(f"    - 总样本数: {len(self.data_list)}")
        print(f"    - 属性: {self.assets}")
        print(f"    - 循环: {self.loop}")
        print(f"    - 缓存: {'已启用' if self.cache_data else '已禁用'}")
    
    @abstractmethod
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """
        加载所有数据样本的列表
        必须由子类实现
        
        返回：
            包含样本信息的字典列表
        """
        raise NotImplementedError("子类必须实现 _load_data_list() 方法")
    
    @abstractmethod
    def _load_data(self, idx: int) -> Dict[str, Any]:
        """
        加载给定索引的点云数据
        必须由子类实现
        
        参数：
            idx: 数据索引
            
        返回：
            包含加载数据的字典（coord、labels 等）
        """
        raise NotImplementedError("子类必须实现 _load_data() 方法")
    
    def __len__(self) -> int:
        """返回考虑循环因子的数据集长度"""
        return len(self.data_list) * self.loop
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        加载并返回一个数据样本
        
        参数：
            idx: 样本索引
            
        返回：
            包含点云数据和标签的字典
        """
        # 处理循环
        data_idx = idx % len(self.data_list)
        
        # 检查缓存
        if self.cache_data and data_idx in self.data_cache:
            data_dict = self.data_cache[data_idx].copy()
        else:
            # 加载数据（由子类实现）
            data_dict = self._load_data(data_idx)
            
            # 如果启用则缓存
            if self.cache_data:
                self.data_cache[data_idx] = data_dict.copy()
        
        # 应用变换
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        
        return data_dict
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        获取样本信息而不加载数据
        
        参数：
            idx: 样本索引
            
        返回：
            样本信息字典
        """
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]
    
    def get_class_distribution(self) -> Optional[Dict[int, int]]:
        """
        获取数据集的类别分布（每个类别的样本点数）
        
        子类应重写此方法以提供特定格式的类别统计
        
        返回：
            类别分布字典 {class_id: count}，如果不支持则返回 None
        """
        return None
    
    def get_sample_weights(self, class_weights: Optional[Dict[int, float]] = None) -> Optional[np.ndarray]:
        """
        计算每个样本的权重（用于 WeightedRandomSampler）
        
        样本权重基于其包含的类别权重之和，这样：
        - 包含稀有类别的样本获得更高权重
        - 包含多个类别的样本获得更高权重
        
        参数：
            class_weights: 类别权重字典 {class_id: weight}
                          如果为 None，则返回 None（不支持加权采样）
        
        返回：
            样本权重数组 [num_samples]，如果不支持则返回 None
        """
        return None
    
    def compute_class_weights(
        self,
        method: str = 'inverse',
        smooth: float = 1.0,
        normalize: bool = True
    ) -> Optional[Dict[int, float]]:
        """
        从数据集的类别分布计算类别权重
        
        这是一个便捷方法，基于 get_class_distribution() 自动计算权重
        子类通常不需要重写此方法，只需实现 get_class_distribution()
        
        参数：
            method: 权重计算方法
                   - 'inverse': 1 / count（反比例）
                   - 'sqrt_inverse': 1 / sqrt(count)（平方根反比例）
                   - 'log_inverse': 1 / log(count + 1)（对数反比例）
                   - 'effective_num': Effective Number of Samples (ENS) 方法
            smooth: 平滑参数，避免权重过大（加到分母上）
            normalize: 是否归一化权重使其和为 1
        
        返回：
            类别权重字典 {class_id: weight}，如果不支持则返回 None
        """
        class_distribution = self.get_class_distribution()
        if class_distribution is None or len(class_distribution) == 0:
            return None
        
        # 转换为数组格式
        num_classes = max(class_distribution.keys()) + 1
        counts = np.zeros(num_classes, dtype=np.float64)
        for class_id, count in class_distribution.items():
            counts[class_id] = count
        
        # 处理空类别
        empty_classes = np.where(counts == 0)[0]
        if len(empty_classes) > 0:
            counts[empty_classes] = 1.0
        
        # 计算权重
        if method == 'inverse':
            weights = 1.0 / (counts + smooth)
        elif method == 'sqrt_inverse':
            weights = 1.0 / np.sqrt(counts + smooth)
        elif method == 'log_inverse':
            weights = 1.0 / np.log(counts + smooth + 1)
        elif method == 'effective_num':
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / (effective_num + 1e-8)
        else:
            raise ValueError(f"未知的权重计算方法: {method}")
        
        # 归一化
        if normalize:
            weights = weights / weights.sum()
        
        # 转换为字典
        return {i: float(weights[i]) for i in range(num_classes) if counts[i] > 0}





