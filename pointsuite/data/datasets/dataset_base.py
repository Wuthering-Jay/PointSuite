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
        "return_number",  # 回波编号
        "number_of_returns",  # 回波总数
        "classification",  # 分类标签
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
            cache_data: 是否在内存中缓存加载的数据
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
        self.loop = (loop if split == 'train' else 1)
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
        print(f"    - 资产: {self.assets}")
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





