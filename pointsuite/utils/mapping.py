"""
类别映射工具

提供统一的类别标签映射处理，支持多种输入格式：
1. Dict[int, int]: 显式映射 {原始ID: 新ID}
2. List[int]: 原始类别ID列表，自动按顺序映射为 0, 1, 2, ...
3. None: 不做映射

示例：
    >>> # 方式1: 显式字典映射
    >>> mapping = ClassMapping({1: 0, 2: 1, 3: 2, 4: 3})
    
    >>> # 方式2: 类别ID列表（自动映射）
    >>> mapping = ClassMapping([1, 2, 3, 4])  # 等价于 {1: 0, 2: 1, 3: 2, 4: 3}
    
    >>> # 方式3: 不映射
    >>> mapping = ClassMapping(None)
    
    >>> # 使用
    >>> new_label = mapping.forward(original_label)  # 原始 -> 模型
    >>> original_label = mapping.backward(new_label)  # 模型 -> 原始
"""

from typing import Dict, List, Optional, Union
import numpy as np


# 类型别名：支持 Dict 或 List 作为输入
ClassMappingInput = Optional[Union[Dict[int, int], List[int]]]


class ClassMapping:
    """
    类别标签映射器
    
    支持将原始标签映射到连续的模型标签（0, 1, 2, ...），
    以及将模型预测反向映射回原始标签。
    
    属性：
        forward_map: 原始标签 -> 模型标签 的映射字典
        backward_map: 模型标签 -> 原始标签 的映射字典
        num_classes: 映射后的类别数量
        original_classes: 原始类别ID列表（有序）
        ignore_label: 不在映射中的标签将被设为此值
    """
    
    def __init__(
        self, 
        mapping: ClassMappingInput = None,
        ignore_label: int = -1
    ):
        """
        初始化类别映射器
        
        参数：
            mapping: 类别映射配置，支持以下格式：
                - None: 不做任何映射（直接使用原始标签）
                - Dict[int, int]: 显式映射 {原始ID: 新ID}
                - List[int]: 原始类别ID列表，自动按顺序映射为 0, 1, 2, ...
            ignore_label: 未映射标签的默认值（默认 -1）
        
        示例：
            >>> ClassMapping({1: 0, 2: 1, 6: 2})  # 显式映射
            >>> ClassMapping([1, 2, 6])           # 等价于上面
            >>> ClassMapping(None)               # 不映射
        """
        self.ignore_label = ignore_label
        self._input_mapping = mapping  # 保存原始输入用于序列化
        
        if mapping is None:
            # 不做映射
            self.forward_map = None
            self.backward_map = None
            self.num_classes = None
            self.original_classes = None
        elif isinstance(mapping, dict):
            # 显式字典映射
            self.forward_map = dict(mapping)
            self.backward_map = {v: k for k, v in mapping.items()}
            self.num_classes = len(set(mapping.values()))
            self.original_classes = sorted(mapping.keys())
        elif isinstance(mapping, (list, tuple)):
            # 列表形式：自动生成连续映射
            self.forward_map = {orig: idx for idx, orig in enumerate(mapping)}
            self.backward_map = {idx: orig for idx, orig in enumerate(mapping)}
            self.num_classes = len(mapping)
            self.original_classes = list(mapping)
        else:
            raise TypeError(
                f"mapping 必须是 None、Dict[int, int] 或 List[int]，"
                f"但收到 {type(mapping).__name__}"
            )
        
        # 验证映射的有效性
        self._validate()
    
    def _validate(self):
        """验证映射的有效性"""
        if self.forward_map is None:
            return
        
        # 检查目标标签是否连续且从0开始
        target_labels = sorted(set(self.forward_map.values()))
        expected = list(range(len(target_labels)))
        
        if target_labels != expected:
            raise ValueError(
                f"映射后的标签必须是从0开始的连续整数，"
                f"但得到 {target_labels}，期望 {expected}"
            )
    
    @property
    def is_identity(self) -> bool:
        """是否为恒等映射（不做任何转换）"""
        return self.forward_map is None
    
    def forward(self, label: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        将原始标签映射到模型标签
        
        参数：
            label: 原始标签（单个值或数组）
            
        返回：
            映射后的标签，未知标签返回 ignore_label
        """
        if self.forward_map is None:
            return label
        
        if isinstance(label, np.ndarray):
            result = np.full_like(label, self.ignore_label)
            for orig, new in self.forward_map.items():
                result[label == orig] = new
            return result
        else:
            return self.forward_map.get(label, self.ignore_label)
    
    def backward(self, label: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        将模型标签反向映射到原始标签
        
        参数：
            label: 模型标签（单个值或数组）
            
        返回：
            原始标签，未知标签返回 ignore_label
        """
        if self.backward_map is None:
            return label
        
        if isinstance(label, np.ndarray):
            result = np.full_like(label, self.ignore_label)
            for new, orig in self.backward_map.items():
                result[label == new] = orig
            return result
        else:
            return self.backward_map.get(label, self.ignore_label)
    
    def to_dict(self) -> Optional[Dict[int, int]]:
        """
        返回字典形式的映射（用于序列化）
        
        返回：
            forward_map 字典，或 None（如果不映射）
        """
        return self.forward_map
    
    def to_reverse_dict(self) -> Optional[Dict[int, int]]:
        """
        返回反向映射字典（用于预测结果后处理）
        
        返回：
            backward_map 字典，或 None（如果不映射）
        """
        return self.backward_map
    
    def __repr__(self) -> str:
        if self.forward_map is None:
            return "ClassMapping(None)"
        return f"ClassMapping({self.original_classes} -> [0..{self.num_classes-1}])"
    
    def __str__(self) -> str:
        if self.forward_map is None:
            return "无映射 (使用原始标签)"
        return f"{self.original_classes} -> [0, {self.num_classes-1}]"
    
    @classmethod
    def from_config(cls, config: ClassMappingInput, ignore_label: int = -1) -> 'ClassMapping':
        """
        从配置创建 ClassMapping（工厂方法）
        
        这是推荐的创建方式，支持所有输入格式。
        
        参数：
            config: 映射配置（None、Dict 或 List）
            ignore_label: 忽略标签值
            
        返回：
            ClassMapping 实例
        """
        return cls(config, ignore_label)


def normalize_class_mapping(
    mapping: ClassMappingInput,
    ignore_label: int = -1
) -> Optional[Dict[int, int]]:
    """
    将各种映射输入格式标准化为字典
    
    这是一个便捷函数，用于兼容旧代码。
    新代码推荐直接使用 ClassMapping 类。
    
    参数：
        mapping: 映射配置（None、Dict 或 List）
        ignore_label: 忽略标签值（此函数中未使用，仅为签名一致性）
        
    返回：
        标准化的映射字典，或 None
    """
    if mapping is None:
        return None
    elif isinstance(mapping, dict):
        return dict(mapping)
    elif isinstance(mapping, (list, tuple)):
        return {orig: idx for idx, orig in enumerate(mapping)}
    else:
        raise TypeError(
            f"mapping 必须是 None、Dict[int, int] 或 List[int]，"
            f"但收到 {type(mapping).__name__}"
        )


def create_reverse_mapping(mapping: ClassMappingInput) -> Optional[Dict[int, int]]:
    """
    从映射配置创建反向映射
    
    参数：
        mapping: 映射配置（None、Dict 或 List）
        
    返回：
        反向映射字典（模型标签 -> 原始标签），或 None
    """
    if mapping is None:
        return None
    
    cm = ClassMapping(mapping)
    return cm.to_reverse_dict()
