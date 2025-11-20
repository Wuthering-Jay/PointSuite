"""
Base Utilities for Metrics

提供混淆矩阵基类和通用工具函数
"""

import torch
from torchmetrics import Metric
from typing import Optional, List, Dict


def create_class_names(
    num_classes: int,
    class_names: Optional[List[str]] = None,
    reverse_class_mapping: Optional[Dict[int, int]] = None
) -> List[str]:
    """
    创建类别名称列表
    
    优先级：
    1. 用户提供的 class_names
    2. 使用 reverse_class_mapping 从连续标签映射回原始标签
    3. 默认使用 "Class 0", "Class 1", ...
    
    Args:
        num_classes: 类别数量（映射后的连续类别数）
        class_names: 用户提供的类别名称列表
        reverse_class_mapping: 连续标签 -> 原始标签的映射
    
    Returns:
        List[str]: 类别名称列表
    """
    if class_names is not None:
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names 长度 ({len(class_names)}) 与 num_classes ({num_classes}) 不匹配"
            )
        return class_names
    
    if reverse_class_mapping is not None:
        names = []
        for i in range(num_classes):
            original_label = reverse_class_mapping.get(i, i)
            names.append(f"Class {original_label}")
        return names
    
    return [f"Class {i}" for i in range(num_classes)]


def convert_preds_to_labels(preds: torch.Tensor) -> torch.Tensor:
    """
    将预测转换为标签
    
    自动检测输入类型:
    - 如果是 1D tensor [N]，假设已经是 labels，直接返回
    - 如果是 2D tensor [N, C]，假设是 logits，执行 argmax
    - 如果是 3D tensor [B, N, C]，flatten 后执行 argmax
    
    Args:
        preds: 预测结果，可以是 labels [N] 或 logits [N, C] 或 [B, N, C]
        
    Returns:
        torch.Tensor: 标签 [N] 或 [B*N]，long类型
    """
    if preds.dim() == 1:
        return preds.long()
    elif preds.dim() == 2:
        return torch.argmax(preds, dim=1).long()
    elif preds.dim() == 3:
        B, N, C = preds.shape
        preds = preds.reshape(B * N, C)
        return torch.argmax(preds, dim=1).long()
    else:
        raise ValueError(f"Unsupported prediction shape: {preds.shape}")


class ConfusionMatrixBase(Metric):
    """
    混淆矩阵基类
    
    所有基于混淆矩阵的指标的统一基类，提供：
    1. 混淆矩阵状态管理
    2. update() 方法实现
    3. TP/FP/FN 计算
    
    子类只需实现 compute() 方法即可
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -1,
        class_names: Optional[List[str]] = None,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        """
        Args:
            num_classes: 类别数量
            ignore_index: 忽略的标签索引
            class_names: 类别名称列表
            dist_sync_on_step: 是否在每步同步
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        # 混淆矩阵 [C, C]
        self.add_state(
            "confusion_matrix",
            default=torch.zeros(num_classes, num_classes),
            dist_reduce_fx="sum"
        )
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        更新混淆矩阵
        
        Args:
            preds: 预测结果 (labels [N] 或 logits [N, C] 或 [B, N, C])
            target: 真实标签 [N] 或 [B, N]
        """
        # 处理 target 维度
        if target.dim() == 2:
            target = target.reshape(-1)
        
        # 转换 preds 为 labels
        pred_labels = convert_preds_to_labels(preds)
        
        # 过滤 ignore_index
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            pred_labels = pred_labels[valid_mask]
            target = target[valid_mask]
        
        # 确保类型正确
        target = target.long()
        pred_labels = pred_labels.long()
        
        # 计算混淆矩阵
        indices = (self.num_classes * target + pred_labels).long()
        cm = torch.bincount(indices, minlength=self.num_classes ** 2)
        cm = cm.reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += cm
    
    def _compute_tp_fp_fn(self):
        """计算 TP, FP, FN"""
        cm = self.confusion_matrix
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        return tp, fp, fn
