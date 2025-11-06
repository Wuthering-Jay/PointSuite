"""
Lovász-Softmax Loss for Semantic Segmentation

优化 IoU 指标的代理损失函数
参考论文: "The Lovász-Softmax loss: A tractable surrogate for the optimization of 
           the intersection-over-union measure in neural networks" (Berman et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    计算 Lovász 扩展的梯度
    
    Args:
        gt_sorted: 排序后的真值标签
    
    Returns:
        梯度权重
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    
    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    
    return jaccard


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor, 
                       classes: str = 'present') -> torch.Tensor:
    """
    Lovász-Softmax loss (扁平化版本)
    
    Args:
        probs: 类别概率 [P, C]
        labels: 真值标签 [P]
        classes: 计算哪些类别的损失
                - 'all': 所有类别
                - 'present': 只计算 batch 中出现的类别
                - list: 指定类别列表
    
    Returns:
        loss: 标量损失值
    """
    if probs.numel() == 0:
        return probs.sum() * 0.
    
    C = probs.size(1)
    losses = []
    
    # 确定要计算的类别
    if classes == 'all':
        class_to_sum = list(range(C))
    elif classes == 'present':
        class_to_sum = torch.unique(labels).tolist()
    else:
        class_to_sum = classes
    
    for c in class_to_sum:
        # 获取该类别的概率和真值
        fg = (labels == c).float()  # foreground
        if fg.sum() == 0:
            continue
        
        errors = (1. - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    
    return torch.stack(losses).mean() if losses else torch.tensor(0., device=probs.device)


class LovaszLoss(nn.Module):
    """
    Lovász-Softmax Loss - IoU 优化的代理损失
    
    特点：
    - 直接优化 IoU 指标
    - 对类别不平衡鲁棒
    - 可以只计算 batch 中出现的类别
    """
    
    def __init__(
        self,
        ignore_index: int = -1,
        classes: str = 'present',
        per_point: bool = False
    ):
        """
        Args:
            ignore_index: 忽略的标签索引
            classes: 计算哪些类别
                    - 'all': 所有类别
                    - 'present': 只计算 batch 中出现的类别（推荐）
            per_point: 是否对每个点单独计算 loss（通常用 False）
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes
        self.per_point = per_point
    
    def forward(self, preds: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        计算 Lovász-Softmax Loss
        
        Args:
            preds: 模型预测 logits [N, C] or [B, N, C]
            batch: 批次数据字典
        
        Returns:
            loss: 标量损失值
        """
        # 获取标签
        target = batch.get('class', batch.get('labels'))
        if target is None:
            raise ValueError("batch 中必须包含 'class' 或 'labels' 键")
        
        target = target.long()
        
        # 处理维度
        if preds.dim() == 3:  # [B, N, C]
            B, N, C = preds.shape
            preds = preds.reshape(B * N, C)
            if target.dim() == 2:
                target = target.reshape(B * N)
        
        # 转换为概率
        probs = F.softmax(preds, dim=1)
        
        # 过滤 ignore_index
        if self.ignore_index >= 0:
            valid_mask = target != self.ignore_index
            probs = probs[valid_mask]
            target = target[valid_mask]
        
        if probs.numel() == 0:
            return preds.sum() * 0.
        
        # 计算 Lovász loss
        if self.per_point:
            # 对每个点单独计算（较少使用）
            losses = []
            for prob, label in zip(probs, target):
                losses.append(lovasz_softmax_flat(
                    prob.unsqueeze(0),
                    label.unsqueeze(0),
                    classes=self.classes
                ))
            return torch.stack(losses).mean()
        else:
            # 整体计算（推荐）
            return lovasz_softmax_flat(probs, target, classes=self.classes)
    
    def extra_repr(self) -> str:
        return f'classes={self.classes}, ignore_index={self.ignore_index}'
