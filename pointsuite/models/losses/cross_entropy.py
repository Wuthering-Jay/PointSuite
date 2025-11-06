"""
Cross Entropy Loss for Semantic Segmentation

支持类别权重和标签平滑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class CrossEntropyLoss(nn.Module):
    """
    交叉熵损失函数，用于语义分割任务
    
    特点：
    - 支持类别权重处理类别不平衡
    - 支持标签平滑
    - 支持 ignore_index
    - 兼容 Task 框架的输入格式
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            weight: 类别权重 [num_classes]，用于处理类别不平衡
            ignore_index: 忽略的标签索引（如背景类）
            label_smoothing: 标签平滑系数 [0, 1)
            reduction: 损失聚合方式 ('none', 'mean', 'sum')
        """
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # 使用 PyTorch 内置的 CrossEntropyLoss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    
    def forward(self, preds: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        计算交叉熵损失
        
        Args:
            preds: 模型预测 logits
                  - [N, C] for point-level prediction
                  - [B, N, C] for batched prediction
            batch: 批次数据字典，包含:
                  - 'class': 点级标签 [N] or [B, N]
                  - 'labels': 备用键名
        
        Returns:
            loss: 标量损失值
        """
        # 获取标签（兼容不同键名）
        target = batch.get('class', batch.get('labels'))
        
        if target is None:
            raise ValueError("batch 中必须包含 'class' 或 'labels' 键")
        
        # 确保标签是 long 类型
        target = target.long()
        
        # 处理维度：如果 preds 是 [N, C]，target 是 [N]，直接计算
        # 如果 preds 是 [B, N, C]，需要 reshape
        if preds.dim() == 3:  # [B, N, C]
            B, N, C = preds.shape
            preds = preds.reshape(B * N, C)
            if target.dim() == 2:  # [B, N]
                target = target.reshape(B * N)
        
        # 计算损失
        loss = self.ce_loss(preds, target)
        
        return loss
    
    def extra_repr(self) -> str:
        """额外的字符串表示"""
        return f'ignore_index={self.ignore_index}, label_smoothing={self.label_smoothing}'
