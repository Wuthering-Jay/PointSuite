"""
Focal Loss for Semantic Segmentation

解决类别不平衡问题，关注难分类样本
参考论文: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss - 关注难分类样本
    
    公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    特点：
    - 通过 γ 调节对难分类样本的关注度
    - 通过 α 平衡正负样本
    - 自动降低易分类样本的权重
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -1,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: 类别权重 [num_classes]，用于平衡类别
            gamma: 聚焦参数，γ=0 时退化为交叉熵
                  - γ>0 降低易分类样本的损失贡献
                  - 推荐值：2.0
            ignore_index: 忽略的标签索引
            reduction: 损失聚合方式
            label_smoothing: 标签平滑系数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, preds: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        计算 Focal Loss
        
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
        original_shape = preds.shape
        if preds.dim() == 3:  # [B, N, C]
            B, N, C = preds.shape
            preds = preds.reshape(B * N, C)
            if target.dim() == 2:
                target = target.reshape(B * N)
        
        # 应用标签平滑
        if self.label_smoothing > 0:
            num_classes = preds.size(-1)
            target_one_hot = F.one_hot(target, num_classes).float()
            target_one_hot = target_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
        
        # 计算交叉熵（不聚合）
        ce_loss = F.cross_entropy(
            preds,
            target,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # 计算 p_t（正确类别的预测概率）
        p = F.softmax(preds, dim=-1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # 应用 Focal Loss 公式
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # 应用类别权重 α
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha.gather(0, target)
            focal_loss = alpha_t * focal_loss
        
        # 过滤 ignore_index
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            focal_loss = focal_loss[mask]
        
        # 聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def extra_repr(self) -> str:
        return f'gamma={self.gamma}, ignore_index={self.ignore_index}'
