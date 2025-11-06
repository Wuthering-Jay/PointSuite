"""
Dice Loss for Semantic Segmentation

基于 Dice 系数的损失函数，特别适合处理类别不平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class DiceLoss(nn.Module):
    """
    Dice Loss - 基于 Dice 系数
    
    Dice 系数: DSC = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss: 1 - DSC
    
    特点：
    - 对类别不平衡鲁棒
    - 直接优化重叠区域
    - 适合分割任务
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        square_denominator: bool = False
    ):
        """
        Args:
            smooth: 平滑项，防止除零
            ignore_index: 忽略的标签索引
            weight: 类别权重 [num_classes]
            reduction: 损失聚合方式
            square_denominator: 是否对分母平方（Tversky loss 的变体）
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.square_denominator = square_denominator
    
    def forward(self, preds: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        计算 Dice Loss
        
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
        
        # 过滤 ignore_index
        if self.ignore_index >= 0:
            valid_mask = target != self.ignore_index
            preds = preds[valid_mask]
            target = target[valid_mask]
        
        if preds.numel() == 0:
            return preds.sum() * 0.
        
        # 转换为概率
        probs = F.softmax(preds, dim=1)
        num_classes = probs.size(1)
        
        # 转换标签为 one-hot
        target_one_hot = F.one_hot(target, num_classes).float()  # [N, C]
        
        # 计算每个类别的 Dice
        dice_losses = []
        for c in range(num_classes):
            # 预测概率和真值
            pred_c = probs[:, c]
            target_c = target_one_hot[:, c]
            
            # 计算交集和并集
            intersection = (pred_c * target_c).sum()
            
            if self.square_denominator:
                # Tversky loss 变体
                denominator = (pred_c ** 2).sum() + (target_c ** 2).sum()
            else:
                # 标准 Dice
                denominator = pred_c.sum() + target_c.sum()
            
            # Dice 系数
            dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
            
            # Dice Loss
            dice_loss = 1. - dice
            
            dice_losses.append(dice_loss)
        
        dice_losses = torch.stack(dice_losses)
        
        # 应用类别权重
        if self.weight is not None:
            if self.weight.device != dice_losses.device:
                self.weight = self.weight.to(dice_losses.device)
            dice_losses = dice_losses * self.weight
        
        # 聚合
        if self.reduction == 'mean':
            return dice_losses.mean()
        elif self.reduction == 'sum':
            return dice_losses.sum()
        else:
            return dice_losses
    
    def extra_repr(self) -> str:
        return f'smooth={self.smooth}, ignore_index={self.ignore_index}'


class DiceCELoss(nn.Module):
    """
    Dice Loss + Cross Entropy Loss 的组合
    
    结合两种损失的优点：
    - Dice Loss: 关注重叠区域，对类别不平衡鲁棒
    - CE Loss: 提供稳定的梯度
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1.0,
        ignore_index: int = -1,
        class_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            dice_weight: Dice Loss 的权重
            ce_weight: CE Loss 的权重
            smooth: Dice Loss 的平滑项
            ignore_index: 忽略的标签索引
            class_weight: 类别权重
            label_smoothing: 标签平滑系数
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss(
            smooth=smooth,
            ignore_index=ignore_index,
            weight=class_weight
        )
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(self, preds: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """计算组合损失"""
        dice = self.dice_loss(preds, batch)
        
        # CE Loss 需要单独处理维度
        target = batch.get('class', batch.get('labels')).long()
        if preds.dim() == 3:
            B, N, C = preds.shape
            preds_flat = preds.reshape(B * N, C)
            if target.dim() == 2:
                target_flat = target.reshape(B * N)
            else:
                target_flat = target
        else:
            preds_flat = preds
            target_flat = target
        
        ce = self.ce_loss(preds_flat, target_flat)
        
        return self.dice_weight * dice + self.ce_weight * ce
    
    def extra_repr(self) -> str:
        return f'dice_weight={self.dice_weight}, ce_weight={self.ce_weight}'
