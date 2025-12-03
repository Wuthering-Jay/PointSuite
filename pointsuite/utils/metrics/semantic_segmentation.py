"""
Semantic Segmentation Metrics

使用 torchmetrics 库实现，支持 DDP 自动同步
所有指标继承自 base.ConfusionMatrixBase，避免代码重复
"""

import torch
from torchmetrics import Metric
from typing import Optional, List, Dict

from .base import ConfusionMatrixBase, create_class_names, convert_preds_to_labels
from ..mapping import create_reverse_mapping


class OverallAccuracy(Metric):
    """
    总体准确率 (Overall Accuracy)
    
    OA = (正确分类的点数) / (总点数)
    """
    
    def __init__(
        self,
        ignore_index: int = -1,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        self.ignore_index = ignore_index
        
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if target.dim() == 2:
            target = target.reshape(-1)
        
        pred_labels = convert_preds_to_labels(preds)
        
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            pred_labels = pred_labels[valid_mask]
            target = target[valid_mask]
        
        correct = (pred_labels == target).sum()
        total = target.numel()
        
        self.correct += correct
        self.total += total
    
    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total


class MeanIoU(ConfusionMatrixBase):
    """
    平均交并比 (Mean Intersection over Union)
    
    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    mIoU = mean(IoU_c for c in classes)
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -1,
        class_names: Optional[List[str]] = None,
        reverse_class_mapping: Optional[Dict[int, int]] = None,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        # 使用 create_class_names 生成名称
        class_names = create_class_names(num_classes, class_names, reverse_class_mapping)
        super().__init__(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_names=class_names,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs
        )
        self.reverse_class_mapping = reverse_class_mapping
    
    def compute(self) -> torch.Tensor:
        """从混淆矩阵计算 mIoU"""
        tp, fp, fn = self._compute_tp_fp_fn()
        
        iou = tp / (tp + fp + fn + 1e-10)
        
        # 只对出现过的类别计算 mIoU
        valid_classes = (tp + fn) > 0
        if valid_classes.sum() == 0:
            return torch.tensor(0.0, device=self.confusion_matrix.device)
        
        return iou[valid_classes].mean()
    
    def compute_per_class_iou(self) -> Dict[str, float]:
        """计算每个类别的 IoU"""
        tp, fp, fn = self._compute_tp_fp_fn()
        iou = tp / (tp + fp + fn + 1e-10)
        
        result = {}
        for i, name in enumerate(self.class_names):
            result[f"IoU/{name}"] = iou[i].item()
        
        return result


class PerClassIoU(ConfusionMatrixBase):
    """
    每个类别的 IoU (Per-Class IoU)
    
    返回字典形式的详细指标，自动从 trainer.datamodule 获取类别名称
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        reverse_class_mapping: Optional[Dict[int, int]] = None,
        ignore_index: int = -1,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_names=class_names,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs
        )
        
        self._user_class_names = class_names
        self._reverse_class_mapping = reverse_class_mapping
        self._cached_class_names = None
    
    def _get_class_names(self) -> List[str]:
        """
        获取类别名称，按优先级：
        1. 用户手动指定的 class_names
        2. 从 trainer.datamodule 获取
        3. 从 reverse_class_mapping 生成
        4. 默认生成
        """
        if self._user_class_names is not None:
            return self._user_class_names
        
        if self._cached_class_names is None:
            try:
                if hasattr(self, '_device'):
                    parent = self
                    trainer = None
                    
                    for _ in range(5):
                        if hasattr(parent, 'trainer'):
                            trainer = parent.trainer
                            break
                        if hasattr(parent, '_trainer'):
                            trainer = parent._trainer
                            break
                        if not hasattr(parent, '_parent'):
                            break
                        parent = parent._parent
                    
                    if trainer is not None and hasattr(trainer, 'datamodule'):
                        dm = trainer.datamodule
                        if dm is not None:
                            if hasattr(dm, 'class_names') and dm.class_names is not None:
                                self._cached_class_names = dm.class_names
                            elif hasattr(dm, 'class_mapping') and dm.class_mapping is not None:
                                # 使用 create_reverse_mapping 处理 Dict 或 List
                                reverse_mapping = create_reverse_mapping(dm.class_mapping)
                                self._cached_class_names = create_class_names(
                                    self.num_classes,
                                    reverse_class_mapping=reverse_mapping
                                )
            except:
                pass
            
            if self._cached_class_names is None:
                self._cached_class_names = create_class_names(
                    self.num_classes,
                    reverse_class_mapping=self._reverse_class_mapping
                )
        
        return self._cached_class_names
    
    def compute(self) -> dict:
        """计算每个类别的详细指标"""
        tp, fp, fn = self._compute_tp_fp_fn()
        
        iou = tp / (tp + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        class_names = self._get_class_names()
        
        return {
            'iou_per_class': iou,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'class_names': class_names
        }


class Precision(ConfusionMatrixBase):
    """精确率 (Precision)"""
    
    def compute(self) -> torch.Tensor:
        tp, fp, fn = self._compute_tp_fp_fn()
        precision = tp / (tp + fp + 1e-10)
        
        valid_classes = (tp + fp) > 0
        if valid_classes.sum() == 0:
            return torch.tensor(0.0, device=self.confusion_matrix.device)
        
        return precision[valid_classes].mean()


class Recall(ConfusionMatrixBase):
    """召回率 (Recall / Sensitivity)"""
    
    def compute(self) -> torch.Tensor:
        tp, fp, fn = self._compute_tp_fp_fn()
        recall = tp / (tp + fn + 1e-10)
        
        valid_classes = (tp + fn) > 0
        if valid_classes.sum() == 0:
            return torch.tensor(0.0, device=self.confusion_matrix.device)
        
        return recall[valid_classes].mean()


class F1Score(ConfusionMatrixBase):
    """F1 分数 (F1-Score)"""
    
    def compute(self) -> torch.Tensor:
        tp, fp, fn = self._compute_tp_fp_fn()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        valid_classes = (tp + fn) > 0
        if valid_classes.sum() == 0:
            return torch.tensor(0.0, device=self.confusion_matrix.device)
        
        return f1[valid_classes].mean()


class SegmentationMetrics(ConfusionMatrixBase):
    """
    统一的语义分割指标类
    
    一次性计算所有指标：OA, mIoU, Precision, Recall, F1
    避免多次重复计算混淆矩阵
    """
    
    def compute(self) -> dict:
        """计算所有指标"""
        cm = self.confusion_matrix
        tp, fp, fn = self._compute_tp_fp_fn()
        
        # IoU, Precision, Recall, F1
        iou = tp / (tp + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Overall Accuracy
        oa = tp.sum() / cm.sum()
        
        # 计算均值（只对出现过的类别）
        valid_classes = (tp + fn) > 0
        
        if valid_classes.sum() == 0:
            mean_iou = torch.tensor(0.0, device=cm.device)
            mean_precision = torch.tensor(0.0, device=cm.device)
            mean_recall = torch.tensor(0.0, device=cm.device)
            mean_f1 = torch.tensor(0.0, device=cm.device)
        else:
            mean_iou = iou[valid_classes].mean()
            mean_precision = precision[valid_classes].mean()
            mean_recall = recall[valid_classes].mean()
            mean_f1 = f1[valid_classes].mean()
        
        return {
            'overall_accuracy': oa,
            'mean_iou': mean_iou,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'iou_per_class': iou,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'class_names': self.class_names
        }
