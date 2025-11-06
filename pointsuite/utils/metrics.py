"""
Metrics for Point Cloud Semantic Segmentation

使用 torchmetrics 库实现，支持 DDP 自动同步
"""

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from typing import Optional, List


class OverallAccuracy(Metric):
    """
    总体准确率 (Overall Accuracy)
    
    OA = (正确分类的点数) / (总点数)
    
    使用 torchmetrics.Metric 基类，自动支持 DDP 同步
    """
    
    def __init__(
        self,
        ignore_index: int = -1,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        """
        Args:
            ignore_index: 忽略的标签索引（如背景类）
            dist_sync_on_step: 是否在每步同步（通常在 epoch 结束同步即可）
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        
        self.ignore_index = ignore_index
        
        # 添加状态变量（分布式训练时自动聚合）
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        更新指标状态
        
        Args:
            preds: 预测 logits [N, C] 或 [B, N, C]
            target: 真实标签 [N] 或 [B, N]
        """
        # 处理维度
        if preds.dim() == 3:
            B, N, C = preds.shape
            preds = preds.reshape(B * N, C)
            if target.dim() == 2:
                target = target.reshape(B * N)
        
        # 获取预测类别
        pred_labels = torch.argmax(preds, dim=1)
        
        # 过滤 ignore_index
        if self.ignore_index >= 0:
            valid_mask = target != self.ignore_index
            pred_labels = pred_labels[valid_mask]
            target = target[valid_mask]
        
        # 统计正确数量
        correct = (pred_labels == target).sum()
        total = target.numel()
        
        self.correct += correct
        self.total += total
    
    def compute(self) -> torch.Tensor:
        """计算最终指标"""
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total


class MeanIoU(Metric):
    """
    平均交并比 (Mean Intersection over Union)
    
    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    mIoU = mean(IoU_c for c in classes)
    
    使用混淆矩阵计算，支持 DDP
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -1,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        """
        Args:
            num_classes: 类别数量
            ignore_index: 忽略的标签索引
            dist_sync_on_step: 是否在每步同步
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # 混淆矩阵 [C, C]
        self.add_state(
            "confusion_matrix",
            default=torch.zeros(num_classes, num_classes),
            dist_reduce_fx="sum"
        )
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """更新混淆矩阵"""
        # 处理维度
        if preds.dim() == 3:
            B, N, C = preds.shape
            preds = preds.reshape(B * N, C)
            if target.dim() == 2:
                target = target.reshape(B * N)
        
        # 获取预测类别
        pred_labels = torch.argmax(preds, dim=1)
        
        # 过滤 ignore_index
        if self.ignore_index >= 0:
            valid_mask = target != self.ignore_index
            pred_labels = pred_labels[valid_mask]
            target = target[valid_mask]
        
        # 计算混淆矩阵
        # confusion[i, j] = 真实类别 i 被预测为 j 的数量
        indices = self.num_classes * target + pred_labels
        cm = torch.bincount(indices, minlength=self.num_classes ** 2)
        cm = cm.reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += cm
    
    def compute(self) -> torch.Tensor:
        """从混淆矩阵计算 mIoU"""
        cm = self.confusion_matrix
        
        # IoU = TP / (TP + FP + FN)
        # TP: 对角线元素
        # FP: 列和 - TP
        # FN: 行和 - TP
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        
        iou = tp / (tp + fp + fn + 1e-10)
        
        # 只对出现过的类别计算 mIoU
        valid_classes = (tp + fn) > 0
        if valid_classes.sum() == 0:
            return torch.tensor(0.0, device=cm.device)
        
        return iou[valid_classes].mean()
    
    def get_per_class_iou(self) -> torch.Tensor:
        """获取每个类别的 IoU"""
        cm = self.confusion_matrix
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        
        iou = tp / (tp + fp + fn + 1e-10)
        return iou


class PerClassIoU(Metric):
    """
    每个类别的 IoU (Per-Class IoU)
    
    返回字典形式的详细指标
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -1,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        """
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
            ignore_index: 忽略的标签索引
            dist_sync_on_step: 是否在每步同步
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        
        # 混淆矩阵
        self.add_state(
            "confusion_matrix",
            default=torch.zeros(num_classes, num_classes),
            dist_reduce_fx="sum"
        )
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """更新混淆矩阵"""
        # 处理维度
        if preds.dim() == 3:
            B, N, C = preds.shape
            preds = preds.reshape(B * N, C)
            if target.dim() == 2:
                target = target.reshape(B * N)
        
        # 获取预测类别
        pred_labels = torch.argmax(preds, dim=1)
        
        # 过滤 ignore_index
        if self.ignore_index >= 0:
            valid_mask = target != self.ignore_index
            pred_labels = pred_labels[valid_mask]
            target = target[valid_mask]
        
        # 更新混淆矩阵
        indices = self.num_classes * target + pred_labels
        cm = torch.bincount(indices, minlength=self.num_classes ** 2)
        cm = cm.reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += cm
    
    def compute(self) -> dict:
        """
        计算每个类别的详细指标
        
        Returns:
            dict: {
                'iou_per_class': [C],
                'precision_per_class': [C],
                'recall_per_class': [C],
                'class_names': List[str]
            }
        """
        cm = self.confusion_matrix
        
        # 计算 TP, FP, FN
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        
        # IoU, Precision, Recall
        iou = tp / (tp + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        return {
            'iou_per_class': iou,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'class_names': self.class_names
        }


class _ConfusionMatrixBase(Metric):
    """
    混淆矩阵基类
    
    所有基于混淆矩阵的指标的统一基类，避免重复代码
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
        """更新混淆矩阵"""
        # 处理维度
        if preds.dim() == 3:
            B, N, C = preds.shape
            preds = preds.reshape(B * N, C)
            if target.dim() == 2:
                target = target.reshape(B * N)
        
        # 获取预测类别
        pred_labels = torch.argmax(preds, dim=1)
        
        # 过滤 ignore_index
        if self.ignore_index >= 0:
            valid_mask = target != self.ignore_index
            pred_labels = pred_labels[valid_mask]
            target = target[valid_mask]
        
        # 计算混淆矩阵
        # confusion[i, j] = 真实类别 i 被预测为 j 的数量
        indices = self.num_classes * target + pred_labels
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


class Precision(_ConfusionMatrixBase):
    """
    精确率 (Precision)
    
    Precision_c = TP_c / (TP_c + FP_c)
    Macro Precision = mean(Precision_c for c in classes)
    
    支持返回每个类别的 Precision 和均值
    """
    
    def compute(self) -> torch.Tensor:
        """计算 Macro Precision (均值)"""
        tp, fp, fn = self._compute_tp_fp_fn()
        
        precision = tp / (tp + fp + 1e-10)
        
        # 只对有预测的类别计算平均
        valid_classes = (tp + fp) > 0
        if valid_classes.sum() == 0:
            return torch.tensor(0.0, device=self.confusion_matrix.device)
        
        return precision[valid_classes].mean()
    
    def get_per_class_precision(self) -> torch.Tensor:
        """获取每个类别的 Precision"""
        tp, fp, fn = self._compute_tp_fp_fn()
        precision = tp / (tp + fp + 1e-10)
        return precision
    
    def get_detailed_results(self) -> dict:
        """
        获取详细结果
        
        Returns:
            dict: {
                'precision_per_class': [C],
                'mean_precision': scalar,
                'class_names': List[str]
            }
        """
        precision_per_class = self.get_per_class_precision()
        
        tp, fp, fn = self._compute_tp_fp_fn()
        valid_classes = (tp + fp) > 0
        
        if valid_classes.sum() == 0:
            mean_precision = torch.tensor(0.0, device=self.confusion_matrix.device)
        else:
            mean_precision = precision_per_class[valid_classes].mean()
        
        return {
            'precision_per_class': precision_per_class,
            'mean_precision': mean_precision,
            'class_names': self.class_names
        }


class Recall(_ConfusionMatrixBase):
    """
    召回率 (Recall / Sensitivity)
    
    Recall_c = TP_c / (TP_c + FN_c)
    Macro Recall = mean(Recall_c for c in classes)
    
    支持返回每个类别的 Recall 和均值
    """
    
    def compute(self) -> torch.Tensor:
        """计算 Macro Recall (均值)"""
        tp, fp, fn = self._compute_tp_fp_fn()
        
        recall = tp / (tp + fn + 1e-10)
        
        # 只对出现过的类别计算平均
        valid_classes = (tp + fn) > 0
        if valid_classes.sum() == 0:
            return torch.tensor(0.0, device=self.confusion_matrix.device)
        
        return recall[valid_classes].mean()
    
    def get_per_class_recall(self) -> torch.Tensor:
        """获取每个类别的 Recall"""
        tp, fp, fn = self._compute_tp_fp_fn()
        recall = tp / (tp + fn + 1e-10)
        return recall
    
    def get_detailed_results(self) -> dict:
        """
        获取详细结果
        
        Returns:
            dict: {
                'recall_per_class': [C],
                'mean_recall': scalar,
                'class_names': List[str]
            }
        """
        recall_per_class = self.get_per_class_recall()
        
        tp, fp, fn = self._compute_tp_fp_fn()
        valid_classes = (tp + fn) > 0
        
        if valid_classes.sum() == 0:
            mean_recall = torch.tensor(0.0, device=self.confusion_matrix.device)
        else:
            mean_recall = recall_per_class[valid_classes].mean()
        
        return {
            'recall_per_class': recall_per_class,
            'mean_recall': mean_recall,
            'class_names': self.class_names
        }


class F1Score(_ConfusionMatrixBase):
    """
    F1 分数 (F1-Score)
    
    F1_c = 2 * (Precision_c * Recall_c) / (Precision_c + Recall_c)
    Macro F1 = mean(F1_c for c in classes)
    
    支持返回每个类别的 F1 和均值
    """
    
    def compute(self) -> torch.Tensor:
        """计算 Macro F1 (均值)"""
        tp, fp, fn = self._compute_tp_fp_fn()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # 只对出现过的类别计算平均
        valid_classes = (tp + fn) > 0
        if valid_classes.sum() == 0:
            return torch.tensor(0.0, device=self.confusion_matrix.device)
        
        return f1[valid_classes].mean()
    
    def get_per_class_f1(self) -> torch.Tensor:
        """获取每个类别的 F1 Score"""
        tp, fp, fn = self._compute_tp_fp_fn()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1
    
    def get_detailed_results(self) -> dict:
        """
        获取详细结果
        
        Returns:
            dict: {
                'f1_per_class': [C],
                'precision_per_class': [C],
                'recall_per_class': [C],
                'mean_f1': scalar,
                'mean_precision': scalar,
                'mean_recall': scalar,
                'class_names': List[str]
            }
        """
        tp, fp, fn = self._compute_tp_fp_fn()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # 计算均值（只对出现过的类别）
        valid_classes = (tp + fn) > 0
        
        if valid_classes.sum() == 0:
            mean_f1 = torch.tensor(0.0, device=self.confusion_matrix.device)
            mean_precision = torch.tensor(0.0, device=self.confusion_matrix.device)
            mean_recall = torch.tensor(0.0, device=self.confusion_matrix.device)
        else:
            mean_f1 = f1[valid_classes].mean()
            mean_precision = precision[valid_classes].mean()
            mean_recall = recall[valid_classes].mean()
        
        return {
            'f1_per_class': f1,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'mean_f1': mean_f1,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'class_names': self.class_names
        }


class SegmentationMetrics(_ConfusionMatrixBase):
    """
    统一的语义分割指标类
    
    一次性计算所有指标：OA, mIoU, Precision, Recall, F1
    避免多次重复计算混淆矩阵
    
    推荐在训练中使用此类，一次性获取所有指标
    """
    
    def compute(self) -> dict:
        """
        计算所有指标
        
        Returns:
            dict: {
                'overall_accuracy': scalar,
                'mean_iou': scalar,
                'mean_precision': scalar,
                'mean_recall': scalar,
                'mean_f1': scalar,
                'iou_per_class': [C],
                'precision_per_class': [C],
                'recall_per_class': [C],
                'f1_per_class': [C],
                'class_names': List[str]
            }
        """
        cm = self.confusion_matrix
        tp, fp, fn = self._compute_tp_fp_fn()
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-10)
        
        # Precision, Recall, F1
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
