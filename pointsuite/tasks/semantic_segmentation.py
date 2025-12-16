"""
语义分割任务模块

该模块实现了点云语义分割的完整任务逻辑，继承自 BaseTask。

功能特性：
- 支持任意 backbone + head 组合
- 自动追踪最佳 mIoU
- 详细的每类指标打印
- 预测结果导出支持

配置示例
--------
.. code-block:: yaml

    model:
        backbone:
            class_path: pointsuite.models.backbones.ptv2.PointTransformerV2
            init_args:
                in_channels: 6
        head:
            class_path: pointsuite.models.heads.SegmentationHead
            init_args:
                in_channels: 512
                num_classes: 8
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional

from .base_task import BaseTask
from ..utils.logger import (
    Colors,
    log_info,
    log_warning,
    print_header,
    print_section,
)
from ..utils.config import import_class


def _display_width(s: str) -> int:
    """
    计算字符串的显示宽度
    
    中文字符占 2 个宽度，其他字符占 1 个宽度。
    
    参数
    ----
    s : str
        输入字符串
        
    返回
    ----
    int
        显示宽度
    """
    width = 0
    for c in s:
        if '\u4e00' <= c <= '\u9fff':
            width += 2
        else:
            width += 1
    return width


def _pad_to_width(s: str, target_width: int) -> str:
    """
    将字符串填充到指定显示宽度
    
    参数
    ----
    s : str
        输入字符串
    target_width : int
        目标宽度
        
    返回
    ----
    str
        填充后的字符串
    """
    current_width = _display_width(s)
    padding = target_width - current_width
    return s + ' ' * max(0, padding)


class SemanticSegmentationTask(BaseTask):
    """
    语义分割任务
    
    继承自 BaseTask，添加了语义分割特定的组件：
    - backbone + head 网络结构
    - 最佳 mIoU 追踪
    - 详细的每类指标打印
    
    Attributes
    ----------
    backbone : nn.Module
        特征提取骨干网络
    head : nn.Module
        分割头
    best_miou : float
        最佳 mIoU 值
    best_miou_epoch : int
        最佳 mIoU 对应的轮次
    """
    
    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        初始化语义分割任务
        
        参数
        ----
        backbone : nn.Module, optional
            已实例化的骨干网络
        head : nn.Module, optional
            已实例化的分割头
        model_config : dict, optional
            模型配置字典，用于从配置实例化 backbone 和 head。
            如果提供，则忽略 backbone 和 head 参数。
            
            格式::
            
                {
                    'backbone': {'class_path': '...', 'init_args': {...}},
                    'head': {'class_path': '...', 'init_args': {...}}
                }
                
        **kwargs
            传递给 BaseTask 的参数
        """
        super().__init__(**kwargs)
        
        # 追踪最佳 mIoU
        self.best_miou = 0.0
        self.best_miou_epoch = -1
        
        # 🔥 关键修改：保存 hyperparameters
        # 如果使用 model_config，我们忽略 backbone 和 head 对象，避免重复保存和警告
        # 如果使用 backbone/head 对象，我们必须保存它们以支持自动重建（尽管会有警告）
        if model_config is not None:
            self.save_hyperparameters(ignore=['backbone', 'head'])
            
            # 从配置实例化
            backbone_cfg = model_config.get('backbone')
            head_cfg = model_config.get('head')
            
            if backbone_cfg:
                backbone_cls = import_class(backbone_cfg['class_path'])
                self.backbone = backbone_cls(**backbone_cfg.get('init_args', {}))
            
            if head_cfg:
                head_cls = import_class(head_cfg['class_path'])
                self.head = head_cls(**head_cfg.get('init_args', {}))
                
        else:
            # 兼容旧方式：直接传入对象
            # 这种情况下我们不 ignore backbone/head，以便 load_from_checkpoint 能工作
            # 用户会看到 PL 的警告，但这是预期的
            self.save_hyperparameters()
            self.backbone = backbone
            self.head = head
            
        # 验证模型是否正确初始化
        if not hasattr(self, 'backbone') or self.backbone is None:
            raise ValueError("Backbone 未初始化！请提供 backbone 对象或 model_config")
        if not hasattr(self, 'head') or self.head is None:
            raise ValueError("Head 未初始化！请提供 head 对象或 model_config")

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        定义模型的单次前向传播。
        
        Args:
            batch (Dict): 来自 DataLoader 的批次数据 (由 collate_fn 产生)。
                          我们的 collate_fn 提供:
                          - 'coord': [N, 3] 点坐标
                          - 'feat': [N, C] 点特征
                          - 'class': [N] 点标签
                          - 'offset': [B] 累积偏移量
        
        Returns:
            torch.Tensor: 模型的原始 Logits 输出 (shape: [N_total_points, num_classes])。
        """
        # 1. Backbone 提取特征
        # 不同 backbone 可能有不同的输入格式：
        # - 简单模型：直接接收 batch['feat']
        # - PointTransformerV2/PointNet++：需要整个 batch 字典
        
        # 检查 backbone 是否需要整个 batch 字典
        # 方法1: 检查参数名是否为 'batch' 或 'data_dict'
        # 方法2: 检查是否为 PointTransformerV2 等已知需要 dict 的模型
        forward_params = self.backbone.forward.__code__.co_varnames if hasattr(self.backbone, 'forward') else []
        needs_dict = ('batch' in forward_params or 
                     'data_dict' in forward_params or
                     'PointTransformerV2' in self.backbone.__class__.__name__)
        
        if needs_dict:
            # Backbone 接收整个 batch 字典
            backbone_output = self.backbone(batch)
        else:
            # Backbone 只接收特征张量（如简单 MLP）
            backbone_output = self.backbone(batch.get('feat', batch.get('coord')))
        
        # 2. 处理 backbone 输出
        # 如果输出是字典（如 PointNet++ 返回 {'feat': ..., 'sa_xyz': ...}）
        if isinstance(backbone_output, dict):
            features = backbone_output['feat']  # 提取特征
        else:
            features = backbone_output
        
        # 3. Head 生成 logits
        logits = self.head(features)
        
        # 返回字典以支持辅助损失 (Auxiliary Loss)
        # 'logits': 标准输出
        # 'features': Head 之前的特征 (Backbone 输出)
        return {
            'logits': logits,
            'features': features
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        执行单个训练步骤。
        """
        # 1. 前向传播
        preds_logits = self.forward(batch)
        
        # 2. 计算损失 (使用 BaseTask 的辅助函数)
        #    BaseTask._calculate_total_loss 默认会调用 loss(preds, batch)
        #    您的损失函数 (例如 CrossEntropyLoss) 需要知道如何从 'preds' (logits)
        #    和 'batch' (包含 'class') 中提取所需信息。
        loss_dict = self._calculate_total_loss(preds_logits, batch)
        
        # 3. 记录训练损失 (PL 会自动添加 'train/' 前缀)
        #    prog_bar=True 会在进度条上显示 'total_loss'
        batch_size = self._get_batch_size(batch)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # 4. 返回总损失
        return loss_dict["total_loss"]

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        执行单个预测步骤（用于生产环境、无真值标签）
        
        与 test_step 的区别：
        - predict_step: 无真值标签，不计算损失和指标，只返回预测结果
        - test_step: 有真值标签，计算损失和指标，可选保存预测结果
        
        使用场景：
        - 新场景预测（无标签）
        - 生产环境部署
        - 需要保存 .las 文件时使用 Trainer.predict() + SemanticPredictLasWriter
        """
        # 1. 前向传播
        preds = self.forward(batch)
        
        # 2. 后处理预测 (支持 Mask3D 等复杂输出)
        #    子类可以覆盖 postprocess_predictions 来自定义行为
        processed_preds = self.postprocess_predictions(preds)
        
        # 3. 返回一个字典，PredictionWriter 回调将处理这个字典
        #    我们返回 CPU 张量以释放 GPU 内存
        results = {
            "logits": processed_preds.cpu(),  # 可以是 logits [N, C] 或 labels [N]
        }
        
        # (可选) 如果需要原始索引 (用于拼接/投票)
        # 我们的数据集可能提供 'indices' 字段
        if "indices" in batch:
            results["indices"] = batch["indices"].cpu()
        
        # 🔥 传递文件信息到 callback（用于预测结果的文件级聚合）
        # 这些信息由 dataset 在 test/predict split 时提供
        if "bin_file" in batch:
            results["bin_file"] = batch["bin_file"]  # 文件标识符（可能是字符串列表）
        if "bin_path" in batch:
            results["bin_path"] = batch["bin_path"]  # 原始数据文件路径
        if "pkl_path" in batch:
            results["pkl_path"] = batch["pkl_path"]  # 元数据文件路径
        
        # 保存坐标信息（用于可视化）
        if "coord" in batch:
            results["coord"] = batch["coord"].cpu()
            
        return results
    
    def _print_validation_metrics(self, print_metrics: Dict[str, Any]) -> None:
        """
        打印语义分割的详细验证指标
        
        参数
        ----
        print_metrics : dict
            计算出的指标字典
        """
        miou_key = 'mean_iou'
        if miou_key not in print_metrics:
            super()._print_validation_metrics(print_metrics)
            return
        
        try:
            current_miou = float(print_metrics[miou_key])
            
            # 更新最佳 mIoU
            if current_miou > self.best_miou:
                self.best_miou = current_miou
                self.best_miou_epoch = self.current_epoch
            
            overall_acc = print_metrics.get('overall_accuracy', None)
            
            # 获取每类指标
            per_class_iou = print_metrics.get('iou_per_class', print_metrics.get('per_class_iou', None))
            per_class_precision = print_metrics.get('precision_per_class', print_metrics.get('per_class_precision', None))
            per_class_recall = print_metrics.get('recall_per_class', print_metrics.get('per_class_recall', None))
            per_class_f1 = print_metrics.get('f1_per_class', print_metrics.get('per_class_f1', None))
            
            # 兼容旧代码：从 metric 对象获取
            if per_class_iou is None and 'mean_iou' in self.val_metrics:
                metric = self.val_metrics['mean_iou']
                if hasattr(metric, 'confusion_matrix'):
                    confmat = metric.confusion_matrix.cpu().numpy()
                    intersection = np.diag(confmat)
                    union = confmat.sum(1) + confmat.sum(0) - np.diag(confmat)
                    per_class_iou = intersection / (union + 1e-10)
                    per_class_precision = intersection / (confmat.sum(0) + 1e-10)
                    per_class_recall = intersection / (confmat.sum(1) + 1e-10)
                    per_class_f1 = 2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall + 1e-10)

            # 打印标题
            display_epoch = self.current_epoch + 1
            print()
            print("=" * 100)
            print(f"Validation Epoch {display_epoch} - Metrics")
            print("=" * 100)
            
            if overall_acc is not None:
                print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
            print(f"Mean IoU (current): {current_miou:.4f}")
            print(f"Mean IoU (best)   : {self.best_miou:.4f} (Epoch {self.best_miou_epoch + 1})")
            if current_miou > self.best_miou - 1e-6:
                print("* New best mIoU achieved!")
            print("=" * 100)
            
            # 打印每类指标
            if per_class_iou is not None:
                self._print_per_class_metrics(
                    per_class_iou, per_class_precision, per_class_recall, per_class_f1,
                    print_metrics, current_miou
                )
            print("=" * 100)
            print()
            
        except Exception as e:
            self._task_logger.warning(f"无法打印详细指标: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_test_metrics(self, print_metrics: Dict[str, Any]) -> None:
        """
        打印语义分割的详细测试指标
        
        参数
        ----
        print_metrics : dict
            计算出的指标字典
        """
        miou_key = 'mean_iou'
        if miou_key not in print_metrics:
            super()._print_test_metrics(print_metrics)
            return
        
        try:
            current_miou = float(print_metrics[miou_key])
            overall_acc = print_metrics.get('overall_accuracy', None)
            
            # 获取每类指标
            per_class_iou = print_metrics.get('iou_per_class', print_metrics.get('per_class_iou', None))
            per_class_precision = print_metrics.get('precision_per_class', print_metrics.get('per_class_precision', None))
            per_class_recall = print_metrics.get('recall_per_class', print_metrics.get('per_class_recall', None))
            per_class_f1 = print_metrics.get('f1_per_class', print_metrics.get('per_class_f1', None))
            
            # 兼容旧代码
            if per_class_iou is None and 'mean_iou' in self.test_metrics:
                metric = self.test_metrics['mean_iou']
                if hasattr(metric, 'confusion_matrix'):
                    confmat = metric.confusion_matrix.cpu().numpy()
                    intersection = np.diag(confmat)
                    union = confmat.sum(1) + confmat.sum(0) - np.diag(confmat)
                    per_class_iou = intersection / (union + 1e-10)
                    per_class_precision = intersection / (confmat.sum(0) + 1e-10)
                    per_class_recall = intersection / (confmat.sum(1) + 1e-10)
                    per_class_f1 = 2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall + 1e-10)

            print()
            print("=" * 100)
            print("Test Results - Metrics")
            print("=" * 100)
            if overall_acc is not None:
                print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
            print(f"Mean IoU: {current_miou:.4f}")
            print("=" * 100)
            
            if per_class_iou is not None:
                self._print_per_class_metrics(
                    per_class_iou, per_class_precision, per_class_recall, per_class_f1,
                    print_metrics, current_miou
                )
            print("=" * 100)
            print()
            
        except Exception as e:
            self._task_logger.warning(f"无法打印详细测试指标: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_per_class_metrics(
        self, 
        per_class_iou, 
        per_class_precision, 
        per_class_recall, 
        per_class_f1,
        print_metrics: Dict[str, Any],
        mean_iou: float
    ):
        """
        打印每个类别的详细指标表格
        
        Args:
            per_class_iou: 每类 IoU 数组
            per_class_precision: 每类 Precision 数组
            per_class_recall: 每类 Recall 数组
            per_class_f1: 每类 F1 数组
            print_metrics: 指标字典（用于获取类别名）
            mean_iou: 平均 IoU
        """
        # 获取类别名
        class_names = print_metrics.get('class_names', None)
        if class_names is None:
            class_names = self.hparams.get('class_names', None) if hasattr(self, 'hparams') else None
        
        # 确保是 numpy 数组
        if isinstance(per_class_iou, torch.Tensor): 
            per_class_iou = per_class_iou.cpu().numpy()
        if isinstance(per_class_precision, torch.Tensor): 
            per_class_precision = per_class_precision.cpu().numpy()
        if isinstance(per_class_recall, torch.Tensor): 
            per_class_recall = per_class_recall.cpu().numpy()
        if isinstance(per_class_f1, torch.Tensor): 
            per_class_f1 = per_class_f1.cpu().numpy()
        
        num_classes = len(per_class_iou)
        
        # 计算最大类别名宽度
        max_name_width = 8  # 最小宽度
        for i in range(num_classes):
            c_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            max_name_width = max(max_name_width, _display_width(c_name))
        max_name_width = min(max_name_width, 20)  # 最大宽度限制
        
        # 表头
        header_class = _pad_to_width("Class", max_name_width)
        print(f"  {header_class}  {'IoU':>8}  {'Precision':>10}  {'Recall':>8}  {'F1-Score':>10}")
        print(f"  {'-'*max_name_width}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
        
        for i in range(num_classes):
            c_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            c_name_padded = _pad_to_width(c_name, max_name_width)
            print(f"  {c_name_padded}  {per_class_iou[i]:8.4f}  {per_class_precision[i]:10.4f}  "
                  f"{per_class_recall[i]:8.4f}  {per_class_f1[i]:10.4f}")
        
        # 计算平均指标
        mean_precision = np.nanmean(per_class_precision)
        mean_recall = np.nanmean(per_class_recall)
        mean_f1 = np.nanmean(per_class_f1)
        
        print(f"  {'-'*max_name_width}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
        mean_label = _pad_to_width("Mean", max_name_width)
        print(f"  {mean_label}  {mean_iou:8.4f}  {mean_precision:10.4f}  "
              f"{mean_recall:8.4f}  {mean_f1:10.4f}")