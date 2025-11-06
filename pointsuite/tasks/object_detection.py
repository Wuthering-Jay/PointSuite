import torch
import torch.nn as nn
from typing import Dict, Any

from .base_task import BaseTask

class ObjectDetectionTask(BaseTask):
    """
    3D 目标检测任务 (LightningModule)。

    展示了框架的扩展性：
    1. 接收一个 backbone 和一个 detection_head。
    2. 期望 DataModule 提供 'gt_boxes' 和 'gt_labels'。
    3. 覆盖 `_calculate_total_loss` 来处理复杂的匹配和回归损失。
    4. 它的 `predict_step` 返回最终的 3D 边界框。
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 detection_head: nn.Module,
                 **kwargs):
        """
        Args:
            backbone (nn.Module): 骨干网络。
            detection_head (nn.Module): 目标检测头 (例如 VoteNet, DETR head)。
            **kwargs: 传递给 BaseTask 的参数 (learning_rate, loss_configs, etc.)
        """
        super().__init__(**kwargs)
        
        self.backbone = backbone
        self.detection_head = detection_head
        
        # 假设 'detection_loss' 是在 YAML 的 loss_configs 中定义的
        # 并且它是一个包含了 "matcher" 和 "set_criterion" 的复杂损失模块
        self.detection_loss_fn = self.losses['detection_loss']

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        前向传播，返回一个包含预测框和类别 logits 的字典。
        """
        # 1. 提取骨干网络特征
        features = self.backbone(batch)
        
        # 2. 将特征传递给检测头
        #    检测头通常返回一个字典
        preds_dict = self.detection_head(features, batch)
        
        return preds_dict # 例如: {'pred_logits': [B, N, C], 'pred_boxes': [B, N, 7]}

    def _calculate_total_loss(self, preds_dict: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        (覆盖 BaseTask)
        计算目标检测的复合损失。
        """
        # 1. 准备真值 (Ground Truth) 
        #    DataModule 应该提供了 'gt_boxes', 'gt_labels' 等
        targets = {
            'gt_boxes': batch['gt_boxes'],
            'gt_labels': batch['gt_labels']
        }
        
        # 2. 调用检测损失函数
        #    这个损失函数内部会执行 "匹配" (Matcher) 和 "损失计算"
        #    (这模仿了 DETR 或 VoteNet 的损失计算方式)
        loss_dict = self.detection_loss_fn(preds_dict, targets)
        
        # 3. 加权 (如果需要)
        #    (假设 self.loss_weights['detection_loss'] = 1.0)
        total_loss = loss_dict['total_loss'] * self.loss_weights['detection_loss']
        
        # 将 'total_loss' 添加回字典以便日志记录
        loss_dict['total_loss'] = total_loss
        return loss_dict
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        执行单个训练步骤。
        """
        # 1. 前向传播
        preds_dict = self.forward(batch)
        
        # 2. 计算损失
        loss_dict = self._calculate_total_loss(preds_dict, batch)
        
        # 3. 记录日志 (BaseTask 会自动处理字典中的 'total_loss' 和其他损失)
        batch_size = self._get_batch_size(batch)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss_dict["total_loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        验证步骤：计算损失并更新 mAP 指标。
        """
        # 1. 前向传播
        preds_dict = self.forward(batch)
        
        # 2. 计算损失
        loss_dict = self._calculate_total_loss(preds_dict, batch)
        
        # 3. 记录损失
        batch_size = self._get_batch_size(batch)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        # 4. (可选) 对预测结果进行后处理 (例如 NMS)，以匹配 mAP 指标的输入
        # post_preds = self.detection_head.post_process(preds_dict)
        
        # 5. 更新指标
        targets = { 'gt_boxes': batch['gt_boxes'], 'gt_labels': batch['gt_labels'] }
        for metric in self.val_metrics.values():
            metric.update(preds_dict, targets) # 假设 mAP 指标能处理原始 logits

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Any]:
        """
        预测步骤：返回经过后处理 (如 NMS) 的最终边界框。
        """
        # 1. 前向传播
        preds_dict = self.forward(batch)
        
        # 2. 调用头部的后处理方法
        final_boxes, final_scores, final_labels = self.detection_head.post_process(preds_dict)
        
        results = {
            "pred_boxes": final_boxes,   # [N_final, 7]
            "pred_scores": final_scores, # [N_final]
            "pred_labels": final_labels, # [N_final]
        }

        # 保存场景信息（用于后处理）
        if "indices" in batch:
            results["indices"] = batch["indices"].cpu()
        if "coord" in batch:
            results["coord"] = batch["coord"].cpu()
            
        return results