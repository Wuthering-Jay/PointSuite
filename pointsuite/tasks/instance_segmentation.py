import torch
import torch.nn as nn
from typing import Dict, Any

from .base_task import BaseTask
# 假设您在 utils 中有一个聚类算法 (例如 MeanShift)
# from ..utils.clustering import mean_shift_cluster

class InstanceSegmentationTask(BaseTask):
    """
    实例分割任务 (LightningModule)。

    展示了框架的扩展性：
    1. 它接收 *两个* 头部 (semantic_head, instance_head)。
    2. 它覆盖了 `_calculate_total_loss` 来处理复合损失。
    3. 它的 `predict_step` 返回 embeddings，由 Callback 负责聚类。
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 semantic_head: nn.Module,
                 instance_head: nn.Module,
                 **kwargs):
        """
        Args:
            backbone (nn.Module): 骨干网络。
            semantic_head (nn.Module): 语义预测头。
            instance_head (nn.Module): 实例嵌入头。
            **kwargs: 传递给 BaseTask 的参数 (learning_rate, loss_configs, etc.)
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.semantic_head = semantic_head
        self.instance_head = instance_head

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        前向传播，返回一个包含 *多头* 输出的字典。
        """
        features = self.backbone(batch)
        
        sem_logits = self.semantic_head(features)
        inst_embeds = self.instance_head(features)
        
        return {
            "sem_logits": sem_logits,   # [N, num_classes]
            "inst_embeds": inst_embeds  # [N, embedding_dim]
        }

    def _calculate_total_loss(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        (覆盖 BaseTask)
        计算实例分割的复合损失。
        """
        # 假设您的 YAML 损失配置中定义了 "semantic_loss" 和 "instance_loss"
        
        # 1. 语义损失
        # 我们的数据集使用 'class' 作为语义标签
        sem_loss = self.losses["semantic_loss"](
            preds["sem_logits"], 
            batch.get("class", batch.get("labels"))  # 兼容不同的标签键名
        )
        
        # 2. 实例损失 (例如 DiscriminativeLoss)
        #    它可能需要 embeddings, 实例ID, 和 语义标签
        inst_loss = self.losses["instance_loss"](
            preds["inst_embeds"], 
            batch.get("instance_id", batch.get("instance_labels")),  # 实例ID
            batch.get("class", batch.get("labels"))  # 语义标签
        )
        
        # 3. 加权总和
        total_loss = (self.loss_weights["semantic_loss"] * sem_loss + 
                      self.loss_weights["instance_loss"] * inst_loss)
        
        return {
            "total_loss": total_loss,
            "loss_sem": sem_loss,
            "loss_inst": inst_loss
        }
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        preds_dict = self.forward(batch)
        loss_dict = self._calculate_total_loss(preds_dict, batch)
        batch_size = self._get_batch_size(batch)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss_dict["total_loss"]

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        预测步骤返回 logits 和 embeddings。
        聚类 (Clustering) 的重活交给 PredictionWriter 回调处理。
        """
        preds_dict = self.forward(batch)
        
        results = {
            "sem_logits": preds_dict["sem_logits"].cpu(),
            "inst_embeds": preds_dict["inst_embeds"].cpu(),
        }
        
        # 保存额外信息用于后处理
        if "indices" in batch:
            results["indices"] = batch["indices"].cpu()
        if "coord" in batch:
            results["coord"] = batch["coord"].cpu()
            
        return results