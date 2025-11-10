import torch
import torch.nn as nn
from typing import Dict, Any

from .base_task import BaseTask

class SemanticSegmentationTask(BaseTask):
    """
    语义分割任务 (LightningModule)。

    它继承自 BaseTask，并添加了特定于语义分割的组件：
    1. 一个 `head` (分割头)。
    2. 修改了 `forward` 逻辑，以连接 backbone 和 head。
    3. 修改了 `predict_step` 以输出最终的 argmax 预测。
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 head: nn.Module,
                 **kwargs): # 接收来自 BaseTask 的所有参数 (learning_rate, loss_configs, etc.)
        """
        Args:
            backbone (nn.Module): 已经实例化的骨干网络 (例如 PT-v2m5)。
            head (nn.Module): 已经实例化的分割头 (例如 SegmentationHead)。
            **kwargs: 传递给 BaseTask 的参数。
        """
        super().__init__(**kwargs)
        
        # 将 backbone 和 head 保存为子模块
        # (注意: BaseTask 并不假定一定有 backbone，所以我们在这里保存)
        self.backbone = backbone
        self.head = head

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
        # - PointNet++：需要整个 batch 字典
        
        if hasattr(self.backbone, 'forward') and 'batch' in self.backbone.forward.__code__.co_varnames:
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
        return logits

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
        执行单个预测步骤 (用于生产)。
        """
        # 1. 前向传播
        preds_logits = self.forward(batch)
        
        # 2. 计算最终的类别预测
        # preds_labels = torch.argmax(preds_logits, dim=-1)  # 使用 -1 以支持 [B, N, C] 或 [N, C]
        
        # 3. 返回一个字典，PredictionWriter 回调将处理这个字典
        #    我们返回 CPU 张量以释放 GPU 内存
        results = {
            # "preds": preds_labels.cpu(),
            "logits": preds_logits.cpu(),  # 也保存 logits 用于后处理
        }
        
        # (可选) 如果需要原始索引 (用于拼接/投票)
        # 我们的数据集可能提供 'indices' 字段
        if "indices" in batch:
            results["indices"] = batch["indices"].cpu()
        
        # 🔥 新增：传递文件信息到 callback，避免推断
        # 这些信息由 BinPklDataset 在 test split 时提供
        if "bin_file" in batch:
            results["bin_file"] = batch["bin_file"]  # 通常是字符串列表
        if "bin_path" in batch:
            results["bin_path"] = batch["bin_path"]
        if "pkl_path" in batch:
            results["pkl_path"] = batch["pkl_path"]
        
        # 保存坐标信息（用于可视化）
        if "coord" in batch:
            results["coord"] = batch["coord"].cpu()
            
        return results