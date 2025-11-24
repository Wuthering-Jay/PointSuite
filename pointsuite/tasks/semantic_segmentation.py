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
                 backbone: nn.Module = None,
                 head: nn.Module = None,
                 model_config: Dict[str, Any] = None,
                 **kwargs): # 接收来自 BaseTask 的所有参数 (learning_rate, loss_configs, etc.)
        """
        Args:
            backbone (nn.Module): 已经实例化的骨干网络 (例如 PT-v2m5)。
            head (nn.Module): 已经实例化的分割头 (例如 SegmentationHead)。
            model_config (Dict): 模型配置字典，用于从配置实例化 backbone 和 head。
                                 如果提供了 model_config，则忽略 backbone 和 head 参数。
                                 格式:
                                 {
                                     'backbone': {'class_path': '...', 'init_args': {...}},
                                     'head': {'class_path': '...', 'init_args': {...}}
                                 }
            **kwargs: 传递给 BaseTask 的参数。
        """
        super().__init__(**kwargs)
        
        # 🔥 关键修改：保存 hyperparameters
        # 如果使用 model_config，我们忽略 backbone 和 head 对象，避免重复保存和警告
        # 如果使用 backbone/head 对象，我们必须保存它们以支持自动重建（尽管会有警告）
        if model_config is not None:
            self.save_hyperparameters(ignore=['backbone', 'head'])
            
            # 从配置实例化
            backbone_cfg = model_config.get('backbone')
            head_cfg = model_config.get('head')
            
            if backbone_cfg:
                backbone_cls = self._import_class(backbone_cfg['class_path'])
                self.backbone = backbone_cls(**backbone_cfg.get('init_args', {}))
            
            if head_cfg:
                head_cls = self._import_class(head_cfg['class_path'])
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