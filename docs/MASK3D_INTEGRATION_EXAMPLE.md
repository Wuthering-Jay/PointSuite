# Mask3D 集成示例

本文档展示如何将 Mask3D 风格的分割头集成到 PointSuite 框架中。

## 架构概述

Mask3D 输出与传统分割头的区别：
- **传统分割头**: 直接输出 `[N, C]` logits → argmax → 类别预测
- **Mask3D**: 输出 `class_logits [N_queries, C]` 和 `mask_logits [N_queries, N]`
  - 需要计算: `class_logits.T @ mask_logits` → `[C, N]` → transpose → `[N, C]` logits
  - 然后才能进行 argmax

## 解决方案

PointSuite 框架现已支持灵活的后处理机制：

### 1. BaseTask 提供的扩展点

```python
# pointsuite/tasks/base_task.py
class BaseTask(pl.LightningModule):
    def postprocess_predictions(self, preds):
        """
        后处理模型预测输出。
        
        子类可以覆盖此方法来支持：
        - Mask3D 风格的 class_logits @ mask_logits
        - 多任务模型的输出选择
        - 自定义的后处理流程
        
        默认行为：
        - 如果 preds 是 dict，尝试提取 'logits', 'labels', 或 'pred'
        - 如果 preds 是 tensor，直接返回
        """
        # 默认实现...
```

### 2. Mask3D 任务实现示例

```python
# pointsuite/tasks/mask3d_segmentation.py
import torch
from typing import Dict, Any
from .base_task import BaseTask


class Mask3DSemanticSegmentation(BaseTask):
    """
    支持 Mask3D 风格分割头的语义分割任务。
    
    Mask3D 输出格式：
        {
            'class_logits': [N_queries, num_classes],  # 查询向量的类别预测
            'mask_logits': [N_queries, N_points],      # 查询向量的掩码预测
        }
    
    最终预测计算：
        point_logits = class_logits.T @ mask_logits  # [num_classes, N_points]
        point_logits = point_logits.T                # [N_points, num_classes]
    """
    
    def postprocess_predictions(self, preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将 Mask3D 的 class_logits 和 mask_logits 转换为点级别的 logits。
        
        Args:
            preds: 模型输出字典，包含:
                - 'class_logits': [N_queries, num_classes] or [B, N_queries, num_classes]
                - 'mask_logits': [N_queries, N_points] or [B, N_queries, N_points]
        
        Returns:
            point_logits: [N_points, num_classes] or [B, N_points, num_classes]
        """
        if not isinstance(preds, dict):
            # 如果不是字典，假设已经是标准格式
            return preds
        
        # 检查是否包含 Mask3D 特有的输出
        if 'class_logits' in preds and 'mask_logits' in preds:
            class_logits = preds['class_logits']  # [N_queries, C] or [B, N_queries, C]
            mask_logits = preds['mask_logits']    # [N_queries, N] or [B, N_queries, N]
            
            # 处理批量维度
            if class_logits.ndim == 3:  # [B, N_queries, C]
                # 批量矩阵乘法: [B, C, N_queries] @ [B, N_queries, N] = [B, C, N]
                point_logits = torch.bmm(
                    class_logits.transpose(1, 2),  # [B, C, N_queries]
                    mask_logits                     # [B, N_queries, N]
                )  # -> [B, C, N]
                point_logits = point_logits.transpose(1, 2)  # [B, N, C]
            else:  # [N_queries, C]
                # 单样本: [C, N_queries] @ [N_queries, N] = [C, N]
                point_logits = class_logits.T @ mask_logits  # [C, N]
                point_logits = point_logits.T                 # [N, C]
            
            return point_logits
        
        # 如果不是 Mask3D 格式，回退到默认行为
        return super().postprocess_predictions(preds)


# 使用示例
if __name__ == "__main__":
    from pointsuite.data.datamodule_bin import BinPklDataModule
    from pytorch_lightning import Trainer
    
    # 1. 创建数据模块
    datamodule = BinPklDataModule(
        data_root="data/dales",
        batch_size=4,
        num_workers=4,
        # ... 其他参数
    )
    
    # 2. 创建 Mask3D 任务
    task = Mask3DSemanticSegmentation(
        model_cfg={
            'backbone': 'mask3d',
            'num_classes': 9,
            'num_queries': 100,
            # ... Mask3D 特定参数
        },
        optimizer_cfg={'lr': 0.001},
        scheduler_cfg=None,
    )
    
    # 3. 训练（自动使用 postprocess_predictions）
    trainer = Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
    )
    trainer.fit(task, datamodule)
```

### 3. Mask3D 分割头实现示例

```python
# pointsuite/models/heads/mask3d_head.py
import torch
import torch.nn as nn


class Mask3DHead(nn.Module):
    """
    简化的 Mask3D 分割头（参考 Mask2Former/Mask3D 架构）。
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_queries: int = 100,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Query embeddings (可学习的查询向量)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 特征投影
        self.feat_proj = nn.Linear(in_channels, hidden_dim)
        
        # 类别预测头
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 掩码预测头
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 简化的 Transformer Decoder (实际 Mask3D 使用更复杂的结构)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
    
    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: [B, N, C] 或 [N, C] - 骨干网络提取的特征
        
        Returns:
            {
                'class_logits': [B, N_queries, num_classes],
                'mask_logits': [B, N_queries, N],
            }
        """
        # 处理维度
        if features.ndim == 2:
            features = features.unsqueeze(0)  # [N, C] -> [1, N, C]
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, C = features.shape
        
        # 投影特征
        feat_proj = self.feat_proj(features)  # [B, N, hidden_dim]
        
        # 生成查询
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, N_queries, hidden_dim]
        
        # Transformer 解码
        decoded_queries = self.decoder(
            queries,           # [B, N_queries, hidden_dim]
            feat_proj          # [B, N, hidden_dim]
        )  # -> [B, N_queries, hidden_dim]
        
        # 类别预测
        class_logits = self.class_head(decoded_queries)  # [B, N_queries, num_classes]
        
        # 掩码预测
        mask_embed = self.mask_head(decoded_queries)  # [B, N_queries, hidden_dim]
        mask_logits = torch.bmm(mask_embed, feat_proj.transpose(1, 2))  # [B, N_queries, N]
        
        # 如果输入是单样本，squeeze batch 维度
        if squeeze_output:
            class_logits = class_logits.squeeze(0)  # [N_queries, num_classes]
            mask_logits = mask_logits.squeeze(0)    # [N_queries, N]
        
        return {
            'class_logits': class_logits,
            'mask_logits': mask_logits,
        }
```

## 框架支持的关键点

### 1. ✅ Metrics 自动处理

所有指标现在都支持两种输入：
- **Labels**: `[N]` - 直接使用
- **Logits**: `[N, C]` 或 `[B, N, C]` - 自动 argmax

```python
# pointsuite/utils/metrics.py
def _convert_preds_to_labels(preds: torch.Tensor) -> torch.Tensor:
    """自动检测并转换 logits 为 labels"""
    if preds.ndim == 1:
        return preds  # 已经是 labels
    elif preds.ndim == 2:
        return torch.argmax(preds, dim=1)  # [N, C] -> [N]
    elif preds.ndim == 3:
        return torch.argmax(preds, dim=2)  # [B, N, C] -> [B, N]
```

### 2. ✅ SegmentationWriter 智能处理

回调函数现在支持：
- **Logits averaging**: 多次预测的 logits 取平均，最后再 argmax
- **Labels voting**: 如果 postprocess 返回 labels，则直接累加（投票）

```python
# pointsuite/utils/callbacks.py (SegmentationWriter)
# 智能判断是否需要 argmax
if mean_logits.ndim == 2 and mean_logits.size(1) > 1:
    # [N, C] logits -> argmax
    final_preds = torch.argmax(mean_logits, dim=1)
else:
    # [N] labels -> 直接使用
    final_preds = mean_logits
```

### 3. ✅ Validation/Test 自动调用

在 `validation_step` 和 `test_step` 中，`postprocess_predictions` 会自动被调用：

```python
# pointsuite/tasks/base_task.py
def validation_step(self, batch, batch_idx):
    preds = self.forward(batch)
    processed_preds = self.postprocess_predictions(preds)  # 自动调用
    
    # Metrics 接收处理后的预测
    self.val_acc.update(processed_preds, targets)
    self.val_iou.update(processed_preds, targets)
```

## 总结

现在的架构完全兼容 Mask3D！你只需要：

1. **继承 BaseTask** 并实现 `postprocess_predictions`
2. **模型返回字典** 包含 `class_logits` 和 `mask_logits`
3. **其他代码无需修改**：
   - Metrics 自动处理 logits
   - SegmentationWriter 自动处理 logits
   - Validation/Test 自动调用 postprocess

这种设计遵循**开闭原则** (Open-Closed Principle)：
- 对扩展开放：轻松添加新的分割头（Mask3D, Mask2Former 等）
- 对修改封闭：无需修改现有 metrics 和 callbacks 代码

## 下一步

如果需要实现真实的 Mask3D：
1. 参考 [Mask3D 官方实现](https://github.com/JonasSchult/Mask3D)
2. 实现更复杂的 Transformer Decoder
3. 添加匈牙利匹配损失 (Hungarian Matching Loss)
4. 实现 mask/class 损失的加权组合

需要帮助实现具体模块吗？
