import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import importlib
from typing import List, Dict, Any

class BaseTask(pl.LightningModule):
    """
    一个抽象的任务基类 (LightningModule)。
    
    它负责处理所有任务共有的逻辑：
    1. 自动从 YAML 配置中实例化损失函数 (losses)。
    2. 自动从 YAML 配置中实例化指标 (metrics)。
    3. 自动在 validation/test epoch 结束时计算和记录所有指标。
    
    注意: 
    我们 *不* 在这里实现 `configure_optimizers`。
    PyTorch Lightning 的 `LightningCLI` 会自动读取您在
    `configs/schedules/` 目录中定义的 `optimizer` 和 `lr_scheduler` 
    配置，并自动为您配置它们。这保持了 Task 模块的简洁。
    """
    
    def __init__(self,
                 learning_rate: float = 1e-3,
                 loss_configs: List[Dict[str, Any]] = None,
                 metric_configs: List[Dict[str, Any]] = None):
        """
        Args:
            learning_rate (float): 学习率。
                                   我们在此处接收 learning_rate (而不是仅在优化器配置中)
                                   主要有两个原因:
                                   1. 日志记录: 'self.save_hyperparameters()' 会自动
                                      将 'learning_rate' 记录到 TensorBoard/Wandb。
                                   2. 灵活性: 允许在不使用 'LightningCLI' 的纯 Python 模式下
                                      轻松访问 'self.hparams.learning_rate' 来配置优化器。
                                   
                                   在 YAML 配置中，我们应将此 'learning_rate' 视为“单一事实来源”，
                                   并在 'optimizer' 配置中使用 YAML 链接 (例如:
                                   lr: ${model.init_args.learning_rate}) 来引用它。
                                   
            loss_configs (List[Dict]): 
                来自 YAML 的损失函数配置列表。
                示例: 
                - class_path: point_suite.models.losses.focal_loss.FocalLoss
                  init_args: { gamma: 2.0 }
                  weight: 1.0 # (可选) 损失的权重
                  
            metric_configs (List[Dict]): 
                来自 YAML 的指标配置列表。
                示例:
                - class_path: point_suite.utils.metrics.OverallAccuracy
                  init_args: { num_classes: 8 }
        """
        super().__init__()
        # 将 learning_rate 保存到 self.hparams，可供 logger 记录
        self.save_hyperparameters("learning_rate")
        
        # --- 1. 动态实例化损失函数 ---
        self.losses = nn.ModuleDict()
        self.loss_weights = {}
        if loss_configs:
            for cfg in loss_configs:
                # 'loss_name' 是我们给这个损失起的名字，例如 'focal_loss'
                loss_name = cfg.get("name", cfg["class_path"].split('.')[-1].lower())
                loss_class = self._import_class(cfg["class_path"])
                init_args = cfg.get("init_args", {})
                
                self.losses[loss_name] = loss_class(**init_args)
                self.loss_weights[loss_name] = cfg.get("weight", 1.0)
                
        # --- 2. 动态实例化指标 ---
        # 我们使用 ModuleDict 来确保指标被正确移动到 GPU
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        if metric_configs:
            for cfg in metric_configs:
                metric_name = cfg.get("name", cfg["class_path"].split('.')[-1].lower())
                metric_class = self._import_class(cfg["class_path"])
                init_args = cfg.get("init_args", {})
                
                # 为 val 和 test 分别创建实例，以避免状态冲突
                self.val_metrics[metric_name] = metric_class(**init_args)
                self.test_metrics[metric_name] = metric_class(**init_args)

    def _import_class(self, class_path: str) -> type:
        """一个辅助函数，用于从字符串路径动态导入类"""
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _calculate_total_loss(self, preds: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        (子类可以覆盖)
        计算所有损失函数的加权总和。
        
        Args:
            preds (Any): 模型的 forward() 输出。
            batch (Dict): 来自 DataLoader 的批次数据。
            
        Returns:
            Dict[str, torch.Tensor]: 包含 'total_loss' 和每个单独损失的字典。
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        for name, loss_fn in self.losses.items():
            # 假设损失函数接收 (preds, batch)
            # 子类可以覆盖此方法以传递不同参数
            loss = loss_fn(preds, batch)
            loss_dict[name] = loss
            total_loss += self.loss_weights[name] * loss
            
        loss_dict["total_loss"] = total_loss
        return loss_dict
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """
        从 batch 中推断 batch_size。
        
        适配我们项目的 collate_fn：
        - 如果有 'batch_index'，使用 max + 1
        - 如果有 'offset'，使用 len(offset)
        - 否则返回 1
        """
        if 'batch_index' in batch:
            return batch['batch_index'].max().item() + 1
        elif 'offset' in batch:
            return len(batch['offset'])
        else:
            return 1

    # --- 验证 (Validation) 逻辑 ---
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        # 1. 前向传播
        preds = self.forward(batch)
        
        # 2. 计算损失
        loss_dict = self._calculate_total_loss(preds, batch)
        
        # 3. 记录损失 (PL 会自动添加 'val/' 前缀)
        batch_size = self._get_batch_size(batch)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        # 4. 更新指标
        for metric in self.val_metrics.values():
            metric.update(preds, batch)

    def on_validation_epoch_end(self):
        # 5. 在 epoch 结束时，计算并记录所有指标
        metric_results = {}
        for name, metric in self.val_metrics.items():
            metric_results[name] = metric.compute()
            metric.reset() # 重置指标状态
        
        self.log_dict(metric_results, on_step=False, on_epoch=True, prog_bar=True)

    # --- 测试 (Test) 逻辑 ---
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        # 逻辑与 validation_step 相同
        preds = self.forward(batch)
        loss_dict = self._calculate_total_loss(preds, batch)
        batch_size = self._get_batch_size(batch)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, batch_size=batch_size)
        for metric in self.test_metrics.values():
            metric.update(preds, batch)

    def on_test_epoch_end(self):
        # 在 epoch 结束时，计算并记录所有指标
        metric_results = {}
        for name, metric in self.test_metrics.items():
            metric_results[name] = metric.compute()
            metric.reset()
        self.log_dict(metric_results, on_step=False, on_epoch=True)