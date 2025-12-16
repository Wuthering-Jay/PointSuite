"""
任务基类模块

该模块提供了 PyTorch Lightning 任务的抽象基类，实现了以下核心功能：
- 自动从 YAML 配置实例化损失函数
- 自动从 YAML 配置实例化评估指标
- 统一的训练、验证和测试流程
- 超参数保存和加载支持

配置要求
--------
配置文件需要包含以下结构：

.. code-block:: yaml

    model:
        backbone:
            class_path: pointsuite.models.backbones.ptv2.PointTransformerV2
            init_args:
                in_channels: 6
                
    losses:
        - class_path: pointsuite.models.losses.FocalLoss
          init_args:
              gamma: 2.0
          loss_weight: 1.0
          
    metrics:
        - class_path: pointsuite.utils.metrics.SegmentationMetrics
          init_args:
              num_classes: 8

使用示例
--------
>>> from pointsuite.tasks import SemanticSegmentationTask
>>> task = SemanticSegmentationTask(config)
>>> trainer.fit(task, datamodule)

继承指南
--------
子类需要实现以下方法：
- forward(): 模型前向传播
- postprocess_predictions(): 处理模型输出（可选）
- _print_validation_metrics(): 自定义验证指标打印（可选）

.. code-block:: python

    class CustomTask(BaseTask):
        def forward(self, batch: Dict[str, Any]) -> Any:
            # 实现前向传播
            return self.model(batch)
"""

from typing import List, Dict, Any, Optional, Union
import gc

import torch
import torch.nn as nn
import yaml

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

from ..utils.logger import (
    Colors,
    log_info,
    log_warning,
    log_error,
    log_debug,
    log_success,
    print_header,
    print_section,
    print_config,
)
from ..utils.config import import_class


class BaseTask(L.LightningModule):
    """
    任务基类
    
    这是一个抽象的 PyTorch Lightning 任务基类，处理所有任务共有的逻辑：
    
    1. 从 YAML 配置自动实例化损失函数
    2. 从 YAML 配置自动实例化评估指标
    3. 在验证/测试结束时自动计算和记录所有指标
    
    注意
    ----
    优化器配置通过 Engine 传入或由 configure_optimizers 提供默认实现。
    
    Attributes
    ----------
    losses : nn.ModuleDict
        损失函数字典
    loss_weights : Dict[str, float]
        损失函数权重
    val_metrics : nn.ModuleDict
        验证阶段指标
    test_metrics : nn.ModuleDict
        测试阶段指标
    class_mapping : Optional[Dict[int, int]]
        类别映射字典
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        loss_configs: Optional[List[Dict[str, Any]]] = None,
        metric_configs: Optional[List[Dict[str, Any]]] = None,
        class_mapping: Optional[Dict[int, int]] = None,
        class_names: Optional[List[str]] = None,
        ignore_label: int = -1
    ) -> None:
        """
        初始化任务基类
        
        参数
        ----
        learning_rate : float, default=1e-3
            学习率。此参数会被保存到 hparams 中用于日志记录，
            同时也可以在不使用 LightningCLI 时用于配置优化器
        loss_configs : list, optional
            损失函数配置列表，每个元素是一个字典：
            
            .. code-block:: python
            
                {
                    'class_path': 'pointsuite.models.losses.FocalLoss',
                    'init_args': {'gamma': 2.0},
                    'loss_weight': 1.0,  # 可选，默认 1.0
                    'loss_source': 'logits'  # 可选，指定使用哪个输出
                }
                
        metric_configs : list, optional
            指标配置列表，格式与 loss_configs 类似
        class_mapping : dict, optional
            原始类别 -> 连续类别的映射，例如 {1: 0, 2: 1, 6: 2}
            此映射会保存到检查点中，在预测时自动加载
        class_names : list, optional
            类别名称列表，用于验证时显示
        ignore_label : int, default=-1
            忽略的标签值
        """
        super().__init__()
        
        # 保存超参数到检查点（排除不可序列化的对象）
        self.save_hyperparameters()
        
        # 保存类别映射（用于预测时的标签恢复）
        self.class_mapping = class_mapping
        self.class_names = class_names
        self.ignore_label = ignore_label
        
        # 标记需要延迟保存 hparams
        self._pending_hparams_save = True
        
        # 初始化损失函数
        self.losses = nn.ModuleDict()
        self.loss_weights: Dict[str, float] = {}
        self.loss_sources: Dict[str, Optional[str]] = {}
        
        if loss_configs:
            self._init_losses(loss_configs)
            
        # 初始化评估指标
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        
        if metric_configs:
            self._init_metrics(metric_configs)
            
        # 训练状态追踪
        self.last_loss: Optional[float] = None
        
    def _init_losses(self, loss_configs: List[Dict[str, Any]]) -> None:
        """
        初始化损失函数
        
        参数
        ----
        loss_configs : list
            损失函数配置列表
        """
        # 保存配置以便后续更新权重
        self._loss_configs = loss_configs
        self._auto_weight_losses: List[str] = []  # 需要自动权重的损失函数名
        
        for cfg in loss_configs:
            # 获取或生成损失函数名称
            loss_name = cfg.get("name", cfg["class_path"].split('.')[-1].lower())
            
            # 动态导入损失函数类
            loss_class = import_class(cfg["class_path"])
            init_args = cfg.get("init_args", {}).copy()
            
            # 检查是否需要自动类别权重
            auto_weight = cfg.get("auto_weight", False)
            if auto_weight:
                self._auto_weight_losses.append(loss_name)
                log_info(f"损失函数 [{loss_name}] 将在训练开始时自动计算类别权重")
            
            # 实例化损失函数
            self.losses[loss_name] = loss_class(**init_args)
            
            # 获取权重（兼容 'loss_weight' 和旧版 'weight'）
            self.loss_weights[loss_name] = cfg.get("loss_weight", cfg.get("weight", 1.0))
            
            # 获取输入源（可选，用于多输出模型）
            self.loss_sources[loss_name] = cfg.get("loss_source")
            
            log_debug(f"初始化损失函数: {loss_name}")
    
    def _update_loss_weights_from_datamodule(self, datamodule) -> None:
        """
        从 datamodule 获取类别权重并更新损失函数
        
        参数
        ----
        datamodule : LightningDataModule
            数据模块实例
        """
        if not self._auto_weight_losses:
            return
        
        # 尝试从 train_dataset 获取类别权重
        class_weights = None
        if hasattr(datamodule, 'train_dataset') and datamodule.train_dataset is not None:
            dataset = datamodule.train_dataset
            if hasattr(dataset, 'class_weights'):
                class_weights = dataset.class_weights
        
        if class_weights is None:
            log_warning("无法从 datamodule 获取类别权重，使用均匀权重")
            return
        
        # 更新需要自动权重的损失函数
        for loss_name in self._auto_weight_losses:
            if loss_name in self.losses:
                loss_fn = self.losses[loss_name]
                # 更新 CrossEntropyLoss 的权重
                if hasattr(loss_fn, 'ce_loss') and hasattr(loss_fn.ce_loss, 'weight'):
                    loss_fn.ce_loss.weight = class_weights.to(self.device)
                    loss_fn.weight = class_weights.to(self.device)
                elif hasattr(loss_fn, 'weight'):
                    loss_fn.weight = class_weights.to(self.device)
                
                # 打印权重信息
                log_info(f"损失函数 [{loss_name}] 类别权重已更新:")
                weights_str = ", ".join([f"{w:.4f}" for w in class_weights.tolist()])
                log_info(f"  [{weights_str}]")
                if self.class_names:
                    for i, (name, w) in enumerate(zip(self.class_names, class_weights.tolist())):
                        log_debug(f"    {i}: {name} -> {w:.4f}")
    
    def on_fit_start(self) -> None:
        """训练开始时的回调"""
        # 从 datamodule 获取类别权重
        if hasattr(self, '_auto_weight_losses') and self._auto_weight_losses:
            if self.trainer and hasattr(self.trainer, 'datamodule'):
                self._update_loss_weights_from_datamodule(self.trainer.datamodule)
            
    def _init_metrics(self, metric_configs: List[Dict[str, Any]]) -> None:
        """
        初始化评估指标
        
        参数
        ----
        metric_configs : list
            指标配置列表
            
        说明
        ----
        为验证和测试分别创建独立的指标实例，以避免状态冲突。
        """
        for cfg in metric_configs:
            # 获取或生成指标名称
            metric_name = cfg.get("name", cfg["class_path"].split('.')[-1].lower())
            
            # 动态导入指标类
            metric_class = import_class(cfg["class_path"])
            init_args = cfg.get("init_args", {})
            
            # 为验证和测试分别创建实例
            self.val_metrics[metric_name] = metric_class(**init_args)
            self.test_metrics[metric_name] = metric_class(**init_args)
            
            log_debug(f"初始化指标: {metric_name}")
            
    def configure_optimizers(self):
        """
        配置优化器
        
        返回
        ----
        optimizer
            优化器实例
            
        说明
        ----
        这是默认实现，使用 AdamW 优化器。
        实际训练中通常由 Engine 通过 configure_optimizer_fn 覆盖此方法。
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.get('learning_rate', 1e-3),
            weight_decay=1e-4
        )
        return optimizer
        
    def on_train_start(self) -> None:
        """
        训练开始时的回调
        
        保存 hparams.yaml（确保中文正确显示）
        """
        if self._pending_hparams_save:
            self._save_hparams_with_unicode()
            self._pending_hparams_save = False
            
    def _save_hparams_with_unicode(self) -> None:
        """
        保存超参数到 YAML 文件（支持 Unicode）
        """
        try:
            import os
            if hasattr(self.logger, 'log_dir') and self.logger.log_dir:
                hparams_file = os.path.join(self.logger.log_dir, 'hparams.yaml')
                with open(hparams_file, 'w', encoding='utf-8') as f:
                    yaml.dump(
                        dict(self.hparams),
                        f,
                        allow_unicode=True,
                        default_flow_style=False,
                        sort_keys=False
                    )
        except Exception as e:
            log_warning(f"无法保存 hparams: {e}")
            
    # =========================================================================
    # 损失计算
    # =========================================================================
    
    def _calculate_total_loss(
        self,
        preds: Any,
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有损失函数的加权总和
        
        参数
        ----
        preds : Any
            模型的前向输出
        batch : dict
            来自 DataLoader 的批次数据
            
        返回
        ----
        dict
            包含 'total_loss' 和每个单独损失的字典
            
        说明
        ----
        损失计算强制在 FP32 下运行，以避免混合精度训练中的数值不稳定问题。
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # 强制在 FP32 下计算损失
        with torch.amp.autocast('cuda', enabled=False):
            # 转换预测结果到 FP32
            preds = self._to_fp32(preds)
            batch = self._to_fp32(batch)
            
            for name, loss_fn in self.losses.items():
                # 确定输入源
                input_preds = self._get_loss_input(preds, name)
                
                # 计算损失
                loss = loss_fn(input_preds, batch)
                
                # 确保损失是 FP32
                if loss.dtype != torch.float32:
                    loss = loss.float()
                    
                loss_dict[name] = loss
                total_loss += self.loss_weights[name] * loss
                
        loss_dict["total_loss"] = total_loss
        return loss_dict
        
    def _to_fp32(self, data: Any) -> Any:
        """
        将数据转换为 FP32
        
        参数
        ----
        data : Any
            输入数据（Tensor 或 Dict）
            
        返回
        ----
        Any
            转换后的数据
        """
        if isinstance(data, torch.Tensor):
            if data.is_floating_point() and data.dtype != torch.float32:
                return data.float()
            return data
        elif isinstance(data, dict):
            return {
                k: self._to_fp32(v) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }
        return data
        
    def _get_loss_input(self, preds: Any, loss_name: str) -> Any:
        """
        获取特定损失函数的输入
        
        参数
        ----
        preds : Any
            模型输出
        loss_name : str
            损失函数名称
            
        返回
        ----
        Any
            损失函数的输入
        """
        source_key = self.loss_sources.get(loss_name)
        
        if not isinstance(preds, dict):
            return preds
            
        if source_key:
            # 使用指定的源
            if source_key not in preds:
                raise ValueError(
                    f"损失函数 '{loss_name}' 请求的源 '{source_key}' 不存在于预测结果中"
                )
            return preds[source_key]
        elif 'logits' in preds:
            # 默认使用 logits
            return preds['logits']
        else:
            # 返回整个字典
            return preds
            
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """
        从批次数据推断批次大小
        
        参数
        ----
        batch : dict
            批次数据
            
        返回
        ----
        int
            批次大小
        """
        if 'batch_index' in batch:
            return batch['batch_index'].max().item() + 1
        elif 'offset' in batch:
            return len(batch['offset'])
        return 1
        
    # =========================================================================
    # 预测后处理
    # =========================================================================
    
    def postprocess_predictions(self, preds: Any) -> torch.Tensor:
        """
        后处理模型预测结果
        
        将模型输出转换为适合指标计算的格式。
        子类可以覆盖此方法以支持复杂的输出处理。
        
        参数
        ----
        preds : Any
            模型的原始输出
            
        返回
        ----
        torch.Tensor
            处理后的预测结果（logits 或 labels）
            
        示例
        ----
        >>> # 标准语义分割（默认）
        >>> def postprocess_predictions(self, preds):
        ...     return preds  # 直接返回 logits
        
        >>> # Mask3D
        >>> def postprocess_predictions(self, preds):
        ...     class_logits = preds['class_logits']
        ...     mask_logits = preds['mask_logits']
        ...     return (class_logits.T @ mask_logits).T
        """
        if isinstance(preds, dict):
            if 'logits' in preds:
                return preds['logits']
            elif 'labels' in preds:
                return preds['labels']
            elif 'pred' in preds:
                return preds['pred']
            else:
                return next(iter(preds.values()))
        return preds
        
    # =========================================================================
    # 训练流程
    # =========================================================================
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        训练步骤
        
        参数
        ----
        batch : dict
            批次数据
        batch_idx : int
            批次索引
            
        返回
        ----
        torch.Tensor
            总损失
        """
        # 确保在训练模式
        if batch_idx == 0:
            self.train()
            
        # 前向传播
        try:
            preds = self(batch)
        except Exception as e:
            self._save_error_batch(batch, batch_idx, e)
            raise
            
        # 计算损失
        loss_dict = self._calculate_total_loss(preds, batch)
        total_loss = loss_dict["total_loss"]
        
        # 保存最新损失供进度条使用
        self.last_loss = total_loss.item()
        if self.trainer is not None:
            self.trainer.live_loss = self.last_loss
            
        # 记录损失
        batch_size = self._get_batch_size(batch)
        for name, loss_value in loss_dict.items():
            self.log(
                f"{name}_step",
                loss_value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=batch_size,
            )
            
        return total_loss
        
    def _save_error_batch(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        error: Exception
    ) -> None:
        """
        保存出错时的批次数据用于调试
        
        参数
        ----
        batch : dict
            出错的批次数据
        batch_idx : int
            批次索引
        error : Exception
            发生的异常
        """
        import pickle
        
        error_path = f'error_batch_{batch_idx}_step_{self.global_step}.pkl'
        
        with open(error_path, 'wb') as f:
            pickle.dump({
                'batch_idx': batch_idx,
                'global_step': self.global_step,
                'batch': {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                },
                'error': str(error)
            }, f)
            
        print_section("训练错误")
        log_error(f"batch_idx={batch_idx}, global_step={self.global_step}")
        log_error(f"错误数据已保存到: {error_path}")
        
    def on_train_epoch_end(self) -> None:
        """
        训练轮次结束时的回调
        
        清理显存以便验证。
        """
        self._cleanup_memory()
        
    def on_validation_start(self) -> None:
        """
        验证开始时的回调
        
        再次清理显存。
        """
        self._cleanup_memory()
        
    def _cleanup_memory(self) -> None:
        """
        清理 GPU 和系统内存
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    # =========================================================================
    # 验证流程
    # =========================================================================
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        验证步骤
        
        参数
        ----
        batch : dict
            批次数据
        batch_idx : int
            批次索引
        """
        # 前向传播
        preds = self.forward(batch)
        
        # 计算损失
        loss_dict = self._calculate_total_loss(preds, batch)
        batch_size = self._get_batch_size(batch)
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size
        )
        
        # 后处理并更新指标
        processed_preds = self.postprocess_predictions(preds)
        target = self._get_target(batch)
        
        for metric in self.val_metrics.values():
            metric.update(processed_preds, target)
            
    def _get_target(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        从批次中获取目标标签
        
        参数
        ----
        batch : dict
            批次数据
            
        返回
        ----
        torch.Tensor
            目标标签
        """
        for key in ['class', 'label', 'labels', 'target']:
            if key in batch:
                return batch[key]
        raise KeyError("批次中未找到目标标签（尝试了 class, label, labels, target）")
        
    def on_validation_epoch_end(self) -> None:
        """
        验证轮次结束时的回调
        
        计算并记录所有指标。
        """
        metric_results = {}
        print_metrics = {}
        
        for name, metric in self.val_metrics.items():
            val = metric.compute()
            
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, (torch.Tensor, float, int)):
                        if isinstance(v, torch.Tensor) and v.numel() > 1:
                            continue
                        metric_results[k] = v
                print_metrics.update(val)
            else:
                metric_results[name] = val
                print_metrics[name] = val
                
            metric.reset()
            
        self.log_dict(metric_results, on_step=False, on_epoch=True, prog_bar=False)
        self._print_validation_metrics(print_metrics)
        
    def _print_validation_metrics(self, print_metrics: Dict[str, Any]) -> None:
        """
        打印验证指标
        
        子类可以覆盖此方法以自定义打印格式。
        
        参数
        ----
        print_metrics : dict
            计算出的指标字典
        """
        display_epoch = self.current_epoch + 1
        
        print_section(f"验证 Epoch {display_epoch} - 指标")
        
        for name, value in print_metrics.items():
            if isinstance(value, (float, int)):
                log_info(f"{name}: {value:.4f}")
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                log_info(f"{name}: {value.item():.4f}")
                
    # =========================================================================
    # 测试流程
    # =========================================================================
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        测试步骤
        
        参数
        ----
        batch : dict
            批次数据
        batch_idx : int
            批次索引
        """
        # 前向传播
        preds = self.forward(batch)
        
        # 计算损失
        loss_dict = self._calculate_total_loss(preds, batch)
        batch_size = self._get_batch_size(batch)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # 后处理并更新指标
        processed_preds = self.postprocess_predictions(preds)
        target = self._get_target(batch)
        
        for metric in self.test_metrics.values():
            metric.update(processed_preds, target)
            
    def on_test_epoch_end(self) -> None:
        """
        测试轮次结束时的回调
        
        计算并记录所有指标。
        """
        metric_results = {}
        print_metrics = {}
        
        for name, metric in self.test_metrics.items():
            val = metric.compute()
            
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, (torch.Tensor, float, int)):
                        if isinstance(v, torch.Tensor) and v.numel() > 1:
                            continue
                        metric_results[k] = v
                print_metrics.update(val)
            else:
                metric_results[name] = val
                print_metrics[name] = val
                
            metric.reset()
            
        self.log_dict(metric_results, on_step=False, on_epoch=True)
        self._print_test_metrics(print_metrics)
        
    def _print_test_metrics(self, print_metrics: Dict[str, Any]) -> None:
        """
        打印测试指标
        
        子类可以覆盖此方法以自定义打印格式。
        
        参数
        ----
        print_metrics : dict
            计算出的指标字典
        """
        print_section("测试结果 - 指标")
        
        for name, value in print_metrics.items():
            if isinstance(value, (float, int)):
                log_info(f"{name}: {value:.4f}")
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                log_info(f"{name}: {value.item():.4f}")
