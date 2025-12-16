"""
语义分割引擎模块

该模块实现了语义分割任务的完整引擎，继承自 BaseEngine 并提供：
- 语义分割模型的训练、验证和测试流程
- 语义分割任务的数据加载器管理
- 语义分割预测结果的保存和评估

配置要求
--------
配置文件需要包含以下结构：

.. code-block:: yaml

    data:
        paths:
            train: "path/to/train"
            val: "path/to/val"  
            test: "path/to/test"
            predict: "path/to/predict"
        classes:
            num_classes: 8
            names: ["ground", "vegetation", "cars", ...]
        processing:
            grid_size: 0.04
            max_points: 100000
            batch_size: 4
            num_workers: 4
        
    trainer:
        training:
            max_epochs: 100
            accelerator: "gpu"
            devices: 1
        optimizer:
            type: "AdamW"
            lr: 0.001
            weight_decay: 0.01
        callbacks:
            model_checkpoint:
                enabled: true
                save_top_k: 3

使用示例
--------
>>> from pointsuite.engine import SemanticSegmentationEngine
>>> engine = SemanticSegmentationEngine(config)
>>> engine.train()
>>> engine.test()

继承指南
--------
如需扩展此引擎以支持特定的语义分割变体，可以：

1. 重写 _create_task() 方法以使用自定义的 Task 类
2. 重写 _create_datamodule() 方法以使用自定义的数据模块
3. 重写 _save_predictions() 方法以自定义预测结果保存逻辑

.. code-block:: python

    class CustomSemanticSegmentationEngine(SemanticSegmentationEngine):
        def _create_task(self) -> LightningModule:
            # 返回自定义的语义分割任务
            return CustomSemanticSegmentationTask(self.config)
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import numpy as np

try:
    import lightning as L
    from lightning import Trainer, LightningModule, LightningDataModule
except ImportError:
    import pytorch_lightning as L
    from pytorch_lightning import Trainer, LightningModule, LightningDataModule

from pointsuite.engine.base import BaseEngine
from pointsuite.tasks import SemanticSegmentationTask
from pointsuite.data import BinPklDataModule
from pointsuite.utils.logger import (
    Colors,
    log_info,
    log_warning,
    log_debug,
    log_error,
    print_header,
    print_section,
)


class SemanticSegmentationEngine(BaseEngine):
    """
    语义分割引擎
    
    该引擎管理语义分割任务的完整生命周期，包括：
    - 模型训练与验证
    - 模型测试与评估
    - 预测结果生成与保存
    
    Attributes
    ----------
    config : dict
        完整配置字典
    _task : Optional[SemanticSegmentationTask]
        语义分割任务实例
    _datamodule : Optional[BinPklDataModule]
        数据模块实例
    _trainer : Optional[Trainer]
        PyTorch Lightning 训练器实例
        
    参数
    ----
    config : dict
        配置字典，需包含 data、trainer、model 等配置节
        
    示例
    ----
    >>> config = load_config("configs/experiments/dales_semseg.yaml")
    >>> engine = SemanticSegmentationEngine(config)
    >>> engine.train()
    """
    
    def __init__(self, config: dict) -> None:
        """
        初始化语义分割引擎
        
        参数
        ----
        config : dict
            配置字典，需包含完整的 data、trainer、model 配置
        """
        super().__init__(config)
        
    def _create_task(self) -> LightningModule:
        """
        创建语义分割任务实例
        
        返回
        ----
        SemanticSegmentationTask
            配置好的语义分割任务实例
            
        说明
        ----
        任务实例会接收完整配置，并从中提取所需的模型、优化器和损失函数配置。
        """
        log_debug("创建语义分割任务")
        
        # 从配置中提取任务相关参数
        model_config = self.config.get("model", {})
        data_config = self.config.get("data", {})
        classes = data_config.get("classes", {})
        task_config = self.config.get("task", {})
        
        # 获取损失函数和指标配置
        loss_configs = self.config.get("losses", [])
        metric_configs = self.config.get("metrics", [])
        
        # 获取任务初始化参数
        task_init_args = task_config.get("init_args", {})
        
        # 构建任务参数
        task_kwargs = {
            "model_config": model_config,
            "class_mapping": classes.get("mapping", task_init_args.get("class_mapping")),
            "class_names": classes.get("names", task_init_args.get("class_names")),
            "ignore_label": classes.get("ignore_label", task_init_args.get("ignore_label", -1)),
            "loss_configs": loss_configs,
            "metric_configs": metric_configs,
        }
        
        # 添加学习率配置（如果有）
        if "learning_rate" in task_init_args:
            task_kwargs["learning_rate"] = task_init_args["learning_rate"]
        
        return SemanticSegmentationTask(**task_kwargs)
        
    def _create_datamodule(self) -> LightningDataModule:
        """
        创建数据模块实例
        
        返回
        ----
        BinPklDataModule
            配置好的数据模块实例
            
        说明
        ----
        数据模块会根据配置自动设置训练、验证、测试和预测数据集。
        """
        log_debug("创建数据模块")
        
        # 从配置中提取数据相关参数
        data_config = self.config.get("data", {})
        paths = data_config.get("paths", {})
        classes = data_config.get("classes", {})
        loader = data_config.get("loader", {})
        transforms_config = data_config.get("transforms", {})
        
        # 动态批处理配置
        dynamic_batch = loader.get("dynamic_batch", {})
        dynamic_enabled = dynamic_batch.get("enabled", False)
        
        # 循环配置
        loop_config = loader.get("loop", {})
        
        # 构建数据模块参数
        datamodule_kwargs = {
            # 数据路径
            "train_data": paths.get("train"),
            "val_data": paths.get("val"),
            "test_data": paths.get("test"),
            "predict_data": paths.get("predict"),
            
            # 类别配置
            "class_mapping": classes.get("mapping"),
            "class_names": classes.get("names"),
            "ignore_label": classes.get("ignore_label", -1),
            
            # 加载器配置
            "assets": loader.get("assets", ["coord", "class"]),
            "batch_size": loader.get("batch_size", 4),
            "num_workers": loader.get("num_workers", 4),
            "pin_memory": loader.get("pin_memory", True),
            "prefetch_factor": loader.get("prefetch_factor", 2),
            "persistent_workers": loader.get("persistent_workers", False),
            
            # 采样模式
            "mode": loader.get("mode", "grid"),
            "max_loops": loader.get("max_loops"),
            
            # 动态批处理
            "use_dynamic_batch": dynamic_enabled,
            "max_points": dynamic_batch.get("max_points_train", 100000),
            "use_dynamic_batch_inference": dynamic_enabled,
            "max_points_inference": dynamic_batch.get("max_points_inference", 200000),
            
            # 循环次数
            "train_loop": loop_config.get("train", 1),
            "val_loop": loop_config.get("val", 1),
            "test_loop": loop_config.get("test", 1),
            "predict_loop": loop_config.get("predict", 1),
        }
        
        # 处理数据增强配置
        from ..data.transforms import build_transforms
        
        if transforms_config.get("train"):
            datamodule_kwargs["train_transforms"] = build_transforms(transforms_config["train"])
        if transforms_config.get("val"):
            datamodule_kwargs["val_transforms"] = build_transforms(transforms_config["val"])
        if transforms_config.get("test"):
            datamodule_kwargs["test_transforms"] = build_transforms(transforms_config["test"])
        if transforms_config.get("predict"):
            datamodule_kwargs["predict_transforms"] = build_transforms(transforms_config["predict"])
        
        return BinPklDataModule(**datamodule_kwargs)
        
    def _print_training_info(self) -> None:
        """
        打印训练配置信息
        
        输出包括：
        - 数据路径信息
        - 类别信息
        - 训练超参数
        - 优化器配置
        """
        data_config = self.config.get("data", {})
        trainer_config = self.config.get("trainer", {})
        
        # 数据信息
        paths = data_config.get("paths", {})
        classes = data_config.get("classes", {})
        processing = data_config.get("processing", {})
        
        print_section("数据配置")
        log_info(f"训练数据: {paths.get('train', 'N/A')}")
        log_info(f"验证数据: {paths.get('val', 'N/A')}")
        log_info(f"类别数量: {classes.get('num_classes', 'N/A')}")
        log_info(f"网格大小: {processing.get('grid_size', 'N/A')}")
        log_info(f"批次大小: {processing.get('batch_size', 'N/A')}")
        
        # 训练信息
        training = trainer_config.get("training", {})
        optimizer = trainer_config.get("optimizer", {})
        
        print_section("训练配置")
        log_info(f"最大轮数: {training.get('max_epochs', 'N/A')}")
        log_info(f"加速器: {training.get('accelerator', 'N/A')}")
        log_info(f"设备数: {training.get('devices', 'N/A')}")
        log_info(f"优化器: {optimizer.get('type', 'N/A')}")
        log_info(f"学习率: {optimizer.get('lr', 'N/A')}")
        
    def _find_best_checkpoint(self) -> Optional[str]:
        """
        查找最佳检查点文件
        
        返回
        ----
        str or None
            最佳检查点的路径，如果未找到则返回 None
            
        说明
        ----
        搜索顺序：
        1. 查找以 "best-" 开头的检查点
        2. 如果没有，返回最新的检查点
        """
        trainer_config = self.config.get("trainer", {})
        callbacks_config = trainer_config.get("callbacks", {})
        checkpoint_config = callbacks_config.get("model_checkpoint", {})
        
        # 获取检查点目录
        checkpoint_dir = checkpoint_config.get("dirpath")
        if not checkpoint_dir or not os.path.exists(checkpoint_dir):
            log_warning("未找到检查点目录")
            return None
            
        checkpoint_dir = Path(checkpoint_dir)
        
        # 查找最佳检查点
        best_checkpoints = list(checkpoint_dir.glob("best-*.ckpt"))
        if best_checkpoints:
            return str(best_checkpoints[0])
            
        # 查找最新检查点
        all_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if all_checkpoints:
            # 按修改时间排序，返回最新的
            latest = max(all_checkpoints, key=lambda p: p.stat().st_mtime)
            return str(latest)
            
        log_warning("未找到任何检查点文件")
        return None
        
    def _get_default_prediction_dir(self) -> str:
        """
        获取默认的预测结果输出目录
        
        返回
        ----
        str
            预测结果输出目录路径
        """
        trainer_config = self.config.get("trainer", {})
        output_dir = trainer_config.get("output_dir", "outputs")
        exp_name = trainer_config.get("exp_name", "experiment")
        
        return os.path.join(output_dir, exp_name, "predictions")
        
    def _save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        output_dir: str,
        save_format: str = "npy"
    ) -> None:
        """
        保存预测结果
        
        参数
        ----
        predictions : list
            预测结果列表，每个元素是包含预测信息的字典
        output_dir : str
            输出目录路径
        save_format : str, default="npy"
            保存格式，支持 "npy"、"txt"
            
        说明
        ----
        每个预测结果会保存为单独的文件，文件名基于原始数据文件名。
        """
        log_info(f"保存 {len(predictions)} 个预测结果")
        
        for idx, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                # 如果预测结果不是字典，尝试转换
                pred = {"predictions": pred}
                
            # 获取或生成文件名
            filename = pred.get("filename", f"prediction_{idx:04d}")
            if isinstance(filename, (list, tuple)):
                filename = filename[0] if filename else f"prediction_{idx:04d}"
                
            # 移除原始扩展名
            filename = Path(filename).stem
            
            # 获取预测标签
            pred_labels = pred.get("predictions", pred.get("pred_labels"))
            if pred_labels is None:
                log_warning(f"预测结果 {idx} 没有有效的预测标签")
                continue
                
            # 转换为 numpy 数组
            if torch.is_tensor(pred_labels):
                pred_labels = pred_labels.cpu().numpy()
                
            # 保存文件
            if save_format == "npy":
                output_path = os.path.join(output_dir, f"{filename}.npy")
                np.save(output_path, pred_labels)
            elif save_format == "txt":
                output_path = os.path.join(output_dir, f"{filename}.txt")
                np.savetxt(output_path, pred_labels, fmt="%d")
            else:
                log_warning(f"不支持的保存格式: {save_format}")
                continue
                
            log_debug(f"保存: {output_path}")
            
        log_info(f"预测结果已保存到: {output_dir}")
        
    def export_model(
        self,
        output_path: str,
        ckpt_path: Optional[str] = None,
        export_format: str = "onnx"
    ) -> None:
        """
        导出模型
        
        参数
        ----
        output_path : str
            导出文件路径
        ckpt_path : str, optional
            检查点路径。如果为 None，使用最佳检查点
        export_format : str, default="onnx"
            导出格式，支持 "onnx"、"torchscript"
            
        说明
        ----
        模型导出用于部署场景，导出后的模型可以在没有 PyTorch Lightning 的环境中运行。
        
        .. warning::
            ONNX 导出可能不支持所有操作，请确保模型中的操作都有对应的 ONNX 实现。
        """
        print_header("模型导出")
        
        task = self.get_task()
        
        # 加载检查点
        if ckpt_path is None:
            ckpt_path = self._find_best_checkpoint()
        if ckpt_path:
            log_info(f"加载检查点: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            task.load_state_dict(checkpoint["state_dict"])
            
        task.eval()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if export_format == "torchscript":
            self._export_torchscript(task, output_path)
        elif export_format == "onnx":
            self._export_onnx(task, output_path)
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")
            
        print_section("导出完成")
        
    def _export_torchscript(self, task: LightningModule, output_path: str) -> None:
        """
        导出为 TorchScript 格式
        
        参数
        ----
        task : LightningModule
            任务实例
        output_path : str
            输出文件路径
        """
        log_info("导出为 TorchScript 格式")
        
        try:
            # 获取模型
            model = task.model if hasattr(task, "model") else task
            
            # 使用 trace 方法导出
            # 注意：这里需要根据实际模型输入进行调整
            scripted = torch.jit.script(model)
            scripted.save(output_path)
            
            log_info(f"TorchScript 模型已保存到: {output_path}")
        except Exception as e:
            log_error(f"TorchScript 导出失败: {e}")
            raise
            
    def _export_onnx(self, task: LightningModule, output_path: str) -> None:
        """
        导出为 ONNX 格式
        
        参数
        ----
        task : LightningModule
            任务实例
        output_path : str
            输出文件路径
        """
        log_info("导出为 ONNX 格式")
        log_warning("ONNX 导出目前不支持，点云模型的稀疏卷积操作可能无法导出")
        
        # ONNX 导出需要具体的输入示例，这里只是占位实现
        # 实际使用时需要根据模型的输入格式进行调整
        raise NotImplementedError("ONNX 导出尚未实现，请使用 TorchScript 导出")
