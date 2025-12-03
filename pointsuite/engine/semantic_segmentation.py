"""
语义分割引擎

基于 BaseEngine 实现语义分割任务的完整流程，包括:
- BinPklDataModule 创建
- SemanticSegmentationTask 创建
- 语义分割特定的回调 (SemanticPredictLasWriter)
"""

import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .base import BaseEngine
from ..data import BinPklDataModule
from ..tasks import SemanticSegmentationTask
from ..utils.callbacks import SemanticPredictLasWriter, AutoEmptyCacheCallback, TextLoggingCallback
from ..utils.logger import Colors, print_config


class SemanticSegmentationEngine(BaseEngine):
    """
    语义分割引擎
    
    处理语义分割任务的完整流程:
    - 数据加载 (BinPklDataModule)
    - 模型训练 (SemanticSegmentationTask)
    - 预测结果保存 (SemanticPredictLasWriter)
    
    使用示例:
        # 方式1: YAML 配置
        >>> engine = SemanticSegmentationEngine.from_config(
        ...     'configs/experiments/dales_semseg.yaml'
        ... )
        >>> engine.run()
        
        # 方式2: Python 配置
        >>> engine = SemanticSegmentationEngine(config={
        ...     'run': {'mode': 'train', 'output_dir': './outputs'},
        ...     'data': {...},
        ...     'model': {...}
        ... })
        >>> engine.run()
        
        # 方式3: 完全编程式
        >>> datamodule = BinPklDataModule(...)
        >>> task = SemanticSegmentationTask(...)
        >>> engine = SemanticSegmentationEngine(
        ...     datamodule=datamodule,
        ...     task=task
        ... )
        >>> engine.train().test().predict()
    """
    
    TASK_TYPE = "语义分割"
    
    def _create_datamodule(self) -> pl.LightningDataModule:
        """
        创建 BinPklDataModule
        
        Returns:
            BinPklDataModule 实例
        """
        data_config = self.config.data.copy()
        
        # 处理变换配置
        train_transforms = data_config.pop('train_transforms', None)
        val_transforms = data_config.pop('val_transforms', None)
        test_transforms = data_config.pop('test_transforms', None)
        predict_transforms = data_config.pop('predict_transforms', None)
        
        # 实例化变换
        if train_transforms and isinstance(train_transforms[0], dict):
            train_transforms = self._instantiate_transforms(train_transforms)
        if val_transforms and isinstance(val_transforms[0], dict):
            val_transforms = self._instantiate_transforms(val_transforms)
        if test_transforms and isinstance(test_transforms[0], dict):
            test_transforms = self._instantiate_transforms(test_transforms)
        if predict_transforms and isinstance(predict_transforms[0], dict):
            predict_transforms = self._instantiate_transforms(predict_transforms)
        
        # 移除不属于 DataModule 的配置
        data_config.pop('num_classes', None)  # 这是派生属性
        
        # 打印数据配置
        print_config({
            '训练数据': data_config.get('train_data', 'N/A'),
            '验证数据': data_config.get('val_data', 'N/A'),
            '测试数据': data_config.get('test_data', 'N/A'),
            '预测数据': data_config.get('predict_data', 'N/A'),
        }, "📁 数据路径")
        
        print_config({
            '类别数量': len(data_config.get('class_mapping', [])),
            '类别名称': ', '.join(data_config.get('class_names', [])),
            '忽略标签': data_config.get('ignore_label', -1),
        }, "🏷️  类别配置")
        
        print_config({
            '采样模式': data_config.get('mode', 'grid'),
            '批次大小': data_config.get('batch_size', 4),
            '最大点数(训练)': f"{data_config.get('max_points', 100000):,}",
            '最大点数(推理)': f"{data_config.get('max_points_inference', 100000):,}",
            'Workers': data_config.get('num_workers', 4),
        }, "⚙️  数据加载配置")
        
        datamodule = BinPklDataModule(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            predict_transforms=predict_transforms,
            **data_config
        )
        
        return datamodule
    
    def _create_task(self) -> pl.LightningModule:
        """
        创建 SemanticSegmentationTask
        
        Returns:
            SemanticSegmentationTask 实例
        """
        model_config = self.config.model.copy()
        task_config = self.config.task.copy() if self.config.task else {}
        
        # 从配置中获取损失函数和指标配置
        loss_configs = self.config._raw.get('losses', [])
        metric_configs = self.config._raw.get('metrics', [])
        
        # 从 data 配置获取类别信息
        data_config = self.config.data
        class_mapping = data_config.get('class_mapping')
        class_names = data_config.get('class_names')
        ignore_label = data_config.get('ignore_label', -1)
        
        # 处理损失函数中的类别权重
        if loss_configs and hasattr(self._datamodule, 'train_dataset'):
            for loss_cfg in loss_configs:
                init_args = loss_cfg.get('init_args', {})
                # 如果需要类别权重但未指定，从 datamodule 获取
                if 'weight' not in init_args and hasattr(self._datamodule.train_dataset, 'class_weights'):
                    init_args['weight'] = self._datamodule.train_dataset.class_weights
        
        # 获取 task 初始化参数
        task_init_args = task_config.get('init_args', {})
        learning_rate = task_init_args.get('learning_rate', 1e-3)
        
        # 打印模型配置
        backbone_name = model_config.get('backbone', {}).get('class_path', 'Unknown').split('.')[-1]
        head_name = model_config.get('head', {}).get('class_path', 'Unknown').split('.')[-1]
        in_channels = model_config.get('backbone', {}).get('init_args', {}).get('in_channels', 'Unknown')
        
        print(f"  {Colors.DIM}├─{Colors.RESET} Backbone: {Colors.GREEN}{backbone_name}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} Head: {Colors.GREEN}{head_name}{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 输入通道: {Colors.YELLOW}{in_channels}{Colors.RESET}")
        
        task = SemanticSegmentationTask(
            model_config=model_config,
            learning_rate=learning_rate,
            class_mapping=class_mapping,
            class_names=class_names,
            ignore_label=ignore_label,
            loss_configs=loss_configs,
            metric_configs=metric_configs,
        )
        
        return task
    
    def _get_default_callbacks(self) -> List[Callback]:
        """
        获取语义分割特定的回调
        
        Returns:
            回调列表
        """
        callbacks = []
        callback_config = self.config._raw.get('callbacks', {})
        
        # SemanticPredictLasWriter
        if 'predict_writer' in callback_config:
            writer_cfg = callback_config['predict_writer']
            callbacks.append(self._instantiate_class(writer_cfg))
        else:
            # 默认预测写入器
            callbacks.append(SemanticPredictLasWriter(
                output_dir=os.path.join(self.config.output_dir, 'predictions'),
                save_logits=False,
                auto_infer_reverse_mapping=True
            ))
        
        # TextLoggingCallback
        callbacks.append(TextLoggingCallback(log_interval=10))
        
        # AutoEmptyCacheCallback
        callbacks.append(AutoEmptyCacheCallback(
            slowdown_threshold=3.0,
            absolute_threshold=1.5,
            clear_interval=0,
            warmup_steps=10,
            verbose=True
        ))
        
        return callbacks
    
    def _print_config(self) -> None:
        """打印语义分割配置"""
        from ..utils.logger import print_header
        print_header("DALES 语义分割训练", "🎯")
        
        print_config({
            '运行模式': self.config.mode,
            '随机种子': self.config.seed,
            '输出目录': self.config.output_dir,
            'Checkpoint': self.config.checkpoint_path or 'N/A',
        }, "⚙️  运行配置")
