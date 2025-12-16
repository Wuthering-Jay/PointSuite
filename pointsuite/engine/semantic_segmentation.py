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
from ..utils.logger import Colors, print_config, log_info, log_warning


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
        
        适配新的配置结构:
        - data.paths: 数据路径
        - data.classes: 类别配置
        - data.loader: 数据加载配置
        - data.transforms: 数据增强配置
        
        Returns:
            BinPklDataModule 实例
        """
        data_config = self.config.data.copy()
        
        # ========================================
        # 解析新配置结构
        # ========================================
        
        # 数据路径 (新结构: data.paths.xxx)
        paths = data_config.get('paths', {})
        train_data = paths.get('train') or data_config.get('train_data')
        val_data = paths.get('val') or data_config.get('val_data')
        test_data = paths.get('test') or data_config.get('test_data')
        predict_data = paths.get('predict') or data_config.get('predict_data')
        
        # 类别配置 (新结构: data.classes.xxx)
        classes = data_config.get('classes', {})
        class_mapping = classes.get('mapping') or data_config.get('class_mapping')
        class_names = classes.get('names') or data_config.get('class_names')
        num_classes = classes.get('num_classes') or data_config.get('num_classes')
        ignore_label = classes.get('ignore_label') if 'ignore_label' in classes else data_config.get('ignore_label', -1)
        
        # 数据加载配置 (新结构: data.loader.xxx)
        loader = data_config.get('loader', {})
        assets = loader.get('assets') or data_config.get('assets', ['coord', 'class'])
        mode = loader.get('mode') or data_config.get('mode', 'grid')
        max_loops = loader.get('max_loops') or data_config.get('max_loops')
        batch_size = loader.get('batch_size') or data_config.get('batch_size', 4)
        num_workers = loader.get('num_workers') or data_config.get('num_workers', 4)
        pin_memory = loader.get('pin_memory', True)
        persistent_workers = loader.get('persistent_workers', True)
        prefetch_factor = loader.get('prefetch_factor', 2)
        
        # 动态批次配置
        dynamic_batch = loader.get('dynamic_batch', {})
        use_dynamic_batch = dynamic_batch.get('enabled') if isinstance(dynamic_batch, dict) else data_config.get('use_dynamic_batch', True)
        max_points = dynamic_batch.get('max_points_train') if isinstance(dynamic_batch, dict) else data_config.get('max_points', 125000)
        max_points_inference = dynamic_batch.get('max_points_inference') if isinstance(dynamic_batch, dict) else data_config.get('max_points_inference', 125000)
        
        # 加权采样
        use_weighted_sampler = loader.get('weighted_sampler') if isinstance(loader, dict) else data_config.get('use_weighted_sampler', True)
        
        # 循环配置
        loop_config = loader.get('loop', {})
        train_loop = loop_config.get('train') if isinstance(loop_config, dict) else data_config.get('train_loop', 5)
        val_loop = loop_config.get('val') if isinstance(loop_config, dict) else data_config.get('val_loop', 5)
        test_loop = loop_config.get('test') if isinstance(loop_config, dict) else data_config.get('test_loop', 1)
        predict_loop = loop_config.get('predict') if isinstance(loop_config, dict) else data_config.get('predict_loop', 1)
        
        # ========================================
        # 处理变换配置
        # ========================================
        transforms = data_config.get('transforms', {})
        
        # 新结构: data.transforms.train/val/test/predict
        train_transforms = transforms.get('train') or data_config.get('train_transforms')
        val_transforms = transforms.get('val') or data_config.get('val_transforms')
        test_transforms = transforms.get('test') or data_config.get('test_transforms')
        predict_transforms = transforms.get('predict') or data_config.get('predict_transforms')
        
        # 实例化变换
        if train_transforms and isinstance(train_transforms, list) and len(train_transforms) > 0 and isinstance(train_transforms[0], dict):
            train_transforms = self._instantiate_transforms(train_transforms)
        if val_transforms and isinstance(val_transforms, list) and len(val_transforms) > 0 and isinstance(val_transforms[0], dict):
            val_transforms = self._instantiate_transforms(val_transforms)
        if test_transforms and isinstance(test_transforms, list) and len(test_transforms) > 0 and isinstance(test_transforms[0], dict):
            test_transforms = self._instantiate_transforms(test_transforms)
        if predict_transforms and isinstance(predict_transforms, list) and len(predict_transforms) > 0 and isinstance(predict_transforms[0], dict):
            predict_transforms = self._instantiate_transforms(predict_transforms)
        
        # ========================================
        # 打印配置信息
        # ========================================
        print_config({
            '训练数据': train_data or 'N/A',
            '验证数据': val_data or 'N/A',
            '测试数据': test_data or 'N/A',
            '预测数据': predict_data or 'N/A',
        }, "数据路径")
        
        print_config({
            '类别数量': num_classes or len(class_mapping or []),
            '类别名称': ', '.join(class_names or []),
            '忽略标签': ignore_label,
        }, "类别配置")
        
        print_config({
            '采样模式': mode,
            '批次大小': batch_size,
            '最大点数(训练)': f"{max_points:,}",
            '最大点数(推理)': f"{max_points_inference:,}",
            'Workers': num_workers,
        }, "数据加载配置")
        
        # ========================================
        # 创建 DataModule
        # ========================================
        datamodule = BinPklDataModule(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            predict_data=predict_data,
            assets=assets,
            class_mapping=class_mapping,
            class_names=class_names,
            ignore_label=ignore_label,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            mode=mode,
            max_loops=max_loops,
            use_dynamic_batch=use_dynamic_batch,
            max_points=max_points,
            use_dynamic_batch_inference=use_dynamic_batch,
            max_points_inference=max_points_inference,
            use_weighted_sampler=use_weighted_sampler,
            train_loop=train_loop,
            val_loop=val_loop,
            test_loop=test_loop,
            predict_loop=predict_loop,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            predict_transforms=predict_transforms,
        )
        
        return datamodule
    
    def _create_task(self) -> pl.LightningModule:
        """
        创建 SemanticSegmentationTask
        
        适配新的配置结构:
        - data.classes: 类别配置
        - model.backbone/head: 模型配置
        - task.init_args: 任务参数
        
        Returns:
            SemanticSegmentationTask 实例
        """
        model_config = self.config.model.copy()
        task_config = self.config.task.copy() if self.config.task else {}
        
        # 从配置中获取损失函数和指标配置
        loss_configs = self.config._raw.get('losses', [])
        metric_configs = self.config._raw.get('metrics', [])
        
        # 从 data 配置获取类别信息 (支持新旧两种结构)
        data_config = self.config.data
        classes = data_config.get('classes', {})
        class_mapping = classes.get('mapping') or data_config.get('class_mapping')
        class_names = classes.get('names') or data_config.get('class_names')
        ignore_label = classes.get('ignore_label') if 'ignore_label' in classes else data_config.get('ignore_label', -1)
        
        # 处理损失函数中的类别权重
        if loss_configs and hasattr(self._datamodule, 'train_dataset'):
            for loss_cfg in loss_configs:
                # 仅对 CrossEntropyLoss 自动注入类别权重
                # 其他 Loss (如 LovaszLoss, FocalLoss) 可能不支持 weight 参数或使用不同参数名 (如 alpha)
                class_path = loss_cfg.get('class_path', '').lower()
                loss_name = loss_cfg.get('name', '').lower()
                
                is_ce_loss = 'crossentropyloss' in class_path or 'ce_loss' in loss_name
                
                if is_ce_loss:
                    init_args = loss_cfg.get('init_args', {})
                    
                    # 检查是否启用了自动权重计算
                    # 1. 显式参数 auto_weight=True (推荐)
                    # 2. 兼容旧逻辑：如果没指定 weight 且 dataset 有权重，则自动注入 (但不推荐)
                    auto_weight = loss_cfg.get('auto_weight', False)
                    
                    if auto_weight:
                        if hasattr(self._datamodule.train_dataset, 'class_weights'):
                            class_weights = self._datamodule.train_dataset.class_weights
                            init_args['weight'] = class_weights
                            
                            # 格式化权重显示
                            weights_str = ", ".join([f"{w:.4f}" for w in class_weights])
                            log_info(f"[{loss_name}] 已启用 auto_weight: 自动注入类别权重")
                            log_info(f"[{loss_name}] 类别权重: {Colors.YELLOW}[{weights_str}]{Colors.RESET}")
                        else:
                            log_warning(f"[{loss_name}] 启用了 auto_weight 但数据集不支持计算权重")
                    
                    # 旧逻辑兼容 (可选，如果想彻底废弃旧逻辑可以删除)
                    elif 'weight' not in init_args and hasattr(self._datamodule.train_dataset, 'class_weights'):
                        # 只有在没有显式设置 auto_weight=False 时才触发隐式注入
                        if 'auto_weight' not in loss_cfg:
                            class_weights = self._datamodule.train_dataset.class_weights
                            init_args['weight'] = class_weights
                            
                            # 格式化权重显示
                            weights_str = ", ".join([f"{w:.4f}" for w in class_weights])
                            log_info(f"[{loss_name}] 隐式自动注入类别权重 (建议使用 auto_weight: true 显式开启)")
                            log_info(f"[{loss_name}] 类别权重: {Colors.YELLOW}[{weights_str}]{Colors.RESET}")
        
        # 获取 task 初始化参数
        task_init_args = task_config.get('init_args', {})
        learning_rate = task_init_args.get('learning_rate', 0.001)
        
        # 打印模型配置
        backbone_name = model_config.get('backbone', {}).get('class_path', 'Unknown').split('.')[-1]
        head_name = model_config.get('head', {}).get('class_path', 'Unknown').split('.')[-1]
        in_channels = model_config.get('backbone', {}).get('init_args', {}).get('in_channels', 'Unknown')
        
        log_info(f"模型: Backbone={Colors.GREEN}{backbone_name}{Colors.RESET}, "
                 f"Head={Colors.GREEN}{head_name}{Colors.RESET}, "
                 f"输入通道={Colors.YELLOW}{in_channels}{Colors.RESET}")
        
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
        
        从新配置结构读取回调配置:
        - trainer.callbacks.predict_writer: 预测写入器配置 (推荐位置)
        - predict_writer: 兼容旧配置位置
        
        Returns:
            回调列表
        """
        callbacks = []
        
        # 从 trainer.callbacks 获取配置
        trainer_config = self.config._raw.get('trainer', {})
        callback_config = trainer_config.get('callbacks', {})
        
        # 获取 predict_writer 配置 (优先级: trainer.callbacks.predict_writer > predict_writer)
        predict_writer_cfg = callback_config.get('predict_writer') or self.config._raw.get('predict_writer')
        
        if predict_writer_cfg:
            # 检查是否启用
            enabled = predict_writer_cfg.get('enabled', True)
            if enabled:
                callbacks.append(self._instantiate_class(predict_writer_cfg))
        else:
            # 默认预测写入器
            callbacks.append(SemanticPredictLasWriter(
                output_dir=os.path.join(self.config.output_dir, 'predictions'),
                save_logits=False,
                auto_infer_reverse_mapping=True
            ))
        
        return callbacks
    
    def _print_config(self) -> None:
        """打印语义分割配置"""
        from ..utils.logger import print_header
        print_header("DALES 语义分割训练")
        
        print_config({
            '运行模式': self.config.mode,
            '随机种子': self.config.seed,
            '输出目录': self.config.output_dir,
            'Checkpoint': self.config.checkpoint_path or 'N/A',
        }, "运行配置")

        # 记录完整配置
        from ..utils.logger import print_section, log_warning
        print_section("完整配置")
        try:
            import yaml
            print(yaml.dump(self.config.to_dict(), allow_unicode=True, default_flow_style=False, sort_keys=False))
        except ImportError:
            log_warning("未安装 PyYAML，无法打印完整配置 YAML")
            print(self.config.to_dict())
