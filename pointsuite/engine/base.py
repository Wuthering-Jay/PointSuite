"""
点云任务引擎基类

提供通用的任务执行流程，包括:
- 配置解析和组件实例化
- DataModule/Task/Trainer 创建
- 训练/测试/预测流程控制

子类需要实现:
- _create_datamodule(): 创建数据模块
- _create_task(): 创建任务模块
- _get_default_callbacks(): 获取默认回调
"""

import os
import sys
import warnings
import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)

from ..utils.config import (
    ExperimentConfig,
    ConfigLoader,
    create_experiment_config,
    load_yaml,
    save_yaml,
    deep_merge,
)
from ..utils.logger import setup_logger, Colors, print_header, print_section, print_config


class BaseEngine(ABC):
    """
    点云任务引擎基类
    
    负责整个实验流程的管理:
    1. 配置加载和解析
    2. 组件实例化 (DataModule, Task, Callbacks)
    3. Trainer 创建
    4. 训练/测试/预测流程执行
    
    使用方式:
        # 方式1: YAML 配置
        >>> engine = SemanticSegmentationEngine.from_config('configs/experiments/dales.yaml')
        >>> engine.run()
        
        # 方式2: Python 配置
        >>> engine = SemanticSegmentationEngine(
        ...     config=ExperimentConfig(...),
        ...     datamodule=my_datamodule,
        ...     task=my_task
        ... )
        >>> engine.run()
        
        # 方式3: 分步执行
        >>> engine.setup()
        >>> engine.train()
        >>> engine.test()
        >>> engine.predict()
    """
    
    # 任务类型标识，子类需要覆盖
    TASK_TYPE: str = "base"
    
    def __init__(
        self,
        config: Optional[Union[ExperimentConfig, Dict, str, Path]] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        task: Optional[pl.LightningModule] = None,
        trainer: Optional[pl.Trainer] = None,
        callbacks: Optional[List[Callback]] = None,
        **kwargs
    ):
        """
        初始化引擎
        
        Args:
            config: 实验配置，可以是:
                   - ExperimentConfig 对象
                   - 配置字典
                   - YAML 配置文件路径
            datamodule: 数据模块 (可选，如果提供则跳过配置创建)
            task: 任务模块 (可选，如果提供则跳过配置创建)
            trainer: Trainer (可选，如果提供则跳过配置创建)
            callbacks: 额外的回调列表
            **kwargs: 额外参数，用于覆盖配置
        """
        # 解析配置
        self.config = self._parse_config(config, **kwargs)
        
        # 组件 (可以预设或后续创建)
        self._datamodule = datamodule
        self._task = task
        self._trainer = trainer
        self._callbacks = callbacks or []
        
        # 状态标志
        self._is_setup = False
        self._logger_initialized = False
        
        # 忽略常见警告
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        cli_args: Optional[List[str]] = None,
        **kwargs
    ) -> 'BaseEngine':
        """
        从 YAML 配置文件创建引擎
        
        Args:
            config_path: YAML 配置文件路径
            cli_args: 命令行参数覆盖
            **kwargs: 额外参数覆盖
            
        Returns:
            引擎实例
        """
        config = create_experiment_config(
            config_path=config_path,
            cli_args=cli_args,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> 'BaseEngine':
        """
        从配置字典创建引擎
        
        Args:
            config_dict: 配置字典
            **kwargs: 额外参数覆盖
            
        Returns:
            引擎实例
        """
        config = create_experiment_config(config_dict=config_dict, **kwargs)
        return cls(config=config)
    
    def _parse_config(
        self,
        config: Optional[Union[ExperimentConfig, Dict, str, Path]],
        **kwargs
    ) -> ExperimentConfig:
        """
        解析配置
        
        Args:
            config: 配置输入
            **kwargs: 额外参数
            
        Returns:
            ExperimentConfig 对象
        """
        if config is None:
            config = ExperimentConfig()
        elif isinstance(config, (str, Path)):
            config = create_experiment_config(config_path=config, **kwargs)
        elif isinstance(config, dict):
            config = create_experiment_config(config_dict=config, **kwargs)
        elif isinstance(config, ExperimentConfig):
            if kwargs:
                # 应用额外参数
                config = create_experiment_config(
                    config_dict=config.to_dict(),
                    **kwargs
                )
        
        return config
    
    # ========================================================================
    # 属性访问
    # ========================================================================
    
    @property
    def datamodule(self) -> pl.LightningDataModule:
        """获取数据模块"""
        if self._datamodule is None:
            raise RuntimeError("DataModule 未初始化，请先调用 setup()")
        return self._datamodule
    
    @property
    def task(self) -> pl.LightningModule:
        """获取任务模块"""
        if self._task is None:
            raise RuntimeError("Task 未初始化，请先调用 setup()")
        return self._task
    
    @property
    def trainer(self) -> pl.Trainer:
        """获取 Trainer"""
        if self._trainer is None:
            raise RuntimeError("Trainer 未初始化，请先调用 setup()")
        return self._trainer
    
    # ========================================================================
    # 抽象方法 (子类必须实现)
    # ========================================================================
    
    @abstractmethod
    def _create_datamodule(self) -> pl.LightningDataModule:
        """
        创建数据模块
        
        子类需要实现此方法，根据配置创建对应的 DataModule
        
        Returns:
            DataModule 实例
        """
        raise NotImplementedError
    
    @abstractmethod
    def _create_task(self) -> pl.LightningModule:
        """
        创建任务模块
        
        子类需要实现此方法，根据配置创建对应的 Task
        
        Returns:
            Task 实例
        """
        raise NotImplementedError
    
    def _get_default_callbacks(self) -> List[Callback]:
        """
        获取默认回调列表
        
        子类可以覆盖此方法添加任务特定的回调
        
        Returns:
            回调列表
        """
        return []
    
    # ========================================================================
    # 组件创建辅助方法
    # ========================================================================
    
    def _import_class(self, class_path: str) -> Type:
        """
        从字符串路径导入类
        
        Args:
            class_path: 类路径 (如 'pointsuite.models.PointTransformerV2')
            
        Returns:
            类对象
        """
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    
    def _instantiate_class(
        self,
        config: Dict[str, Any],
        **override_kwargs
    ) -> Any:
        """
        从配置实例化类
        
        Args:
            config: 配置字典，包含 'class_path' 和可选的 'init_args'
            **override_kwargs: 覆盖参数
            
        Returns:
            实例化的对象
        """
        class_path = config['class_path']
        init_args = config.get('init_args', {}).copy()
        init_args.update(override_kwargs)
        
        cls = self._import_class(class_path)
        return cls(**init_args)
    
    def _instantiate_transforms(self, transform_configs: List[Dict]) -> List:
        """
        从配置实例化变换列表
        
        Args:
            transform_configs: 变换配置列表
            
        Returns:
            变换对象列表
        """
        transforms = []
        for cfg in transform_configs:
            if isinstance(cfg, dict) and 'class_path' in cfg:
                transforms.append(self._instantiate_class(cfg))
            else:
                # 已经是变换对象
                transforms.append(cfg)
        return transforms
    
    def _create_callbacks(self) -> List[Callback]:
        """
        创建回调列表
        
        Returns:
            回调列表
        """
        callbacks = []
        callback_config = self.config._raw.get('callbacks', {})
        
        # ModelCheckpoint
        if 'model_checkpoint' in callback_config:
            ckpt_cfg = callback_config['model_checkpoint']
            callbacks.append(ModelCheckpoint(
                dirpath=os.path.join(self.config.output_dir, 'checkpoints'),
                **ckpt_cfg
            ))
        else:
            # 默认检查点
            callbacks.append(ModelCheckpoint(
                dirpath=os.path.join(self.config.output_dir, 'checkpoints'),
                monitor='mean_iou',
                mode='max',
                save_top_k=3,
                save_last=True,
                filename='{epoch:02d}-{mean_iou:.4f}',
                verbose=True
            ))
        
        # EarlyStopping
        if 'early_stopping' in callback_config:
            es_cfg = callback_config['early_stopping']
            callbacks.append(EarlyStopping(**es_cfg))
        
        # LearningRateMonitor
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        # 任务特定的默认回调
        callbacks.extend(self._get_default_callbacks())
        
        # 用户额外的回调
        callbacks.extend(self._callbacks)
        
        # 从配置实例化的自定义回调
        if 'custom_callbacks' in callback_config:
            for cb_cfg in callback_config['custom_callbacks']:
                callbacks.append(self._instantiate_class(cb_cfg))
        
        return callbacks
    
    def _create_trainer(self, callbacks: List[Callback]) -> pl.Trainer:
        """
        创建 Trainer
        
        Args:
            callbacks: 回调列表
            
        Returns:
            Trainer 实例
        """
        trainer_config = self.config.trainer.copy()
        
        # 设置默认根目录
        if trainer_config.get('default_root_dir') is None:
            trainer_config['default_root_dir'] = self.config.output_dir
        
        # 确定加速器
        if trainer_config.get('accelerator') == 'auto':
            trainer_config['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'
        
        # 禁用 logger (我们使用自定义日志)
        trainer_config.setdefault('logger', False)
        
        return pl.Trainer(
            callbacks=callbacks,
            **trainer_config
        )
    
    def _configure_optimizer(self, task: pl.LightningModule) -> None:
        """
        配置优化器
        
        如果配置中指定了优化器，通过 monkey patch 覆盖 task 的 configure_optimizers
        
        Args:
            task: 任务模块
        """
        optimizer_config = self.config._raw.get('optimizer')
        scheduler_config = self.config._raw.get('lr_scheduler')
        
        if optimizer_config is None:
            return
        
        # 创建新的 configure_optimizers 方法
        def configure_optimizers(self_task):
            # 实例化优化器
            optimizer_cls = self._import_class(optimizer_config['class_path'])
            optimizer_args = optimizer_config.get('init_args', {}).copy()
            
            # 处理 lr 引用
            if 'lr' in optimizer_args and isinstance(optimizer_args['lr'], str):
                # 如果是字符串引用，尝试解析
                optimizer_args['lr'] = self.config.get('task.init_args.learning_rate', 1e-3)
            
            optimizer = optimizer_cls(self_task.parameters(), **optimizer_args)
            
            if scheduler_config is None:
                return optimizer
            
            # 实例化学习率调度器
            scheduler_cls = self._import_class(scheduler_config['class_path'])
            scheduler_args = scheduler_config.get('init_args', {}).copy()
            
            # 特殊处理: T_max 为 null 时自动设置
            if scheduler_args.get('T_max') is None and hasattr(self_task.trainer, 'estimated_stepping_batches'):
                scheduler_args['T_max'] = self_task.trainer.estimated_stepping_batches
            
            scheduler = scheduler_cls(optimizer, **scheduler_args)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': scheduler_config.get('interval', 'step'),
                    'frequency': scheduler_config.get('frequency', 1),
                }
            }
        
        # Monkey patch
        import types
        task.configure_optimizers = types.MethodType(configure_optimizers, task)
    
    # ========================================================================
    # 主要流程方法
    # ========================================================================
    
    def setup(self, stage: Optional[str] = None) -> 'BaseEngine':
        """
        初始化所有组件
        
        Args:
            stage: 阶段 ('fit', 'test', 'predict', None 表示全部)
            
        Returns:
            self (支持链式调用)
        """
        if self._is_setup:
            return self
        
        # 设置随机种子
        pl.seed_everything(self.config.seed)
        
        # 设置日志
        if not self._logger_initialized:
            setup_logger(self.config.output_dir)
            self._logger_initialized = True
        
        # 打印配置
        self._print_config()
        
        # 创建 DataModule
        print_section("📦 初始化 DataModule")
        if self._datamodule is None:
            self._datamodule = self._create_datamodule()
        self._datamodule.setup(stage='fit' if stage is None else stage)
        
        # 创建 Task
        print_section("🧠 初始化模型")
        if self._task is None:
            self._task = self._create_task()
        
        # 配置优化器
        self._configure_optimizer(self._task)
        
        # 创建 Callbacks
        callbacks = self._create_callbacks()
        
        # 创建 Trainer
        print_section("🔧 初始化 Trainer")
        if self._trainer is None:
            self._trainer = self._create_trainer(callbacks)
        
        self._print_trainer_info()
        
        self._is_setup = True
        return self
    
    def train(self, ckpt_path: Optional[str] = None) -> 'BaseEngine':
        """
        执行训练
        
        Args:
            ckpt_path: checkpoint 路径 (用于 resume)
            
        Returns:
            self (支持链式调用)
        """
        if not self._is_setup:
            self.setup()
        
        print_header("开始训练", "🏋️")
        
        mode = self.config.mode
        
        if mode == 'train':
            self.trainer.fit(self.task, self.datamodule)
            
        elif mode == 'resume':
            ckpt = ckpt_path or self.config.checkpoint_path
            if ckpt is None:
                raise ValueError("resume 模式需要指定 checkpoint_path")
            print(f"  从 checkpoint 继续训练: {Colors.CYAN}{ckpt}{Colors.RESET}")
            self.trainer.fit(self.task, self.datamodule, ckpt_path=ckpt)
            
        elif mode == 'finetune':
            ckpt = ckpt_path or self.config.checkpoint_path
            if ckpt is None:
                raise ValueError("finetune 模式需要指定 checkpoint_path")
            print(f"  加载预训练权重: {Colors.CYAN}{ckpt}{Colors.RESET}")
            self._load_pretrained_weights(ckpt)
            self.trainer.fit(self.task, self.datamodule)
        
        return self
    
    def test(self, ckpt_path: Optional[str] = None) -> 'BaseEngine':
        """
        执行测试
        
        Args:
            ckpt_path: checkpoint 路径
            
        Returns:
            self (支持链式调用)
        """
        if not self._is_setup:
            self.setup()
        
        print_header("开始测试", "🧪")
        
        # 确定 checkpoint
        if ckpt_path is None:
            if self.config.mode == 'test':
                ckpt_path = self.config.checkpoint_path
            else:
                ckpt_path = 'best'
        
        self.trainer.test(self.task, self.datamodule, ckpt_path=ckpt_path)
        return self
    
    def predict(self, ckpt_path: Optional[str] = None) -> 'BaseEngine':
        """
        执行预测
        
        Args:
            ckpt_path: checkpoint 路径
            
        Returns:
            self (支持链式调用)
        """
        if not self._is_setup:
            self.setup()
        
        print_header("开始预测", "🔮")
        
        # 确定 checkpoint
        if ckpt_path is None:
            if self.config.mode == 'test':
                ckpt_path = self.config.checkpoint_path
            else:
                ckpt_path = 'best'
        
        self.trainer.predict(self.task, datamodule=self.datamodule, ckpt_path=ckpt_path)
        return self
    
    def run(self) -> 'BaseEngine':
        """
        根据配置执行完整流程
        
        根据 run.mode 决定执行的流程:
        - train: 训练 -> 测试 -> 预测
        - resume: 继续训练 -> 测试 -> 预测
        - finetune: 微调 -> 测试 -> 预测
        - test: 测试 -> 预测
        - predict: 仅预测
        
        Returns:
            self (支持链式调用)
        """
        self.setup()
        
        mode = self.config.mode
        
        if mode in ['train', 'resume', 'finetune']:
            self.train()
            if self.config.data.get('test_data'):
                self.test()
            if self.config.data.get('predict_data'):
                self.predict()
                
        elif mode == 'test':
            self.test()
            if self.config.data.get('predict_data'):
                self.predict()
                
        elif mode == 'predict':
            self.predict()
            
        else:
            raise ValueError(f"未知的运行模式: {mode}")
        
        self._print_completion()
        return self
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _load_pretrained_weights(self, ckpt_path: str) -> None:
        """
        加载预训练权重 (用于 finetune)
        
        Args:
            ckpt_path: checkpoint 路径
        """
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        
        # 移除 'model.' 前缀 (如果有)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        missing_keys, unexpected_keys = self.task.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"  {Colors.YELLOW}缺失的键: {missing_keys[:5]}...{Colors.RESET}")
        if unexpected_keys:
            print(f"  {Colors.YELLOW}未预期的键: {unexpected_keys[:5]}...{Colors.RESET}")
        print(f"  {Colors.GREEN}✓ 权重加载完成{Colors.RESET}")
    
    def _print_config(self) -> None:
        """打印配置信息"""
        print_header(f"{self.TASK_TYPE.upper()} 任务", "🎯")
        
        print_config({
            '运行模式': self.config.mode,
            '随机种子': self.config.seed,
            '输出目录': self.config.output_dir,
        }, "⚙️  运行配置")
    
    def _print_trainer_info(self) -> None:
        """打印 Trainer 信息"""
        device_name = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'
        print(f"  {Colors.DIM}├─{Colors.RESET} 设备: {Colors.GREEN}{device_name}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 精度: {Colors.GREEN}{self.trainer.precision}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} Epochs: {Colors.GREEN}{self.config.trainer.get('max_epochs', 100)}{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 检查点目录: {Colors.CYAN}{self.config.output_dir}{Colors.RESET}")
    
    def _print_completion(self) -> None:
        """打印完成信息"""
        print()
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}  🎉 任务完成!{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 检查点目录: {Colors.CYAN}{self.trainer.default_root_dir}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 输出目录: {Colors.CYAN}{self.config.output_dir}{Colors.RESET}")
        
        if hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback:
            if self.trainer.checkpoint_callback.best_model_path:
                print(f"  {Colors.DIM}├─{Colors.RESET} 最佳模型: {Colors.GREEN}{self.trainer.checkpoint_callback.best_model_path}{Colors.RESET}")
            
            if self.trainer.checkpoint_callback.best_model_score is not None:
                print(f"  {Colors.DIM}└─{Colors.RESET} 最佳分数: {Colors.GREEN}{self.trainer.checkpoint_callback.best_model_score:.4f}{Colors.RESET}")
            else:
                print(f"  {Colors.DIM}└─{Colors.RESET} 最佳分数: {Colors.DIM}N/A{Colors.RESET}")
        
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    
    # ========================================================================
    # 保存/加载配置
    # ========================================================================
    
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        保存当前配置到文件
        
        Args:
            path: 保存路径，如果为 None 则保存到 output_dir/config.yaml
        """
        if path is None:
            path = Path(self.config.output_dir) / 'config.yaml'
        
        self.config.save(path)
        print(f"配置已保存到: {path}")
