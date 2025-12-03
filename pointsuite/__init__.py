"""
PointSuite - 点云深度学习框架

一个基于 PyTorch Lightning 的点云处理框架，支持:
- 语义分割 (Semantic Segmentation)
- 实例分割 (Instance Segmentation) [TODO]
- 目标检测 (Object Detection) [TODO]

主要模块:
- data: 数据加载和预处理
- models: 模型架构 (backbone, head, losses)
- tasks: PyTorch Lightning 任务模块
- engine: 任务执行引擎
- utils: 工具函数

使用示例:
    # 方式1: YAML 配置
    from pointsuite.engine import SemanticSegmentationEngine
    engine = SemanticSegmentationEngine.from_config('configs/experiments/dales.yaml')
    engine.run()
    
    # 方式2: Python 配置
    from pointsuite import BinPklDataModule, SemanticSegmentationTask
    datamodule = BinPklDataModule(...)
    task = SemanticSegmentationTask(...)
    trainer.fit(task, datamodule)
"""

__version__ = "0.1.0"

# 核心组件
from .data import BinPklDataModule
from .tasks import SemanticSegmentationTask, BaseTask
from .engine import SemanticSegmentationEngine, BaseEngine

# 工具
from .utils import (
    ExperimentConfig,
    create_experiment_config,
    setup_logger,
)

__all__ = [
    # 版本
    '__version__',
    # 数据
    'BinPklDataModule',
    # 任务
    'BaseTask',
    'SemanticSegmentationTask',
    # 引擎
    'BaseEngine',
    'SemanticSegmentationEngine',
    # 工具
    'ExperimentConfig',
    'create_experiment_config',
    'setup_logger',
]
