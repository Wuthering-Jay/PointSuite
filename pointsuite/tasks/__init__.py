"""
PointSuite 任务模块

本包提供与 PyTorch Lightning 兼容的任务 (LightningModule)，
用于不同的点云任务

主要组件：
- BaseTask: 任务的抽象基类
- SemanticSegmentationTask: 语义分割任务
- InstanceSegmentationTask: 实例分割任务
- ObjectDetectionTask: 目标检测任务
"""

from .base_task import BaseTask
from .semantic_segmentation import SemanticSegmentationTask

try:
    from .instance_segmentation import InstanceSegmentationTask
except ImportError:
    InstanceSegmentationTask = None

try:
    from .object_detection import ObjectDetectionTask
except ImportError:
    ObjectDetectionTask = None

__all__ = [
    'BaseTask',
    'SemanticSegmentationTask',
    'InstanceSegmentationTask',
    'ObjectDetectionTask',
]
