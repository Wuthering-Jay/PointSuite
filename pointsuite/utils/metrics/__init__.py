"""
Metrics Package for Point Cloud Tasks

组织结构:
- base.py: 基础混淆矩阵类和通用工具
- semantic_segmentation.py: 语义分割指标
- 未来可扩展: instance_segmentation.py, object_detection.py 等
"""

# 导出基础工具
from .base import create_class_names, convert_preds_to_labels, ConfusionMatrixBase

# 导出语义分割指标
from .semantic_segmentation import (
    OverallAccuracy,
    MeanIoU,
    PerClassIoU,
    Precision,
    Recall,
    F1Score,
    SegmentationMetrics
)

__all__ = [
    # Base utilities
    'create_class_names',
    'convert_preds_to_labels',
    'ConfusionMatrixBase',
    
    # Semantic Segmentation
    'OverallAccuracy',
    'MeanIoU',
    'PerClassIoU',
    'Precision',
    'Recall',
    'F1Score',
    'SegmentationMetrics',
]
