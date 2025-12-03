"""
目标检测引擎

TODO: 实现目标检测任务的完整流程
- 数据加载 (支持 3D 边界框标注)
- 模型训练 (VoteNet, 3DETR 等)
- 预测结果保存 (边界框输出)
"""

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .base import BaseEngine


class ObjectDetectionEngine(BaseEngine):
    """
    目标检测引擎
    
    TODO: 实现以下功能:
    - _create_datamodule(): 创建支持边界框标注的数据模块
    - _create_task(): 创建目标检测任务 (ObjectDetectionTask)
    - _get_default_callbacks(): 添加目标检测特定的回调
    """
    
    TASK_TYPE = "目标检测"
    
    def _create_datamodule(self) -> pl.LightningDataModule:
        """创建数据模块"""
        raise NotImplementedError("目标检测 DataModule 尚未实现")
    
    def _create_task(self) -> pl.LightningModule:
        """创建任务模块"""
        raise NotImplementedError("目标检测 Task 尚未实现")
    
    def _get_default_callbacks(self) -> List[Callback]:
        """获取默认回调"""
        return []
