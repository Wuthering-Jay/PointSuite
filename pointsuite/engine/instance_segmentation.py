"""
实例分割引擎

TODO: 实现实例分割任务的完整流程
- 数据加载 (支持实例标注)
- 模型训练 (Mask3D, PointGroup 等)
- 预测结果保存 (实例级输出)
"""

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .base import BaseEngine


class InstanceSegmentationEngine(BaseEngine):
    """
    实例分割引擎
    
    TODO: 实现以下功能:
    - _create_datamodule(): 创建支持实例标注的数据模块
    - _create_task(): 创建实例分割任务 (InstanceSegmentationTask)
    - _get_default_callbacks(): 添加实例分割特定的回调
    """
    
    TASK_TYPE = "实例分割"
    
    def _create_datamodule(self) -> pl.LightningDataModule:
        """创建数据模块"""
        raise NotImplementedError("实例分割 DataModule 尚未实现")
    
    def _create_task(self) -> pl.LightningModule:
        """创建任务模块"""
        raise NotImplementedError("实例分割 Task 尚未实现")
    
    def _get_default_callbacks(self) -> List[Callback]:
        """获取默认回调"""
        return []
