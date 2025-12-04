"""
PointSuite Engine 模块

提供任务执行引擎:
- BaseEngine: 引擎基类
- SemanticSegmentationEngine: 语义分割引擎
- InstanceSegmentationEngine: 实例分割引擎
- ObjectDetectionEngine: 目标检测引擎
- run_experiment: 运行实验的便捷函数
- get_engine_class: 根据任务类型获取引擎类
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseEngine
from .semantic_segmentation import SemanticSegmentationEngine
from .instance_segmentation import InstanceSegmentationEngine
from .object_detection import ObjectDetectionEngine


def get_engine_class(task_type: str):
    """
    根据任务类型获取对应的 Engine 类
    
    Args:
        task_type: 任务类型 ('semantic_segmentation', 'instance_segmentation', 'object_detection')
        
    Returns:
        Engine 类
    """
    task_type = task_type.lower().replace('-', '_')
    
    if task_type in ['semantic_segmentation', 'semseg', 'segmentation']:
        return SemanticSegmentationEngine
    elif task_type in ['instance_segmentation', 'insseg']:
        return InstanceSegmentationEngine
    elif task_type in ['object_detection', 'detection', 'det']:
        return ObjectDetectionEngine
    else:
        raise ValueError(
            f"未知的任务类型: {task_type}\n"
            f"支持的类型: semantic_segmentation, instance_segmentation, object_detection"
        )


def infer_task_type(config_path: Union[str, Path]) -> str:
    """
    从配置文件路径推断任务类型
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        任务类型字符串
    """
    config_name = Path(config_path).stem.lower()
    
    if 'semseg' in config_name or 'semantic' in config_name:
        return 'semantic_segmentation'
    elif 'insseg' in config_name or 'instance' in config_name:
        return 'instance_segmentation'
    elif 'det' in config_name or 'detection' in config_name:
        return 'object_detection'
    else:
        # 默认为语义分割
        return 'semantic_segmentation'


def run_experiment(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
    cli_args: Optional[List[str]] = None,
    **kwargs
) -> Any:
    """
    运行实验的便捷函数
    
    支持多种配置方式:
    1. YAML 配置文件
    2. Python 字典
    3. 关键字参数
    
    Args:
        config_path: YAML 配置文件路径
        config_dict: 配置字典
        task_type: 任务类型，如果为 None 则从配置推断
        cli_args: 命令行参数列表
        **kwargs: 额外的配置覆盖
        
    Returns:
        Engine 实例
        
    示例:
        # 从 YAML 运行
        >>> engine = run_experiment('configs/experiments/dales_semseg.yaml')
        
        # 从字典运行
        >>> engine = run_experiment(config_dict={
        ...     'run': {'mode': 'train'},
        ...     'data': {...},
        ...     'model': {...}
        ... })
        
        # 覆盖配置
        >>> engine = run_experiment(
        ...     'configs/experiments/dales_semseg.yaml',
        ...     mode='test',
        ...     checkpoint_path='path/to/ckpt'
        ... )
    """
    # 推断任务类型
    if task_type is None:
        if config_path:
            task_type = infer_task_type(config_path)
        elif config_dict:
            task_type = config_dict.get('task', {}).get('type', 'semantic_segmentation')
        else:
            task_type = 'semantic_segmentation'
    
    # 获取 Engine 类
    EngineClass = get_engine_class(task_type)
    
    # 创建并运行 Engine
    if config_path:
        engine = EngineClass.from_config(
            config_path=config_path,
            cli_args=cli_args,
            **kwargs
        )
    elif config_dict:
        engine = EngineClass.from_dict(config_dict, **kwargs)
    else:
        engine = EngineClass(**kwargs)
    
    engine.run()
    return engine


__all__ = [
    'BaseEngine',
    'SemanticSegmentationEngine',
    'InstanceSegmentationEngine',
    'ObjectDetectionEngine',
    'get_engine_class',
    'infer_task_type',
    'run_experiment',
]
