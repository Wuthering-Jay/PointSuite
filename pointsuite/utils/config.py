"""
配置管理工具

提供 YAML 配置文件的加载、合并和解析功能
支持:
- 多级配置继承 (experiment -> data/model/trainer)
- 配置变量引用 (${xxx.yyy})
- 命令行参数覆盖
- 配置验证

示例配置结构:
    experiment.yaml:
        defaults:
            - data: dales.yaml
            - model: ptv2_small.yaml
            - trainer: default.yaml
        
        run:
            mode: train  # train/resume/finetune/test/predict
            seed: 42
            output_dir: ./outputs/dales1
"""

import os
import yaml
import re
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 YAML 文件
    
    Args:
        path: YAML 文件路径
        
    Returns:
        解析后的配置字典
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_yaml(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置字典
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典，override 会覆盖 base 中的值
    
    Args:
        base: 基础字典
        override: 覆盖字典
        
    Returns:
        合并后的字典
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def resolve_references(config: Dict, root: Dict = None) -> Dict:
    """
    解析配置中的变量引用 (${xxx.yyy} 格式)
    
    支持:
    - ${data.num_classes}: 引用 data 节点下的 num_classes
    - ${model.backbone.in_channels}: 多级引用
    - ${run.output_dir}/checkpoints: 字符串内引用
    
    Args:
        config: 配置字典
        root: 根配置字典（用于变量查找）
        
    Returns:
        解析后的配置字典
    """
    if root is None:
        root = config
    
    result = copy.deepcopy(config)
    
    def get_value(path: str, data: Dict) -> Any:
        """从配置中获取路径对应的值"""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"配置路径不存在: {path}")
        return value
    
    def resolve_string(s: str) -> Any:
        """解析字符串中的变量引用"""
        pattern = r'\$\{([^}]+)\}'
        
        # 检查是否整个字符串就是一个引用
        match = re.fullmatch(pattern, s)
        if match:
            # 整个字符串是引用，返回原始类型
            return get_value(match.group(1), root)
        
        # 否则进行字符串替换
        def replace(m):
            value = get_value(m.group(1), root)
            return str(value)
        
        return re.sub(pattern, replace, s)
    
    def resolve_value(value: Any) -> Any:
        """递归解析值"""
        if isinstance(value, str):
            return resolve_string(value)
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        else:
            return value
    
    return resolve_value(result)


class ConfigLoader:
    """
    配置加载器
    
    负责加载和合并多级配置文件，支持:
    - experiment.yaml 作为入口，引用 data/model/trainer 配置
    - 配置继承和覆盖
    - 变量引用解析
    - 命令行参数覆盖
    
    使用示例:
        >>> loader = ConfigLoader('configs/experiments/dales.yaml')
        >>> config = loader.load()
        >>> print(config['data']['num_classes'])
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        config_dir: Optional[Union[str, Path]] = None,
        cli_overrides: Optional[Dict[str, Any]] = None
    ):
        """
        初始化配置加载器
        
        Args:
            config_path: 主配置文件路径 (通常是 experiment.yaml)
            config_dir: 配置文件根目录，用于解析相对路径
                       如果为 None，使用 config_path 所在目录
            cli_overrides: 命令行参数覆盖
        """
        self.config_path = Path(config_path)
        self.config_dir = Path(config_dir) if config_dir else self.config_path.parent
        self.cli_overrides = cli_overrides or {}
        
    def load(self) -> Dict[str, Any]:
        """
        加载并合并所有配置
        
        处理流程:
        1. 加载主配置文件
        2. 解析 defaults 部分，加载子配置
        3. 合并所有配置
        4. 解析变量引用
        5. 应用命令行覆盖
        
        Returns:
            合并后的完整配置字典
        """
        # 1. 加载主配置
        main_config = load_yaml(self.config_path)
        
        # 2. 处理 defaults (子配置引用)
        defaults = main_config.pop('defaults', [])
        
        merged_config = {}
        for default in defaults:
            if isinstance(default, dict):
                # 格式: - data: dales.yaml
                for category, filename in default.items():
                    sub_config = self._load_sub_config(category, filename)
                    merged_config = deep_merge(merged_config, {category: sub_config})
            elif isinstance(default, str):
                # 格式: - base.yaml (直接继承)
                base_config = self._load_sub_config(None, default)
                merged_config = deep_merge(merged_config, base_config)
        
        # 3. 合并主配置 (覆盖 defaults)
        merged_config = deep_merge(merged_config, main_config)
        
        # 4. 解析变量引用
        merged_config = resolve_references(merged_config)
        
        # 5. 应用命令行覆盖
        merged_config = deep_merge(merged_config, self.cli_overrides)
        
        return merged_config
    
    def _load_sub_config(self, category: Optional[str], filename: str) -> Dict[str, Any]:
        """
        加载子配置文件
        
        Args:
            category: 配置类别 (data/model/trainer)，用于确定子目录
            filename: 配置文件名
            
        Returns:
            子配置字典
        """
        if category:
            # 在 configs/{category}/ 目录下查找
            sub_path = self.config_dir / category / filename
        else:
            # 在配置目录下直接查找
            sub_path = self.config_dir / filename
        
        if not sub_path.exists():
            # 尝试相对于主配置文件的路径
            sub_path = self.config_path.parent / filename
        
        if not sub_path.exists():
            raise FileNotFoundError(f"子配置文件不存在: {sub_path}")
        
        return load_yaml(sub_path)


def parse_cli_args(args: List[str]) -> Dict[str, Any]:
    """
    解析命令行参数为配置覆盖字典
    
    支持格式:
    - --data.batch_size=16
    - --model.backbone.in_channels 5
    - --run.mode train
    
    Args:
        args: 命令行参数列表
        
    Returns:
        配置覆盖字典
    """
    overrides = {}
    i = 0
    
    while i < len(args):
        arg = args[i]
        
        if arg.startswith('--'):
            key = arg[2:]
            
            # 处理 --key=value 格式
            if '=' in key:
                key, value = key.split('=', 1)
            elif i + 1 < len(args) and not args[i + 1].startswith('--'):
                # 处理 --key value 格式
                i += 1
                value = args[i]
            else:
                # 布尔标志
                value = True
            
            # 尝试转换类型
            value = _convert_value(value)
            
            # 设置嵌套键
            _set_nested_key(overrides, key, value)
        
        i += 1
    
    return overrides


def _convert_value(value: str) -> Any:
    """
    尝试将字符串值转换为适当的类型
    
    Args:
        value: 字符串值
        
    Returns:
        转换后的值
    """
    if not isinstance(value, str):
        return value
    
    # 布尔值
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.lower() == 'none':
        return None
    
    # 整数
    try:
        return int(value)
    except ValueError:
        pass
    
    # 浮点数
    try:
        return float(value)
    except ValueError:
        pass
    
    # 列表 (JSON 格式)
    if value.startswith('[') and value.endswith(']'):
        try:
            import json
            return json.loads(value)
        except:
            pass
    
    return value


def _set_nested_key(d: Dict, key: str, value: Any) -> None:
    """
    设置嵌套字典的键值
    
    Args:
        d: 字典
        key: 点分隔的键路径 (如 'data.batch_size')
        value: 要设置的值
    """
    keys = key.split('.')
    current = d
    
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    current[keys[-1]] = value


@dataclass
class ExperimentConfig:
    """
    实验配置数据类
    
    提供对配置的结构化访问，同时保持灵活性
    """
    # 运行配置
    mode: str = 'train'  # train/resume/finetune/test/predict
    seed: int = 42
    output_dir: str = './outputs'
    checkpoint_path: Optional[str] = None
    
    # 数据配置
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 模型配置
    model: Dict[str, Any] = field(default_factory=dict)
    
    # 训练器配置
    trainer: Dict[str, Any] = field(default_factory=dict)
    
    # 任务配置
    task: Dict[str, Any] = field(default_factory=dict)
    
    # 原始配置
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ExperimentConfig':
        """
        从字典创建配置对象
        
        Args:
            config: 配置字典
            
        Returns:
            ExperimentConfig 实例
        """
        run_config = config.get('run', {})
        
        return cls(
            mode=run_config.get('mode', 'train'),
            seed=run_config.get('seed', 42),
            output_dir=run_config.get('output_dir', './outputs'),
            checkpoint_path=run_config.get('checkpoint_path'),
            data=config.get('data', {}),
            model=config.get('model', {}),
            trainer=config.get('trainer', {}),
            task=config.get('task', {}),
            _raw=config
        )
    
    @classmethod
    def from_yaml(
        cls,
        config_path: Union[str, Path],
        cli_overrides: Optional[Dict[str, Any]] = None
    ) -> 'ExperimentConfig':
        """
        从 YAML 文件加载配置
        
        Args:
            config_path: YAML 配置文件路径
            cli_overrides: 命令行参数覆盖
            
        Returns:
            ExperimentConfig 实例
        """
        loader = ConfigLoader(config_path, cli_overrides=cli_overrides)
        config = loader.load()
        return cls.from_dict(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return {
            'run': {
                'mode': self.mode,
                'seed': self.seed,
                'output_dir': self.output_dir,
                'checkpoint_path': self.checkpoint_path,
            },
            'data': self.data,
            'model': self.model,
            'trainer': self.trainer,
            'task': self.task,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        保存配置到文件
        
        Args:
            path: 保存路径
        """
        save_yaml(self.to_dict(), path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分隔的路径
        
        Args:
            key: 配置键 (如 'data.batch_size')
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._raw
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


def create_experiment_config(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    cli_args: Optional[List[str]] = None,
    **kwargs
) -> ExperimentConfig:
    """
    创建实验配置的便捷函数
    
    支持多种配置来源:
    1. YAML 文件
    2. Python 字典
    3. 命令行参数
    4. 关键字参数
    
    优先级 (从低到高):
    YAML 文件 < Python 字典 < 命令行参数 < 关键字参数
    
    Args:
        config_path: YAML 配置文件路径
        config_dict: 配置字典
        cli_args: 命令行参数列表
        **kwargs: 额外的配置覆盖
        
    Returns:
        ExperimentConfig 实例
        
    示例:
        >>> # 从 YAML 文件加载
        >>> config = create_experiment_config('configs/experiments/dales.yaml')
        
        >>> # 从 YAML 加载并覆盖
        >>> config = create_experiment_config(
        ...     'configs/experiments/dales.yaml',
        ...     cli_args=['--run.mode', 'test', '--data.batch_size', '8']
        ... )
        
        >>> # 纯 Python 配置
        >>> config = create_experiment_config(config_dict={
        ...     'run': {'mode': 'train', 'seed': 42},
        ...     'data': {...},
        ...     'model': {...}
        ... })
    """
    config = {}
    
    # 1. 从 YAML 加载
    if config_path is not None:
        cli_overrides = parse_cli_args(cli_args) if cli_args else {}
        loader = ConfigLoader(config_path, cli_overrides=cli_overrides)
        config = loader.load()
    
    # 2. 合并 Python 字典
    if config_dict is not None:
        config = deep_merge(config, config_dict)
    
    # 3. 应用关键字参数
    if kwargs:
        for key, value in kwargs.items():
            _set_nested_key(config, key, value)
    
    return ExperimentConfig.from_dict(config)
