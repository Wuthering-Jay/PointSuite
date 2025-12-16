"""
PointSuite 主入口

支持多种运行方式:
1. YAML 配置文件运行
2. 命令行参数覆盖
3. Python 编程式调用

使用示例:
    # YAML 配置运行
    python main.py --config configs/experiments/dales_semseg.yaml
    
    # 带命令行覆盖
    python main.py --config configs/experiments/dales_semseg.yaml --run.mode test
    
    # 编程式调用
    from pointsuite.engine import run_experiment
    run_experiment('configs/experiments/dales_semseg.yaml', mode='train')
"""

import os
import sys
import argparse

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='PointSuite - 点云深度学习框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从配置文件运行
  python main.py --config configs/experiments/dales_semseg.yaml
  
  # 测试模式
  python main.py --config configs/experiments/dales_semseg.yaml --run.mode test --run.checkpoint_path path/to/ckpt
  
  # 覆盖配置
  python main.py --config configs/experiments/dales_semseg.yaml --data.batch_size 8 --trainer.max_epochs 50
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/experiments/dales_semseg_standalone.yaml',
        help='实验配置文件路径 (YAML)'
    )
    
    parser.add_argument(
        '--task', '-t',
        type=str,
        default=None,
        choices=['semantic_segmentation', 'instance_segmentation', 'object_detection'],
        help='任务类型 (如果不指定则从配置文件名推断)'
    )
    
    # 解析已知参数，剩余参数作为配置覆盖
    args, unknown = parser.parse_known_args()
    
    return args, unknown


def main():
    """主入口函数"""
    args, cli_overrides = parse_args()
    
    from pointsuite.utils.logger import log_info, Colors
    
    log_info(f"配置文件: {Colors.CYAN}{args.config}{Colors.RESET}")
    if cli_overrides:
        log_info(f"命令行覆盖: {Colors.YELLOW}{cli_overrides}{Colors.RESET}")
    
    from pointsuite.engine import run_experiment
    
    run_experiment(
        config_path=args.config,
        task_type=args.task,
        cli_args=cli_overrides
    )


if __name__ == '__main__':
    main()
