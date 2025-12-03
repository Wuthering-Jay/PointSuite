"""
PointSuite 使用示例

展示三种使用方式:
1. YAML 配置文件运行
2. Python 编程式运行
3. 混合模式 (YAML + Python 覆盖)

运行方式:
    # 方式1: 通过 main.py 运行 YAML 配置
    python main.py --config configs/experiments/dales_semseg.yaml
    
    # 方式2: 运行此示例脚本
    python examples/run_experiment.py
"""

import os
import sys

# 确保项目根目录在 path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def example_yaml_config():
    """
    示例1: 使用 YAML 配置文件运行
    
    最简洁的方式，适合生产环境
    """
    from pointsuite.engine import SemanticSegmentationEngine
    
    # 从 YAML 配置创建引擎并运行
    engine = SemanticSegmentationEngine.from_config(
        'configs/experiments/dales_semseg.yaml'
    )
    engine.run()
    
    return engine


def example_yaml_with_override():
    """
    示例2: YAML 配置 + 命令行覆盖
    
    适合实验调参
    """
    from pointsuite.engine import SemanticSegmentationEngine
    
    # 从 YAML 配置创建，同时覆盖部分参数
    engine = SemanticSegmentationEngine.from_config(
        'configs/experiments/dales_semseg.yaml',
        cli_args=[
            '--run.mode', 'train',
            '--run.seed', '123',
            '--trainer.max_epochs', '50',
            '--data.batch_size', '8',
        ]
    )
    engine.run()
    
    return engine


def example_python_config():
    """
    示例3: 纯 Python 配置
    
    最灵活的方式，适合开发调试
    """
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    from pointsuite.data import BinPklDataModule
    from pointsuite.data.transforms import (
        CenterShift, RandomDropout, RandomRotate, RandomScale,
        RandomFlip, RandomJitter, Collect, ToTensor
    )
    from pointsuite.tasks import SemanticSegmentationTask
    from pointsuite.utils.callbacks import (
        SemanticPredictLasWriter, AutoEmptyCacheCallback, TextLoggingCallback
    )
    from pointsuite.utils.logger import setup_logger
    
    # ========================================================================
    # 配置
    # ========================================================================
    
    # 数据路径
    TRAIN_DATA = r"E:\data\DALES\dales_las\bin_logical\train"
    VAL_DATA = r"E:\data\DALES\dales_las\bin_logical\val"
    TEST_DATA = r"E:\data\DALES\dales_las\bin_logical\test"
    PREDICT_DATA = r"E:\data\DALES\dales_las\bin_logical\test"
    OUTPUT_DIR = r"./outputs/dales_python"
    
    # 类别配置
    CLASS_MAPPING = [1, 2, 3, 4, 5, 6, 7, 8]
    CLASS_NAMES = ['地面', '植被', '车辆', '卡车', '电线', '篱笆', '杆状物', '建筑']
    NUM_CLASSES = len(CLASS_MAPPING)
    IGNORE_LABEL = -1
    
    # 训练配置
    MAX_EPOCHS = 100
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3
    
    # 设置日志和随机种子
    setup_logger(OUTPUT_DIR)
    pl.seed_everything(42)
    
    # ========================================================================
    # 数据增强
    # ========================================================================
    
    train_transforms = [
        CenterShift(),
        RandomDropout(dropout_ratio=0.2, p=0.5),
        RandomRotate(angle=[-1, 1], axis='z', p=0.5),
        RandomScale(scale=[0.9, 1.1]),
        RandomFlip(p=0.5),
        RandomJitter(sigma=0.005, clip=0.02),
        Collect(keys=['coord', 'class'], feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    val_transforms = [
        CenterShift(),
        Collect(keys=['coord', 'class'], feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    test_transforms = val_transforms
    
    predict_transforms = [
        CenterShift(),
        Collect(
            keys=['coord', 'indices', 'bin_file', 'bin_path', 'pkl_path'],
            feat_keys={'feat': ['coord', 'echo']}
        ),
        ToTensor(),
    ]
    
    # ========================================================================
    # 创建 DataModule
    # ========================================================================
    
    datamodule = BinPklDataModule(
        train_data=TRAIN_DATA,
        val_data=VAL_DATA,
        test_data=TEST_DATA,
        predict_data=PREDICT_DATA,
        assets=['coord', 'class', 'echo'],
        class_mapping=CLASS_MAPPING,
        class_names=CLASS_NAMES,
        ignore_label=IGNORE_LABEL,
        batch_size=BATCH_SIZE,
        num_workers=4,
        mode='grid',
        use_dynamic_batch=True,
        max_points=125000,
        use_dynamic_batch_inference=True,
        max_points_inference=125000,
        use_weighted_sampler=True,
        train_loop=5,
        val_loop=5,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=test_transforms,
        predict_transforms=predict_transforms,
    )
    
    datamodule.setup(stage='fit')
    
    # ========================================================================
    # 创建模型
    # ========================================================================
    
    model_config = {
        'backbone': {
            'class_path': 'pointsuite.models.PointTransformerV2',
            'init_args': {
                'in_channels': 5,
                'patch_embed_depth': 1,
                'patch_embed_channels': 24,
                'patch_embed_groups': 6,
                'patch_embed_neighbours': 24,
                'enc_depths': (2, 2, 2, 2),
                'enc_channels': (48, 96, 192, 256),
                'enc_groups': (6, 12, 24, 32),
                'enc_neighbours': (32, 32, 32, 32),
                'dec_depths': (1, 1, 1, 1),
                'dec_channels': (24, 48, 96, 192),
                'dec_groups': (4, 6, 12, 24),
                'dec_neighbours': (32, 32, 32, 32),
                'grid_sizes': (1.5, 3.75, 9.375, 23.4375),
                'attn_qkv_bias': True,
                'pe_multiplier': False,
                'pe_bias': True,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.3,
                'unpool_backend': 'interp',
            }
        },
        'head': {
            'class_path': 'pointsuite.models.SegHead',
            'init_args': {
                'in_channels': 24,
                'num_classes': NUM_CLASSES
            }
        }
    }
    
    loss_configs = [
        {
            'name': 'ce_loss',
            'class_path': 'pointsuite.models.losses.CrossEntropyLoss',
            'init_args': {
                'ignore_index': IGNORE_LABEL,
                'weight': datamodule.train_dataset.class_weights,
            },
            'weight': 1.0,
        }
    ]
    
    metric_configs = [
        {
            'name': 'seg_metrics',
            'class_path': 'pointsuite.utils.metrics.semantic_segmentation.SegmentationMetrics',
            'init_args': {
                'num_classes': NUM_CLASSES,
                'ignore_index': IGNORE_LABEL,
                'class_names': CLASS_NAMES
            },
        }
    ]
    
    task = SemanticSegmentationTask(
        model_config=model_config,
        learning_rate=LEARNING_RATE,
        class_mapping=CLASS_MAPPING,
        class_names=CLASS_NAMES,
        ignore_label=IGNORE_LABEL,
        loss_configs=loss_configs,
        metric_configs=metric_configs,
    )
    
    # ========================================================================
    # 配置优化器
    # ========================================================================
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    import types
    task.configure_optimizers = types.MethodType(configure_optimizers, task)
    
    # ========================================================================
    # 创建 Trainer
    # ========================================================================
    
    callbacks = [
        ModelCheckpoint(
            monitor='mean_iou',
            mode='max',
            save_top_k=3,
            save_last=True,
            filename='{epoch:02d}-{mean_iou:.4f}',
            verbose=True
        ),
        EarlyStopping(
            monitor='mean_iou',
            patience=20,
            mode='max',
            verbose=True
        ),
        SemanticPredictLasWriter(
            output_dir=OUTPUT_DIR,
            save_logits=False,
            auto_infer_reverse_mapping=True
        ),
        TextLoggingCallback(log_interval=10),
        AutoEmptyCacheCallback(
            slowdown_threshold=3.0,
            absolute_threshold=1.5,
            clear_interval=0,
            warmup_steps=10,
            verbose=True
        ),
    ]
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision='16-mixed',
        log_every_n_steps=10,
        default_root_dir=OUTPUT_DIR,
        logger=False,
        callbacks=callbacks,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        enable_progress_bar=False,
        enable_model_summary=True,
    )
    
    # ========================================================================
    # 训练
    # ========================================================================
    
    trainer.fit(task, datamodule)
    trainer.test(task, datamodule, ckpt_path='best')
    trainer.predict(task, datamodule=datamodule, ckpt_path='best')
    
    return trainer


def example_engine_with_components():
    """
    示例4: 使用 Engine 但预设组件
    
    适合需要自定义组件的场景
    """
    import torch
    import pytorch_lightning as pl
    
    from pointsuite.engine import SemanticSegmentationEngine
    from pointsuite.data import BinPklDataModule
    from pointsuite.tasks import SemanticSegmentationTask
    from pointsuite.data.transforms import CenterShift, Collect, ToTensor
    
    # 创建自定义 DataModule
    datamodule = BinPklDataModule(
        train_data=r"E:\data\DALES\dales_las\bin_logical\train",
        val_data=r"E:\data\DALES\dales_las\bin_logical\val",
        assets=['coord', 'class', 'echo'],
        class_mapping=[1, 2, 3, 4, 5, 6, 7, 8],
        batch_size=4,
        train_transforms=[
            CenterShift(),
            Collect(keys=['coord', 'class'], feat_keys={'feat': ['coord', 'echo']}),
            ToTensor(),
        ],
        val_transforms=[
            CenterShift(),
            Collect(keys=['coord', 'class'], feat_keys={'feat': ['coord', 'echo']}),
            ToTensor(),
        ],
    )
    
    # 使用预设的 DataModule 创建 Engine
    engine = SemanticSegmentationEngine(
        config={
            'run': {
                'mode': 'train',
                'output_dir': './outputs/dales_custom',
                'seed': 42,
            },
            'model': {
                'backbone': {
                    'class_path': 'pointsuite.models.PointTransformerV2',
                    'init_args': {'in_channels': 5}
                },
                'head': {
                    'class_path': 'pointsuite.models.SegHead',
                    'init_args': {'in_channels': 24, 'num_classes': 8}
                }
            },
            'trainer': {
                'max_epochs': 10,
                'precision': '16-mixed',
            }
        },
        datamodule=datamodule,  # 使用预设的 DataModule
    )
    
    engine.run()
    return engine


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PointSuite 使用示例')
    parser.add_argument(
        '--example',
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help='选择示例 (1: YAML, 2: YAML+覆盖, 3: Python, 4: Engine+组件)'
    )
    
    args = parser.parse_args()
    
    if args.example == 1:
        print("运行示例1: YAML 配置")
        example_yaml_config()
    elif args.example == 2:
        print("运行示例2: YAML + 命令行覆盖")
        example_yaml_with_override()
    elif args.example == 3:
        print("运行示例3: 纯 Python 配置")
        example_python_config()
    elif args.example == 4:
        print("运行示例4: Engine + 预设组件")
        example_engine_with_components()
