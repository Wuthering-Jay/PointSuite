"""快速验证调试脚本"""
import sys
sys.path.insert(0, r'E:\code\PointSuite')

import torch
import pytorch_lightning as pl
from pointsuite.data import BinPklDataModule1
from pointsuite.tasks import SemanticSegmentationTask
from pointsuite.data.transforms import *

# 简化配置
CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
CLASS_NAMES = ['地面', '植被', '车辆', '卡车', '电线', '篱笆', '杆状物', '建筑']
NUM_CLASSES = 8

val_transforms = [
    CenterShift(),
    Collect(keys=['coord', 'class'], feat_keys={'feat': ['coord', 'echo']}),
    ToTensor(),
]

datamodule = BinPklDataModule1(
    train_data=r"E:\data\DALES\dales_las\bin_logical\train",
    val_data=r"E:\data\DALES\dales_las\bin_logical\test",
    assets=['coord', 'echo', 'class'],
    class_mapping=CLASS_MAPPING,
    ignore_label=-1,
    batch_size=4,
    num_workers=0,
    mode='voxel',
    max_loops=10,
    use_dynamic_batch=True,
    max_points=150000,
    use_dynamic_batch_inference=True,
    max_points_inference=150000,
    train_loop=1,
    val_loop=1,
    use_weighted_sampler=False,  # 简化
    train_transforms=val_transforms,
    val_transforms=val_transforms,
)

# 模型配置
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
            'unpool_backend': "interp",
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

# 损失函数配置
loss_configs = [
    {
        "name": "ce_loss",
        "class_path": "pointsuite.models.losses.CrossEntropyLoss",
        "init_args": {"ignore_index": -1},
        "weight": 1.0,
    },
]

# 指标配置
metric_configs = [
    {
        "name": "seg_metrics",
        "class_path": "pointsuite.utils.metrics.semantic_segmentation.SegmentationMetrics",
        "init_args": {
            "num_classes": NUM_CLASSES, 
            "ignore_index": -1,
            "class_names": CLASS_NAMES
        },
    },
]

from pointsuite.utils.callbacks import SemanticPredictLasWriter1, AutoEmptyCacheCallback, TextLoggingCallback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

task = SemanticSegmentationTask(
    model_config=model_config,
    learning_rate=1e-3,
    loss_configs=loss_configs,
    metric_configs=metric_configs,
)

print(f"\n[DEBUG] task.val_metrics: {list(task.val_metrics.keys())}")

callbacks = [
    ModelCheckpoint(
        monitor='mean_iou', 
        mode='max', 
        save_top_k=3,
        save_last=True,
        filename='dales1-{epoch:02d}-{mean_iou:.4f}', 
        verbose=True
    ),
    EarlyStopping(
        monitor='mean_iou', 
        patience=20, 
        mode='max', 
        verbose=True, 
        check_on_train_epoch_end=False
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

# 极简 Trainer - 模拟 train_dales1.py 的配置
trainer = pl.Trainer(
    max_epochs=1,
    limit_train_batches=5,  # 只训练 5 个 batch
    limit_val_batches=3,    # 只验证 3 个 batch
    devices=1,
    accelerator='gpu',
    precision="16-mixed",
    logger=False,
    enable_progress_bar=False,  # 禁用进度条，与 train_dales1.py 一致
    num_sanity_val_steps=2,  # 与 train_dales1.py 一致
    callbacks=callbacks,
)

print("\n开始训练...")
trainer.fit(task, datamodule)
print("\n训练完成!")
