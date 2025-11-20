"""
DALES 数据集训练脚本（纯 Python 配置）

新功能演示：
- ✅ 自动类别权重计算和加权采样
- ✅ 中文类别名称支持（验证日志 + hparams.yaml）
- ✅ CSV 日志记录（文本格式，便于查看）
- ✅ 动态批次采样
- ✅ 多文件 LAS 预测支持
- ✅ 梯度累积（模拟大batch训练）

梯度累积说明：
- 原理：每N个batch计算一次梯度，累积N次后才更新参数
- 优势：在显存受限时模拟更大的batch size
- 与动态batch完全兼容：
  * 动态batch控制每个batch的点数（max_points）
  * 梯度累积控制更新频率（accumulate_grad_batches）
  * 等效batch = max_points × accumulate_grad_batches

配置建议：
- 小显存(8GB):  max_points=100K, accumulate=4  → 400K点/更新
- 中显存(16GB): max_points=150K, accumulate=2  → 300K点/更新
- 大显存(24GB): max_points=200K, accumulate=1  → 200K点/更新

推理加速优化：
- 多进程加载: NUM_WORKERS=4 (避免数据加载成为瓶颈)
- 大batch推理: max_points_inference=600K (无梯度，可用3-4倍训练batch)
- 自动优化: Lightning 2.5+ 默认使用 inference_mode (比 no_grad 更快)
- TF32加速: 全局启用，训练和推理都生效
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointsuite.data import BinPklDataModule
from pointsuite.data.transforms import (
    RandomRotate, RandomScale, RandomFlip, RandomJitter,
    AddExtremeOutliers, Collect, ToTensor
)
from pointsuite.tasks import SemanticSegmentationTask
from pointsuite.models import PointTransformerV2, SegHead
from pointsuite.utils.callbacks import SegmentationWriter
from pointsuite.utils.progress_bar import CustomProgressBar


def main():
    # ========================================================================
    # 配置
    # ========================================================================
    
    # 数据
    TRAIN_DATA = r"E:\data\DALES\dales_las\bin\test"
    TEST_DATA = r"E:\data\DALES\dales_las\bin\test"
    OUTPUT_DIR = r"E:\data\DALES\dales_las\bin\result"
    
    CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    CLASS_NAMES = ['地面', '植被', '车辆', '卡车', '电线', '篱笆', '杆状物', '建筑']
    NUM_CLASSES = 8
    IGNORE_LABEL = -1
    
    # 训练
    MAX_EPOCHS = 5
    BATCH_SIZE = 4
    NUM_WORKERS = 4  # 多进程数据加载，加速训练和推理
    LEARNING_RATE = 0.001
    MAX_POINTS = 250000
    MAX_POINTS_INFERENCE = 500000  # 推理时使用更大batch（无梯度，显存占用少）
    ACCUMULATE_GRAD_BATCHES = 4  # 梯度累积：每4个batch更新一次参数，模拟更大batch
    
    pl.seed_everything(42)
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("\n" + "=" * 80)
    print(f"DALES 语义分割训练 - {NUM_CLASSES} 类")
    print("=" * 80)
    
    # ========================================================================
    # 数据增强
    # ========================================================================
    
    train_transforms = [
        RandomRotate(angle=[-180, 180], axis='z', p=0.5),
        RandomScale(scale=[0.9, 1.1]),
        RandomFlip(p=0.5),
        RandomJitter(sigma=0.01, clip=0.05),
        AddExtremeOutliers(
            ratio=0.001, height_range=(-10, 100), height_mode='bimodal',
            intensity_range=(0, 1), color_value=(128, 128, 128),
            class_label='ignore', p=0.5
        ),
        Collect(keys=['coord', 'class'], offset_key={'offset': 'coord'},
                feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    val_transforms = [
        Collect(keys=['coord', 'class'], offset_key={'offset': 'coord'},
                feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    predict_transforms = [
        Collect(keys=['coord', 'indices', 'bin_file', 'bin_path', 'pkl_path'],
                offset_key={'offset': 'coord'}, feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    # ========================================================================
    # DataModule
    # ========================================================================
    
    datamodule = BinPklDataModule(
        train_data=TRAIN_DATA,
        val_data=TRAIN_DATA,
        test_data=TEST_DATA,
        predict_data=TEST_DATA,
        assets=['coord', 'echo', 'class'],
        class_mapping=CLASS_MAPPING,
        class_names=CLASS_NAMES,
        ignore_label=IGNORE_LABEL,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        use_dynamic_batch=True,
        max_points=MAX_POINTS,
        use_dynamic_batch_inference=True,
        max_points_inference=MAX_POINTS_INFERENCE,
        use_weighted_sampler=True,  # 启用加权采样
        class_weights=None,  # None = 自动从数据集计算
        train_loop=1,
        val_loop=1,
        test_loop=2,
        predict_loop=2,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=val_transforms,
        predict_transforms=predict_transforms,
    )
    
    datamodule.print_info()
    
    # ========================================================================
    # 模型
    # ========================================================================
    
    backbone = PointTransformerV2(
        in_channels=5,
        patch_embed_depth=1,
        patch_embed_channels=24,
        patch_embed_groups=6,
        patch_embed_neighbours=24,
        enc_depths=(1, 1, 1, 1),
        enc_channels=(48, 96, 192, 256),
        enc_groups=(6, 12, 24, 32),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(24, 48, 96, 192),
        dec_groups=(4, 6, 12, 24),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(1, 2.5, 7.5, 15),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        unpool_backend="interp",
    )
    
    head = SegHead(in_channels=24, num_classes=NUM_CLASSES)
    
    # 损失函数（不使用类别权重，让加权采样处理类别不平衡）
    loss_configs = [
        {
            "name": "ce_loss",
            "class_path": "pointsuite.models.losses.CrossEntropyLoss",
            "init_args": {"ignore_index": IGNORE_LABEL},
            "weight": 1.0,
        },
        {
            "name": "lovasz_loss",
            "class_path": "pointsuite.models.losses.LovaszLoss",
            "init_args": {"mode": "multiclass", "ignore_index": IGNORE_LABEL},
            "weight": 1.0,
        },
    ]
    
    metric_configs = [
        {
            "name": "overall_accuracy",
            "class_path": "torchmetrics.classification.MulticlassAccuracy",
            "init_args": {"num_classes": NUM_CLASSES, "ignore_index": IGNORE_LABEL, "average": "micro"},
        },
        {
            "name": "mean_iou",
            "class_path": "torchmetrics.classification.MulticlassJaccardIndex",
            "init_args": {"num_classes": NUM_CLASSES, "ignore_index": IGNORE_LABEL, "average": "macro"},
        },
    ]
    
    task = SemanticSegmentationTask(
        backbone=backbone,
        head=head,
        learning_rate=LEARNING_RATE,
        class_mapping=CLASS_MAPPING,
        class_names=CLASS_NAMES,
        loss_configs=loss_configs,
        metric_configs=metric_configs,
    )
    
    # 优化器
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
    
    import types
    task.configure_optimizers = types.MethodType(configure_optimizers, task)
    
    # ========================================================================
    # 回调和 Trainer
    # ========================================================================
    
    callbacks = [
        ModelCheckpoint(monitor='mean_iou', mode='max', save_top_k=3,
                       filename='dales-{epoch:02d}-{mean_iou:.4f}', verbose=True),
        EarlyStopping(monitor='mean_iou', patience=20, mode='max', verbose=True),
        LearningRateMonitor(logging_interval='step'),
        SegmentationWriter(output_dir=OUTPUT_DIR, save_logits=False, auto_infer_reverse_mapping=True),
        CustomProgressBar(refresh_rate=1),  # 自定义进度条
    ]
    
    csv_logger = CSVLogger(save_dir='./outputs/dales', name='csv_logs', version=None)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision="32-true",
        log_every_n_steps=10,
        default_root_dir='./outputs/dales',
        logger=[csv_logger],
        callbacks=callbacks,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,  # 梯度累积
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=2,
    )
    
    print(f"\n设备: {trainer.accelerator} | 精度: {trainer.precision} | Epochs: {MAX_EPOCHS}")
    print(f"梯度累积: {ACCUMULATE_GRAD_BATCHES} batches | 等效batch: ~{MAX_POINTS * ACCUMULATE_GRAD_BATCHES / 1000:.0f}K points/update")
    print(f"推理优化: max_points={MAX_POINTS/1000:.0f}K (训练) → {MAX_POINTS_INFERENCE/1000:.0f}K (推理) | workers={NUM_WORKERS}")
    
    # ========================================================================
    # 训练流程
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    trainer.fit(task, datamodule)
    
    print("\n" + "=" * 80)
    print("开始测试")
    print("=" * 80)
    trainer.test(task, datamodule)
    
    print("\n" + "=" * 80)
    print("开始预测")
    print("=" * 80)
    trainer.predict(task, datamodule)
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"检查点: {trainer.default_root_dir}")
    print(f"预测结果: {OUTPUT_DIR}")
    print(f"最佳模型: {trainer.checkpoint_callback.best_model_path}")
    print(f"最佳 MeanIoU: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    main()
