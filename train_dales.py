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
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# 忽略 Windows 下 num_workers 的警告
warnings.filterwarnings("ignore", ".*does not have many workers.*")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointsuite.data import BinPklDataModule
from pointsuite.data.transforms import *
from pointsuite.tasks import SemanticSegmentationTask
from pointsuite.utils.callbacks import SemanticPredictLasWriter, AutoEmptyCacheCallback, TextLoggingCallback
from pointsuite.utils.logger import setup_logger
# from pointsuite.utils.progress_bar import CustomProgressBar


def main():
    # ========================================================================
    # 配置
    # ========================================================================
    
    # 数据
    TRAIN_DATA = r"E:\data\DALES\dales_las\bin\train"
    TEST_DATA = r"E:\data\DALES\dales_las\bin\test"
    OUTPUT_DIR = r"E:\data\DALES\dales_las\bin\result"
    
    # 设置日志 (捕获所有终端输出)
    log_file_path = setup_logger(OUTPUT_DIR)
    
    CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    CLASS_NAMES = ['地面', '植被', '车辆', '卡车', '电线', '篱笆', '杆状物', '建筑']
    NUM_CLASSES = 8
    IGNORE_LABEL = -1
    
    # 训练
    MAX_EPOCHS = 5
    BATCH_SIZE = 4 
    NUM_WORKERS = 0  # 多进程数据加载，加速训练和推理
    LEARNING_RATE = 1e-3
    MAX_POINTS = 120000
    MAX_POINTS_INFERENCE = 120000  # 推理时使用更大batch（无梯度，显存占用少）
    ACCUMULATE_GRAD_BATCHES = 2  # 梯度累积：每4个batch更新一次参数，模拟更大batch
    
    pl.seed_everything(42)
    
    # if torch.cuda.is_available():
    #     torch.set_float32_matmul_precision('high')
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True
    
    print("\n" + "=" * 80)
    print(f"DALES 语义分割训练 - {NUM_CLASSES} 类")
    print("=" * 80)
    
    # ========================================================================
    # 数据增强
    # ========================================================================
    
    train_transforms = [
        CenterShift(),  # 中心化坐标
        RandomDropout(dropout_ratio=0.2, p=0.5),
        RandomRotate(angle=[-1, 1], axis='z', p=0.5),
        RandomScale(scale=[0.9, 1.1]),
        RandomFlip(p=0.5),
        RandomJitter(sigma=0.005, clip=0.02),
        # AddExtremeOutliers(
        #     ratio=0.001, height_range=(-10, 100), height_mode='bimodal',
        #     intensity_range=(0, 1), color_value=(128, 128, 128),
        #     class_label='ignore', p=0.5
        # ),
        Collect(keys=['coord', 'class'],
                feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    val_transforms = [
        CenterShift(),  # 中心化坐标
        RandomDropout(dropout_ratio=0.2, p=0.5),
        RandomRotate(angle=[-1, 1], axis='z', p=0.5),
        RandomScale(scale=[0.9, 1.1]),
        RandomFlip(p=0.5),
        RandomJitter(sigma=0.005, clip=0.02),
        # AddExtremeOutliers(
        #     ratio=0.001, height_range=(-10, 100), height_mode='bimodal',
        #     intensity_range=(0, 1), color_value=(128, 128, 128),
        #     class_label='ignore', p=0.5
        # ),
        Collect(keys=['coord', 'class'],
                feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    predict_transforms = [
        CenterShift(),  # 中心化坐标
        Collect(keys=['coord', 'indices', 'bin_file', 'bin_path', 'pkl_path'],
                feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    # ========================================================================
    # DataModule
    # ========================================================================
    
    datamodule = BinPklDataModule(
        train_data=TRAIN_DATA,
        val_data=TEST_DATA,
        test_data=None,
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
        test_loop=1,
        predict_loop=1,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=val_transforms,
        predict_transforms=predict_transforms,
    )
    
    # 🔥 手动 setup 以便访问数据集并计算权重
    datamodule.setup(stage='fit')
    
    # ========================================================================
    # 模型
    # ========================================================================
    
    # 使用配置字典定义模型结构，而不是直接实例化对象
    # 这样可以避免 PyTorch Lightning 的 "attribute is already saved" 警告
    # 并且让 checkpoint 更轻量、更规范
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

    
    loss_configs = [
        {
            "name": "ce_loss",
            "class_path": "pointsuite.models.losses.CrossEntropyLoss",
            "init_args": {
                "ignore_index": IGNORE_LABEL,
                "weight": datamodule.train_dataset.class_weights, # 直接调用属性
            },
            "weight": 1.0,
        },
        {
            "name": "lac_loss",
            "class_path": "pointsuite.models.losses.LACLoss",
            "init_args": {"k_neighbors":16, "ignore_index": IGNORE_LABEL},
            "weight": 1.0,
        },
    ]
    
    metric_configs = [
        {
            "name": "seg_metrics",
            "class_path": "pointsuite.utils.metrics.semantic_segmentation.SegmentationMetrics",
            "init_args": {
                "num_classes": NUM_CLASSES, 
                "ignore_index": IGNORE_LABEL,
                "class_names": CLASS_NAMES
            },
        },
    ]
    
    task = SemanticSegmentationTask(
        model_config=model_config,  # 传入配置字典
        learning_rate=LEARNING_RATE,
        class_mapping=CLASS_MAPPING,
        class_names=CLASS_NAMES,
        ignore_label=IGNORE_LABEL,
        loss_configs=loss_configs,
        metric_configs=metric_configs,
    )
    
    # 优化器
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay= 1e-4)
        
        # 使用 Trainer 的 estimated_stepping_batches 自动获取总优化步数
        # 这会自动考虑 max_epochs, dataloader 长度以及 accumulate_grad_batches
        # 避免了手动估算 steps_per_epoch
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps, 
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step", 
                "frequency": 1
            }
        }
    
    import types
    task.configure_optimizers = types.MethodType(configure_optimizers, task)
    
    # ========================================================================
    # 回调和 Trainer
    # ========================================================================
    
    callbacks = [
        # 保存最佳模型 (Top 3) 和 最后一个模型 (last.ckpt)
        ModelCheckpoint(
            monitor='mean_iou', 
            mode='max', 
            save_top_k=1,
            save_last=True,  # 🔥 保存最后一个模型为 last.ckpt
            filename='dales-{epoch:02d}-{mean_iou:.4f}', 
            verbose=True
        ),
        EarlyStopping(monitor='mean_iou', patience=20, mode='max', verbose=True, 
                     check_on_train_epoch_end=False),  # 🔥 修复：在验证结束时检查，而不是训练结束时
        # LearningRateMonitor(logging_interval='step'), # ❌ 移除：因为禁用了 logger，无法使用此回调
        SemanticPredictLasWriter(output_dir=OUTPUT_DIR, save_logits=False, auto_infer_reverse_mapping=True),
        # CustomProgressBar(refresh_rate=1),  # 自定义进度条
        TextLoggingCallback(log_interval=10), # 静态文本日志 (不再需要 log_file 参数，因为全局捕获了)
        AutoEmptyCacheCallback(slowdown_threshold=3.0, absolute_threshold=1.5, clear_interval=0, warmup_steps=10, verbose=True),  # 自动清理显存
    ]
    
    # 移除 CSVLogger 和 TensorBoardLogger，改用 TextLoggingCallback 记录到文件
    # csv_logger = CSVLogger(save_dir='./outputs/dales', name='csv_logs', version=None)
    # tb_logger = TensorBoardLogger(save_dir='./outputs/dales', name='tb_logs', version=None)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision="16-mixed",
        log_every_n_steps=10,
        default_root_dir='./outputs/dales',
        logger=False, # 🔥 禁用默认 Logger
        callbacks=callbacks,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,  # 梯度累积
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_progress_bar=False, # 禁用默认进度条
        enable_model_summary=True,
        num_sanity_val_steps=2,
    )
    
    print(f"\n设备: {trainer.accelerator} | 精度: {trainer.precision} | Epochs: {MAX_EPOCHS}")
    print(f"梯度累积: {ACCUMULATE_GRAD_BATCHES} batches | 等效batch: ~{MAX_POINTS * ACCUMULATE_GRAD_BATCHES / 1000:.0f}K points/update")
    print(f"推理优化: max_points={MAX_POINTS/1000:.0f}K (训练) → {MAX_POINTS_INFERENCE/1000:.0f}K (推理) | workers={NUM_WORKERS}")
    # ========================================================================
    # 训练流程
    # ========================================================================
    
    # 1. 断点恢复 (Resume): 恢复完整的训练状态 (模型权重 + 优化器 + Epoch)
    #    用于训练中断后继续训练
    #    例如: ckpt_path = "outputs/dales/csv_logs/version_0/checkpoints/last.ckpt"
    ckpt_path = None 
    
    # 2. 预训练权重 (Pretrained): 仅加载模型权重，从头开始训练 (重置 Epoch 和 优化器)
    #    用于微调 (Fine-tuning) 或迁移学习
    #    例如: pretrained_path = "outputs/dales/csv_logs/version_0/checkpoints/best.ckpt"
    pretrained_path = None

    # 加载预训练权重 (如果指定)
    if pretrained_path is not None and ckpt_path is None:
        print(f"\n[Info] 加载预训练权重: {pretrained_path}")
        # strict=False 允许权重不完全匹配 (例如微调时修改了 head)
        # 注意: 这里我们加载权重到当前的 task 实例中
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        
        # 处理可能的 key 不匹配 (例如有些 checkpoint 有 'model.' 前缀)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v # 去掉 'model.' 前缀
            else:
                new_state_dict[k] = v
                
        missing_keys, unexpected_keys = task.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"  - 缺失的键 (将随机初始化): {missing_keys[:5]} ...")
        if unexpected_keys:
            print(f"  - 未预期的键 (将被忽略): {unexpected_keys[:5]} ...")
        print(f"  - 权重加载完成 (Epoch 将从 0 开始)")

    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    trainer.fit(task, datamodule, ckpt_path=ckpt_path)
    
    if datamodule.test_data is not None:
        print("\n" + "=" * 80)
        print("开始测试")
        print("=" * 80)
        trainer.test(task, datamodule)
    else:
        print("\n" + "=" * 80)
        print("跳过测试 (未提供测试数据)")
        print("=" * 80)
    
    if datamodule.predict_data is not None:
        print("\n" + "=" * 80)
        print("开始预测")
        print("=" * 80)
        # 🔥 显式调用 predict
        # 使用 "best" 自动加载最佳 checkpoint
        trainer.predict(task, datamodule=datamodule, ckpt_path="best")
        
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"检查点: {trainer.default_root_dir}")
    print(f"预测结果: {OUTPUT_DIR}")
    
    if trainer.checkpoint_callback.best_model_path:
        print(f"最佳模型: {trainer.checkpoint_callback.best_model_path}")
    
    if trainer.checkpoint_callback.best_model_score is not None:
        print(f"最佳 MeanIoU: {trainer.checkpoint_callback.best_model_score:.4f}")
    else:
        print("最佳 MeanIoU: N/A (未生成或未记录)")

if __name__ == '__main__':
    main()
