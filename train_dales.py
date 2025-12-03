"""
DALES 数据集训练脚本 (逻辑索引格式 - tile_las1.py)

适配新的 bin+pkl 逻辑索引数据格式：
- 🔥 体素模式 (voxel): 每个 segment 根据体素化索引采样
  - train/val: 每个体素随机取 1 个点，每 epoch 不同
  - test/predict: 模运算采样确保全覆盖
- 📍 局部坐标: 自动转换为 0~50m 范围，避免 float32 精度损失
- 🎯 完美批次控制: 固定体素数 = 固定显存占用

功能特性：
- ✅ 自动类别权重计算和加权采样
- ✅ 中文类别名称支持
- ✅ 动态批次采样 (按体素数控制)
- ✅ 多文件 LAS 预测支持
- ✅ 梯度累积
- ✅ 局部坐标自动转换

配置建议：
- 小显存(8GB):  max_points=80K,  accumulate=4  → 320K体素/更新
- 中显存(16GB): max_points=120K, accumulate=2  → 240K体素/更新
- 大显存(24GB): max_points=160K, accumulate=1  → 160K体素/更新
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
from pointsuite.utils.logger import setup_logger, Colors, print_header, print_section, print_config


def main():
    # ========================================================================
    # 配置
    # ========================================================================
    
    # 数据路径 (使用 tile_las1.py 生成的逻辑索引格式)
    TRAIN_DATA = r"E:\data\DALES\dales_las\bin_logical\test"
    VAL_DATA = r"E:\data\DALES\dales_las\bin_logical\test"   # 使用 test 作为验证
    TEST_DATA = r"E:\data\DALES\dales_las\bin_logical\test"
    PREDICT_DATA = r"E:\data\DALES\dales_las\bin_logical\test"
    OUTPUT_DIR = r"E:\data\DALES\dales_las\bin_logical\result"
    
    # 设置日志 (捕获所有终端输出)
    log_file_path = setup_logger(OUTPUT_DIR)
    
    # 类别配置 (DALES 8类)
    # 🔥 支持两种格式：
    #   - 列表形式：[1, 2, 3, 4, 5, 6, 7, 8] 自动映射为 [0, 1, 2, ..., 7]
    #   - 字典形式：{1: 0, 2: 1, ...} 显式指定映射
    CLASS_MAPPING = [1, 2, 3, 4, 5, 6, 7, 8]  # 等价于 {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
    CLASS_NAMES = ['地面', '植被', '车辆', '卡车', '电线', '篱笆', '杆状物', '建筑']
    NUM_CLASSES = len(CLASS_MAPPING)  # 自动计算类别数
    IGNORE_LABEL = -1
    
    # 训练配置
    MAX_EPOCHS = 1
    BATCH_SIZE = 4 
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-3
    
    # 🔥 内存优化：禁用 persistent_workers 避免多进程缓存复制
    # 每个 worker 进程会复制 dataset 对象，包括其中的缓存
    # persistent_workers=True 会让这些进程常驻，累积大量内存
    PERSISTENT_WORKERS = True
    
    # 体素模式配置
    MODE = 'grid'           # 'grid' 或 'full'
    MAX_LOOPS = None         # test/predict 时最大采样轮数 (None = 每轮1点，和训练时一致)
    MAX_POINTS = 125000     # 每批次最大点数 (体素模式下 = 体素数)
    MAX_POINTS_INFERENCE = 125000  # 推理时批次点数
    
    # ========================================================================
    # 🔥 运行模式配置
    # ========================================================================
    # RUN_MODE 控制运行流程：
    #   'train'    : 从头训练 → 验证 → 测试 → 预测
    #   'resume'   : 从 checkpoint 继续训练 → 验证 → 测试 → 预测 (恢复 optimizer/epoch 状态)
    #   'finetune' : 加载预训练权重，从头训练 → 验证 → 测试 → 预测 (只加载模型权重)
    #   'test'     : 跳过训练，直接 测试 → 预测
    # 
    RUN_MODE = 'train'  # 'train' | 'resume' | 'finetune' | 'test'
    
    # checkpoint 路径 (用于 resume/finetune/test 模式)
    CKPT_PATH = r"E:\code\PointSuite\outputs\dales1\checkpoints\dales1-epoch=09-mean_iou=0.7765.ckpt"
    # CKPT_PATH = None
    # 示例: CKPT_PATH = r"E:\code\PointSuite\outputs\dales1\checkpoints\dales1-epoch=05-mean_iou=0.7830.ckpt"
    
    # 梯度累积
    ACCUMULATE_GRAD_BATCHES = 2
    
    # 随机种子
    pl.seed_everything(42)
    
    # ========================================================================
    # 打印配置
    # ========================================================================
    
    print_header("DALES 语义分割训练 (逻辑索引格式)", "🎯")
    
    print_config({
        '训练数据': TRAIN_DATA,
        '验证数据': VAL_DATA,
        '测试数据': TEST_DATA,
        '预测数据': PREDICT_DATA,
        '输出目录': OUTPUT_DIR,
    }, "📁 数据路径")
    
    print_config({
        '类别数量': NUM_CLASSES,
        '类别名称': ', '.join(CLASS_NAMES),
        '忽略标签': IGNORE_LABEL,
    }, "🏷️  类别配置")
    
    print_config({
        '采样模式': MODE,
        '最大轮数': MAX_LOOPS if MAX_LOOPS else '自动',
        '批次大小': BATCH_SIZE,
        '最大点数(训练)': f'{MAX_POINTS:,}',
        '最大点数(推理)': f'{MAX_POINTS_INFERENCE:,}',
        '梯度累积': ACCUMULATE_GRAD_BATCHES,
        '等效批次': f'~{MAX_POINTS * ACCUMULATE_GRAD_BATCHES / 1000:.0f}K 点/更新',
        '学习率': LEARNING_RATE,
        '最大Epoch': MAX_EPOCHS,
        'Workers': NUM_WORKERS,
    }, "⚙️  训练配置")
    
    # ========================================================================
    # 数据增强
    # ========================================================================
    
    train_transforms = [
        CenterShift(),  # 中心化坐标 (在局部坐标系下)
        RandomDropout(dropout_ratio=0.2, p=0.5),
        RandomRotate(angle=[-1, 1], axis='z', p=0.5),
        RandomScale(scale=[0.9, 1.1]),
        RandomFlip(p=0.5),
        RandomJitter(sigma=0.005, clip=0.02),
        # GridSample(grid_size=0.5,hash_type='fnv',mode='train'),
        Collect(keys=['coord', 'class'],
                feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    val_transforms = [
        CenterShift(),  # 中心化坐标 (在局部坐标系下)
        RandomDropout(dropout_ratio=0.2, p=0.5),
        RandomRotate(angle=[-1, 1], axis='z', p=0.5),
        RandomScale(scale=[0.9, 1.1]),
        RandomFlip(p=0.5),
        RandomJitter(sigma=0.005, clip=0.02),
        # GridSample(grid_size=0.5,hash_type='fnv',mode='train'),
        Collect(keys=['coord', 'class'],
                feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    test_transforms = [
        CenterShift(),
        Collect(keys=['coord', 'class'],
                feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]

    # 预测时需要保留更多信息
    predict_transforms = [
        CenterShift(),
        Collect(keys=['coord', 'indices', 'bin_file', 'bin_path', 'pkl_path'],
                feat_keys={'feat': ['coord', 'echo']}),
        ToTensor(),
    ]
    
    # ========================================================================
    # DataModule
    # ========================================================================
    
    print_section("📦 初始化 DataModule")
    
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
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        
        # 🔥 逻辑索引格式特有配置
        mode=MODE,
        max_loops=MAX_LOOPS,
        
        # 动态批次
        use_dynamic_batch=True,
        max_points=MAX_POINTS,
        use_dynamic_batch_inference=True,
        max_points_inference=MAX_POINTS_INFERENCE,
        
        # 加权采样
        use_weighted_sampler=True,
        
        # 循环配置
        train_loop=5,
        val_loop=5,
        test_loop=1,
        predict_loop=1,
        
        # 变换
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=test_transforms,
        predict_transforms=predict_transforms,
    )
    
    # 手动 setup 以便访问数据集
    datamodule.setup(stage='fit')
    
    # 打印数据集信息
    if hasattr(datamodule, 'train_dataset') and datamodule.train_dataset is not None:
        print(f"  {Colors.DIM}├─{Colors.RESET} 训练样本数: {Colors.GREEN}{len(datamodule.train_dataset)}{Colors.RESET}")
    if hasattr(datamodule, 'val_dataset') and datamodule.val_dataset is not None:
        print(f"  {Colors.DIM}├─{Colors.RESET} 验证样本数: {Colors.GREEN}{len(datamodule.val_dataset)}{Colors.RESET}")
        # 🔥 检查验证 dataloader
        val_loader = datamodule.val_dataloader()
        print(f"  {Colors.DIM}├─{Colors.RESET} 验证 batch 数: {Colors.GREEN}{len(val_loader)}{Colors.RESET}")
    print(f"  {Colors.DIM}└─{Colors.RESET} 采样模式: {Colors.YELLOW}{MODE}{Colors.RESET}")
    
    # ========================================================================
    # 模型
    # ========================================================================
    
    print_section("🧠 初始化模型")
    
    # 模型配置
    model_config = {
        'backbone': {
            'class_path': 'pointsuite.models.PointTransformerV2',
            'init_args': {
                'in_channels': 5,  # coord(3) + echo(2)
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
    
    print(f"  {Colors.DIM}├─{Colors.RESET} Backbone: {Colors.GREEN}PointTransformerV2{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} Head: {Colors.GREEN}SegHead{Colors.RESET}")
    print(f"  {Colors.DIM}└─{Colors.RESET} 输入通道: {Colors.YELLOW}5{Colors.RESET} (coord + echo)")
    
    # 损失函数配置
    loss_configs = [
        {
            "name": "ce_loss",
            "class_path": "pointsuite.models.losses.CrossEntropyLoss",
            "init_args": {
                "ignore_index": IGNORE_LABEL,
                "weight": datamodule.train_dataset.class_weights,
            },
            "weight": 1.0,
        },
        # {
        #     "name": "lac_loss",
        #     "class_path": "pointsuite.models.losses.LACLoss",
        #     "init_args": {"k_neighbors": 16, "ignore_index": IGNORE_LABEL},
        #     "weight": 1.0,
        # },
        # {
        #     "name": "lovasz_loss",
        #     "class_path": "pointsuite.models.losses.LovaszLoss",
        #     "init_args": {"ignore_index": IGNORE_LABEL, "mode": "multiclass"},
        #     "weight": 1.0,
        # }
    ]
    
    # 指标配置
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
    
    # 创建任务
    task = SemanticSegmentationTask(
        model_config=model_config,
        learning_rate=LEARNING_RATE,
        class_mapping=CLASS_MAPPING,
        class_names=CLASS_NAMES,
        ignore_label=IGNORE_LABEL,
        loss_configs=loss_configs,
        metric_configs=metric_configs,
    )
    
    # 自定义优化器
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
    
    print_section("🔧 初始化 Trainer")
    
    callbacks = [
        # 模型检查点
        ModelCheckpoint(
            monitor='mean_iou', 
            mode='max', 
            save_top_k=3,
            save_last=True,
            filename='dales1-{epoch:02d}-{mean_iou:.4f}', 
            verbose=True
        ),
        
        # 早停
        EarlyStopping(
            monitor='mean_iou', 
            patience=20, 
            mode='max', 
            verbose=True, 
            check_on_train_epoch_end=False
        ),
        
        # 🔥 使用新的 LAS Writer (适配逻辑索引格式)
        SemanticPredictLasWriter(
            output_dir=OUTPUT_DIR, 
            save_logits=False, 
            auto_infer_reverse_mapping=True
        ),
        
        # 文本日志
        TextLoggingCallback(log_interval=10),
        
        # 自动显存清理
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
        precision="16-mixed",
        log_every_n_steps=10,
        default_root_dir='./outputs/dales1',
        logger=False,
        callbacks=callbacks,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_progress_bar=False,
        enable_model_summary=True,
        num_sanity_val_steps=2,
        limit_train_batches=None,
        check_val_every_n_epoch=1,  # 🔥 显式设置每个 epoch 验证一次
        val_check_interval=1.0,      # 🔥 每个 epoch 结束时验证
    )
    
    device_name = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'
    print(f"  {Colors.DIM}├─{Colors.RESET} 设备: {Colors.GREEN}{device_name}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 精度: {Colors.GREEN}{trainer.precision}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} Epochs: {Colors.GREEN}{MAX_EPOCHS}{Colors.RESET}")
    print(f"  {Colors.DIM}└─{Colors.RESET} 检查点目录: {Colors.CYAN}./outputs/dales1{Colors.RESET}")
    
    # ========================================================================
    # 训练流程
    # ========================================================================
    
    print_section(f"🚀 运行模式: {RUN_MODE.upper()}")
    
    # 根据 RUN_MODE 决定流程
    if RUN_MODE == 'train':
        # 从头训练
        print(f"  从头开始训练")
        print_header("开始训练", "🏋️")
        trainer.fit(task, datamodule)
        best_ckpt = "best"
        
    elif RUN_MODE == 'resume':
        # 从 checkpoint 继续训练 (恢复 optimizer/epoch 状态)
        if CKPT_PATH is None:
            raise ValueError("RUN_MODE='resume' 时必须指定 CKPT_PATH")
        print(f"  从 checkpoint 继续训练: {Colors.CYAN}{CKPT_PATH}{Colors.RESET}")
        print_header("继续训练", "🏋️")
        trainer.fit(task, datamodule, ckpt_path=CKPT_PATH)
        best_ckpt = "best"
        
    elif RUN_MODE == 'finetune':
        # 加载预训练权重，从头训练
        if CKPT_PATH is None:
            raise ValueError("RUN_MODE='finetune' 时必须指定 CKPT_PATH")
        print(f"  加载预训练权重: {Colors.CYAN}{CKPT_PATH}{Colors.RESET}")
        
        checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
                
        missing_keys, unexpected_keys = task.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"  {Colors.YELLOW}缺失的键: {missing_keys[:5]}...{Colors.RESET}")
        if unexpected_keys:
            print(f"  {Colors.YELLOW}未预期的键: {unexpected_keys[:5]}...{Colors.RESET}")
        print(f"  {Colors.GREEN}✓ 权重加载完成{Colors.RESET}")
        
        print_header("开始微调", "🏋️")
        trainer.fit(task, datamodule)
        best_ckpt = "best"
        
    elif RUN_MODE == 'test':
        # 跳过训练，直接测试/预测
        if CKPT_PATH is None:
            raise ValueError("RUN_MODE='test' 时必须指定 CKPT_PATH")
        print(f"  跳过训练，直接测试")
        print(f"  使用 checkpoint: {Colors.CYAN}{CKPT_PATH}{Colors.RESET}")
        best_ckpt = CKPT_PATH
        
    else:
        raise ValueError(f"未知的 RUN_MODE: {RUN_MODE}，可选: 'train', 'resume', 'finetune', 'test'")
    
    # ---------- 测试 ----------
    if datamodule.test_data is not None:
        print_header("开始测试", "🧪")
        trainer.test(task, datamodule, ckpt_path=best_ckpt)
    else:
        print_section("跳过测试 (未提供测试数据)")
    
    # ---------- 预测 ----------
    if datamodule.predict_data is not None:
        print_header("开始预测", "🔮")
        trainer.predict(task, datamodule=datamodule, ckpt_path=best_ckpt)
    else:
        print_section("跳过预测 (未提供预测数据)")
    
    # ========================================================================
    # 完成
    # ========================================================================
    
    print()
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  🎉 训练完成!{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 检查点目录: {Colors.CYAN}{trainer.default_root_dir}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 预测结果: {Colors.CYAN}{OUTPUT_DIR}{Colors.RESET}")
    
    if trainer.checkpoint_callback.best_model_path:
        print(f"  {Colors.DIM}├─{Colors.RESET} 最佳模型: {Colors.GREEN}{trainer.checkpoint_callback.best_model_path}{Colors.RESET}")
    
    if trainer.checkpoint_callback.best_model_score is not None:
        print(f"  {Colors.DIM}└─{Colors.RESET} 最佳 MeanIoU: {Colors.GREEN}{trainer.checkpoint_callback.best_model_score:.4f}{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}└─{Colors.RESET} 最佳 MeanIoU: {Colors.DIM}N/A{Colors.RESET}")
    
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")


if __name__ == '__main__':
    main()
