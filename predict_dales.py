"""
仅运行预测的脚本 - 使用已训练的checkpoint

用法：
    python predict_dales.py

需要修改的配置：
    - CHECKPOINT_PATH: 你的checkpoint路径
    - TEST_DATA: 测试数据路径
    - OUTPUT_DIR: 预测结果输出路径
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointsuite.data import BinPklDataModule
from pointsuite.data.transforms import ToTensor, Collect, CenterShift
from pointsuite.tasks import SemanticSegmentationTask
from pointsuite.models import PointTransformerV2, SegHead
from pointsuite.utils.callbacks import SemanticPredictLasWriter, TextLoggingCallback


def main():
    # ========================================================================
    # 配置 - 根据你的实际情况修改
    # ========================================================================
    
    # 🔥 重要：修改为你实际的路径
    CHECKPOINT_PATH = r"E:\code\PointSuite\outputs\dales\csv_logs\version_42\checkpoints\dales-epoch=09-mean_iou=0.8094.ckpt"  # 修改这里！
    TEST_DATA = r"E:\data\DALES\dales_las\bin\test"  # 修改这里！
    OUTPUT_DIR = r"E:\data\DALES\dales_las\bin\result"  # 修改这里！
    
    # Predict 配置
    NUM_WORKERS = 0
    MAX_POINTS_INFERENCE = 300000  # 推理时使用更大batch
    
    pl.seed_everything(42)
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("\n" + "=" * 80)
    print(f"DALES 预测 - 使用已训练模型")
    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"测试数据: {TEST_DATA}")
    print(f"输出目录: {OUTPUT_DIR}")

    # ========================================================================
    # 1. 从 Checkpoint 加载模型 (获取配置信息)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("从 Checkpoint 加载模型...")
    print("=" * 80)
    
    # 检查 checkpoint 文件是否存在
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ 错误: Checkpoint 文件不存在: {CHECKPOINT_PATH}")
        print("\n请检查以下内容：")
        print("1. 确认训练是否完成并保存了checkpoint")
        print("2. 查看 outputs/dales/checkpoints/ 目录")
        print("3. 修改 CHECKPOINT_PATH 为实际的文件路径")
        return
    
    # 从 checkpoint 加载模型
    # 注意：load_from_checkpoint 需要知道如何实例化 backbone 和 head
    # 如果 checkpoint 中保存了这些参数（通过 save_hyperparameters），则会自动加载
    # 但如果 backbone 和 head 是作为对象传入 __init__ 的，PL 可能无法自动重建它们
    # 因此我们需要手动实例化它们并传入
    
    # 1. 先实例化 backbone 和 head (使用与训练时相同的配置)
    # 这里我们假设使用 PointTransformerV2 和 SegHead，参数需要与训练时一致
    # 如果你不确定参数，可以查看 hparams.yaml 或 checkpoint 中的 hyper_parameters
    
    # 为了通用性，我们尝试直接加载。如果失败，说明需要手动传入 backbone/head
    try:
        task = SemanticSegmentationTask.load_from_checkpoint(
            CHECKPOINT_PATH,
            strict=False
        )
    except TypeError as e:
        print(f"\n⚠️  自动加载失败: {e}")
        print("尝试手动实例化 Backbone 和 Head...")
        
        # 这里需要硬编码训练时的配置，或者从配置文件读取
        # 假设是 DALES 的配置：
        backbone = PointTransformerV2(
            in_channels=4,  # coord(3) + intensity(1)
            num_classes=8,
            patch_embed_depth=2,
            enc_depths=[2, 2, 6, 2],
            dec_depths=[1, 1, 1, 1],
            enc_channels=[32, 64, 128, 256],
            dec_channels=[32, 64, 128, 256],
            num_heads=[2, 4, 8, 16],
            patch_embed_channels=32,
            grid_size=0.05,
            in_grid_size=0.02
        )
        
        head = SegHead(
            in_channels=32,
            num_classes=8,
            dropout=0.5
        )
        
        task = SemanticSegmentationTask.load_from_checkpoint(
            CHECKPOINT_PATH,
            backbone=backbone,
            head=head,
            strict=False
        )

    # 从模型中提取配置信息
    class_mapping = task.hparams.get('class_mapping')
    class_names = task.hparams.get('class_names')
    
    # 尝试获取类别数量
    if hasattr(task.head, 'num_classes'):
        num_classes = task.head.num_classes
    elif class_mapping:
        num_classes = len(set(class_mapping.values()))
    else:
        num_classes = -1  # 让 Writer 自动推断
        
    # 构建反向映射 (用于将预测结果映射回原始标签)
    reverse_mapping = None
    if class_mapping:
        reverse_mapping = {v: k for k, v in class_mapping.items()}
    
    print(f"✓ 模型加载成功")
    print(f"  - 自动提取类别数: {num_classes}")
    print(f"  - 自动提取 Class Mapping: {class_mapping is not None}")
    
    # ========================================================================
    # 2. 数据模块 - Predict transforms（不要数据增强！）
    # ========================================================================
    
    predict_transforms = [
        CenterShift(),
        ToTensor(),
        Collect(keys=['coord', 'feat'], feat_keys=['coord', 'intensity'])
    ]
    
    # 预测时不需要加载 classification，也不需要 class_mapping
    # 只要提供 coord 和 intensity 给模型即可
    datamodule = BinPklDataModule(
        predict_data=TEST_DATA,
        assets=['coord', 'intensity'],  # 仅加载需要的特征，不加载标签
        batch_size=1,  # Predict 时batch_size无关紧要
        num_workers=NUM_WORKERS,
        # class_mapping=None,  # 预测不需要映射输入标签
        # ignore_label=None,   # 预测不需要 ignore_label
        predict_loop=1,
        predict_transforms=predict_transforms,
        use_dynamic_batch_inference=True,
        max_points_inference=MAX_POINTS_INFERENCE,
        pin_memory=True,
    )
    
    # ========================================================================
    # 3. Trainer 和 Callbacks
    # ========================================================================
    
    callbacks = [
        SemanticPredictLasWriter(
            output_dir=OUTPUT_DIR, 
            num_classes=num_classes,
            save_logits=False, 
            reverse_class_mapping=reverse_mapping, # 传入从模型提取的反向映射
            auto_infer_reverse_mapping=False #既然已经传入了，就不需要自动推断了
        ),
        TextLoggingCallback(log_interval=10),
    ]
    
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision="bf16-mixed", # 使用 bf16-mixed 加速预测 (与训练一致)
        logger=False,  # Predict 时不需要 logger
        callbacks=callbacks,
        enable_progress_bar=False,  # 使用 TextLoggingCallback 代替
        enable_model_summary=False,
    )
    
    print(f"\n设备: {trainer.accelerator}")
    print(f"精度: {trainer.precision}")
    print(f"推理配置: max_points={MAX_POINTS_INFERENCE/1000:.0f}K | workers={NUM_WORKERS}")
    
    # ========================================================================
    # 4. 运行预测
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("开始预测")
    print("=" * 80)
    
    trainer.predict(task, datamodule)
    
    print("\n" + "=" * 80)
    print("预测完成！")
    print("=" * 80)
    print(f"预测结果已保存到: {OUTPUT_DIR}")
    print(f"\n请使用 CloudCompare 或其他工具查看生成的 .las 文件")
    print("检查类别分布是否正常（不应该99%是一个类别）")


if __name__ == '__main__':
    main()
