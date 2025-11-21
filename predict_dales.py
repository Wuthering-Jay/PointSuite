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
from pointsuite.data.transforms import ToTensor, Collect
from pointsuite.tasks import SemanticSegmentationTask
from pointsuite.models import PointTransformerV2, SegHead
from pointsuite.utils.callbacks import SegmentationWriter
from pointsuite.utils.progress_bar import CustomProgressBar


def main():
    # ========================================================================
    # 配置 - 根据你的实际情况修改
    # ========================================================================
    
    # 🔥 重要：修改为你实际的路径
    CHECKPOINT_PATH = r"outputs/dales/checkpoints/dales-epoch=XX-mean_iou=0.XXXX.ckpt"  # 修改这里！
    TEST_DATA = r"E:\data\DALES\dales_las\bin\test"  # 修改这里！
    OUTPUT_DIR = r"E:\data\DALES\dales_las\bin\result"  # 修改这里！
    
    CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    CLASS_NAMES = ['地面', '植被', '车辆', '卡车', '电线', '篱笆', '杆状物', '建筑']
    NUM_CLASSES = 8
    IGNORE_LABEL = -1
    
    # Predict 配置
    NUM_WORKERS = 4
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
    # 数据模块 - Predict transforms（不要数据增强！）
    # ========================================================================
    
    predict_transforms = [
        ToTensor(),
        Collect(keys=['coord', 'feat'], feat_keys=['coord', 'intensity'])
    ]
    
    datamodule = BinPklDataModule(
        predict_data=TEST_DATA,
        assets=['coord', 'intensity', 'classification'],
        batch_size=1,  # Predict 时batch_size无关紧要
        num_workers=NUM_WORKERS,
        class_mapping=CLASS_MAPPING,
        class_names=CLASS_NAMES,
        ignore_label=IGNORE_LABEL,
        predict_loop=1,
        predict_transforms=predict_transforms,
        use_dynamic_batch_inference=True,
        max_points_inference=MAX_POINTS_INFERENCE,
        pin_memory=True,
    )
    
    # ========================================================================
    # 从 Checkpoint 加载模型
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
    task = SemanticSegmentationTask.load_from_checkpoint(
        CHECKPOINT_PATH,
        strict=False  # 如果模型结构略有不同，使用 strict=False
    )
    
    print(f"✓ 模型加载成功")
    print(f"  - 类别数: {NUM_CLASSES}")
    print(f"  - Class Mapping: {CLASS_MAPPING}")
    
    # ========================================================================
    # Trainer 和 Callbacks
    # ========================================================================
    
    callbacks = [
        SegmentationWriter(
            output_dir=OUTPUT_DIR, 
            save_logits=False, 
            auto_infer_reverse_mapping=True
        ),
        CustomProgressBar(refresh_rate=1),
    ]
    
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision="32-true",
        logger=False,  # Predict 时不需要 logger
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    print(f"\n设备: {trainer.accelerator}")
    print(f"推理配置: max_points={MAX_POINTS_INFERENCE/1000:.0f}K | workers={NUM_WORKERS}")
    
    # ========================================================================
    # 运行预测
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
