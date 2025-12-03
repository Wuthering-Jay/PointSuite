"""
简单的训练脚本示例

使用方法：
    python train_example.py

注意：这是一个简化的示例，展示如何手动编写训练脚本。
      生产环境建议使用 LightningCLI + 配置文件的方式。
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# 导入 PointSuite 组件
from pointsuite.data import BinPklDataModule
from pointsuite.data.transforms import (
    CenterShift, AutoNormalizeHNorm, RandomRotate, RandomScale,
    Collect, ToTensor
)
from pointsuite.tasks import SemanticSegmentationTask
from pointsuite.models.backbones import PointTransformerV2m5
from pointsuite.models.heads import SegmentationHead


def create_datamodule():
    """创建数据模块"""
    
    # 定义训练 transforms
    train_transforms = [
        CenterShift(apply_z=True),
        RandomRotate(angle=[-180, 180], axis='z', p=0.5),
        RandomScale(scale=[0.9, 1.1], p=0.5),
        AutoNormalizeHNorm(clip_range=None),
        Collect(
            keys=['coord', 'class'],
            offset_key={'offset': 'coord'},
            feat_keys={'feat': ['coord', 'h_norm', 'intensity', 'echo']}
        ),
        ToTensor()
    ]
    
    # 定义验证/测试 transforms
    val_transforms = [
        CenterShift(apply_z=True),
        AutoNormalizeHNorm(clip_range=None),
        Collect(
            keys=['coord', 'class'],
            offset_key={'offset': 'coord'},
            feat_keys={'feat': ['coord', 'h_norm', 'intensity', 'echo']}
        ),
        ToTensor()
    ]
    
    # 创建 DataModule
    datamodule = BinPklDataModule(
        # 数据路径 - 修改为你的数据路径
        train_data='data/train',
        val_data='data/val',
        test_data='data/test',
        
        # DataLoader 参数
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        
        # Dataset 参数
        assets=['coord', 'intensity', 'echo', 'h_norm', 'classification'],
        ignore_label=-1,
        
        # Loop 参数
        train_loop=1,
        val_loop=1,
        test_loop=1,
        
        # 动态 Batch
        use_dynamic_batch=True,
        max_points=500000,
        use_dynamic_batch_inference=True,
        max_points_inference=800000,
        
        # Transforms
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=val_transforms,
    )
    
    return datamodule


def create_model(num_classes=8, in_channels=6, class_mapping=None):
    """创建模型"""
    
    # 创建 Backbone
    backbone = PointTransformerV2m5(
        in_channels=in_channels,  # coord(3) + h_norm(1) + intensity(1) + echo(1) = 6
        num_classes=num_classes,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=16,
        enc_depths=[2, 2, 6, 2],
        enc_channels=[48, 96, 192, 384],
        enc_num_head=[3, 6, 12, 24],
        enc_patch_size=[128, 128, 128, 128],
        dec_depths=[1, 1, 1, 1],
        dec_channels=[48, 96, 192, 384],
        dec_num_head=[3, 6, 12, 24],
        dec_patch_size=[128, 128, 128, 128],
    )
    
    # 创建 Head
    head = SegmentationHead(
        in_channels=48,  # backbone 输出通道
        num_classes=num_classes,
        hidden_channels=64
    )
    
    # 创建 Task
    model = SemanticSegmentationTask(
        backbone=backbone,
        head=head,
        learning_rate=0.001,
        
        # 🔥 重要：传入 class_mapping，将被保存到 checkpoint
        class_mapping=class_mapping,
        
        # 损失函数配置
        loss_configs=[
            {
                'type': 'pointsuite.models.losses.CrossEntropyLoss',
                'weight': 1.0,
                'init_args': {'ignore_index': -1}
            }
        ],
        
        # 指标配置
        metric_configs=[
            {
                'type': 'pointsuite.utils.metrics.OverallAccuracy',
                'name': 'OA',
                'init_args': {'ignore_index': -1}
            },
            {
                'type': 'pointsuite.utils.metrics.MeanIoU',
                'name': 'mIoU',
                'init_args': {'num_classes': num_classes, 'ignore_index': -1}
            }
        ]
    )
    
    return model


def main():
    """主函数"""
    
    # 设置随机种子
    pl.seed_everything(42)
    
    print("="*60)
    print("PointSuite 训练示例")
    print("="*60)
    
    # 🔥 定义类别映射（如果需要）
    # 如果你的类别标签不连续，需要定义映射
    class_mapping = None  # 默认不使用映射
    # class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}  # 示例：5个类别
    
    # 1. 创建 DataModule
    print("\n[1/4] 创建 DataModule...")
    datamodule = create_datamodule()
    # 如果使用 class_mapping，需要传入 DataModule
    if class_mapping is not None:
        datamodule.class_mapping = class_mapping
    datamodule.print_info()  # 打印数据信息
    
    # 2. 创建 Model
    print("\n[2/4] 创建 Model...")
    num_classes = 8  # 如果使用 class_mapping，应该是映射后的类别数
    model = create_model(
        num_classes=num_classes,
        in_channels=6,
        class_mapping=class_mapping  # 🔥 传入 class_mapping，保存到 checkpoint
    )
    print(f"✓ Model created: {model.__class__.__name__}")
    print(f"  - Backbone: {model.backbone.__class__.__name__}")
    print(f"  - Head: {model.head.__class__.__name__}")
    print(f"  - Learning rate: {model.learning_rate}")
    if class_mapping is not None:
        print(f"  - Class mapping: {class_mapping}")
        print(f"  - 将被保存到 checkpoint，预测时自动加载")
    
    # 3. 创建 Trainer
    print("\n[3/4] 创建 Trainer...")
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        
        # 梯度相关
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        
        # 验证相关
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        
        # 日志相关
        log_every_n_steps=50,
        
        # 回调
        callbacks=[
            ModelCheckpoint(
                dirpath='checkpoints/',
                filename='epoch={epoch}-val_loss={val/total_loss:.4f}',
                monitor='val/total_loss',
                mode='min',
                save_top_k=3,
                save_last=True,
                auto_insert_metric_name=False
            ),
            EarlyStopping(
                monitor='val/total_loss',
                patience=20,
                mode='min',
                verbose=True
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        
        # 日志器
        logger=True,  # 使用默认 TensorBoard logger
    )
    print(f"✓ Trainer created")
    print(f"  - Max epochs: {trainer.max_epochs}")
    print(f"  - Devices: {trainer.num_devices}")
    print(f"  - Precision: {trainer.precision}")
    
    # 4. 训练
    print("\n[4/4] 开始训练...")
    print("="*60)
    
    try:
        trainer.fit(model, datamodule)
        print("\n✅ 训练完成!")
        
        # 5. 测试
        print("\n[5/5] 开始测试...")
        trainer.test(model, datamodule, ckpt_path='best')
        print("\n✅ 测试完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
