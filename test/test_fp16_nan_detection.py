"""
检测 FP16 训练中 NaN 产生的位置
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pointsuite.data.datamodule_bin import BinPklDataModule
from pointsuite.models.backbones.point_transformer_v2m5 import PointTransformerV2
from pointsuite.models.heads.seg_head import SegHead

# 配置
IGNORE_LABEL = -1
BATCH_SIZE = 1
NUM_WORKERS = 0

print("="*80)
print("FP16 NaN 检测测试")
print("="*80)

# 加载数据
datamodule = BinPklDataModule(
    train_data={'type': 'BinPklDataset', 'data_root': 'E:/data/DALES/dales_las/bin/train'},
    val_data={'type': 'BinPklDataset', 'data_root': 'E:/data/DALES/dales_las/bin/train'},
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    in_channels=5,
    num_classes=8,
    voxel_size=0.1,
    voxel_max=200000,
    ignore_label=IGNORE_LABEL,
    class_mapping={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7},
    shuffle=False,
)

datamodule.setup('fit')
train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))

# 打印批次信息
print("\n批次数据:")
print(f"  coord: {batch['coord'].shape}, dtype: {batch['coord'].dtype}")
print(f"  feat: {batch['feat'].shape}, dtype: {batch['feat'].dtype}")
print(f"  class: {batch['class'].shape}, min: {batch['class'].min()}, max: {batch['class'].max()}")
print(f"  offset: {batch['offset']}")

# 创建浅层模型 (深度=1)
backbone = PointTransformerV2(
    in_channels=5,
    enc_depths=(1, 1, 1, 1),
    enc_channels=(96, 192, 384, 512),
    enc_num_head=(3, 6, 12, 24),
    enc_patch_size=(48, 48, 48, 48),
    dec_depths=(1, 1, 1, 1),
    dec_channels=(48, 96, 192, 384),
    dec_num_head=(3, 6, 12, 24),
    dec_patch_size=(48, 48, 48, 48),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.3,
    pre_norm=True,
    shuffle_orders=True,
    enable_rpe=False,
    enable_flash=False,
    upcast_attention=True,
    upcast_softmax=True,
    cls_mode=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
)

head = SegHead(
    in_channels=48,
    num_classes=8,
    cls_mode=False,
)

backbone = backbone.cuda()
head = head.cuda()
backbone.train()
head.train()

print("\n" + "="*80)
print("测试: PointTransformerV2 FP16 前向传播 + NaN 检测")
print("="*80)

# 移动数据到 GPU
batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# 注册钩子检测 NaN
def check_nan_hook(module, input, output):
    module_name = module.__class__.__name__
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f"[NaN detected] {module_name}: {output.shape}, NaN count: {torch.isnan(output).sum().item()}")
            print(f"  Input has NaN: {torch.isnan(input[0]).any().item() if isinstance(input[0], torch.Tensor) else 'N/A'}")
            print(f"  Output min/max: {torch.nanmin(output).item():.4f} / {torch.nanmax(output).item():.4f}")

# 注册钩子到所有模块
for name, module in backbone.named_modules():
    module.register_forward_hook(check_nan_hook)
for name, module in head.named_modules():
    module.register_forward_hook(check_nan_hook)

try:
    # 前向传播 (FP16)
    print("\n开始前向传播...")
    with torch.cuda.amp.autocast():
        feat = backbone(batch_cuda)
        print(f"\nBackbone 输出:")
        print(f"  Shape: {feat.shape}, dtype: {feat.dtype}")
        print(f"  有 NaN: {torch.isnan(feat).any().item()}")
        if not torch.isnan(feat).any():
            print(f"  Min/Max: {feat.min().item():.4f} / {feat.max().item():.4f}")
        
        logits = head(feat)
        print(f"\nHead 输出:")
        print(f"  Shape: {logits.shape}, dtype: {logits.dtype}")
        print(f"  有 NaN: {torch.isnan(logits).any().item()}")
        if not torch.isnan(logits).any():
            print(f"  Min/Max: {logits.min().item():.4f} / {logits.max().item():.4f}")
        else:
            print(f"  NaN 数量: {torch.isnan(logits).sum().item()} / {logits.numel()}")
    
    print("\n[OK] 前向传播完成")
    
except Exception as e:
    print(f"\n[ERROR] 前向传播失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)
