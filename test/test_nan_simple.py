"""
简化的 NaN 检测测试
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointsuite.data import BinPklDataModule
from pointsuite.data.transforms import Collect, ToTensor
from pointsuite.models import PointTransformerV2, SegHead

print("="*80)
print("FP16 NaN 检测测试")
print("="*80)

# 配置
TRAIN_DATA = r"E:\data\DALES\dales_las\bin\train"
CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
NUM_CLASSES = 8
IGNORE_LABEL = -1

# 创建 DataModule
val_transforms = [
    Collect(
        keys=['coord', 'class'],
        offset_key={'offset': 'coord'},
        feat_keys={'feat': ['coord', 'echo']}
    ),
    ToTensor(),
]

datamodule = BinPklDataModule(
    train_data=TRAIN_DATA,
    val_data=TRAIN_DATA,
    assets=['coord', 'echo', 'class'],
    class_mapping=CLASS_MAPPING,
    ignore_label=IGNORE_LABEL,
    batch_size=1,
    num_workers=0,
    use_dynamic_batch=True,
    max_points=80000,
    train_loop=1,
    val_loop=1,
    train_transforms=val_transforms,
    val_transforms=val_transforms,
)

datamodule.setup('fit')
train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))

print("\n批次数据:")
print(f"  coord: {batch['coord'].shape}, dtype: {batch['coord'].dtype}")
print(f"  feat: {batch['feat'].shape}, dtype: {batch['feat'].dtype}")
print(f"  class: {batch['class'].shape}, min: {batch['class'].min()}, max: {batch['class'].max()}")

# 创建浅层模型
print("\n创建模型...")
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
).cuda()

head = SegHead(
    in_channels=48,
    num_classes=NUM_CLASSES,
    cls_mode=False,
).cuda()

backbone.train()
head.train()

# 移动数据到 GPU
batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

print("\n"+"="*80)
print("测试 FP16 前向传播")
print("="*80)

# 注册钩子检测 NaN
nan_modules = []
def check_nan_hook(module, input, output):
    if isinstance(output, torch.Tensor) and torch.isnan(output).any():
        nan_modules.append((module.__class__.__name__, output.shape, torch.isnan(output).sum().item()))

# 只注册关键模块的钩子
for module in backbone.modules():
    if isinstance(module, (nn.Linear, nn.LayerNorm, nn.BatchNorm1d)):
        module.register_forward_hook(check_nan_hook)

for module in head.modules():
    if isinstance(module, (nn.Linear, nn.LayerNorm, nn.BatchNorm1d)):
        module.register_forward_hook(check_nan_hook)

try:
    with torch.cuda.amp.autocast():
        print("运行 Backbone...")
        feat = backbone(batch_cuda)
        print(f"  输出: {feat.shape}, dtype: {feat.dtype}")
        print(f"  有 NaN: {torch.isnan(feat).any().item()}")
        
        if torch.isnan(feat).any():
            print(f"  NaN 数量: {torch.isnan(feat).sum().item()} / {feat.numel()}")
            print(f"  NaN 模块:")
            for name, shape, count in nan_modules:
                print(f"    - {name}: {shape}, NaN count: {count}")
        else:
            print(f"  Min/Max: {feat.min().item():.4f} / {feat.max().item():.4f}")
            
            print("\n运行 Head...")
            logits = head(feat)
            print(f"  输出: {logits.shape}, dtype: {logits.dtype}")
            print(f"  有 NaN: {torch.isnan(logits).any().item()}")
            
            if torch.isnan(logits).any():
                print(f"  NaN 数量: {torch.isnan(logits).sum().item()} / {logits.numel()}")
            else:
                print(f"  Min/Max: {logits.min().item():.4f} / {logits.max().item():.4f}")
    
    print("\n[SUCCESS] 前向传播完成")
    
except Exception as e:
    print(f"\n[ERROR] 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)
