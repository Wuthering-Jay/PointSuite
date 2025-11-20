"""
逐步测试各个组件的 FP16 兼容性
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointsuite.data import BinPklDataModule
from pointsuite.data.transforms import Collect, ToTensor
from pointsuite.models import PointTransformerV2, SegHead
from pointsuite.tasks import SemanticSegmentationTask

print("=" * 80)
print("FP16 组件兼容性测试")
print("=" * 80)

# 准备测试数据
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
    class_names=['地面', '植被', '车辆', '卡车', '电线', '篱笆', '杆状物', '建筑'],
    ignore_label=IGNORE_LABEL,
    batch_size=2,
    num_workers=0,
    use_dynamic_batch=True,
    max_points=100000,
    train_loop=1,
    val_loop=1,
    train_transforms=val_transforms,
    val_transforms=val_transforms,
)

datamodule.setup('fit')
train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))

print(f"\n测试数据:")
print(f"  coord: {batch['coord'].shape}, dtype: {batch['coord'].dtype}")
print(f"  feat: {batch['feat'].shape}, dtype: {batch['feat'].dtype}")
print(f"  class: {batch['class'].shape}, min: {batch['class'].min()}, max: {batch['class'].max()}")
print(f"  offset: {batch['offset']}")

# ============================================================================
# 测试 1: 纯 MLP 网络（不使用 pointops）
# ============================================================================
print("\n" + "=" * 80)
print("测试 1: 纯 MLP 网络（FP32）")
print("=" * 80)

class SimpleMLP(nn.Module):
    def __init__(self, in_channels=5, num_classes=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, batch):
        feat = batch['feat']  # [N, 5]
        return self.mlp(feat)  # [N, 8]

mlp_fp32 = SimpleMLP().cuda()
try:
    with torch.no_grad():
        batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = mlp_fp32(batch_cuda)
    print(f"✓ FP32 MLP 前向传播成功: {logits.shape}")
except Exception as e:
    print(f"✗ FP32 MLP 失败: {e}")

# ============================================================================
# 测试 2: 纯 MLP 网络（FP16 autocast）
# ============================================================================
print("\n" + "=" * 80)
print("测试 2: 纯 MLP 网络（FP16 autocast）")
print("=" * 80)

mlp_fp16 = SimpleMLP().cuda()
try:
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = mlp_fp16(batch_cuda)
    print(f"✓ FP16 MLP 前向传播成功: {logits.shape}, dtype: {logits.dtype}")
except Exception as e:
    print(f"✗ FP16 MLP 失败: {e}")

# ============================================================================
# 测试 3: MLP + Loss（FP16 + 反向传播）
# ============================================================================
print("\n" + "=" * 80)
print("测试 3: MLP + Loss + 反向传播（FP16）")
print("=" * 80)

mlp_train = SimpleMLP().cuda()
optimizer = torch.optim.Adam(mlp_train.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()

try:
    batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # 前向传播
    with torch.cuda.amp.autocast():
        logits = mlp_train(batch_cuda)
    
    # Loss 计算必须在 FP32 进行，避免数值不稳定
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    loss = loss_fn(logits.float(), batch_cuda['class'])
    
    print(f"  前向传播成功, loss: {loss.item():.4f}, loss dtype: {loss.dtype}")
    
    # 反向传播
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"✓ FP16 MLP + Loss + 反向传播成功")
except Exception as e:
    print(f"✗ FP16 MLP + Loss + 反向传播失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试 4: 包含标签 -1 的 Loss 计算
# ============================================================================
print("\n" + "=" * 80)
print("测试 4: 检查标签中是否有 -1")
print("=" * 80)

labels = batch['class']
unique_labels = torch.unique(labels).tolist()
has_negative = any(l < 0 for l in unique_labels)

print(f"  唯一标签: {unique_labels}")
print(f"  是否有负数标签: {has_negative}")

if has_negative:
    print(f"  包含 -1 标签的点数: {(labels == -1).sum().item()} / {labels.numel()}")

# ============================================================================
# 测试 5: PointTransformerV2 骨干网络（不使用 pointops 的简化版本）
# ============================================================================
print("\n" + "=" * 80)
print("测试 5: 使用真实的 PointTransformerV2 (FP32)")
print("=" * 80)

backbone_fp32 = PointTransformerV2(
    in_channels=5,
    patch_embed_depth=1,
    patch_embed_channels=48,
    patch_embed_groups=6,
    patch_embed_neighbours=8,
    enc_depths=(1, 1, 1, 1),
    enc_channels=(96, 192, 384, 512),
    enc_groups=(12, 24, 48, 64),
    enc_neighbours=(16, 16, 16, 16),
    dec_depths=(1, 1, 1, 1),
    dec_channels=(48, 96, 192, 384),
    dec_groups=(6, 12, 24, 48),
    dec_neighbours=(16, 16, 16, 16),
    grid_sizes=(1.0, 3, 9, 27),
    attn_qkv_bias=True,
    pe_multiplier=False,
    pe_bias=True,
    attn_drop_rate=0.0,
    drop_path_rate=0.2,
    unpool_backend="map",
).cuda()

head_fp32 = SegHead(in_channels=48, num_classes=NUM_CLASSES).cuda()

try:
    with torch.no_grad():
        batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        feat = backbone_fp32(batch_cuda)
        logits = head_fp32(feat)
    print(f"✓ FP32 PointTransformerV2 前向传播成功: {logits.shape}")
except Exception as e:
    print(f"✗ FP32 PointTransformerV2 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试 6: PointTransformerV2 (FP16 autocast)
# ============================================================================
print("\n" + "=" * 80)
print("测试 6: PointTransformerV2 (FP16 autocast)")
print("=" * 80)

backbone_fp16 = PointTransformerV2(
    in_channels=5,
    patch_embed_depth=1,
    patch_embed_channels=48,
    patch_embed_groups=6,
    patch_embed_neighbours=8,
    enc_depths=(1, 1, 1, 1),
    enc_channels=(96, 192, 384, 512),
    enc_groups=(12, 24, 48, 64),
    enc_neighbours=(16, 16, 16, 16),
    dec_depths=(1, 1, 1, 1),
    dec_channels=(48, 96, 192, 384),
    dec_groups=(6, 12, 24, 48),
    dec_neighbours=(16, 16, 16, 16),
    grid_sizes=(1.0, 3, 9, 27),
    attn_qkv_bias=True,
    pe_multiplier=False,
    pe_bias=True,
    attn_drop_rate=0.0,
    drop_path_rate=0.2,
    unpool_backend="map",
).cuda()

head_fp16 = SegHead(in_channels=48, num_classes=NUM_CLASSES).cuda()

try:
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            feat = backbone_fp16(batch_cuda)
            logits = head_fp16(feat)
    print(f"✓ FP16 PointTransformerV2 前向传播成功: {logits.shape}, dtype: {logits.dtype}")
except Exception as e:
    print(f"✗ FP16 PointTransformerV2 前向传播失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试 7: PointTransformerV2 + Loss + 反向传播 (FP16)
# ============================================================================
print("\n" + "=" * 80)
print("测试 7: PointTransformerV2 + Loss + 反向传播 (FP16)")
print("=" * 80)

backbone_train = PointTransformerV2(
    in_channels=5,
    patch_embed_depth=1,
    patch_embed_channels=48,
    patch_embed_groups=6,
    patch_embed_neighbours=8,
    enc_depths=(1,),  # 只用一层来快速测试
    enc_channels=(96,),
    enc_groups=(12,),
    enc_neighbours=(16,),
    dec_depths=(1,),
    dec_channels=(48,),
    dec_groups=(6,),
    dec_neighbours=(16,),
    grid_sizes=(1.0,),
    attn_qkv_bias=True,
    pe_multiplier=False,
    pe_bias=True,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    unpool_backend="map",
).cuda()

head_train = SegHead(in_channels=48, num_classes=NUM_CLASSES).cuda()
optimizer = torch.optim.Adam(list(backbone_train.parameters()) + list(head_train.parameters()), lr=0.001)
scaler = torch.cuda.amp.GradScaler()

try:
    batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # 前向传播
    with torch.cuda.amp.autocast():
        feat = backbone_train(batch_cuda)
        logits = head_train(feat)
    
    # 检查 NaN
    if torch.isnan(logits).any():
        print(f"  ⚠️  警告: logits 包含 NaN!")
        print(f"    NaN 数量: {torch.isnan(logits).sum().item()} / {logits.numel()}")
        print(f"    feat NaN: {torch.isnan(feat).any().item()}")
        print(f"    feat min/max: {feat.min().item():.4f} / {feat.max().item():.4f}")
        print(f"    logits min/max: {torch.nanmin(logits).item():.4f} / {torch.nanmax(logits).item():.4f}")
    
    # Loss 计算必须在 FP32 进行
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    loss = loss_fn(logits.float(), batch_cuda['class'])
    
    print(f"  前向传播成功, loss: {loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    print(f"  反向传播成功")
    
    scaler.step(optimizer)
    scaler.update()
    
    print(f"✓ FP16 PointTransformerV2 完整训练步骤成功")
except Exception as e:
    print(f"✗ FP16 PointTransformerV2 训练失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
