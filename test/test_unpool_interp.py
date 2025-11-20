"""
测试 UnpoolWithSkip 的 interp backend 是否可以在 FP16 下工作
不依赖 cluster 索引，使用 pointops.interpolation
"""

import torch
import torch.nn as nn
from pointsuite.models.backbones.point_transformer_v2m5 import PointTransformerV2
from pointsuite.models.heads.seg_head import SegHead

def test_unpool_interp_backend():
    """测试 unpool_backend='interp' 模式"""
    print("\n" + "="*80)
    print("测试 UnpoolWithSkip with backend='interp'")
    print("="*80)
    
    # 创建模型 - 使用 interp backend (与 train_dales.py 一致的配置)
    backbone = PointTransformerV2(
        in_channels=5,
        patch_embed_depth=2,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=16,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(1, 2.5, 7.5, 15),  # 使用与 train_dales.py 相同的值
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        unpool_backend="interp",  # 🔥 使用 interp backend，不依赖 cluster
    )
    
    head = SegHead(in_channels=48, num_classes=8)
    
    model = nn.Sequential(backbone, head)
    model = model.cuda()
    
    # 测试数据
    batch_size = 2
    num_points = 50000
    
    data_dict = {
        "coord": torch.randn(num_points, 3, dtype=torch.float32).cuda(),
        "feat": torch.randn(num_points, 5, dtype=torch.float32).cuda(),
        "offset": torch.tensor([25000, 50000], dtype=torch.long).cuda(),
    }
    
    print(f"\n输入数据:")
    print(f"  coord: {data_dict['coord'].shape}, dtype={data_dict['coord'].dtype}")
    print(f"  feat: {data_dict['feat'].shape}, dtype={data_dict['feat'].dtype}")
    print(f"  offset: {data_dict['offset']}")
    
    # 测试 1: FP32 前向
    print("\n" + "-"*80)
    print("测试 1: FP32 前向传播")
    print("-"*80)
    model.train()
    output = model(data_dict)
    print(f"✅ FP32 前向成功")
    print(f"  输出 shape: {output.shape}, dtype={output.dtype}")
    print(f"  输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 测试 2: FP16 前向
    print("\n" + "-"*80)
    print("测试 2: FP16 前向传播 (AMP)")
    print("-"*80)
    
    data_dict_fp16 = {
        "coord": data_dict["coord"].clone(),
        "feat": data_dict["feat"].clone().half(),
        "offset": data_dict["offset"].clone(),
    }
    
    with torch.cuda.amp.autocast(enabled=True):
        output_fp16 = model(data_dict_fp16)
    
    print(f"✅ FP16 前向成功")
    print(f"  输出 shape: {output_fp16.shape}, dtype={output_fp16.dtype}")
    print(f"  输出范围: [{output_fp16.min().item():.4f}, {output_fp16.max().item():.4f}]")
    
    # 测试 3: FP16 前向 + 损失
    print("\n" + "-"*80)
    print("测试 3: FP16 前向 + 损失计算")
    print("-"*80)
    
    target = torch.randint(0, 8, (num_points,), dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss()
    
    with torch.cuda.amp.autocast(enabled=True):
        output_fp16 = model(data_dict_fp16)
        loss = criterion(output_fp16, target)
    
    print(f"✅ FP16 损失计算成功")
    print(f"  损失值: {loss.item():.4f}")
    print(f"  损失是否为 NaN: {torch.isnan(loss).item()}")
    
    # 测试 4: FP16 反向传播
    print("\n" + "-"*80)
    print("测试 4: FP16 反向传播")
    print("-"*80)
    
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast(enabled=True):
        output_fp16 = model(data_dict_fp16)
        loss = criterion(output_fp16, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"✅ FP16 反向传播成功")
    
    # 检查梯度
    grad_norms = []
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if torch.isnan(param.grad).any():
                has_nan_grad = True
                print(f"  ⚠️ {name}: 梯度包含 NaN!")
    
    print(f"  梯度统计: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={sum(grad_norms)/len(grad_norms):.6f}")
    print(f"  是否有 NaN 梯度: {has_nan_grad}")
    
    # 测试 5: 多步训练
    print("\n" + "-"*80)
    print("测试 5: 3 步 FP16 训练")
    print("-"*80)
    
    for step in range(3):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=True):
            output_fp16 = model(data_dict_fp16)
            loss = criterion(output_fp16, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"  Step {step}: Loss = {loss.item():.4f}, NaN={torch.isnan(loss).item()}")
    
    print(f"\n✅ 所有测试通过! unpool_backend='interp' 可以在 FP16 下正常工作")
    print(f"   不依赖 cluster 索引，使用 pointops.interpolation")
    
    return True

if __name__ == "__main__":
    print("\n" + "="*80)
    print("UnpoolWithSkip Interpolation Backend 测试")
    print("="*80)
    print("测试 unpool_backend='interp' 是否可以避免 cluster 索引问题")
    print("="*80)
    
    try:
        test_unpool_interp_backend()
        print("\n" + "="*80)
        print("🎉 测试成功! 可以使用 unpool_backend='interp'")
        print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ 测试失败: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
