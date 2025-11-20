"""
测试 pointops.interpolation 是否支持 FP16
"""

import torch
import pointops

def test_interpolation_fp16():
    """测试 pointops.interpolation 在 FP16 下是否工作"""
    print("\n" + "="*80)
    print("测试 pointops.interpolation FP16 支持")
    print("="*80)
    
    # 创建测试数据
    n1, n2 = 1000, 2000
    c = 64
    
    # 源点云和目标点云
    coord1 = torch.randn(n1, 3).cuda()
    coord2 = torch.randn(n2, 3).cuda()
    feat = torch.randn(n1, c).cuda()
    
    offset1 = torch.tensor([n1], dtype=torch.long).cuda()
    offset2 = torch.tensor([n2], dtype=torch.long).cuda()
    
    # 测试 1: FP32
    print("\n" + "-"*80)
    print("测试 1: FP32 (原始精度)")
    print("-"*80)
    try:
        result_fp32 = pointops.interpolation(
            coord1.float(), coord2.float(), feat.float(), offset1, offset2
        )
        print(f"✅ FP32 成功")
        print(f"  输入 feat: {feat.float().shape}, dtype={feat.float().dtype}")
        print(f"  输出: {result_fp32.shape}, dtype={result_fp32.dtype}")
        print(f"  输出范围: [{result_fp32.min().item():.4f}, {result_fp32.max().item():.4f}]")
    except Exception as e:
        print(f"❌ FP32 失败: {e}")
        return False
    
    # 测试 2: FP16 输入
    print("\n" + "-"*80)
    print("测试 2: FP16 输入 (coord + feat 都是 FP16)")
    print("-"*80)
    try:
        result_fp16 = pointops.interpolation(
            coord1.half(), coord2.half(), feat.half(), offset1, offset2
        )
        print(f"✅ FP16 输入成功")
        print(f"  输入 feat: {feat.half().shape}, dtype={feat.half().dtype}")
        print(f"  输出: {result_fp16.shape}, dtype={result_fp16.dtype}")
        print(f"  输出范围: [{result_fp16.min().item():.4f}, {result_fp16.max().item():.4f}]")
    except Exception as e:
        print(f"❌ FP16 输入失败: {e}")
        print(f"   可能原因: CUDA kernel 不支持 FP16")
        return False
    
    # 测试 3: FP16 + AMP
    print("\n" + "-"*80)
    print("测试 3: 在 AMP 上下文中使用")
    print("-"*80)
    try:
        with torch.cuda.amp.autocast(enabled=True):
            # 输入是 FP32，autocast 会自动处理
            result_amp = pointops.interpolation(
                coord1, coord2, feat, offset1, offset2
            )
        print(f"✅ AMP 自动转换成功")
        print(f"  输出: {result_amp.shape}, dtype={result_amp.dtype}")
        print(f"  输出范围: [{result_amp.min().item():.4f}, {result_amp.max().item():.4f}]")
    except Exception as e:
        print(f"❌ AMP 失败: {e}")
        return False
    
    # 测试 4: 梯度测试
    print("\n" + "-"*80)
    print("测试 4: FP16 反向传播")
    print("-"*80)
    try:
        feat_grad = feat.half().clone().requires_grad_(True)
        result = pointops.interpolation(
            coord1.half(), coord2.half(), feat_grad, offset1, offset2
        )
        loss = result.sum()
        loss.backward()
        
        print(f"✅ FP16 反向传播成功")
        print(f"  梯度 shape: {feat_grad.grad.shape}, dtype={feat_grad.grad.dtype}")
        print(f"  梯度范围: [{feat_grad.grad.min().item():.4f}, {feat_grad.grad.max().item():.4f}]")
        print(f"  是否有 NaN: {torch.isnan(feat_grad.grad).any().item()}")
    except Exception as e:
        print(f"❌ FP16 反向传播失败: {e}")
        return False
    
    print("\n" + "="*80)
    print("🎉 所有测试通过! pointops.interpolation 完全支持 FP16")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_interpolation_fp16()
    if success:
        print("\n结论: pointops.interpolation 原生支持 FP16，不需要额外的类型转换!")
    else:
        print("\n结论: pointops.interpolation 不支持 FP16，需要转换到 FP32")
