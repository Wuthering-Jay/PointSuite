"""
测试 segment_csr 在 FP16 下的反向传播
"""

import torch
from torch_scatter import segment_csr

def test_segment_csr_backward():
    """测试 segment_csr 的反向传播在 FP16 下是否有问题"""
    
    device = torch.device("cuda:0")
    
    # 创建测试数据
    N = 10000
    feat = torch.randn(N, 64, device=device)
    
    # 创建聚类索引（模拟 voxel_grid 的输出）
    cluster_size = 100
    n_clusters = N // cluster_size
    
    # 创建 sorted_cluster_indices 和 idx_ptr
    sorted_cluster_indices = torch.arange(N, device=device)
    counts = torch.full((n_clusters,), cluster_size, device=device)
    idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    
    print("=" * 80)
    print("测试 segment_csr 在不同精度下的反向传播")
    print("=" * 80)
    
    # 测试 FP32
    print("\n[FP32 测试]")
    feat_fp32 = feat.float().clone().requires_grad_(True)
    output_fp32 = segment_csr(
        feat_fp32[sorted_cluster_indices], 
        idx_ptr, 
        reduce="mean"
    )
    loss_fp32 = output_fp32.sum()
    
    try:
        loss_fp32.backward()
        print(f"✅ FP32 反向传播成功")
        print(f"   - 输入形状: {feat_fp32.shape}, 输出形状: {output_fp32.shape}")
        print(f"   - 梯度形状: {feat_fp32.grad.shape}")
        print(f"   - 梯度范围: [{feat_fp32.grad.min():.6f}, {feat_fp32.grad.max():.6f}]")
    except Exception as e:
        print(f"❌ FP32 反向传播失败: {e}")
    
    # 测试 FP16
    print("\n[FP16 测试]")
    feat_fp16 = feat.half().clone().requires_grad_(True)
    
    try:
        # 尝试直接使用 FP16
        output_fp16 = segment_csr(
            feat_fp16[sorted_cluster_indices], 
            idx_ptr, 
            reduce="mean"
        )
        loss_fp16 = output_fp16.sum()
        print(f"✅ FP16 前向传播成功")
        print(f"   - 输入形状: {feat_fp16.shape}, 输出形状: {output_fp16.shape}")
        
        # 尝试反向传播
        loss_fp16.backward()
        print(f"✅ FP16 反向传播成功")
        print(f"   - 梯度形状: {feat_fp16.grad.shape}")
        print(f"   - 梯度范围: [{feat_fp16.grad.min():.6f}, {feat_fp16.grad.max():.6f}]")
        
    except Exception as e:
        print(f"❌ FP16 反向传播失败: {e}")
    
    # 测试 AMP (autocast)
    print("\n[AMP (autocast) 测试]")
    feat_amp = feat.float().clone().requires_grad_(True)
    
    try:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output_amp = segment_csr(
                feat_amp[sorted_cluster_indices], 
                idx_ptr, 
                reduce="mean"
            )
            loss_amp = output_amp.sum()
        
        print(f"✅ AMP 前向传播成功")
        print(f"   - 输入 dtype: {feat_amp.dtype}, 输出 dtype: {output_amp.dtype}")
        
        loss_amp.backward()
        print(f"✅ AMP 反向传播成功")
        print(f"   - 梯度 dtype: {feat_amp.grad.dtype}")
        print(f"   - 梯度范围: [{feat_amp.grad.min():.6f}, {feat_amp.grad.max():.6f}]")
        
    except Exception as e:
        print(f"❌ AMP 反向传播失败: {e}")
    
    # 测试索引是否正确
    print("\n[索引验证]")
    print(f"sorted_cluster_indices 范围: [{sorted_cluster_indices.min()}, {sorted_cluster_indices.max()}]")
    print(f"feat 长度: {len(feat)}")
    print(f"idx_ptr 范围: [{idx_ptr.min()}, {idx_ptr.max()}]")
    print(f"是否所有索引都在范围内: {(sorted_cluster_indices >= 0).all() and (sorted_cluster_indices < len(feat)).all()}")


if __name__ == "__main__":
    # 设置 CUDA_LAUNCH_BLOCKING 以便捕获详细错误
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    test_segment_csr_backward()
