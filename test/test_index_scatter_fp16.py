"""
测试 PyTorch 索引操作在 FP16 下的反向传播
重现 ScatterGatherKernel 错误
"""

import torch
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

def test_voxel_grid_segment_csr_fp16():
    """测试完整的 voxel_grid + segment_csr 流程在 FP16 下"""
    
    device = torch.device("cuda:0")
    
    # 模拟真实数据
    N = 100000  # 10万点
    coord = torch.randn(N, 3, device=device) * 10  # 坐标范围 [-30, 30]
    feat = torch.randn(N, 48, device=device)  # 48维特征
    batch = torch.zeros(N, dtype=torch.long, device=device)  # 单个 batch
    
    grid_size = 2.5  # 使用实际的 grid_size
    
    print("=" * 80)
    print("测试 voxel_grid + segment_csr + 索引 的 FP16 反向传播")
    print("=" * 80)
    print(f"数据规模: N={N}, feat_dim={feat.shape[1]}, grid_size={grid_size}")
    
    # ==================== FP32 测试 ====================
    print("\n[FP32 测试]")
    coord_fp32 = coord.float().clone()
    feat_fp32 = feat.float().clone().requires_grad_(True)
    
    # 计算 voxel_grid
    start = torch.zeros(3, device=device)
    cluster_fp32 = voxel_grid(
        pos=coord_fp32 - start, size=grid_size, batch=batch, start=0
    )
    
    # 排序和聚合
    unique, cluster_fp32, counts = torch.unique(
        cluster_fp32, sorted=True, return_inverse=True, return_counts=True
    )
    _, sorted_cluster_indices = torch.sort(cluster_fp32)
    idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    
    print(f"  - Cluster 数量: {len(unique)}")
    print(f"  - sorted_cluster_indices 范围: [{sorted_cluster_indices.min()}, {sorted_cluster_indices.max()}]")
    print(f"  - feat 长度: {len(feat_fp32)}")
    
    # 使用索引 + segment_csr
    feat_indexed_fp32 = feat_fp32[sorted_cluster_indices]  # 索引操作
    feat_pooled_fp32 = segment_csr(feat_indexed_fp32, idx_ptr, reduce="max")
    
    loss_fp32 = feat_pooled_fp32.sum()
    
    try:
        loss_fp32.backward()
        print(f"✅ FP32 反向传播成功")
        print(f"   - feat_pooled 形状: {feat_pooled_fp32.shape}")
        print(f"   - 梯度形状: {feat_fp32.grad.shape}")
    except Exception as e:
        print(f"❌ FP32 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== FP16 测试 ====================
    print("\n[FP16 测试]")
    coord_fp16 = coord.float().clone()  # coord 保持 FP32
    feat_fp16 = feat.half().clone().requires_grad_(True)  # feat 使用 FP16
    
    # 计算 voxel_grid（coord 必须是 FP32）
    cluster_fp16 = voxel_grid(
        pos=coord_fp16 - start, size=grid_size, batch=batch, start=0
    )
    
    # 排序和聚合
    unique, cluster_fp16, counts = torch.unique(
        cluster_fp16, sorted=True, return_inverse=True, return_counts=True
    )
    _, sorted_cluster_indices = torch.sort(cluster_fp16)
    idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    
    try:
        # 使用索引 + segment_csr
        feat_indexed_fp16 = feat_fp16[sorted_cluster_indices]  # 索引操作（FP16）
        print(f"✅ 索引操作成功: feat_indexed 形状 {feat_indexed_fp16.shape}, dtype {feat_indexed_fp16.dtype}")
        
        feat_pooled_fp16 = segment_csr(feat_indexed_fp16, idx_ptr, reduce="max")
        print(f"✅ segment_csr 成功: feat_pooled 形状 {feat_pooled_fp16.shape}, dtype {feat_pooled_fp16.dtype}")
        
        loss_fp16 = feat_pooled_fp16.sum()
        print(f"✅ loss 计算成功: {loss_fp16.item()}")
        
        # 尝试反向传播
        print("\n  开始反向传播...")
        loss_fp16.backward()
        print(f"✅ FP16 反向传播成功！")
        print(f"   - 梯度形状: {feat_fp16.grad.shape}")
        print(f"   - 梯度 dtype: {feat_fp16.grad.dtype}")
        
    except Exception as e:
        print(f"❌ FP16 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== AMP 测试 ====================
    print("\n[AMP (autocast) 测试]")
    coord_amp = coord.float().clone()
    feat_amp = feat.float().clone().requires_grad_(True)
    
    try:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # voxel_grid（在 autocast 外部）
            cluster_amp = voxel_grid(
                pos=coord_amp - start, size=grid_size, batch=batch, start=0
            )
            
            # 排序和聚合
            unique, cluster_amp, counts = torch.unique(
                cluster_amp, sorted=True, return_inverse=True, return_counts=True
            )
            _, sorted_cluster_indices = torch.sort(cluster_amp)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            
            # 索引 + segment_csr（在 autocast 内部）
            feat_indexed_amp = feat_amp[sorted_cluster_indices]
            print(f"  在 autocast 中: feat_indexed dtype = {feat_indexed_amp.dtype}")
            
            feat_pooled_amp = segment_csr(feat_indexed_amp, idx_ptr, reduce="max")
            print(f"  在 autocast 中: feat_pooled dtype = {feat_pooled_amp.dtype}")
            
            loss_amp = feat_pooled_amp.sum()
        
        print(f"✅ AMP 前向传播成功")
        
        # 反向传播
        print("  开始反向传播...")
        loss_amp.backward()
        print(f"✅ AMP 反向传播成功！")
        print(f"   - 梯度 dtype: {feat_amp.grad.dtype}")
        
    except Exception as e:
        print(f"❌ AMP 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    test_voxel_grid_segment_csr_fp16()
