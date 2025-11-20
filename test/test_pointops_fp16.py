"""
测试 pointops 是否支持 FP16
"""
import torch
import pointops

print("=" * 80)
print("测试 pointops 对 FP16 的支持")
print("=" * 80)

# 创建测试数据
n_points = 1000
n_channels = 32
k_neighbors = 16

# FP32 测试
print("\n1. FP32 测试:")
coord_fp32 = torch.randn(n_points, 3).cuda()
feat_fp32 = torch.randn(n_points, n_channels).cuda()
offset_fp32 = torch.tensor([n_points], dtype=torch.int32).cuda()

try:
    # KNN查询
    idx, _ = pointops.knn_query(k_neighbors, coord_fp32, offset_fp32)
    print(f"   ✓ knn_query 成功: idx shape = {idx.shape}, dtype = {idx.dtype}")
    
    # Grouping
    grouped = pointops.grouping(idx, feat_fp32, coord_fp32, with_xyz=False)
    print(f"   ✓ grouping 成功: grouped shape = {grouped.shape}, dtype = {grouped.dtype}")
except Exception as e:
    print(f"   ✗ FP32 失败: {e}")

# FP16 测试
print("\n2. FP16 测试:")
coord_fp16 = coord_fp32.half()
feat_fp16 = feat_fp32.half()

try:
    # KNN查询（坐标用FP16）
    idx, _ = pointops.knn_query(k_neighbors, coord_fp16, offset_fp32)
    print(f"   ✓ knn_query 成功: idx shape = {idx.shape}, dtype = {idx.dtype}")
    
    # Grouping（特征用FP16）
    grouped = pointops.grouping(idx, feat_fp16, coord_fp16, with_xyz=False)
    print(f"   ✓ grouping 成功: grouped shape = {grouped.shape}, dtype = {grouped.dtype}")
except Exception as e:
    print(f"   ✗ FP16 失败: {e}")

# 混合精度测试（坐标FP32，特征FP16）
print("\n3. 混合精度测试 (coord FP32, feat FP16):")
try:
    idx, _ = pointops.knn_query(k_neighbors, coord_fp32, offset_fp32)
    print(f"   ✓ knn_query 成功 (coord FP32)")
    
    grouped = pointops.grouping(idx, feat_fp16, coord_fp32, with_xyz=False)
    print(f"   ✓ grouping 成功 (feat FP16): grouped dtype = {grouped.dtype}")
except Exception as e:
    print(f"   ✗ 混合精度失败: {e}")

# 反向传播测试
print("\n4. FP16 反向传播测试:")
coord_fp16 = torch.randn(n_points, 3, requires_grad=True).cuda().half()
feat_fp16 = torch.randn(n_points, n_channels, requires_grad=True).cuda().half()
offset_fp32 = torch.tensor([n_points], dtype=torch.int32).cuda()

try:
    idx, _ = pointops.knn_query(k_neighbors, coord_fp16, offset_fp32)
    grouped = pointops.grouping(idx, feat_fp16, coord_fp16, with_xyz=False)
    loss = grouped.sum()
    loss.backward()
    print(f"   ✓ FP16 反向传播成功")
    print(f"   - feat_fp16.grad: {feat_fp16.grad is not None}, dtype = {feat_fp16.grad.dtype if feat_fp16.grad is not None else 'None'}")
    print(f"   - coord_fp16.grad: {coord_fp16.grad is not None}, dtype = {coord_fp16.grad.dtype if coord_fp16.grad is not None else 'None'}")
except Exception as e:
    print(f"   ✗ FP16 反向传播失败: {type(e).__name__}: {e}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
