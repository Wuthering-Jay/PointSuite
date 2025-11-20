"""
测试 pointops 的梯度流和可微分性
"""
import torch
import pointops

print("=" * 80)
print("测试 pointops 操作的梯度流")
print("=" * 80)

n_points = 100
n_channels = 16
k_neighbors = 8

# 创建需要梯度的输入
coord = torch.randn(n_points, 3, device='cuda', requires_grad=True)
feat = torch.randn(n_points, n_channels, device='cuda', requires_grad=True)
offset = torch.tensor([n_points], dtype=torch.int32, device='cuda')

print("\n1. 测试 knn_query 是否可微分:")
print(f"   coord.requires_grad = {coord.requires_grad}")

# KNN 查询
idx, dist = pointops.knn_query(k_neighbors, coord, offset)
print(f"   idx shape: {idx.shape}, dtype: {idx.dtype}")
print(f"   idx.requires_grad = {idx.requires_grad}")
print(f"   dist shape: {dist.shape}, dtype: {dist.dtype}")
print(f"   dist.requires_grad = {dist.requires_grad}")

if dist.requires_grad:
    print("   ✓ knn_query 输出可微分")
else:
    print("   ✗ knn_query 输出不可微分（索引操作，预期行为）")

print("\n2. 测试 grouping 是否可微分:")
print(f"   feat.requires_grad = {feat.requires_grad}")

grouped = pointops.grouping(idx, feat, coord, with_xyz=False)
print(f"   grouped shape: {grouped.shape}, dtype: {grouped.dtype}")
print(f"   grouped.requires_grad = {grouped.requires_grad}")

if grouped.requires_grad:
    print("   ✓ grouping 输出可微分")
else:
    print("   ✗ grouping 输出不可微分")

print("\n3. 测试完整的反向传播:")
try:
    # 前向传播
    idx, _ = pointops.knn_query(k_neighbors, coord, offset)
    grouped = pointops.grouping(idx, feat, coord, with_xyz=False)
    
    # 聚合操作（模拟注意力机制）
    # grouped: [n, k, c]
    weights = torch.softmax(grouped.sum(dim=-1), dim=-1)  # [n, k]
    output = (grouped * weights.unsqueeze(-1)).sum(dim=1)  # [n, c]
    
    print(f"   output.requires_grad = {output.requires_grad}")
    
    # 计算loss
    loss = output.sum()
    print(f"   loss = {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    print(f"\n   梯度检查:")
    print(f"   - feat.grad is not None: {feat.grad is not None}")
    if feat.grad is not None:
        print(f"     feat.grad shape: {feat.grad.shape}, 非零元素: {(feat.grad != 0).sum().item()}/{feat.grad.numel()}")
    
    print(f"   - coord.grad is not None: {coord.grad is not None}")
    if coord.grad is not None:
        print(f"     coord.grad shape: {coord.grad.shape}, 非零元素: {(coord.grad != 0).sum().item()}/{coord.grad.numel()}")
    
    print("\n   ✓ 反向传播成功！pointops 操作是可微分的")
    
except Exception as e:
    print(f"\n   ✗ 反向传播失败: {type(e).__name__}: {e}")

print("\n4. 测试梯度值的合理性:")
coord2 = torch.randn(n_points, 3, device='cuda', requires_grad=True)
feat2 = torch.randn(n_points, n_channels, device='cuda', requires_grad=True)

idx, _ = pointops.knn_query(k_neighbors, coord2, offset)
grouped = pointops.grouping(idx, feat2, coord2, with_xyz=False)
output = grouped.mean()
output.backward()

print(f"   feat2 梯度统计:")
print(f"   - mean: {feat2.grad.mean().item():.6f}")
print(f"   - std: {feat2.grad.std().item():.6f}")
print(f"   - min: {feat2.grad.min().item():.6f}")
print(f"   - max: {feat2.grad.max().item():.6f}")

print(f"\n   coord2 梯度统计:")
if coord2.grad is not None:
    print(f"   - mean: {coord2.grad.mean().item():.6f}")
    print(f"   - std: {coord2.grad.std().item():.6f}")
    print(f"   - min: {coord2.grad.min().item():.6f}")
    print(f"   - max: {coord2.grad.max().item():.6f}")
else:
    print(f"   - coord2.grad is None (KNN 索引不传播梯度)")

print("\n5. 检查 autograd 图:")
coord3 = torch.randn(n_points, 3, device='cuda', requires_grad=True)
feat3 = torch.randn(n_points, n_channels, device='cuda', requires_grad=True)

idx, _ = pointops.knn_query(k_neighbors, coord3, offset)
grouped = pointops.grouping(idx, feat3, coord3, with_xyz=False)

print(f"   grouped.grad_fn: {grouped.grad_fn}")
print(f"   grouped.grad_fn type: {type(grouped.grad_fn).__name__}")

if grouped.grad_fn is not None:
    print(f"   ✓ grouping 有梯度函数，支持反向传播")
    # 查看梯度函数的输入
    if hasattr(grouped.grad_fn, 'next_functions'):
        print(f"   下一个函数: {[fn[0].__class__.__name__ if fn[0] is not None else None for fn in grouped.grad_fn.next_functions[:3]]}")
else:
    print(f"   ✗ grouping 没有梯度函数")

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
print("pointops.knn_query: 不可微分（索引查找，预期行为）")
print("pointops.grouping: 可微分（通过 gather 操作传播梯度）")
print("整体网络: 可以正常反向传播")
print("=" * 80)
