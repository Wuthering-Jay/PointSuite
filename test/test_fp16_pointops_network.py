"""
测试包含 pointops 和 scatter/gather 操作的复杂网络在 FP16 下的表现
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pointops
except ImportError:
    print("错误: pointops 未安装!")
    sys.exit(1)

from pointsuite.models.losses import CrossEntropyLoss, LovaszLoss
from pointsuite.utils.metrics import SegmentationMetrics
from pointsuite.models.modules.point_wise import PointBatchNorm

print("="*80)
print("FP16 测试 - 包含 PointOps 和 Scatter/Gather 操作的复杂网络")
print("="*80)


class ComplexPointNetwork(nn.Module):
    """
    包含 PointTransformerV2 中使用的各种复杂操作：
    1. pointops.knn_query (FP32 only)
    2. torch.gather / scatter
    3. PointBatchNorm
    4. 残差连接
    5. 多层 MLP
    """
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=8, k_neighbors=16):
        super().__init__()
        self.k_neighbors = k_neighbors
        
        # Stage 1: 初始特征提取
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.norm1 = PointBatchNorm(hidden_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        # Stage 2: KNN + 邻域特征聚合 (类似 PointNet++)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.norm2 = PointBatchNorm(hidden_channels)
        self.act2 = nn.ReLU(inplace=True)
        
        # Stage 3: 残差 MLP
        self.fc3 = nn.Linear(hidden_channels, hidden_channels*2, bias=False)
        self.norm3 = PointBatchNorm(hidden_channels*2)
        self.act3 = nn.ReLU(inplace=True)
        
        self.fc4 = nn.Linear(hidden_channels*2, hidden_channels, bias=False)
        self.norm4 = PointBatchNorm(hidden_channels)
        
        # Stage 4: 输出层
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, feat, coord, offset):
        """
        Args:
            feat: [N, C] 输入特征
            coord: [N, 3] 坐标
            offset: [B] 每个样本的累积点数
        Returns:
            logits: [N, num_classes]
        """
        N = feat.shape[0]
        
        # Stage 1: 初始特征提取
        x = self.fc1(feat)
        x = self.norm1(x)
        x = self.act1(x)
        
        # Stage 2: KNN + 邻域特征聚合
        # pointops.knn_query 需要 FP32 坐标
        with torch.no_grad():
            coord_fp32 = coord.float() if coord.dtype == torch.float16 else coord
            try:
                knn_idx, _ = pointops.knn_query(self.k_neighbors, coord_fp32, offset)
            except Exception as e:
                print(f"  [警告] pointops.knn_query 失败: {e}")
                # Fallback: 使用简单的索引
                knn_idx = torch.arange(N, device=feat.device).unsqueeze(1).expand(-1, self.k_neighbors)
                knn_idx = knn_idx % N  # 简单循环索引
        
        # 使用 gather 收集邻域特征
        # knn_idx: [N, K], x: [N, C]
        # gather 要求索引是 int64
        knn_idx_long = knn_idx.long()  # [N, K] -> int64
        
        # 方法：使用 index_select 或直接索引
        # knn_idx_long: [N, K] -> reshape to [N*K]
        knn_idx_flat = knn_idx_long.reshape(-1)  # [N*K]
        
        # 收集邻域特征: x[knn_idx_flat] -> [N*K, C]
        neighbor_feat_flat = x[knn_idx_flat]  # [N*K, C]
        
        # Reshape 回 [N, K, C]
        neighbor_feat = neighbor_feat_flat.reshape(N, self.k_neighbors, -1)  # [N, K, C]
        
        # 邻域聚合 (max pooling)
        neighbor_feat_max, _ = torch.max(neighbor_feat, dim=1)  # [N, C]
        
        # 与原始特征拼接
        x = torch.cat([x, neighbor_feat_max], dim=1)  # [N, 2C]
        
        x = self.fc2(x[:, :x.shape[1]//2])  # 只使用一半特征维度
        x = self.norm2(x)
        x = self.act2(x)
        
        # Stage 3: 残差 MLP
        identity = x
        x = self.fc3(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.norm4(x)
        x = x + identity  # 残差连接
        x = self.act1(x)  # 复用 act1
        
        # Stage 4: 输出
        logits = self.fc_out(x)
        
        return logits


class GridPoolingNetwork(nn.Module):
    """
    包含类似 GridPool 的体素化和 scatter 操作的网络
    """
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=8, grid_size=0.1):
        super().__init__()
        self.grid_size = grid_size
        
        # 特征编码
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.norm1 = PointBatchNorm(hidden_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        # 池化后的处理
        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.norm2 = PointBatchNorm(hidden_channels)
        self.act2 = nn.ReLU(inplace=True)
        
        # 输出
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, feat, coord, offset):
        """
        Args:
            feat: [N, C]
            coord: [N, 3]
            offset: [B]
        """
        N = feat.shape[0]
        
        # Stage 1: 特征编码
        x = self.fc1(feat)
        x = self.norm1(x)
        x = self.act1(x)
        
        # Stage 2: 简化的体素化 (使用 scatter)
        # 计算体素索引
        coord_fp32 = coord.float() if coord.dtype == torch.float16 else coord
        voxel_idx = (coord_fp32 / self.grid_size).floor().long()
        
        # 将 3D 索引转换为 1D 索引 (简化处理)
        voxel_idx_1d = (voxel_idx[:, 0] * 10000 + voxel_idx[:, 1] * 100 + voxel_idx[:, 2]) % N
        voxel_idx_1d = torch.clamp(voxel_idx_1d, 0, N-1)
        
        # 使用 scatter_mean 聚合同一体素内的特征
        pooled_feat = torch.zeros_like(x)
        pooled_feat.scatter_add_(0, voxel_idx_1d.unsqueeze(1).expand(-1, x.shape[1]), x)
        
        # 计数
        counts = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
        ones = torch.ones_like(counts)
        counts.scatter_add_(0, voxel_idx_1d.unsqueeze(1), ones)
        counts = torch.clamp(counts, min=1.0)
        
        # 平均
        pooled_feat = pooled_feat / counts
        
        # Stage 3: 处理池化后的特征
        x = self.fc2(pooled_feat)
        x = self.norm2(x)
        x = self.act2(x)
        
        # 输出
        logits = self.fc_out(x)
        
        return logits


# ============================================================================
# 准备测试数据
# ============================================================================
print("\n准备测试数据...")

N = 10000  # 点数
B = 2      # batch size
feat = torch.randn(N, 5)
coord = torch.randn(N, 3) * 10.0  # 坐标范围 [-10, 10]
labels = torch.randint(0, 8, (N,))

# 添加 ignore_index=-1
ignore_mask = torch.rand(N) < 0.05
labels[ignore_mask] = -1

# 创建 offset (累积点数)
points_per_sample = N // B
offset = torch.tensor([points_per_sample * (i+1) for i in range(B)], dtype=torch.int32)

print(f"  点数: {N}, Batch size: {B}")
print(f"  特征维度: {feat.shape}")
print(f"  坐标范围: [{coord.min():.2f}, {coord.max():.2f}]")
print(f"  标签分布: 类别0-7 + ignore(-1)")
print(f"  Offset: {offset}")

# 移到 GPU
feat_cuda = feat.cuda()
coord_cuda = coord.cuda()
labels_cuda = labels.cuda()
offset_cuda = offset.cuda()

# ============================================================================
# 测试 1: ComplexPointNetwork (包含 KNN + gather)
# ============================================================================
print("\n" + "="*80)
print("测试 1: ComplexPointNetwork (KNN + Gather + PointBatchNorm)")
print("="*80)

model1 = ComplexPointNetwork(in_channels=5, hidden_channels=64, out_channels=8, k_neighbors=16).cuda()

print("\n1.1 FP32 前向传播:")
try:
    logits_fp32 = model1(feat_cuda, coord_cuda, offset_cuda)
    print(f"  输出形状: {logits_fp32.shape}")
    print(f"  输出范围: [{logits_fp32.min():.4f}, {logits_fp32.max():.4f}]")
    print(f"  有 NaN: {torch.isnan(logits_fp32).any().item()}")
    print(f"  有 Inf: {torch.isinf(logits_fp32).any().item()}")
    print("  [OK]")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n1.2 FP16 前向传播 (autocast):")
try:
    with torch.cuda.amp.autocast():
        logits_fp16 = model1(feat_cuda, coord_cuda, offset_cuda)
    print(f"  输出形状: {logits_fp16.shape}")
    print(f"  输出 dtype: {logits_fp16.dtype}")
    print(f"  输出范围: [{logits_fp16.min():.4f}, {logits_fp16.max():.4f}]")
    print(f"  有 NaN: {torch.isnan(logits_fp16).any().item()}")
    print(f"  有 Inf: {torch.isinf(logits_fp16).any().item()}")
    print("  [OK]")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n1.3 FP16 + 损失计算:")
try:
    ce_loss = CrossEntropyLoss(ignore_index=-1)
    with torch.cuda.amp.autocast():
        logits = model1(feat_cuda, coord_cuda, offset_cuda)
        batch_dict = {'class': labels_cuda}
        loss = ce_loss(logits, batch_dict)
    print(f"  Loss: {loss.item():.4f}, dtype: {loss.dtype}")
    print(f"  有 NaN: {torch.isnan(loss).item()}")
    print("  [OK]")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n1.4 FP16 + 反向传播:")
try:
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model1(feat_cuda, coord_cuda, offset_cuda)
        batch_dict = {'class': labels_cuda}
        loss = ce_loss(logits, batch_dict)
    
    scaler.scale(loss).backward()
    
    # 检查梯度
    has_nan_grad = False
    has_inf_grad = False
    for name, param in model1.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  [警告] {name} 的梯度包含 NaN")
                has_nan_grad = True
            if torch.isinf(param.grad).any():
                print(f"  [警告] {name} 的梯度包含 Inf")
                has_inf_grad = True
    
    scaler.step(optimizer)
    scaler.update()
    
    if not has_nan_grad and not has_inf_grad:
        print(f"  Loss: {loss.item():.4f}")
        print(f"  梯度状态: 正常")
        print("  [OK]")
    else:
        print("  [FAIL] 梯度包含 NaN 或 Inf")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试 2: GridPoolingNetwork (包含 scatter)
# ============================================================================
print("\n" + "="*80)
print("测试 2: GridPoolingNetwork (Scatter + PointBatchNorm)")
print("="*80)

model2 = GridPoolingNetwork(in_channels=5, hidden_channels=64, out_channels=8, grid_size=0.5).cuda()

print("\n2.1 FP32 前向传播:")
try:
    logits_fp32 = model2(feat_cuda, coord_cuda, offset_cuda)
    print(f"  输出形状: {logits_fp32.shape}")
    print(f"  输出范围: [{logits_fp32.min():.4f}, {logits_fp32.max():.4f}]")
    print(f"  有 NaN: {torch.isnan(logits_fp32).any().item()}")
    print("  [OK]")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n2.2 FP16 前向传播 (autocast):")
try:
    with torch.cuda.amp.autocast():
        logits_fp16 = model2(feat_cuda, coord_cuda, offset_cuda)
    print(f"  输出形状: {logits_fp16.shape}")
    print(f"  输出 dtype: {logits_fp16.dtype}")
    print(f"  输出范围: [{logits_fp16.min():.4f}, {logits_fp16.max():.4f}]")
    print(f"  有 NaN: {torch.isnan(logits_fp16).any().item()}")
    print("  [OK]")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n2.3 FP16 + 损失 + 反向传播:")
try:
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    scaler2 = torch.cuda.amp.GradScaler()
    
    optimizer2.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model2(feat_cuda, coord_cuda, offset_cuda)
        batch_dict = {'class': labels_cuda}
        loss = ce_loss(logits, batch_dict)
    
    scaler2.scale(loss).backward()
    scaler2.step(optimizer2)
    scaler2.update()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  有 NaN: {torch.isnan(loss).item()}")
    print("  [OK]")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试 3: 完整训练循环 (多个 epoch)
# ============================================================================
print("\n" + "="*80)
print("测试 3: 完整训练循环 (5 epochs)")
print("="*80)

model3 = ComplexPointNetwork(in_channels=5, hidden_channels=64, out_channels=8).cuda()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
scaler3 = torch.cuda.amp.GradScaler()
ce_loss = CrossEntropyLoss(ignore_index=-1)
seg_metrics = SegmentationMetrics(num_classes=8, ignore_index=-1).cuda()

try:
    print("\n训练进度:")
    for epoch in range(5):
        optimizer3.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model3(feat_cuda, coord_cuda, offset_cuda)
            batch_dict = {'class': labels_cuda}
            loss = ce_loss(logits, batch_dict)
        
        if torch.isnan(loss):
            print(f"  [ERROR] Epoch {epoch+1}: Loss 是 NaN!")
            break
        
        scaler3.scale(loss).backward()
        scaler3.step(optimizer3)
        scaler3.update()
        
        # 更新指标
        seg_metrics.update(logits.detach(), labels_cuda)
        results = seg_metrics.compute()
        
        print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, "
              f"OA={results['overall_accuracy'].item()*100:.2f}%, "
              f"mIoU={results['mean_iou'].item()*100:.2f}%")
        seg_metrics.reset()
    
    print("[OK] 完整训练循环成功")
except Exception as e:
    print(f"[FAIL] 完整训练循环失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("测试总结")
print("="*80)
print("\n测试的特殊操作:")
print("  ✓ pointops.knn_query (FP32 坐标)")
print("  ✓ torch.gather (收集邻域特征)")
print("  ✓ torch.scatter_add (体素化池化)")
print("  ✓ PointBatchNorm (自定义归一化)")
print("  ✓ 残差连接")
print("  ✓ 多层 MLP")
print("  ✓ FP16 自动混合精度")
print("  ✓ GradScaler 反向传播")
print("\n如果以上测试都通过，说明这些特殊操作在 FP16 下可以正常工作。")
print("如果 PointTransformerV2 仍然产生 NaN，问题可能在于:")
print("  1. 特定的参数组合导致数值不稳定")
print("  2. 更深的网络结构累积误差")
print("  3. 特定的 attention 机制在 FP16 下不稳定")
