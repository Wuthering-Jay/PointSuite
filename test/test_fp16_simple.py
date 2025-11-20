"""
最简单的 FP16 测试 - 使用随机数据
"""
import torch
import torch.nn as nn

print("="*80)
print("FP16 基础测试 - 随机数据")
print("="*80)

# 简单的 MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
    
    def forward(self, x):
        return self.layers(x)

# 创建模型
model = SimpleMLP().cuda()
model.train()

# 创建随机数据
N = 50000  # 点数
feat = torch.randn(N, 5).cuda()  # 随机特征
labels = torch.randint(0, 8, (N,)).cuda()  # 随机标签

# 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()

print(f"\n数据: {feat.shape}, 标签: {labels.shape}")
print(f"特征范围: [{feat.min().item():.4f}, {feat.max().item():.4f}]")

# 测试 1: FP32 训练
print("\n" + "="*80)
print("测试 1: FP32 训练")
print("="*80)

try:
    logits = model(feat)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    
    print(f"Logits: {logits.shape}, Loss: {loss.item():.4f}")
    print(f"有 NaN: {torch.isnan(loss).item()}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("[OK] FP32 训练成功")
except Exception as e:
    print(f"[FAIL] FP32 训练失败: {e}")

# 测试 2: FP16 前向传播
print("\n" + "="*80)
print("测试 2: FP16 前向传播")
print("="*80)

try:
    with torch.cuda.amp.autocast():
        logits = model(feat)
    
    print(f"Logits: {logits.shape}, dtype: {logits.dtype}")
    print(f"Logits 范围: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"有 NaN: {torch.isnan(logits).any().item()}")
    
    print("[OK] FP16 前向传播成功")
except Exception as e:
    print(f"[FAIL] FP16 前向传播失败: {e}")

# 测试 3: FP16 训练（完整流程）
print("\n" + "="*80)
print("测试 3: FP16 训练（完整流程）")
print("="*80)

try:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        logits = model(feat)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
    
    print(f"Logits dtype: {logits.dtype}")
    print(f"Loss: {loss.item():.4f}, dtype: {loss.dtype}")
    print(f"有 NaN: {torch.isnan(loss).item()}")
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print("[OK] FP16 训练成功")
except Exception as e:
    print(f"[FAIL] FP16 训练失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 4: 多次迭代
print("\n" + "="*80)
print("测试 4: FP16 多次迭代（10次）")
print("="*80)

try:
    for i in range(10):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(feat)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if i % 2 == 0:
            print(f"  迭代 {i+1}: Loss = {loss.item():.4f}")
    
    print("[OK] FP16 多次迭代成功")
except Exception as e:
    print(f"[FAIL] FP16 多次迭代失败（在第 {i+1} 次）: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)
