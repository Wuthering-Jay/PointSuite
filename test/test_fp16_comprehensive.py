"""
全面测试 FP16 环境中的各个组件（不涉及 PointTransformerV2）
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

print("="*80)
print("FP16 组件全面测试")
print("="*80)

# ============================================================================
# 1. 准备测试数据和模型
# ============================================================================
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

# 创建数据 - 包含 ignore_index=-1 的情况
N = 50000
feat = torch.randn(N, 5)
labels = torch.randint(0, 8, (N,))

# 随机将 5% 的标签设置为 -1 (ignore_index)
ignore_mask = torch.rand(N) < 0.05
labels[ignore_mask] = -1

print(f"\n数据统计:")
print(f"  总点数: {N}")
print(f"  特征维度: {feat.shape}")
print(f"  标签范围: [{labels.min()}, {labels.max()}]")
print(f"  ignore_index=-1 的点数: {(labels == -1).sum()} ({(labels == -1).float().mean()*100:.2f}%)")
print(f"  各类别分布: {torch.bincount(labels[labels >= 0])}")

# ============================================================================
# 测试 1: torch.cuda.amp (手动混合精度)
# ============================================================================
print("\n" + "="*80)
print("测试 1: torch.cuda.amp 手动混合精度")
print("="*80)

model1 = SimpleMLP().cuda()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
scaler1 = torch.cuda.amp.GradScaler()

feat_cuda = feat.cuda()
labels_cuda = labels.cuda()

try:
    for i in range(5):
        optimizer1.zero_grad()
        
        # 前向传播 (FP16)
        with torch.cuda.amp.autocast():
            logits = model1(feat_cuda)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fn(logits, labels_cuda)
        
        # 检查 NaN
        if torch.isnan(loss):
            print(f"  [ERROR] 迭代 {i+1}: Loss 是 NaN!")
            break
        
        # 反向传播
        scaler1.scale(loss).backward()
        scaler1.step(optimizer1)
        scaler1.update()
        
        if i % 2 == 0:
            print(f"  迭代 {i+1}: Loss = {loss.item():.4f}, Scale = {scaler1.get_scale()}")
    
    print("[OK] torch.cuda.amp 测试成功")
except Exception as e:
    print(f"[FAIL] torch.cuda.amp 测试失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试 2: Lightning Trainer precision="16-mixed"
# ============================================================================
print("\n" + "="*80)
print("测试 2: Lightning Trainer precision='16-mixed'")
print("="*80)

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleMLP()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # 计算精度 (忽略 -1 标签)
        mask = y != -1
        if mask.sum() > 0:
            pred = logits[mask].argmax(dim=1)
            acc = (pred == y[mask]).float().mean()
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# 创建 DataLoader
dataset = TensorDataset(feat, labels)
train_loader = DataLoader(dataset, batch_size=10000, shuffle=False)

lit_model = LitModel()

try:
    trainer = pl.Trainer(
        max_epochs=2,
        precision="16-mixed",
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    trainer.fit(lit_model, train_loader)
    print("[OK] Lightning 16-mixed 测试成功")
except Exception as e:
    print(f"[FAIL] Lightning 16-mixed 测试失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试 3: 不同的损失函数
# ============================================================================
print("\n" + "="*80)
print("测试 3: 不同损失函数 (CrossEntropy, NLL, Focal)")
print("="*80)

model3 = SimpleMLP().cuda()

# 3.1 CrossEntropyLoss
print("\n3.1 CrossEntropyLoss (ignore_index=-1):")
try:
    with torch.cuda.amp.autocast():
        logits = model3(feat_cuda)
        loss_ce = nn.CrossEntropyLoss(ignore_index=-1)(logits, labels_cuda)
    print(f"  Loss: {loss_ce.item():.4f}, 有 NaN: {torch.isnan(loss_ce).item()}")
except Exception as e:
    print(f"  [ERROR] {e}")

# 3.2 CrossEntropyLoss with weights
print("\n3.2 CrossEntropyLoss (带类别权重):")
try:
    weights = torch.tensor([1.0, 1.5, 1.2, 1.3, 1.1, 1.4, 1.6, 1.2]).cuda()
    with torch.cuda.amp.autocast():
        logits = model3(feat_cuda)
        loss_weighted = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)(logits, labels_cuda)
    print(f"  Loss: {loss_weighted.item():.4f}, 有 NaN: {torch.isnan(loss_weighted).item()}")
except Exception as e:
    print(f"  [ERROR] {e}")

# 3.3 NLLLoss (需要 log_softmax)
print("\n3.3 NLLLoss (ignore_index=-1):")
try:
    with torch.cuda.amp.autocast():
        logits = model3(feat_cuda)
        log_probs = torch.log_softmax(logits, dim=1)
        loss_nll = nn.NLLLoss(ignore_index=-1)(log_probs, labels_cuda)
    print(f"  Loss: {loss_nll.item():.4f}, 有 NaN: {torch.isnan(loss_nll).item()}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ============================================================================
# 测试 4: 精度评价指标
# ============================================================================
print("\n" + "="*80)
print("测试 4: 精度评价指标 (FP16)")
print("="*80)

model4 = SimpleMLP().cuda()

with torch.cuda.amp.autocast():
    logits = model4(feat_cuda)

# 转换为 FP32 进行评价
logits_fp32 = logits.float()
pred = logits_fp32.argmax(dim=1)

# 忽略 -1 标签的精度计算
mask = labels_cuda != -1
if mask.sum() > 0:
    accuracy = (pred[mask] == labels_cuda[mask]).float().mean()
    print(f"  总体精度 (忽略 -1): {accuracy.item()*100:.2f}%")
    
    # 每个类别的精度
    print("\n  各类别精度:")
    for cls in range(8):
        cls_mask = (labels_cuda == cls)
        if cls_mask.sum() > 0:
            cls_acc = (pred[cls_mask] == labels_cuda[cls_mask]).float().mean()
            print(f"    类别 {cls}: {cls_acc.item()*100:.2f}% ({cls_mask.sum()} 个样本)")

# 混淆矩阵统计
print("\n  预测分布 (忽略 -1):")
valid_pred = pred[mask]
print(f"    预测类别分布: {torch.bincount(valid_pred, minlength=8)}")

# ============================================================================
# 测试 5: 极端情况 - 全是 ignore_index
# ============================================================================
print("\n" + "="*80)
print("测试 5: 极端情况 - 全是 ignore_index=-1")
print("="*80)

all_ignore_labels = torch.full((1000,), -1).cuda()
test_feat = torch.randn(1000, 5).cuda()

model5 = SimpleMLP().cuda()

try:
    with torch.cuda.amp.autocast():
        logits = model5(test_feat)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fn(logits, all_ignore_labels)
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  有 NaN: {torch.isnan(loss).item()}")
    print("[OK] 全 ignore_index 测试成功")
except Exception as e:
    print(f"[FAIL] 全 ignore_index 测试失败: {e}")

# ============================================================================
# 测试 6: 混合 - ignore_index=-1 的反向传播
# ============================================================================
print("\n" + "="*80)
print("测试 6: ignore_index=-1 的反向传播")
print("="*80)

model6 = SimpleMLP().cuda()
optimizer6 = torch.optim.Adam(model6.parameters(), lr=0.001)
scaler6 = torch.cuda.amp.GradScaler()

# 创建一个批次，50% 是 -1
batch_size = 10000
batch_feat = torch.randn(batch_size, 5).cuda()
batch_labels = torch.randint(0, 8, (batch_size,)).cuda()
batch_labels[:batch_size//2] = -1  # 前一半设置为 -1

print(f"  批次大小: {batch_size}")
print(f"  ignore_index=-1 的比例: {(batch_labels == -1).float().mean()*100:.2f}%")

try:
    optimizer6.zero_grad()
    
    with torch.cuda.amp.autocast():
        logits = model6(batch_feat)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fn(logits, batch_labels)
    
    print(f"  Loss: {loss.item():.4f}")
    
    scaler6.scale(loss).backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model6.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()
            print(f"    {name}: grad_norm = {grad_norm:.4f}, has_NaN = {has_nan}")
    
    if has_grad:
        scaler6.step(optimizer6)
        scaler6.update()
        print("[OK] ignore_index 反向传播测试成功")
    else:
        print("[WARNING] 没有梯度生成")
        
except Exception as e:
    print(f"[FAIL] ignore_index 反向传播测试失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试 7: 数据类型转换
# ============================================================================
print("\n" + "="*80)
print("测试 7: FP16 与 FP32 之间的数据类型转换")
print("="*80)

model7 = SimpleMLP().cuda()

# FP16 前向传播
with torch.cuda.amp.autocast():
    logits_fp16 = model7(feat_cuda[:1000])

print(f"  FP16 输出 dtype: {logits_fp16.dtype}")
print(f"  FP16 输出范围: [{logits_fp16.min().item():.4f}, {logits_fp16.max().item():.4f}]")

# 转换为 FP32
logits_fp32 = logits_fp16.float()
print(f"  FP32 输出 dtype: {logits_fp32.dtype}")
print(f"  FP32 输出范围: [{logits_fp32.min().item():.4f}, {logits_fp32.max().item():.4f}]")

# 在 FP32 上计算损失
loss_fp32 = nn.CrossEntropyLoss(ignore_index=-1)(logits_fp32, labels_cuda[:1000])
print(f"  FP32 Loss: {loss_fp32.item():.4f}, 有 NaN: {torch.isnan(loss_fp32).item()}")

print("[OK] 数据类型转换测试成功")

print("\n" + "="*80)
print("所有测试完成")
print("="*80)
