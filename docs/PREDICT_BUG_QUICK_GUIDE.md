# 预测99%为类别1问题 - 已修复！

## 问题根源（已确认）

**真正的bug**：`SegmentationWriter` 在 `write_on_batch_end` 方法中没有正确使用 `offset` 信息切分数据。

### 原始代码的致命错误

```python
# ❌ 错误的代码
bin_files = prediction['bin_file']  # 长度 = Batch Size (例如 4)
logits = prediction['logits'].cpu()  # [总点数, C] (例如 [200000, 8])
indices = prediction['indices'].cpu()  # [总点数]

# 错误循环：只循环了 Batch Size 次，而不是总点数！
for i in range(len(bin_files)):  # 只循环 4 次
    file_groups[bin_basename]['logits'].append(logits[i])  # 只保存了前4个点！
    file_groups[bin_basename]['indices'].append(indices[i])
```

**后果**：
- Batch Size = 4，总点数 = 200,000
- 只保存了前 4 个点的预测
- 剩余 199,996 个点在投票时 `counts == 0`，被强制赋值为 0
- 连续标签 0 映射为原始标签 1（地面）
- 结果：99% 的点都是类别 1

### 修复后的代码

```python
# ✅ 正确的代码
# 1. 获取 offset 信息
offsets = batch['offset'].cpu().numpy()  # [n1, n1+n2, n1+n2+n3, ...]

# 2. 使用 offset 切分数据
start_idx = 0
for i, end_idx in enumerate(offsets):
    # 切片获取该样本的所有点
    sample_logits = logits[start_idx:end_idx]  # 正确获取所有点！
    sample_indices = indices[start_idx:end_idx]
    
    file_groups[bin_basename]['logits'].append(sample_logits)
    file_groups[bin_basename]['indices'].append(sample_indices)
    
    start_idx = end_idx
```

## 已修复的内容

✅ **核心修复**：`pointsuite/utils/callbacks.py` 的 `write_on_batch_end` 方法
- 正确使用 `batch['offset']` 切分数据
- 所有点的预测现在都会被保存

✅ **调试增强**：
- 添加类别分布打印（映射前后）
- Test 和 Predict 阶段的分布对比
- 性能优化（向量化反向映射）

## 验证修复

无需重新训练！直接运行 predict：

```bash
python train_dales.py
```

或者创建单独的预测脚本：

```python
# predict_only.py
import torch
import pytorch_lightning as pl
from train_dales import task, datamodule  # 导入配置

trainer = pl.Trainer(...)
trainer.predict(
    task, 
    datamodule, 
    ckpt_path="outputs/dales/checkpoints/best.ckpt"  # 使用已训练的模型
)
```
