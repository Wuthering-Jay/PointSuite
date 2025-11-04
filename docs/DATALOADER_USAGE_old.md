# DynamicBatchSampler 使用指南（已验证）

## 完整使用示例

```python
import torch
from torch.utils.data import DataLoader
from pointsuite.datasets import BinPklDataset


def collate_fn(batch):
    """
    自定义 collate 函数，用于处理不同点数的点云样本
    
    功能:
    - 将多个样本的点云拼接成一个大点云
    - 自动添加 offset 字段，标记每个样本的边界
    """
    import numpy as np
    from collections.abc import Mapping, Sequence
    
    if isinstance(batch[0], Mapping):
        keys = batch[0].keys()
        result = {}
        num_points_per_sample = []
        
        for key in keys:
            values = [torch.from_numpy(d[key]) if isinstance(d[key], np.ndarray) else d[key] 
                     for d in batch]
            
            # 拼接点云数据
            if key in ['coord', 'intensity', 'classification', 'color']:
                result[key] = torch.cat(values, dim=0)
                if key == 'coord':
                    num_points_per_sample = [v.shape[0] for v in values]
            else:
                result[key] = values
        
        # 添加 offset
        if len(num_points_per_sample) > 0:
            offset = torch.cumsum(torch.tensor([0] + num_points_per_sample), dim=0).int()
            result['offset'] = offset[1:]
        
        return result
    else:
        return torch.utils.data.dataloader.default_collate(batch)


# 创建数据集
dataset = BinPklDataset(
    data_root='path/to/your/data',
    split='train',
    assets=['coord', 'intensity', 'classification'],
    cache_data=False,  # 大数据集建议 False
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,  # Windows 建议 0，Linux 可用 2-4
    collate_fn=collate_fn,
    pin_memory=True,  # 使用 GPU 时建议开启
)

# 训练循环
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # batch 结构:
        # {
        #     'coord': Tensor[N, 3],           # 拼接后的所有点坐标
        #     'intensity': Tensor[N],          # 拼接后的所有点强度
        #     'classification': Tensor[N],     # 拼接后的所有点标签
        #     'offset': Tensor[batch_size]     # 每个样本的结束位置
        # }
        
        coord = batch['coord']              # [N, 3]
        intensity = batch['intensity']      # [N]
        labels = batch['classification']    # [N]
        offset = batch['offset']            # [batch_size]
        
        # 根据 offset 可以分离出每个样本
        # 样本 0: coord[0:offset[0]]
        # 样本 1: coord[offset[0]:offset[1]]
        # 样本 2: coord[offset[1]:offset[2]]
        # ...
        
        # GPU 训练
        if torch.cuda.is_available():
            coord = coord.cuda()
            intensity = intensity.cuda()
            labels = labels.cuda()
            offset = offset.cuda()
        
        # 前向传播
        output = model(coord, intensity, offset)
        loss = criterion(output, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Offset 使用说明

`offset` 字段非常重要，它标记了每个样本在拼接后大点云中的结束位置。

例如，batch_size=4 时：
- 样本0有 10000 个点
- 样本1有 15000 个点
- 样本2有 12000 个点
- 样本3有 13000 个点

则：
- `offset = [10000, 25000, 37000, 50000]`
- 拼接后的点云总共有 50000 个点

分离样本：
```python
start = 0
for i, end in enumerate(offset):
    sample_coord = coord[start:end]      # 第 i 个样本的坐标
    sample_labels = labels[start:end]    # 第 i 个样本的标签
    start = end
```

## 常见问题

### 1. 内存不足
**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小 batch_size (例如从 8 改为 2)
- 减小每个样本的点数（在数据预处理时设置更小的 window_size）
- 使用 gradient accumulation

### 2. 数据加载慢
**症状**: GPU 利用率低，大部分时间在等待数据

**解决方案**:
- 开启 `cache_data=True`（如果内存足够）
- 增加 `num_workers`（Linux）
- 使用 SSD 存储数据
- 开启 `pin_memory=True`

### 3. Windows 多进程错误
**症状**: `RuntimeError: DataLoader worker (pid xxxx) is killed`

**解决方案**:
- 设置 `num_workers=0`
- 或者使用 `if __name__ == '__main__'` 保护主程序

### 4. Shuffle 不生效
**症状**: 每个 epoch 的数据顺序相同

**解决方案**:
- 确保 `shuffle=True`
- 在每个 epoch 开始前不要重新创建 DataLoader

## 性能优化建议

### 小数据集 (< 1GB)
```python
dataset = BinPklDataset(
    data_root='path/to/data',
    cache_data=True,    # ✓ 缓存所有数据
)

dataloader = DataLoader(
    dataset,
    batch_size=8,       # ✓ 可以用较大的 batch
    num_workers=0,
    pin_memory=True,
)
```

### 大数据集 (> 10GB)
```python
dataset = BinPklDataset(
    data_root='path/to/data',
    cache_data=False,   # ✓ 不缓存，按需加载
)

dataloader = DataLoader(
    dataset,
    batch_size=2,       # ✓ 小 batch size
    num_workers=2,      # ✓ Linux 下使用多进程
    pin_memory=True,
    prefetch_factor=2,  # ✓ 预加载
)
```

## 测试

运行测试以确保 DataLoader 工作正常：

```bash
# 测试基础数据集功能
python test/test_bin_dataset.py

# 测试 DataLoader 集成
python test/test_dataloader.py
```
