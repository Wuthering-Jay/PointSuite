# 快速H5读取 - 速查表

## 📖 基本读取

### 1. 读取Header信息
```python
import h5py

with h5py.File('file.h5', 'r') as f:
    header = f['header']
    
    # 基本信息
    num_points = header.attrs['num_points']
    num_segments = f['segments'].attrs['num_segments']
    
    # 坐标参数
    x_scale = header.attrs['x_scale']
    x_offset = header.attrs['x_offset']
    
    # 可用字段
    fields = header.attrs['available_fields'].split(',')
```

### 2. 读取单个Segment
```python
with h5py.File('file.h5', 'r') as f:
    seg = f['segments']['segment_0000']
    
    # 读取坐标
    x = seg['x'][:]
    y = seg['y'][:]
    z = seg['z'][:]
    xyz = np.stack([x, y, z], axis=1)  # Shape: (N, 3)
    
    # 读取分类
    labels = seg['classification'][:]
    
    # 读取其他字段
    intensity = seg['intensity'][:]
    gps_time = seg['gps_time'][:]
```

### 3. 批量读取Segments
```python
# ⚠️ 保持文件打开以提升性能
with h5py.File('file.h5', 'r') as f:
    data = []
    for i in range(10):
        seg = f['segments'][f'segment_{i:04d}']
        xyz = np.stack([seg['x'][:], seg['y'][:], seg['z'][:]], axis=1)
        labels = seg['classification'][:]
        data.append((xyz, labels))
```

## 🚀 训练使用（推荐）

### 单文件训练
```python
from tools.h5_dataset_fast import FastH5Dataset, collate_fn
from torch.utils.data import DataLoader

# 创建Dataset（预加载到内存）
dataset = FastH5Dataset(
    'file.h5',
    preload=True  # 速度：5000+ seg/s
)

# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,  # ⚠️ 预加载时必须用0
    collate_fn=collate_fn
)

# 训练循环
for batch_xyz, batch_labels in dataloader:
    # batch_xyz: List[Tensor[N_i, 3]]
    # batch_labels: List[Tensor[N_i]]
    
    for xyz, labels in zip(batch_xyz, batch_labels):
        # xyz: Tensor[N, 3]
        # labels: Tensor[N]
        pass
```

### 多文件训练
```python
from tools.h5_dataset_fast import FastMultiH5Dataset, collate_fn

# 查找所有H5文件
h5_files = sorted(Path('h5_dir').glob('*.h5'))

# 创建Dataset
dataset = FastMultiH5Dataset(
    [str(f) for f in h5_files],
    preload_strategy="all"  # 或 "none" 或 "first-10"
)

# DataLoader配置相同
dataloader = DataLoader(dataset, batch_size=16, num_workers=0, collate_fn=collate_fn)
```

## 📊 性能对比

### 不同读取模式
| 模式 | 代码 | 速度 | 适用场景 |
|------|------|------|---------|
| 反复打开 | 每次`with h5py.File` | 580 seg/s | ❌ 不推荐 |
| 保持打开 | 外层`with h5py.File` | 700 seg/s | 推理 |
| 预加载 | `FastH5Dataset(preload=True)` | **5000+ seg/s** | ✅ 训练 |

### num_workers设置
| preload | num_workers | 速度 | 说明 |
|---------|------------|------|------|
| True | 0 | **5000+ seg/s** | ✅ 最快 |
| True | 4 | 100 seg/s | ❌ 序列化开销 |
| False | 0 | 650 seg/s | 单进程I/O |
| False | 4 | 700 seg/s | 并行I/O（提升小） |

**结论**：预加载时**必须**用`num_workers=0`！

## 🔍 常用操作

### 获取Segment信息
```python
with h5py.File('file.h5', 'r') as f:
    seg = f['segments']['segment_0000']
    
    # 点数
    num_points = len(seg['x'])
    # 或
    num_points = seg.attrs['num_points']
    
    # 字段列表
    fields = list(seg.keys())
```

### 统计类别分布
```python
with h5py.File('file.h5', 'r') as f:
    all_labels = []
    for i in range(10):  # 前10个segments
        labels = f['segments'][f'segment_{i:04d}']['classification'][:]
        all_labels.append(labels)
    
    all_labels = np.concatenate(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    
    for label, count in zip(unique, counts):
        print(f"类别 {label}: {count} 点")
```

### 筛选特定类别的点
```python
with h5py.File('file.h5', 'r') as f:
    seg = f['segments']['segment_0000']
    
    xyz = np.stack([seg['x'][:], seg['y'][:], seg['z'][:]], axis=1)
    labels = seg['classification'][:]
    
    # 只保留类别1和2
    mask = np.isin(labels, [1, 2])
    xyz_filtered = xyz[mask]
    labels_filtered = labels[mask]
```

### 查找大Segments
```python
with h5py.File('file.h5', 'r') as f:
    num_segs = f['segments'].attrs['num_segments']
    
    large_segments = []
    for i in range(num_segs):
        seg = f['segments'][f'segment_{i:04d}']
        if len(seg['x']) > 100000:
            large_segments.append(i)
    
    print(f"大segments (>100k点): {large_segments}")
```

## ⚠️ 注意事项

### 1. 内存管理
```python
# ✅ 好：19个文件约10GB，64GB RAM可全预加载
dataset = FastMultiH5Dataset(files, preload_strategy="all")

# ✅ 好：内存有限时按需加载（仍有650 seg/s）
dataset = FastMultiH5Dataset(files, preload_strategy="none")

# ✅ 好：折中方案
dataset = FastMultiH5Dataset(files, preload_strategy="first-10")
```

### 2. 字段大小写
```python
# H5中字段名是小写
with h5py.File('file.h5', 'r') as f:
    seg = f['segments']['segment_0000']
    x = seg['x'][:]  # ✅ 小写
    # X = seg['X'][:]  # ❌ 会报错
```

### 3. 数据类型
```python
# 坐标和GPS时间是float64
xyz = seg['x'][:]  # dtype: float64

# 分类是int32
labels = seg['classification'][:]  # dtype: int32

# 强度、颜色是uint16
intensity = seg['intensity'][:]  # dtype: uint16
```

## 📝 完整示例

### 最小训练代码
```python
from tools.h5_dataset_fast import FastMultiH5Dataset, collate_fn
from torch.utils.data import DataLoader
from pathlib import Path

# 数据准备
h5_files = sorted(Path('h5_dir').glob('*.h5'))
dataset = FastMultiH5Dataset(
    [str(f) for f in h5_files],
    preload_strategy="all"
)
dataloader = DataLoader(dataset, batch_size=16, num_workers=0, collate_fn=collate_fn)

# 训练
for epoch in range(10):
    for batch_xyz, batch_labels in dataloader:
        # 你的训练代码
        for xyz, labels in zip(batch_xyz, batch_labels):
            # xyz: Tensor[N, 3]
            # labels: Tensor[N]
            
            # Forward
            output = model(xyz)
            loss = criterion(output, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
```

### 最小推理代码
```python
import h5py
import numpy as np

with h5py.File('file.h5', 'r') as f:
    num_segs = f['segments'].attrs['num_segments']
    
    for i in range(num_segs):
        seg = f['segments'][f'segment_{i:04d}']
        xyz = np.stack([seg['x'][:], seg['y'][:], seg['z'][:]], axis=1)
        
        # 推理
        pred = model.predict(xyz)
```

## 🔗 相关文件

- **完整示例**: `example_h5_fast_reading.py`
- **Dataset类**: `h5_dataset_fast.py`
- **格式对比**: `H5_FORMAT_COMPARISON.md`
- **工具总览**: `README.md`
