# H5文件格式说明文档

## 文件概述

H5文件是通过`tile_h5.py`从LAS文件生成的，用于深度学习训练的点云数据格式。每个H5文件对应一个LAS文件，包含了所有点的数据以及分块（segment）信息。

## 文件结构

```
file.h5
├── header/                    # LAS文件头信息
│   ├── @point_format         # 点格式ID (int)
│   ├── @version_major        # 版本号主版本 (int)
│   ├── @version_minor        # 版本号次版本 (int)
│   ├── @x_scale              # X坐标缩放因子 (float)
│   ├── @y_scale              # Y坐标缩放因子 (float)
│   ├── @z_scale              # Z坐标缩放因子 (float)
│   ├── @x_offset             # X坐标偏移量 (float)
│   ├── @y_offset             # Y坐标偏移量 (float)
│   ├── @z_offset             # Z坐标偏移量 (float)
│   ├── @crs                  # 坐标参考系统 (可选)
│   └── vlrs/                 # VLR记录 (可选)
│
├── data/                      # 全局点云数据
│   ├── @available_fields     # 可用字段列表 (JSON string)
│   ├── x[N]                  # X坐标 (float64, gzip压缩)
│   ├── y[N]                  # Y坐标 (float64, gzip压缩)
│   ├── z[N]                  # Z坐标 (float64, gzip压缩)
│   ├── classification[N]     # 分类标签 (int32, gzip压缩)
│   ├── intensity[N]          # 强度 (uint16, gzip压缩, 可选)
│   ├── return_number[N]      # 回波编号 (uint8, gzip压缩, 可选)
│   ├── number_of_returns[N]  # 总回波数 (uint8, gzip压缩, 可选)
│   ├── red[N]                # 红色通道 (uint16, gzip压缩, 可选)
│   ├── green[N]              # 绿色通道 (uint16, gzip压缩, 可选)
│   ├── blue[N]               # 蓝色通道 (uint16, gzip压缩, 可选)
│   ├── gps_time[N]           # GPS时间 (float64, gzip压缩, 可选)
│   ├── scan_angle_rank[N]    # 扫描角 (int8, gzip压缩, 可选)
│   ├── user_data[N]          # 用户数据 (uint8, gzip压缩, 可选)
│   └── point_source_id[N]    # 点源ID (uint16, gzip压缩, 可选)
│
├── label_statistics/          # 标签统计信息
│   ├── @label_0              # 标签0的点数
│   ├── @label_1              # 标签1的点数
│   └── ...                   # 其他标签
│
└── segments/                  # 分块信息
    ├── @num_segments         # 总分块数
    ├── segment_0000/         # 第一个分块
    │   ├── @num_points       # 该分块的点数
    │   ├── indices[M]        # 点在全局数据中的索引 (int64, 已排序)
    │   └── unique_labels[K]  # 该分块包含的唯一标签
    ├── segment_0001/         # 第二个分块
    └── ...
```

其中 N = 总点数, M = 该分块点数, K = 该分块唯一标签数

## 数据读取方法

### 1. 读取文件头信息

```python
import h5py

with h5py.File('file.h5', 'r') as f:
    # 读取LAS版本和格式
    point_format = int(f['header'].attrs['point_format'])
    version_major = int(f['header'].attrs['version_major'])
    version_minor = int(f['header'].attrs['version_minor'])
    
    # 读取坐标变换参数
    x_scale = float(f['header'].attrs['x_scale'])
    y_scale = float(f['header'].attrs['y_scale'])
    z_scale = float(f['header'].attrs['z_scale'])
    x_offset = float(f['header'].attrs['x_offset'])
    y_offset = float(f['header'].attrs['y_offset'])
    z_offset = float(f['header'].attrs['z_offset'])
    
    print(f"LAS {version_major}.{version_minor}, Point Format {point_format}")
```

### 2. 读取基本分块信息

```python
import h5py
import json

with h5py.File('file.h5', 'r') as f:
    # 获取分块总数
    num_segments = f['segments'].attrs['num_segments']
    
    # 获取可用字段
    available_fields = json.loads(f['data'].attrs['available_fields'])
    
    print(f"总分块数: {num_segments}")
    print(f"可用字段: {available_fields}")
```

### 3. 读取单个分块的数据

```python
import h5py
import numpy as np

def read_segment(h5_path, segment_idx):
    """读取指定分块的点云数据"""
    with h5py.File(h5_path, 'r') as f:
        # 获取该分块的点索引
        indices = f['segments'][f'segment_{segment_idx:04d}']['indices'][:]
        
        # 读取坐标
        x = f['data']['x'][indices]
        y = f['data']['y'][indices]
        z = f['data']['z'][indices]
        
        # 读取分类标签
        labels = f['data']['classification'][indices]
        
        # 读取其他字段（如果存在）
        intensity = None
        if 'intensity' in f['data']:
            intensity = f['data']['intensity'][indices]
        
        return {
            'xyz': np.stack([x, y, z], axis=1),  # [N, 3]
            'labels': labels,                     # [N]
            'intensity': intensity                # [N] or None
        }

# 使用示例
data = read_segment('file.h5', 0)
print(f"分块0包含 {len(data['xyz'])} 个点")
```

### 4. 批量读取多个分块

```python
import h5py
import numpy as np

def read_multiple_segments(h5_path, segment_indices):
    """批量读取多个分块"""
    results = []
    
    with h5py.File(h5_path, 'r') as f:
        for seg_idx in segment_indices:
            indices = f['segments'][f'segment_{seg_idx:04d}']['indices'][:]
            
            data = {
                'xyz': np.stack([
                    f['data']['x'][indices],
                    f['data']['y'][indices],
                    f['data']['z'][indices]
                ], axis=1),
                'labels': f['data']['classification'][indices]
            }
            results.append(data)
    
    return results

# 读取前10个分块
segments = read_multiple_segments('file.h5', range(10))
```

### 5. 获取分块的标签分布

```python
import h5py

with h5py.File('file.h5', 'r') as f:
    seg_group = f['segments']['segment_0000']
    
    # 获取该分块包含的唯一标签
    unique_labels = seg_group['unique_labels'][:]
    
    # 获取点数
    num_points = seg_group.attrs['num_points']
    
    print(f"分块包含 {num_points} 个点")
    print(f"唯一标签: {unique_labels}")
```

### 6. 获取全局标签统计

```python
import h5py

with h5py.File('file.h5', 'r') as f:
    stats_group = f['label_statistics']
    
    # 遍历所有标签统计
    label_counts = {}
    for attr_name in stats_group.attrs:
        if attr_name.startswith('label_'):
            label_id = int(attr_name.split('_')[1])
            count = stats_group.attrs[attr_name]
            label_counts[label_id] = count
    
    print("标签分布:")
    for label_id, count in sorted(label_counts.items()):
        print(f"  标签 {label_id}: {count} 个点")
```

## 深度学习数据加载示例

### PyTorch DataLoader

```python
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class H5SegmentDataset(Dataset):
    """从H5文件加载点云分块的PyTorch数据集"""
    
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        # 预先读取所有segment的索引信息（内存占用小）
        with h5py.File(h5_path, 'r') as f:
            self.num_segments = f['segments'].attrs['num_segments']
    
    def __len__(self):
        return self.num_segments
    
    def __getitem__(self, idx):
        # 每次读取一个segment
        with h5py.File(self.h5_path, 'r') as f:
            indices = f['segments'][f'segment_{idx:04d}']['indices'][:]
            
            # 读取坐标和标签
            xyz = np.stack([
                f['data']['x'][indices],
                f['data']['y'][indices],
                f['data']['z'][indices]
            ], axis=1).astype(np.float32)
            
            labels = f['data']['classification'][indices].astype(np.int64)
            
            # 读取其他特征（可选）
            features = []
            if 'intensity' in f['data']:
                features.append(f['data']['intensity'][indices])
            if 'red' in f['data']:
                features.append(f['data']['red'][indices])
                features.append(f['data']['green'][indices])
                features.append(f['data']['blue'][indices])
            
            if features:
                features = np.stack(features, axis=1).astype(np.float32)
            else:
                features = None
        
        # 应用数据增强
        if self.transform:
            xyz, labels, features = self.transform(xyz, labels, features)
        
        # 返回字典
        sample = {
            'coord': torch.from_numpy(xyz),
            'label': torch.from_numpy(labels)
        }
        if features is not None:
            sample['feat'] = torch.from_numpy(features)
        
        return sample

# 使用示例
dataset = H5SegmentDataset('file.h5')
dataloader = DataLoader(
    dataset, 
    batch_size=8,
    shuffle=True,
    num_workers=4,  # 多进程加载
    pin_memory=True
)

for batch in dataloader:
    coords = batch['coord']  # [B, N, 3]
    labels = batch['label']  # [B, N]
    # 训练模型...
```

## 性能优化建议

### 1. 使用chunking和compression

文件已使用gzip压缩和chunking：
- **Chunking**: 8192点/块，优化随机访问
- **Compression**: gzip level 4，平衡速度和压缩率
- **Shuffle filter**: 提升数值数据压缩率

### 2. 多进程数据加载

```python
# PyTorch
dataloader = DataLoader(dataset, num_workers=4)

# 或手动多进程
from concurrent.futures import ProcessPoolExecutor

def load_segment(h5_path, idx):
    # ... 读取逻辑
    pass

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(load_segment, [h5_path]*10, range(10)))
```

### 3. 预加载索引信息

```python
# 在初始化时一次性读取所有索引
with h5py.File('file.h5', 'r') as f:
    all_indices = []
    for i in range(num_segments):
        indices = f['segments'][f'segment_{i:04d}']['indices'][:]
        all_indices.append(indices)
    
# 后续使用预加载的索引
```

### 4. 缓存频繁访问的数据

对于小数据集，可以一次性加载到内存：

```python
class CachedH5Dataset(Dataset):
    def __init__(self, h5_path):
        # 一次性加载所有数据到内存
        with h5py.File(h5_path, 'r') as f:
            self.data = {
                'x': f['data']['x'][:],
                'y': f['data']['y'][:],
                'z': f['data']['z'][:],
                'labels': f['data']['classification'][:]
            }
            
            self.indices_list = []
            num_seg = f['segments'].attrs['num_segments']
            for i in range(num_seg):
                self.indices_list.append(
                    f['segments'][f'segment_{i:04d}']['indices'][:]
                )
    
    def __getitem__(self, idx):
        indices = self.indices_list[idx]
        return {
            'xyz': np.stack([
                self.data['x'][indices],
                self.data['y'][indices],
                self.data['z'][indices]
            ], axis=1),
            'labels': self.data['labels'][indices]
        }
```

## 注意事项

1. **索引已排序**: segment的indices已经排序，H5读取效率更高
2. **数据类型**: 注意转换为正确的Python/numpy类型（int(), float()）
3. **文件打开模式**: 使用'r'模式只读，避免意外修改
4. **上下文管理器**: 始终使用`with`语句确保文件正确关闭
5. **并发访问**: H5文件支持多进程读取（只读模式下）
6. **内存管理**: 大文件避免一次性加载所有数据

## 工具函数

项目中提供的工具：
- `tile_h5.py`: LAS → H5转换
- `h5_to_las.py`: H5 → LAS转换（可视化）
- `read_h5_example.py`: 读取和检查H5文件的示例

## 相关文档

- `COMPRESSION_FIX.md`: 压缩方式说明
- `H5_TO_LAS_MEMORY.md`: 内存优化说明
