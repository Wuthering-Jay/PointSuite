# Dataset Architecture Refactoring

## Overview
重构了数据集架构，将其分为抽象基类和具体实现，提高代码的可维护性和可扩展性。

## 文件说明

### 1. dataset_base.py - 抽象基类
**职责**: 定义数据集的通用接口和行为

**关键特性**:
- 继承自 `torch.utils.data.Dataset` 和 `abc.ABC`
- 提供数据缓存机制 (`cache_data` 参数)
- 实现 `__len__` 和 `__getitem__` 方法
- 支持 transform 数据增强
- 支持 loop 参数（训练时重复数据集）
- 提供 `get_sample_info()` 获取样本元数据

**抽象方法** (子类必须实现):
```python
@abstractmethod
def _load_data_list(self) -> List[Dict[str, Any]]:
    """加载所有数据样本的列表"""
    pass

@abstractmethod
def _load_data(self, idx: int) -> Dict[str, Any]:
    """加载指定索引的数据"""
    pass
```

**参数**:
- `data_root`: 数据根目录
- `split`: 数据集划分 ('train', 'val', 'test')
- `assets`: 要加载的数据属性列表（如 ['coord', 'intensity', 'classification']）
- `transform`: 数据变换函数
- `ignore_label`: 忽略的标签值
- `loop`: 数据集循环次数
- `cache_data`: 是否缓存加载的数据

### 2. dataset_bin.py - Bin+Pkl格式实现
**职责**: 实现读取bin+pkl文件格式的具体逻辑

**数据格式**:
- `.bin`: 包含所有点云数据的二进制文件（numpy structured array）
- `.pkl`: 包含元数据的pickle文件，包括：
  - segments: 每个segment的信息（start_idx, end_idx, num_points, bounds等）
  - dtype: 数据类型描述
  - label_counts: 标签分布
  - 原始LAS文件的header
  - 处理参数（window_size, grid_size等）

**关键方法**:
```python
def _load_data_list(self):
    """扫描所有pkl文件，根据min_points/max_points过滤segments"""
    
def _load_data(self, idx):
    """使用memmap加载指定segment的数据"""
    
def get_segment_info(self, idx):
    """获取segment的详细元数据"""
    
def get_file_metadata(self, idx):
    """获取文件级别的元数据"""
    
def get_stats(self):
    """获取数据集统计信息"""
    
def print_stats(self):
    """打印数据集统计信息"""
```

**支持的Assets**:
- `coord`: XYZ坐标 (float32, shape: [N, 3])
- `intensity`: 强度值 (float32, 归一化到[0,1])
- `classification`: 分类标签 (int64)
- `color`: RGB颜色 (float32, 归一化到[0,1], shape: [N, 3])
- `return_number`: 回波编号 (int64)
- `number_of_returns`: 总回波数 (int64)

**特殊参数**:
- `min_points`: 过滤掉点数少于该值的segments
- `max_points`: 过滤掉点数多于该值的segments

## 使用示例

### 基础使用
```python
from pointsuite.datasets.dataset_bin import BinPklDataset

# 创建数据集
dataset = BinPklDataset(
    data_root='path/to/data',
    split='train',
    assets=['coord', 'intensity', 'classification'],
    min_points=100,
    max_points=None,
    cache_data=False
)

# 加载数据
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"Coordinates shape: {sample['coord'].shape}")
```

### 使用工厂函数
```python
from pointsuite.datasets.dataset_bin import create_dataset

dataset = create_dataset(
    data_root='path/to/data',
    split='train',
    min_points=100,
)
```

### 查看统计信息
```python
# 打印详细统计
dataset.print_stats()

# 获取统计字典
stats = dataset.get_stats()
print(f"Total samples: {stats['num_samples']}")
print(f"Average points per sample: {stats['num_points']['mean']:.1f}")
```

### 访问元数据
```python
# 样本基本信息
info = dataset.get_sample_info(0)
print(f"File: {info['file_name']}")
print(f"Points: {info['num_points']}")

# Segment详细信息
seg_info = dataset.get_segment_info(0)
print(f"Segment ID: {seg_info['segment_id']}")
print(f"Bounds: {seg_info['x_min']}, {seg_info['x_max']}")

# 文件级别元数据
file_meta = dataset.get_file_metadata(0)
print(f"Grid size: {file_meta.get('grid_size')}")
```

### 数据缓存
```python
# 启用缓存（适合小数据集）
dataset = BinPklDataset(
    data_root='path/to/data',
    cache_data=True  # 数据会被缓存在内存中
)
```

### 数据增强
```python
def my_transform(data):
    # 随机旋转、平移等
    coord = data['coord']
    # ... apply transforms ...
    data['coord'] = coord
    return data

dataset = BinPklDataset(
    data_root='path/to/data',
    transform=my_transform
)
```

## 数据加载流程

```
初始化
  │
  ├─> _load_data_list()        # 扫描所有文件，建立样本列表
  │     ├─> 查找所有 .pkl 文件
  │     ├─> 加载每个pkl的metadata
  │     ├─> 对每个segment应用过滤器
  │     └─> 构建 data_list
  │
获取样本 (dataset[idx])
  │
  ├─> 检查缓存
  │     ├─> 如果已缓存，直接返回
  │     └─> 否则继续
  │
  ├─> _load_data(idx)          # 加载实际数据
  │     ├─> 获取样本信息
  │     ├─> 打开 .pkl 获取segment信息
  │     ├─> 使用memmap打开 .bin 文件
  │     ├─> 提取segment的点云数据
  │     └─> 根据assets提取相应字段
  │
  ├─> 应用 transform (如果有)
  │
  ├─> 保存到缓存 (如果启用)
  │
  └─> 返回数据字典
```

## 性能优化

1. **使用 memmap**: 避免一次性加载整个文件到内存
2. **过滤器**: 使用 min_points/max_points 提前过滤不需要的samples
3. **缓存**: 对于小数据集，启用 cache_data 可以显著提速
4. **Assets**: 只加载需要的字段，减少I/O开销

## 扩展新格式

如果需要支持其他数据格式（如HDF5、NPZ等），只需：

1. 创建新的类继承 `DatasetBase`
2. 实现 `_load_data_list()` 和 `_load_data()` 方法
3. 可选：添加格式特定的辅助方法

示例:
```python
from pointsuite.datasets.dataset_base import DatasetBase

class HDF5Dataset(DatasetBase):
    def _load_data_list(self):
        # 扫描HDF5文件，构建样本列表
        pass
    
    def _load_data(self, idx):
        # 从HDF5文件加载数据
        pass
```

## 测试

运行测试脚本验证功能：
```bash
python tools1/test_dataset_refactored.py
```

测试内容包括：
1. 基础加载和迭代
2. 过滤功能
3. 元数据访问
4. 统计信息
5. 缓存性能
6. 不同asset组合

## 注意事项

1. **数据路径**: 确保 bin 和 pkl 文件在同一目录且文件名匹配
2. **内存使用**: 大数据集建议关闭缓存，使用memmap按需加载
3. **Assets顺序**: assets列表的顺序不影响功能
4. **标签处理**: 使用 ignore_label 参数标记要忽略的标签值
5. **数据类型**: 坐标和特征会自动转换为 float32，标签转换为 int64
