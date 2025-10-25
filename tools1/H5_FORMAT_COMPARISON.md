# H5格式对比：旧版 vs 快速版

## 格式对比

### 旧版格式（tile_h5.py）

```
file.h5
├── header/
│   ├── attrs: 元数据
│   └── vlrs/
├── data/                    # 全局数据数组
│   ├── x: [N] float64      # 所有点的x坐标
│   ├── y: [N] float64      # 所有点的y坐标
│   ├── z: [N] float64      # 所有点的z坐标
│   ├── classification: [N] # 所有点的分类
│   ├── intensity: [N]      # 可选字段
│   └── ...
├── label_statistics/
└── segments/
    ├── segment_0000/
    │   ├── indices: [M]    # 指向data数组的索引
    │   └── unique_labels
    ├── segment_0001/
    └── ...
```

**读取方式**：
```python
indices = f['segments']['segment_0000']['indices'][:]
x = f['data']['x'][indices]  # Fancy indexing
```

### 快速格式（tile_h5_fast_parallel.py）

```
file.h5
├── header/
│   └── attrs: 元数据
└── segments/
    ├── segment_0000/
    │   ├── x: [M] float64           # 直接存储该segment的x
    │   ├── y: [M] float64           # 直接存储该segment的y
    │   ├── z: [M] float64           # 直接存储该segment的z
    │   ├── classification: [M]      # 直接存储该segment的分类
    │   └── attrs: num_points
    ├── segment_0001/
    └── ...
```

**读取方式**：
```python
x = f['segments']['segment_0000']['x'][:]  # 直接连续读取
```

## 性能对比（19个文件，4931个segments，3.59亿点）

| 指标 | 旧版 | 快速版 | 提升 |
|------|------|--------|------|
| **生成速度** | ~60秒 | **34秒** | 1.8x ⚡ |
| **文件大小** | 350MB | 500MB | +43% |
| **压缩** | gzip-4 | 无 | - |
| **按需读取** | 1.5 seg/s | **650 seg/s** | **433x** 🚀 |
| **预加载读取** | - | **5445 seg/s** | **3630x** 🚀🚀🚀 |
| **随机读取延迟** | 2829ms | **1.5ms** | 1886x ⚡ |

## 核心差异

### 1. 存储方式
- **旧版**：全局数组 + 索引（类似数据库）
- **快速版**：每个segment独立存储（类似文件系统）

### 2. 读取性能
- **旧版**：Fancy indexing需要访问多个不连续的chunk
  ```python
  # indices = [100, 5000, 10000, ...] 
  # 需要解压多个8KB chunks，即使只读少量点
  x = f['data']['x'][indices]  # 慢！
  ```

- **快速版**：连续内存读取
  ```python
  # 一次读取连续的内存块
  x = f['segments']['segment_0000']['x'][:]  # 快！
  ```

### 3. 压缩策略
- **旧版**：gzip压缩 + chunking
  - 优点：节省空间（~65%压缩率）
  - 缺点：随机访问需要解压多个chunks

- **快速版**：无压缩 + contiguous layout
  - 优点：极快随机访问（0复制）
  - 缺点：文件较大（+43%）

## 工具对应关系

| 功能 | 旧版工具 | 快速版工具 |
|------|---------|-----------|
| **LAS→H5** | `tile_h5.py` | `tile_h5_fast_parallel.py` |
| **H5→LAS** | `h5_to_las_parallel.py` | `h5_fast_to_las.py` |
| **Dataset类** | `h5_dataset.py` | `h5_dataset_fast.py` |
| **多文件** | `multi_h5_dataset.py` | `h5_dataset_fast.py` (FastMultiH5Dataset) |

## 使用建议

### 选择旧版的场景
- ✅ 磁盘空间紧张
- ✅ 主要顺序访问数据
- ✅ 数据归档/长期存储
- ✅ 网络传输（文件更小）

### 选择快速版的场景（推荐）
- ✅ **大规模训练**（需要快速随机读取）
- ✅ 内存充足（可全预加载）
- ✅ 磁盘空间充足
- ✅ **追求极致性能**
- ✅ 频繁跨文件随机访问

## 转换指南

### 从旧版迁移到快速版

```bash
# 方法1：重新生成（推荐）
python tools/tile_h5_fast_parallel.py \
    --input /path/to/las_files \
    --output /path/to/h5_fast \
    --workers 8

# 方法2：旧版H5 → LAS → 快速版H5
# Step 1: H5转LAS
python tools/h5_to_las_parallel.py old_file.h5 --workers 8

# Step 2: LAS转快速H5
python tools/tile_h5_fast_parallel.py \
    --input ./old_file_segments \
    --output ./new_fast.h5 \
    --workers 8
```

### 从快速版导出到旧版

快速版不能直接转换为旧版格式，但可以通过LAS中转：

```bash
# Step 1: 快速H5 → LAS
python tools/h5_fast_to_las.py fast_file.h5 --workers 8

# Step 2: LAS → 旧版H5
python tools/tile_h5.py \
    --input ./fast_file_segments \
    --output ./old_format.h5 \
    --workers 8
```

## 数据集使用示例

### 旧版格式

```python
from h5_dataset import H5PointCloudDataset, MultiH5Dataset

# 单文件
dataset = H5PointCloudDataset(
    "file.h5",
    preload=True  # 预加载可达900 seg/s
)

# 多文件
dataset = MultiH5Dataset(
    h5_files,
    preload_all=True  # LRU缓存
)

# 训练
dataloader = DataLoader(dataset, batch_size=16, num_workers=0)
```

### 快速格式

```python
from h5_dataset_fast import FastH5Dataset, FastMultiH5Dataset

# 单文件
dataset = FastH5Dataset(
    "file.h5",
    preload=True  # 可达5000+ seg/s
)

# 多文件
dataset = FastMultiH5Dataset(
    h5_files,
    preload_strategy="all"  # 或 "none" 或 "first-20"
)

# 训练（推荐配置）
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    num_workers=0,  # 预加载时用0
    shuffle=True
)
```

## 性能调优建议

### 旧版格式
```python
# 配置1：小内存场景
dataset = H5PointCloudDataset("file.h5", preload=False)
loader = DataLoader(dataset, num_workers=4)  # 多进程I/O
# 性能：~10 seg/s

# 配置2：大内存场景（推荐）
dataset = H5PointCloudDataset("file.h5", preload=True)
loader = DataLoader(dataset, num_workers=0)  # 避免序列化开销
# 性能：~900 seg/s
```

### 快速格式
```python
# 配置1：按需加载（内存有限）
dataset = FastMultiH5Dataset(files, preload_strategy="none")
loader = DataLoader(dataset, num_workers=0)
# 性能：~650 seg/s（仍然很快！）

# 配置2：全预加载（推荐，需要~10GB RAM）
dataset = FastMultiH5Dataset(files, preload_strategy="all")
loader = DataLoader(dataset, num_workers=0)
# 性能：~5445 seg/s（极致速度！）

# 配置3：部分预加载（平衡方案）
dataset = FastMultiH5Dataset(files, preload_strategy="first-10")
loader = DataLoader(dataset, num_workers=0)
# 前10个文件：5445 seg/s，其他：650 seg/s
```

## 总结

快速格式是**为深度学习训练量身定制**的格式：

- ✅ 生成更快（34秒 vs 60秒）
- ✅ 读取快433-3630倍
- ✅ 简化了代码（无需indices排序）
- ✅ 支持极高效的随机访问
- ⚠️ 文件增大43%（可接受的trade-off）

**对于大规模训练场景，强烈推荐使用快速格式！**
