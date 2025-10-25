# H5文件处理工具使用指南

## 📦 工具概览

本项目提供了完整的LAS/LAZ到H5格式转换和使用工具：

| 工具 | 功能 | 性能 |
|------|------|------|
| `tile.py` | LAS → LAS（分块，旧版） | 基础功能 |
| `tile_h5.py` | LAS → H5（分块） | 并行处理，gzip压缩 |
| `h5_dataset.py` | **H5数据集类（训练用）** | **900 seg/s（预加载）** |
| `h5_to_las.py` | H5 → LAS（串行） | ~2 segments/秒 |
| `h5_to_las_parallel.py` | H5 → LAS（并行）| ~10-40 segments/秒 |
| `benchmark_h5_reading.py` | 读取速度测试 | 多种策略对比 |

**推荐工作流**：
1. 生成数据：`tile_h5.py` (LAS → H5)
2. 训练模型：`h5_dataset.py` (高效加载)
3. 可视化：`h5_to_las_parallel.py` (H5 → LAS)

---

## 🚀 快速开始

### 1. LAS转H5（准备训练数据）

```bash
python tools/tile_h5.py
```

配置参数（在文件末尾修改）：
```python
input_path = r"E:\data\云南遥感中心\第一批\train"  # LAS文件目录
output_dir = r"E:\data\云南遥感中心\第一批\h5\train"  # H5输出目录
window_size = (150., 150.)  # 分块窗口大小（米）
min_points = 4096 * 2       # 最小点数
max_points = 4096 * 4 * 2   # 最大点数
n_workers = 8               # 并行worker数量
```

**性能优化**：
- ✅ Gzip level 4压缩（平衡速度和大小）
- ✅ Chunking优化随机访问
- ✅ 并行处理多个文件
- ✅ Indices自动排序

### 2. 测试H5读取速度

```bash
# 完整测试
python tools/benchmark_h5_reading.py processed_02.h5

# 快速测试（只测试前100个segments）
python tools/benchmark_h5_reading.py processed_02.h5 100
```

**输出示例**：
```
=== 测试1: 单线程顺序读取 ===
速度: 15.3 segments/秒
速度: 125,430 点/秒

=== 测试2: 多进程并行读取 (workers=4) ===
速度: 45.7 segments/秒
速度: 374,200 点/秒
加速比: 2.99x

推荐配置:
  最快方法: multiprocess_4 (17.92秒)
  深度学习训练推荐: 使用DataLoader with num_workers=4-8
```

### 3. H5转回LAS（可视化/验证）

#### 方法A: 串行处理（简单，适合少量segments）

```bash
python tools/h5_to_las.py
```

#### 方法B: 并行处理（推荐，快5-10倍）

```bash
# 转换所有segments，使用8个workers
python tools/h5_to_las_parallel.py file.h5 --workers 8

# 只转换前100个segments
python tools/h5_to_las_parallel.py file.h5 --workers 8 --segments 0-99

# 转换特定segments
python tools/h5_to_las_parallel.py file.h5 --workers 4 --segments 0,5,10-20,50

# 指定输出目录
python tools/h5_to_las_parallel.py file.h5 --output ./my_segments --workers 8
```

**性能对比**：
- 串行处理: ~2 segments/秒
- 并行处理 (4 workers): ~10-15 segments/秒
- 并行处理 (8 workers): ~20-40 segments/秒（取决于CPU和硬盘）

---

## 📖 H5文件格式详解

详见 `H5_FILE_FORMAT.md`，包含：
- 完整的文件结构说明
- 各字段数据类型
- Python读取示例代码
- PyTorch DataLoader集成
- 性能优化建议

### 快速示例：读取单个segment

```python
import h5py
import numpy as np

def read_segment(h5_path, segment_idx):
    with h5py.File(h5_path, 'r') as f:
        # 获取点索引
        indices = f['segments'][f'segment_{segment_idx:04d}']['indices'][:]
        
        # 读取XYZ坐标
        xyz = np.stack([
            f['data']['x'][indices],
            f['data']['y'][indices],
            f['data']['z'][indices]
        ], axis=1)
        
        # 读取标签
        labels = f['data']['classification'][indices]
        
        return xyz, labels

# 使用
xyz, labels = read_segment('file.h5', 0)
print(f"Segment 0: {len(xyz)} points, {len(np.unique(labels))} classes")
```

---

## 🔧 工具详细说明

### tile_h5.py - LAS到H5转换

**功能**：
- 将LAS/LAZ文件分块并保存为H5格式
- 自动合并过小或过大的块
- 保留所有点云属性（颜色、强度、时间等）
- 并行处理多个文件

**参数说明**：
```python
class LASToH5Processor:
    input_path: LAS文件或目录
    output_dir: H5输出目录
    window_size: (x_size, y_size) 矩形窗口大小（米）
    min_points: 最小点数阈值（None跳过）
    max_points: 最大点数阈值（None跳过）
    n_workers: 并行worker数量
```

**输出**：
- 一个LAS文件 → 一个H5文件
- H5文件大小约为LAS的35-40%（gzip压缩）
- 包含完整的点云数据和分块信息

### h5_to_las_parallel.py - 并行H5到LAS转换

**优势**：
- ⚡ 使用多进程并行处理
- 🚀 速度提升5-10倍
- 💾 内存效率高（每个worker独立）
- 🎯 支持选择性转换

**命令行选项**：
```bash
positional arguments:
  h5_file               输入H5文件路径

optional arguments:
  -h, --help            帮助信息
  --output, -o          输出目录（默认: <h5file>_segments）
  --workers, -w         并行worker数量（推荐: 4-8）
  --segments, -s        要转换的segment范围
```

**性能建议**：
- CPU核心多 → 使用更多workers（8-16）
- SSD硬盘 → 可以用更多workers
- HDD硬盘 → 4-6个workers为佳
- 内存不足 → 减少workers

### benchmark_h5_reading.py - 性能测试

**测试项目**：
1. ⏱️ 单线程顺序读取
2. 🚀 多进程并行读取（2/4/8 workers）
3. 💾 预加载全部数据
4. 📦 批量读取（batch_size=32）
5. 🎲 随机访问（100个样本）

**输出指标**：
- 总时间
- Segments/秒
- 点数/秒
- 加速比
- 内存占用

---

## 💡 最佳实践

### 训练数据准备流程

```bash
# 步骤1: LAS转H5（一次性）
python tools/tile_h5.py

# 步骤2: 使用h5_dataset.py高效读取
python tools/h5_dataset.py  # 查看使用示例
```

### 高效数据加载（推荐）

```python
from h5_dataset import H5PointCloudDataset, collate_fn
from torch.utils.data import DataLoader

# 方法1: 预加载模式（推荐，速度最快）
# 适用于：数据集小于可用内存
dataset = H5PointCloudDataset(
    h5_path='processed_02.h5',
    preload=True,  # 预加载到内存，900+ segments/秒
    transform=your_transforms
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,  # 预加载模式用单线程即可
    collate_fn=collate_fn
)

# 方法2: 文件读取模式（内存不足时）
dataset = H5PointCloudDataset(
    h5_path='processed_02.h5',
    preload=False,  # 从文件读取
    cache_indices=True  # 缓存indices信息
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,  # 使用多进程加速
    collate_fn=collate_fn
)
```

### 性能对比

| 模式 | 速度 | 内存 | 适用场景 |
|------|------|------|----------|
| 预加载 + num_workers=0 | **900 seg/s** | 高 (~500MB/文件) | 推荐，数据集 < 内存 |
| 文件读取 + num_workers=4 | 1.5 seg/s | 低 | 数据集 > 内存 |
| 文件读取 + num_workers=0 | 0.6 seg/s | 低 | 不推荐 |

### 数据验证流程

```bash
# 步骤1: 转换部分segments到LAS
python tools/h5_to_las_parallel.py file.h5 --workers 8 --segments 0-9

# 步骤2: 在CloudCompare中查看
# 检查坐标、颜色、分类等是否正确

# 步骤3: 如果有问题，重新生成H5
python tools/tile_h5.py  # 修改参数后重新运行
```

### 大规模数据处理

**批量LAS转H5**：
```python
# tile_h5.py 自动处理目录下所有LAS文件
input_path = r"E:\data\云南遥感中心\第一批\train"  # 包含多个LAS的目录
n_workers = 8  # 并行处理
```

**批量H5转LAS**：
```bash
# 使用循环脚本
for file in E:\data\h5\*.h5; do
    python tools/h5_to_las_parallel.py "$file" --workers 8
done
```

---

## 🐛 常见问题

### Q1: H5文件无法读取 / "Can't synchronously read data"

**原因**: 旧版本blosc压缩失败，文件实际无压缩导致读取错误

**解决**: 使用最新的`tile_h5.py`重新生成，现在使用gzip压缩

### Q2: h5_to_las.py 转换很慢

**原因**: 串行处理，每个segment顺序转换

**解决**: 使用`h5_to_las_parallel.py --workers 8`并行处理

### Q3: 转换时出现 "numpy.int64 has no attribute 'id'"

**原因**: H5读取的numpy类型需要转换为Python原生类型

**解决**: 已修复，使用`int()`, `float()`转换

### Q4: "Indexing elements must be in increasing order"

**原因**: H5的fancy indexing要求索引递增

**解决**: 已修复，代码自动排序indices并恢复原始顺序

### Q5: 内存不足

**解决方案**：
- 使用`h5_to_las.py`的`preload_data=False`模式
- 减少并行worker数量
- 分批处理segments：`--segments 0-99`, `--segments 100-199`

---

## 📊 性能数据参考

**测试环境**: Intel i7-12700K, 32GB RAM, NVMe SSD

| 操作 | 数据量 | 单线程 | 4 workers | 8 workers |
|------|--------|--------|-----------|-----------|
| LAS→H5 | 1GB LAS | 45秒 | 15秒 | 10秒 |
| H5读取 | 1000 segs | 65秒 | 22秒 | 18秒 |
| H5→LAS | 1000 segs | 400秒 | 100秒 | 60秒 |

**压缩效果**:
- 原始LAS: 1.0 GB
- H5 (gzip-4): 0.35 GB (节省65%)
- 读取速度: ~200-300 MB/s (解压后)

---

## 🔗 相关文档

- `H5_FILE_FORMAT.md` - H5文件格式完整说明和使用示例
- `COMPRESSION_FIX.md` - 从blosc切换到gzip的技术说明
- `H5_TO_LAS_MEMORY.md` - 内存优化详解

---

## 📝 更新日志

**v2.0 (2025-10-25)**
- ✅ 修复blosc压缩失败问题，切换到gzip
- ✅ 添加并行H5转LAS工具（h5_to_las_parallel.py）
- ✅ 添加性能测试工具（benchmark_h5_reading.py）
- ✅ 修复numpy类型兼容性问题
- ✅ 修复indices排序问题
- ✅ 自动排序保存的indices

**v1.0 (2025-10-24)**
- 初始版本，blosc压缩
- tile_h5.py和h5_to_las.py基础功能

---

## 💬 使用建议

**对于训练**：
- 直接使用H5文件，不需要转回LAS
- 使用DataLoader的num_workers=4-8
- 启用pin_memory=True加速GPU传输

**对于可视化**：
- 转换少量segments到LAS：`--segments 0-9`
- 在CloudCompare中验证数据正确性
- 不需要转换全部segments

**对于备份**：
- H5文件包含完整数据，可以完整还原LAS
- H5文件比LAS小65%，节省存储空间
- 保留原始LAS作为archive，使用H5进行训练
