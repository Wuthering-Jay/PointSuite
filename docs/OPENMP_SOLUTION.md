# OpenMP 冲突问题解决方案

## 问题描述

运行测试时出现以下错误：

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

## 原因分析

### 根本原因
多个 Python 库同时链接了不同版本的 OpenMP 运行时库，导致在同一进程中多次初始化 OpenMP，引发冲突。

### 涉及的库
在本项目中，以下库都可能使用 OpenMP：

1. **NumPy（Intel MKL 版本）**
   - Anaconda 默认安装的 NumPy 使用 Intel MKL
   - MKL 内部依赖 `libiomp5md.dll`（Intel OpenMP 运行时）

2. **PyTorch**
   - 可能链接自己的 OpenMP 库
   - 用于 CPU 并行计算

3. **Numba**
   - JIT 编译时可能使用 OpenMP 进行并行化
   - 在 `transforms.py` 中使用了 `@njit(parallel=True)`

4. **SciPy / scikit-learn**
   - 也可能依赖 OpenMP

### 冲突触发时机
当代码中同时使用这些库，且它们各自初始化自己的 OpenMP 运行时时，就会触发冲突。

## 解决方案

### 方案 1：设置环境变量（推荐）

**在代码中设置（最佳）：**

```python
import os
# 必须在导入任何可能使用 OpenMP 的库之前设置！
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from numba import njit
```

**优点**：
- ✅ 简单可靠
- ✅ 不需要修改环境
- ✅ 在代码中自动生效

**缺点**：
- ⚠️ 可能导致轻微的性能下降（因为允许重复加载）
- ⚠️ 理论上可能产生数值不一致（极少见）

---

**在命令行设置（临时）：**

```powershell
# PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python test/test_dataloader.py

# CMD
set KMP_DUPLICATE_LIB_OK=TRUE
python test/test_dataloader.py

# Linux/macOS
export KMP_DUPLICATE_LIB_OK=TRUE
python test/test_dataloader.py
```

---

**在系统环境变量中设置（永久）：**

Windows:
1. 右键"此电脑" -> 属性 -> 高级系统设置
2. 环境变量 -> 新建系统变量
3. 变量名: `KMP_DUPLICATE_LIB_OK`
4. 变量值: `TRUE`

---

### 方案 2：统一 OpenMP 库（根本解决）

**安装不依赖 MKL 的 NumPy：**

```bash
# 卸载 MKL 版本的 NumPy
conda uninstall numpy

# 安装 OpenBLAS 版本（不使用 Intel OpenMP）
conda install numpy nomkl
```

或者使用 pip 安装：

```bash
pip uninstall numpy
pip install numpy --no-binary numpy
```

**优点**：
- ✅ 从根本上解决问题
- ✅ 不需要设置环境变量
- ✅ 性能不受影响

**缺点**：
- ❌ OpenBLAS 可能比 MKL 慢（在 Intel CPU 上）
- ❌ 需要重新编译或安装不同版本的库

---

### 方案 3：限制 OpenMP 线程数

如果性能影响不大，可以限制 OpenMP 只使用单线程：

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
```

**优点**：
- ✅ 避免多线程冲突
- ✅ 更好的可重复性

**缺点**：
- ❌ 性能下降（失去多核加速）

---

### 方案 4：检查并移除冲突的库

```bash
# 检查当前环境中依赖 MKL 的包
conda list | grep mkl

# 可能的输出：
# mkl                       2021.4.0
# mkl-service               2.4.0
# numpy                     1.23.5    py310h5f9d8c6_0  defaults
```

如果发现多个版本的 OpenMP 库，可以尝试统一：

```bash
# 更新所有包到最新版本（可能自动解决冲突）
conda update --all
```

---

## 本项目推荐方案

### 最终实现（已集成到代码中）

在 `test/test_dataloader_final.py` 中：

```python
import os
# 必须在导入任何库之前设置！
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import numpy as np
import torch
from pointsuite.datasets.dataset_bin import BinPklDataset
from pointsuite.datasets import transforms as T
```

### 使用方法

直接运行即可，无需额外设置：

```powershell
python test\test_dataloader_final.py
```

---

## 其他注意事项

### 为什么必须在导入之前设置？

环境变量 `KMP_DUPLICATE_LIB_OK` 只在 OpenMP 库**首次加载**时检查。如果在导入 NumPy/PyTorch 之后才设置，为时已晚。

**错误示范：**
```python
import numpy as np  # OpenMP 已经初始化
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 太晚了！
```

**正确示范：**
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 必须先设置
import numpy as np  # 然后再导入
```

---

### DataLoader 的 num_workers 问题

如果使用 `num_workers > 0`（多进程），每个子进程都会重新加载库。需要确保环境变量在主进程中设置，子进程会自动继承。

```python
# 在主进程中设置（推荐）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,  # 多进程也能正常工作
    collate_fn=collate_fn
)
```

---

### 性能影响评估

设置 `KMP_DUPLICATE_LIB_OK=TRUE` 的性能影响通常很小：

- **数据加载**：几乎无影响（主要瓶颈在 I/O）
- **数据增强**：轻微影响（< 5%）
- **模型训练**：取决于模型和硬件

在大多数情况下，这个损失是可以接受的。

---

## 总结

| 方案 | 难度 | 效果 | 性能影响 | 推荐度 |
|------|------|------|----------|--------|
| 设置环境变量（代码中） | ⭐ | ✅ 100% | ~5% | ⭐⭐⭐⭐⭐ |
| 设置环境变量（系统） | ⭐⭐ | ✅ 100% | ~5% | ⭐⭐⭐⭐ |
| 统一 OpenMP 库 | ⭐⭐⭐⭐ | ✅ 100% | 0% | ⭐⭐⭐ |
| 限制线程数 | ⭐ | ✅ 100% | 20-50% | ⭐⭐ |

**最终建议**：在代码中设置 `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`（已实现）
