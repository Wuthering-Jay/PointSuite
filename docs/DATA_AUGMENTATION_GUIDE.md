# 数据增强完整指南

## 新增功能概览

### 1. 自动归一化 Transforms

不再需要在 `dataset_bin.py` 中硬编码位数，所有归一化在 transforms 中自动完成。

### 2. 支持的特征增强

- ✅ **Intensity（强度）**: 自动检测 8/16 位，归一化和增强
- ✅ **Color（颜色）**: 自动检测 8/16 位，归一化和增强
- ✅ **h_norm（归一化高程）**: 多种归一化和增强方法

---

## 一、Intensity 增强

### 1.1 自动归一化（推荐）

```python
from pointsuite.data.transforms import AutoNormalizeIntensity

# 自动检测位数并归一化
transform = AutoNormalizeIntensity(target_range=(0, 1))
```

**自动检测逻辑**：
- `max <= 1.0` → 已归一化
- `max <= 255` → 8 位，除以 255
- `max <= 65535` → 16 位，除以 65535
- 其他 → 使用 min-max 归一化

### 1.2 指定位数归一化

```python
from pointsuite.data.transforms import NormalizeIntensity

# 16 位归一化
transform_16bit = NormalizeIntensity(max_value=65535.0)

# 8 位归一化
transform_8bit = NormalizeIntensity(max_value=255.0)
```

### 1.3 Intensity 增强

```python
from pointsuite.data.transforms import (
    RandomIntensityScale,      # 随机缩放
    RandomIntensityShift,       # 随机偏移
    RandomIntensityNoise,       # 随机噪声
    RandomIntensityGamma,       # Gamma 校正
    RandomIntensityDrop,        # 随机丢弃
    StandardNormalizeIntensity, # 标准化（Z-score）
    MinMaxNormalizeIntensity,   # MinMax 归一化
)

# 完整的 Intensity 增强管道
intensity_transforms = [
    AutoNormalizeIntensity(target_range=(0, 1)),        # 自动归一化到 [0, 1]
    RandomIntensityScale(scale=(0.8, 1.2), p=0.95),     # 随机亮度
    RandomIntensityNoise(sigma=0.01, p=0.5),            # 添加噪声
    RandomIntensityGamma(gamma_range=(0.8, 1.2), p=0.5),# Gamma 校正
]
```

---

## 二、Color 增强

### 2.1 自动归一化（推荐）

```python
from pointsuite.data.transforms import AutoNormalizeColor

# 自动归一化到 [0, 255]（大部分颜色增强期望的范围）
transform = AutoNormalizeColor(target_range=(0, 255))
```

**自动检测逻辑**：
- `max <= 1.0` → 归一化到 [0, 1]，映射到 [0, 255]
- `max <= 255` → 8 位，保持
- `max <= 65535` → 16 位，转换到 8 位 [0, 255]
- 其他 → 使用 min-max 归一化

### 2.2 指定位数归一化

```python
from pointsuite.data.transforms import NormalizeColor

# 16 位转 8 位
transform_16to8 = NormalizeColor(source_bits=16, target_range=(0, 255))

# 8 位保持
transform_8bit = NormalizeColor(source_bits=8, target_range=(0, 255))
```

### 2.3 Color 增强

```python
from pointsuite.data.transforms import (
    ChromaticJitter,          # 色度抖动
    ChromaticTranslation,     # 颜色平移
    ChromaticAutoContrast,    # 自动对比度
    RandomColorJitter,        # 完整的颜色抖动（亮度、对比度、饱和度、色调）
    HueSaturationTranslation, # 色调饱和度变换
    RandomColorDrop,          # 颜色丢弃
    RandomColorGrayScale,     # 灰度化
)

# 完整的 Color 增强管道
color_transforms = [
    AutoNormalizeColor(target_range=(0, 255)),             # 自动归一化到 [0, 255]
    ChromaticJitter(p=0.95, std=0.005),                    # 色度抖动
    ChromaticTranslation(p=0.95, ratio=0.05),              # 颜色平移
    RandomColorJitter(                                      # 综合颜色增强
        brightness=0.4, 
        contrast=0.4, 
        saturation=0.4, 
        hue=0.1, 
        p=0.5
    ),
]
```

---

## 三、h_norm（归一化高程）增强

### 3.1 归一化

```python
from pointsuite.data.transforms import (
    AutoNormalizeHNorm,        # 自动归一化（裁剪异常值）
    StandardNormalizeHNorm,    # 标准化（Z-score）
    MinMaxNormalizeHNorm,      # MinMax 归一化
)

# 自动归一化（裁剪负值和过大值）
auto_norm = AutoNormalizeHNorm(clip_range=(0, 50))  # 限制在 0-50m

# 标准化（零均值、单位方差）
z_score = StandardNormalizeHNorm()

# MinMax 归一化到 [0, 1]
minmax = MinMaxNormalizeHNorm(target_range=(0, 1))
```

### 3.2 数据增强

```python
from pointsuite.data.transforms import (
    RandomHNormScale,    # 随机缩放
    RandomHNormShift,    # 随机偏移
    RandomHNormNoise,    # 随机噪声
    LogTransformHNorm,   # 对数变换
    BinHNorm,            # 分桶编码
)

# h_norm 增强示例
hnorm_transforms = [
    AutoNormalizeHNorm(clip_range=(0, 50)),        # 裁剪到合理范围
    RandomHNormScale(scale=(0.95, 1.05), p=0.3),   # 轻微缩放
    RandomHNormShift(shift=(-0.2, 0.2), p=0.3),    # 轻微偏移
    RandomHNormNoise(sigma=0.1, p=0.3),            # 添加噪声
]
```

### 3.3 高级变换

```python
# 对数变换（处理大范围高度）
log_transform = LogTransformHNorm(epsilon=1e-6)

# 分桶编码（离散化高度）
binning = BinHNorm(bins=10, range=(0, 20))  # 10 个高度等级
```

---

## 四、完整使用示例

### 4.1 基础配置

```python
from pointsuite.data import BinPklDataModule
from pointsuite.data.transforms import (
    # 坐标变换
    CenterShift,
    RandomRotate,
    RandomScale,
    RandomFlip,
    
    # Intensity
    AutoNormalizeIntensity,
    RandomIntensityScale,
    
    # Color
    AutoNormalizeColor,
    ChromaticJitter,
    
    # h_norm
    AutoNormalizeHNorm,
    
    # 收集和转换
    Collect,
    ToTensor
)

train_transforms = [
    # 1. 坐标变换
    CenterShift(apply_z=True),
    RandomRotate(axis='z', p=0.5),
    RandomScale(scale=[0.95, 1.05]),
    RandomFlip(p=0.5),
    
    # 2. 自动归一化（关键步骤）
    AutoNormalizeIntensity(target_range=(0, 1)),
    AutoNormalizeColor(target_range=(0, 255)),
    AutoNormalizeHNorm(clip_range=(0, 50)),
    
    # 3. 数据增强
    RandomIntensityScale(scale=(0.8, 1.2), p=0.95),
    ChromaticJitter(p=0.95, std=0.005),
    
    # 4. 收集特征
    Collect(
        keys=['coord', 'class'],
        offset_key={'offset': 'coord'},
        feat_keys={'feat': ['coord', 'intensity', 'color', 'h_norm']}
    ),
    
    # 5. 转换为 Tensor
    ToTensor()
]

# 创建 DataModule
datamodule = BinPklDataModule(
    data_root='path/to/data',
    train_files=['train.pkl'],
    assets=['coord', 'intensity', 'color', 'h_norm', 'class'],
    train_transforms=train_transforms,
    batch_size=8,
    num_workers=4
)
```

### 4.2 高级配置（完整增强）

```python
from pointsuite.data.transforms import *

train_transforms = [
    # ===== 坐标变换 =====
    CenterShift(apply_z=True),
    RandomRotate(angle=[-1, 1], axis='z', p=0.5),
    RandomScale(scale=[0.95, 1.05], anisotropic=False),
    RandomFlip(p=0.5),
    RandomJitter(sigma=0.01, clip=0.05),
    
    # ===== 自动归一化（必须在增强前）=====
    AutoNormalizeIntensity(target_range=(0, 1)),
    AutoNormalizeColor(target_range=(0, 255)),
    AutoNormalizeHNorm(clip_range=(0, 50)),
    
    # ===== Intensity 增强 =====
    RandomIntensityScale(scale=(0.8, 1.2), p=0.95),
    RandomIntensityNoise(sigma=0.01, p=0.5),
    RandomIntensityGamma(gamma_range=(0.8, 1.2), p=0.5),
    RandomIntensityDrop(drop_ratio=0.05, p=0.2),
    
    # ===== Color 增强 =====
    ChromaticJitter(p=0.95, std=0.005),
    ChromaticTranslation(p=0.95, ratio=0.05),
    RandomColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
        p=0.5
    ),
    
    # ===== h_norm 增强 =====
    RandomHNormScale(scale=(0.95, 1.05), p=0.3),
    RandomHNormShift(shift=(-0.2, 0.2), p=0.3),
    RandomHNormNoise(sigma=0.1, p=0.3),
    
    # ===== 采样 =====
    RandomDropout(dropout_ratio=0.2, dropout_application_ratio=0.5),
    
    # ===== 收集特征 =====
    Collect(
        keys=['coord', 'class'],
        offset_key={'offset': 'coord'},
        feat_keys={'feat': ['coord', 'intensity', 'color', 'h_norm']}
    ),
    
    # ===== 转换为 Tensor =====
    ToTensor()
]
```

### 4.3 验证集配置（无增强）

```python
val_transforms = [
    # 只做必要的归一化
    CenterShift(apply_z=True),
    
    # 自动归一化（保持与训练集一致）
    AutoNormalizeIntensity(target_range=(0, 1)),
    AutoNormalizeColor(target_range=(0, 255)),
    AutoNormalizeHNorm(clip_range=(0, 50)),
    
    # 收集特征（与训练集保持一致）
    Collect(
        keys=['coord', 'class'],
        offset_key={'offset': 'coord'},
        feat_keys={'feat': ['coord', 'intensity', 'color', 'h_norm']}
    ),
    
    ToTensor()
]
```

---

## 五、最佳实践

### 5.1 自动归一化的顺序

**关键规则**：自动归一化必须在任何增强之前！

```python
# ✅ 正确顺序
transforms = [
    CenterShift(),                              # 坐标变换
    AutoNormalizeIntensity(),                   # 归一化
    RandomIntensityScale(),                     # 增强
    Collect(...),
]

# ❌ 错误顺序
transforms = [
    CenterShift(),
    RandomIntensityScale(),                     # 在归一化前增强会出错！
    AutoNormalizeIntensity(),                   # 太晚了
    Collect(...),
]
```

### 5.2 训练集和验证集一致性

```python
# 确保归一化参数一致
normalize_params = {
    'intensity': {'target_range': (0, 1)},
    'color': {'target_range': (0, 255)},
    'h_norm': {'clip_range': (0, 50)},
}

# 训练集
train_transforms = [
    AutoNormalizeIntensity(**normalize_params['intensity']),
    AutoNormalizeColor(**normalize_params['color']),
    AutoNormalizeHNorm(**normalize_params['h_norm']),
    # ... 增强 ...
]

# 验证集（相同的归一化，无增强）
val_transforms = [
    AutoNormalizeIntensity(**normalize_params['intensity']),
    AutoNormalizeColor(**normalize_params['color']),
    AutoNormalizeHNorm(**normalize_params['h_norm']),
    # 无增强
]
```

### 5.3 特征维度计算

```python
# 计算最终特征维度
feat_keys = ['coord', 'intensity', 'color', 'h_norm']
# coord: 3, intensity: 1, color: 3, h_norm: 1
# 总维度 = 3 + 1 + 3 + 1 = 8

Collect(
    keys=['coord', 'class'],
    feat_keys={'feat': feat_keys}  # 输出 [N, 8]
)
```

### 5.4 增强概率建议

| 增强类型 | 推荐概率 | 说明 |
|---------|---------|------|
| RandomRotate | 0.5 | 旋转不变性 |
| RandomScale | 1.0 | 轻微缩放总是有益 |
| RandomFlip | 0.5 | 翻转不变性 |
| RandomIntensityScale | 0.95 | 模拟不同光照 |
| ChromaticJitter | 0.95 | 颜色鲁棒性 |
| RandomHNormScale | 0.3 | 高程误差模拟（轻度） |
| RandomHNormNoise | 0.3 | 局部误差（轻度） |

---

## 六、常见问题

### Q1: 为什么要在 transforms 而不是 dataset 中归一化？

**A**: 
- ✅ **灵活性**：不同数据集可能有不同位数（8/16 位）
- ✅ **可配置**：训练时可以轻松调整归一化策略
- ✅ **透明性**：数据加载和预处理分离清晰
- ✅ **兼容性**：支持多种数据格式，无需修改 dataset

### Q2: AutoNormalize 会不会每次都重新检测位数？

**A**: 是的，每个样本都会检测。但检测非常快（只需要 `max()` 操作），相比数据加载和增强时间可忽略。

### Q3: h_norm 增强会不会影响模型性能？

**A**: 
- ✅ **轻度增强（推荐）**：`RandomHNormScale(0.95-1.05, p=0.3)` 提升鲁棒性
- ⚠️ **重度增强**：过大的偏移/噪声可能影响分类精度
- 💡 **建议**：从轻度开始，根据验证集调整

### Q4: 如果不需要某个特征怎么办？

```python
# 只使用 coord 和 intensity
datamodule = BinPklDataModule(
    assets=['coord', 'intensity', 'class'],  # 不加载 color 和 h_norm
    train_transforms=[
        AutoNormalizeIntensity(),
        Collect(
            keys=['coord', 'class'],
            feat_keys={'feat': ['coord', 'intensity']}  # 只拼接需要的
        ),
        ToTensor()
    ]
)
```

### Q5: 如何验证归一化是否正确？

```python
# 加载一个样本
dataset = datamodule.train_dataset
sample = dataset[0]

# 检查范围
print(f"Intensity range: [{sample['intensity'].min():.3f}, {sample['intensity'].max():.3f}]")
print(f"Color range: [{sample['color'].min():.3f}, {sample['color'].max():.3f}]")
print(f"h_norm range: [{sample['h_norm'].min():.3f}, {sample['h_norm'].max():.3f}]")

# 期望输出（如果使用推荐配置）:
# Intensity range: [0.000, 1.000]
# Color range: [0.000, 255.000]
# h_norm range: [0.000, 50.000]
```

---

## 七、迁移指南

### 从旧版本迁移

#### 旧版本（在 dataset 中归一化）
```python
# dataset_bin.py
intensity = intensity / 65535.0  # 硬编码 16 位

# transforms
train_transforms = [
    RandomIntensityScale(),  # 假设已归一化
    Collect(...),
]
```

#### 新版本（在 transforms 中归一化）
```python
# dataset_bin.py
intensity = segment_points['intensity']  # 原始值

# transforms
train_transforms = [
    AutoNormalizeIntensity(),    # 自动检测并归一化
    RandomIntensityScale(),       # 现在可以安全增强
    Collect(...),
]
```

### 必须修改的地方

1. **删除 dataset 中的归一化代码**（已完成）
2. **在 transforms 开头添加 AutoNormalize**
3. **更新 Collect 参数**（如果使用了旧的 API）

---

## 八、性能优化建议

### 8.1 自动归一化的开销

| 操作 | 时间（10k 点）|
|-----|-------------|
| intensity.max() | < 0.1 ms |
| 归一化除法 | ~0.5 ms |
| **总开销** | **< 0.6 ms** |

相比数据加载（50-100ms）和 TIN+Raster（6-9ms），自动归一化的开销可忽略。

### 8.2 推荐配置

```python
# 高性能配置
datamodule = BinPklDataModule(
    cache_data=False,              # 大数据集不缓存
    num_workers=8,                 # 多进程加载
    batch_size=16,                 # 根据 GPU 内存调整
    train_transforms=[
        # 快速归一化
        AutoNormalizeIntensity(),
        AutoNormalizeColor(),
        AutoNormalizeHNorm(),
        
        # 关键增强
        RandomRotate(axis='z', p=0.5),
        RandomIntensityScale(p=0.95),
        ChromaticJitter(p=0.95),
        
        Collect(...),
        ToTensor()
    ]
)
```

---

## 总结

✅ **自动归一化**：无需关心数据位数，自动检测和处理  
✅ **完整增强**：Intensity、Color、h_norm 全覆盖  
✅ **灵活配置**：轻松调整归一化和增强策略  
✅ **性能优异**：归一化开销可忽略  

现在您可以专注于模型训练，而不是数据预处理！🎉
