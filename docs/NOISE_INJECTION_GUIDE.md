# 噪点注入增强完全指南

## 为什么需要噪点注入？

真实世界的激光雷达点云数据总是包含噪声：
- 🌩️ **大气噪声**：飞鸟、云、灰尘、雨雪
- 🔻 **地面反射**：水面、玻璃、光滑表面的错误反射
- 📡 **多路径反射**：建筑物、金属表面的二次反射
- 🌳 **植被遮挡**：树叶间隙产生的伪点
- ⚡ **传感器误差**：距离测量误差、角度误差

通过在训练时注入噪点，可以：
- ✅ **提升鲁棒性**：模型学会识别和忽略噪声
- ✅ **防止过拟合**：增加数据多样性
- ✅ **提高泛化能力**：适应不同质量的数据
- ✅ **模拟真实场景**：训练数据更接近实际应用

---

## 一、h_norm 不裁剪策略（推荐）

### 1.1 为什么不裁剪更好？

```python
from pointsuite.data.transforms import AutoNormalizeHNorm

# ✅ 推荐：不裁剪（默认）
transform = AutoNormalizeHNorm(clip_range=None)
```

**保留负值和极大值的优势**：

1. **真实信息保留**
   - 负值 ≠ 错误！可能是：
     - 🏗️ 地下室、地下停车场
     - 🚇 隧道入口
     - ⛰️ 坑洞、凹陷
   - 极大值可能是：
     - 🏢 高层建筑
     - 🗼 塔、天线
     - 🌳 高大树木

2. **抗噪能力增强**
   - 模型学习到噪声的分布特征
   - 不会因为轻微的异常值而失败
   - 更鲁棒的决策边界

3. **灵活性提升**
   - 不同场景有不同的高度范围
   - 模型自适应数据分布
   - 避免硬编码假设

### 1.2 何时需要裁剪？

```python
# 只在明确知道数据范围时裁剪
transform = AutoNormalizeHNorm(clip_range=(-5, 100))
```

**裁剪的场景**：
- 📊 **明确的数据质量问题**：已知存在大量异常值
- 🎯 **特定应用需求**：只关注特定高度范围（如地面物体）
- 💾 **内存/计算限制**：需要限制特征范围

### 1.3 统计验证

```python
# 查看 h_norm 分布
import numpy as np
import matplotlib.pyplot as plt

dataset = datamodule.train_dataset
sample = dataset[0]
h_norm = sample['h_norm']

print(f"h_norm 统计:")
print(f"  最小值: {h_norm.min():.2f}")
print(f"  最大值: {h_norm.max():.2f}")
print(f"  均值: {h_norm.mean():.2f}")
print(f"  中位数: {np.median(h_norm):.2f}")
print(f"  负值比例: {(h_norm < 0).sum() / len(h_norm) * 100:.2f}%")
print(f"  > 50m 比例: {(h_norm > 50).sum() / len(h_norm) * 100:.2f}%")

# 绘制分布
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(h_norm, bins=100, alpha=0.7)
plt.xlabel('h_norm (m)')
plt.ylabel('Count')
plt.title('h_norm Distribution')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(h_norm, bins=100, alpha=0.7, cumulative=True, density=True)
plt.xlabel('h_norm (m)')
plt.ylabel('Cumulative Probability')
plt.title('h_norm Cumulative Distribution')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 二、极端噪点注入

### 2.1 AddExtremeOutliers - 全局噪点

模拟全局随机分布的极端噪点（大气噪声、传感器误差等）。

#### 基础使用

```python
from pointsuite.data.transforms import AddExtremeOutliers

# 添加 1% 的极端噪点
transform = AddExtremeOutliers(
    ratio=0.01,                    # 噪点占总点数的 1%
    height_range=(-10, 100),       # 高度范围：地下 10m 到高空 100m
    height_mode='uniform',         # 均匀分布
    intensity_range=(0, 0.3),      # 弱强度（模拟大气噪声）
    color_value=(128, 128, 128),   # 灰色（未知颜色）
    class_label='ignore',          # 标记为噪声（-1）
    p=0.5                          # 50% 概率应用
)
```

#### 高级配置

```python
# 配置 1：模拟飞鸟/云（高空噪点）
bird_noise = AddExtremeOutliers(
    ratio=0.005,                   # 0.5%
    height_range=(50, 200),        # 50-200m 高空
    height_mode='high',            # 只在高空
    intensity_range=(0.1, 0.4),    # 弱反射
    color_value='random',          # 随机颜色
    class_label=0,                 # 未分类
    p=0.3
)

# 配置 2：模拟地面反射（低空/地下噪点）
ground_reflection = AddExtremeOutliers(
    ratio=0.01,
    height_range=(-5, 0),          # 地下 5m 到地面
    height_mode='low',             # 只在低空
    intensity_range=(0.5, 1.0),    # 强反射（镜面反射）
    color_value='inherit',         # 继承附近点的颜色
    class_label='ignore',
    p=0.4
)

# 配置 3：模拟多路径反射（双峰分布）
multipath_noise = AddExtremeOutliers(
    ratio=0.02,
    height_range=(-10, 100),
    height_mode='bimodal',         # 高空+低空双峰
    intensity_range=(0.2, 0.8),
    color_value=(200, 200, 200),   # 浅灰色
    class_label='ignore',
    p=0.5
)

# 配置 4：固定数量的噪点
fixed_noise = AddExtremeOutliers(
    num_outliers=100,              # 固定 100 个噪点
    height_range=(-20, 150),
    height_mode='uniform',
    intensity_range=(0, 1),
    color_value='random',
    class_label=None,              # 继承附近点的标签
    p=1.0                          # 总是应用
)
```

#### 参数详解

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `ratio` | 噪点占总点数比例 | 0.005-0.02 (0.5%-2%) |
| `num_outliers` | 固定噪点数量（优先于 ratio） | 50-200 |
| `height_range` | Z 坐标范围（米） | (-10, 100) |
| `height_mode` | 高度分布模式 | 'uniform', 'bimodal', 'high', 'low' |
| `intensity_range` | 强度范围 | (0, 1) 归一化后 |
| `color_value` | 颜色设置 | (128, 128, 128), 'random', 'inherit' |
| `class_label` | 分类标签 | 'ignore'(-1), 0, None |
| `p` | 应用概率 | 0.3-0.5 |

---

### 2.2 AddLocalNoiseClusters - 局部噪点簇

模拟局部聚集的噪点簇（更真实的噪声模式）。

#### 基础使用

```python
from pointsuite.data.transforms import AddLocalNoiseClusters

# 添加 3 个局部噪点簇
transform = AddLocalNoiseClusters(
    num_clusters=3,                # 3 个簇
    points_per_cluster=(10, 30),   # 每个簇 10-30 个点
    cluster_radius=2.0,            # 簇半径 2 米
    height_offset=(-2, 2),         # 高度偏移 ±2 米
    intensity_range=(0.2, 0.6),
    color_value='random',
    class_label='ignore',
    p=0.3
)
```

#### 高级配置

```python
# 配置 1：模拟玻璃反射簇
glass_reflection = AddLocalNoiseClusters(
    num_clusters=5,
    points_per_cluster=(15, 40),
    cluster_radius=1.5,            # 较小的簇
    height_offset=(-1, 3),         # 略高于原点
    intensity_range=(0.6, 1.0),    # 强反射
    color_value='inherit',         # 继承颜色（看起来像真实物体）
    class_label='ignore',
    p=0.4
)

# 配置 2：模拟植被噪声
vegetation_noise = AddLocalNoiseClusters(
    num_clusters=8,                # 更多小簇
    points_per_cluster=(5, 15),    # 较少点
    cluster_radius=0.5,            # 很小的簇
    height_offset=(-0.5, 0.5),     # 轻微偏移
    intensity_range=(0.3, 0.7),
    color_value=(100, 150, 100),   # 绿色调
    class_label=None,              # 继承标签（可能被误分类为植被）
    p=0.3
)

# 配置 3：模拟水面反射
water_reflection = AddLocalNoiseClusters(
    num_clusters=3,
    points_per_cluster=(20, 50),   # 较大的簇
    cluster_radius=3.0,            # 较大范围
    height_offset=(-5, -1),        # 水面下
    intensity_range=(0.4, 0.8),
    color_value=(100, 100, 150),   # 蓝色调
    class_label='ignore',
    p=0.2
)
```

#### 参数详解

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `num_clusters` | 噪点簇数量 | 3-8 |
| `points_per_cluster` | 每簇点数范围 | (5, 30) |
| `cluster_radius` | 簇半径（米） | 0.5-3.0 |
| `height_offset` | 相对簇中心的高度偏移 | (-2, 2) |
| `intensity_range` | 强度范围 | (0, 1) |
| `color_value` | 颜色设置 | 'random', 'inherit', RGB |
| `class_label` | 分类标签 | 'ignore', int, None |
| `p` | 应用概率 | 0.2-0.4 |

---

## 三、完整使用示例

### 3.1 基础配置（轻度噪点）

```python
from pointsuite.data import BinPklDataModule
from pointsuite.data.transforms import *

train_transforms = [
    # 坐标变换
    CenterShift(apply_z=True),
    RandomRotate(axis='z', p=0.5),
    
    # 归一化（不裁剪 h_norm）
    AutoNormalizeIntensity(),
    AutoNormalizeColor(),
    AutoNormalizeHNorm(clip_range=None),  # 不裁剪！
    
    # 轻度噪点注入
    AddExtremeOutliers(
        ratio=0.005,              # 0.5% 噪点
        height_range=(-10, 100),
        height_mode='uniform',
        class_label='ignore',
        p=0.3                     # 30% 概率
    ),
    
    # 常规增强
    RandomIntensityScale(p=0.95),
    ChromaticJitter(p=0.95),
    
    Collect(
        keys=['coord', 'class'],
        offset_key={'offset': 'coord'},
        feat_keys={'feat': ['coord', 'intensity', 'color', 'h_norm']}
    ),
    ToTensor()
]
```

### 3.2 高级配置（完整噪点模拟）

```python
train_transforms = [
    # 坐标变换
    CenterShift(apply_z=True),
    RandomRotate(axis='z', p=0.5),
    RandomScale(scale=[0.95, 1.05]),
    RandomFlip(p=0.5),
    
    # 归一化
    AutoNormalizeIntensity(target_range=(0, 1)),
    AutoNormalizeColor(target_range=(0, 255)),
    AutoNormalizeHNorm(clip_range=None),  # 保留所有值
    
    # ===== 多种噪点注入（模拟真实场景）=====
    
    # 1. 大气噪声（飞鸟、云、灰尘）
    AddExtremeOutliers(
        ratio=0.003,
        height_range=(50, 200),
        height_mode='high',
        intensity_range=(0.1, 0.3),
        color_value='random',
        class_label=0,
        p=0.2
    ),
    
    # 2. 地面反射噪声
    AddExtremeOutliers(
        ratio=0.005,
        height_range=(-5, 0),
        height_mode='low',
        intensity_range=(0.5, 1.0),
        color_value='inherit',
        class_label='ignore',
        p=0.3
    ),
    
    # 3. 多路径反射（双峰分布）
    AddExtremeOutliers(
        ratio=0.01,
        height_range=(-10, 100),
        height_mode='bimodal',
        intensity_range=(0.2, 0.8),
        color_value=(200, 200, 200),
        class_label='ignore',
        p=0.4
    ),
    
    # 4. 玻璃反射噪点簇
    AddLocalNoiseClusters(
        num_clusters=3,
        points_per_cluster=(15, 30),
        cluster_radius=1.5,
        height_offset=(-1, 3),
        intensity_range=(0.6, 1.0),
        color_value='inherit',
        class_label='ignore',
        p=0.3
    ),
    
    # 5. 植被遮挡噪声
    AddLocalNoiseClusters(
        num_clusters=5,
        points_per_cluster=(5, 15),
        cluster_radius=0.5,
        height_offset=(-0.5, 0.5),
        color_value=(100, 150, 100),
        class_label=None,
        p=0.2
    ),
    
    # 常规增强
    RandomIntensityScale(scale=(0.8, 1.2), p=0.95),
    RandomIntensityNoise(sigma=0.01, p=0.5),
    ChromaticJitter(p=0.95, std=0.005),
    RandomHNormNoise(sigma=0.1, p=0.3),
    
    # 采样
    RandomDropout(dropout_ratio=0.2, dropout_application_ratio=0.5),
    
    Collect(
        keys=['coord', 'class'],
        offset_key={'offset': 'coord'},
        feat_keys={'feat': ['coord', 'intensity', 'color', 'h_norm']}
    ),
    ToTensor()
]

datamodule = BinPklDataModule(
    data_root='path/to/data',
    assets=['coord', 'intensity', 'color', 'h_norm', 'class'],
    train_transforms=train_transforms,
    batch_size=8
)
```

### 3.3 验证集配置（无噪点注入）

```python
val_transforms = [
    CenterShift(apply_z=True),
    
    # 归一化（与训练集一致）
    AutoNormalizeIntensity(),
    AutoNormalizeColor(),
    AutoNormalizeHNorm(clip_range=None),
    
    # ❌ 不注入噪点
    # ❌ 不做数据增强
    
    Collect(
        keys=['coord', 'class'],
        offset_key={'offset': 'coord'},
        feat_keys={'feat': ['coord', 'intensity', 'color', 'h_norm']}
    ),
    ToTensor()
]
```

---

## 四、最佳实践

### 4.1 噪点类型选择

| 场景 | 推荐噪点类型 | 配置 |
|-----|------------|------|
| 🏙️ **城市场景** | AddExtremeOutliers (bimodal) + AddLocalNoiseClusters | 模拟建筑反射 |
| 🌲 **森林场景** | AddLocalNoiseClusters (小簇) | 模拟植被遮挡 |
| 🏔️ **山区场景** | AddExtremeOutliers (high) | 模拟大气噪声 |
| 🌊 **水体场景** | AddExtremeOutliers (low) + AddLocalNoiseClusters | 模拟水面反射 |
| 🏢 **室内场景** | AddLocalNoiseClusters | 模拟玻璃/镜面反射 |

### 4.2 噪点比例建议

```python
# 轻度噪点（推荐入门）
noise_light = {
    'AddExtremeOutliers': {'ratio': 0.005, 'p': 0.3},
    'AddLocalNoiseClusters': {'num_clusters': 2, 'p': 0.2},
}

# 中度噪点（推荐默认）
noise_medium = {
    'AddExtremeOutliers': {'ratio': 0.01, 'p': 0.4},
    'AddLocalNoiseClusters': {'num_clusters': 3-5, 'p': 0.3},
}

# 重度噪点（挑战模型）
noise_heavy = {
    'AddExtremeOutliers': {'ratio': 0.02, 'p': 0.5},
    'AddLocalNoiseClusters': {'num_clusters': 5-8, 'p': 0.4},
}
```

### 4.3 class_label 设置策略

```python
# 策略 1：标记为噪声（推荐）
class_label='ignore'  # -1，训练时忽略

# 策略 2：固定标签（测试模型鲁棒性）
class_label=0  # 未分类

# 策略 3：继承标签（最难，测试模型辨识能力）
class_label=None  # 从附近点继承，噪点可能被误认为真实物体
```

### 4.4 应用概率调优

```python
# 开始训练：低概率
train_transforms_early = [
    ...,
    AddExtremeOutliers(..., p=0.2),
    AddLocalNoiseClusters(..., p=0.1),
]

# 中期训练：中等概率
train_transforms_mid = [
    ...,
    AddExtremeOutliers(..., p=0.4),
    AddLocalNoiseClusters(..., p=0.3),
]

# 后期训练：高概率（挑战模型）
train_transforms_late = [
    ...,
    AddExtremeOutliers(..., p=0.6),
    AddLocalNoiseClusters(..., p=0.5),
]
```

---

## 五、效果验证

### 5.1 可视化噪点

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载样本
dataset = datamodule.train_dataset
sample = dataset[0]

coord = sample['coord'].numpy()
h_norm = sample['h_norm'].numpy()

# 识别可能的噪点（基于 h_norm 极值）
is_noise = (h_norm < -5) | (h_norm > 50)

fig = plt.figure(figsize=(14, 6))

# 原始点云
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(coord[~is_noise, 0], coord[~is_noise, 1], coord[~is_noise, 2],
           c='blue', s=1, alpha=0.3, label='正常点')
ax1.scatter(coord[is_noise, 0], coord[is_noise, 1], coord[is_noise, 2],
           c='red', s=10, alpha=0.8, label='可能的噪点')
ax1.set_title('点云（标记可能的噪点）')
ax1.legend()

# h_norm 分布
ax2 = fig.add_subplot(122)
ax2.hist(h_norm, bins=100, alpha=0.7, color='skyblue')
ax2.axvline(0, color='green', linestyle='--', label='地面')
ax2.axvline(-5, color='red', linestyle='--', label='噪点阈值')
ax2.axvline(50, color='red', linestyle='--')
ax2.set_xlabel('h_norm (m)')
ax2.set_ylabel('Count')
ax2.set_title('h_norm 分布')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"总点数: {len(coord)}")
print(f"可能的噪点数: {is_noise.sum()} ({is_noise.sum()/len(coord)*100:.2f}%)")
```

### 5.2 统计分析

```python
# 分析多个样本
n_samples = 100
noise_ratios = []

for i in range(n_samples):
    sample = dataset[i]
    h_norm = sample['h_norm'].numpy()
    is_noise = (h_norm < -5) | (h_norm > 50)
    noise_ratios.append(is_noise.sum() / len(h_norm))

print(f"噪点比例统计（{n_samples} 个样本）:")
print(f"  平均: {np.mean(noise_ratios)*100:.2f}%")
print(f"  中位数: {np.median(noise_ratios)*100:.2f}%")
print(f"  最小: {np.min(noise_ratios)*100:.2f}%")
print(f"  最大: {np.max(noise_ratios)*100:.2f}%")
```

---

## 六、常见问题

### Q1: 噪点会不会影响模型收敛？

**A**: 适度的噪点（0.5%-2%）不会影响收敛，反而提升泛化能力。如果收敛困难：
- 降低噪点比例
- 降低应用概率 `p`
- 从轻度噪点开始，逐步增加

### Q2: 如何确定合适的噪点比例？

**A**: 
1. 从 0.5% 开始
2. 监控验证集性能
3. 逐步增加到 1%-2%
4. 如果验证集性能下降，回退到上一个值

### Q3: class_label 应该设为什么？

**A**: 推荐策略：
- **训练初期**：`class_label='ignore'`（最简单）
- **训练中期**：`class_label=0`（未分类）
- **挑战模型**：`class_label=None`（继承，最难）

### Q4: AddExtremeOutliers 和 AddLocalNoiseClusters 可以同时使用吗？

**A**: 可以！这样更真实：
```python
transforms = [
    ...,
    AddExtremeOutliers(ratio=0.005, p=0.3),      # 全局噪点
    AddLocalNoiseClusters(num_clusters=3, p=0.2), # 局部簇
    ...
]
```

### Q5: 噪点注入的性能开销？

**A**: 
- AddExtremeOutliers: ~1-2 ms（10k 点，1% 噪点）
- AddLocalNoiseClusters: ~2-5 ms（5 个簇）
- 总开销：< 5% 的数据加载时间

### Q6: h_norm 不裁剪会导致模型难以训练吗？

**A**: 不会！现代深度学习模型（PointNet++, Transformer 等）对输入范围有很好的适应性。反而：
- ✅ 保留完整信息帮助模型理解场景
- ✅ 异常值提供额外的判别信息
- ✅ 提升模型的鲁棒性

如果确实有问题，可以使用 `StandardNormalizeHNorm` 标准化。

---

## 七、总结

### ✅ 推荐配置

```python
# 最佳实践：平衡性能和鲁棒性
train_transforms = [
    CenterShift(apply_z=True),
    RandomRotate(axis='z', p=0.5),
    
    # 归一化（不裁剪）
    AutoNormalizeIntensity(),
    AutoNormalizeColor(),
    AutoNormalizeHNorm(clip_range=None),  # ⭐ 不裁剪
    
    # 噪点注入（中度）
    AddExtremeOutliers(
        ratio=0.01,
        height_mode='bimodal',
        class_label='ignore',
        p=0.4
    ),
    AddLocalNoiseClusters(
        num_clusters=3,
        class_label='ignore',
        p=0.3
    ),
    
    # 常规增强
    RandomIntensityScale(p=0.95),
    ChromaticJitter(p=0.95),
    RandomHNormNoise(sigma=0.1, p=0.3),
    
    Collect(...),
    ToTensor()
]
```

### 🎯 关键要点

1. **h_norm 不裁剪**：保留所有信息，增强鲁棒性
2. **适度噪点**：0.5%-2% 的噪点比例
3. **多种噪点类型**：结合全局和局部噪点
4. **渐进式训练**：从轻度噪点开始，逐步增加
5. **验证集不注入**：只在训练时使用噪点增强

现在您的模型将更加鲁棒，能够处理真实世界中的各种噪声！🎉
