# 数据集变换管道说明

## 重要变更

从最新版本开始，`BinPklDataset._load_data()` 方法**不再预先拼接所有特征**。各个特征（coord、intensity、color 等）在数据字典中保持独立，以便 transforms 能够正确处理它们。

## 数据流程

### 1. 数据加载阶段（`_load_data`）

```python
# dataset_bin.py 返回的 data_dict 结构：
{
    'coord': np.array([N, 3]),           # 坐标
    'intensity': np.array([N,]),         # 强度值 [0, 1]
    'color': np.array([N, 3]),           # RGB 颜色 [0, 255]
    'is_first': np.array([N,]),          # 首次回波 [-1, 1]
    'is_last': np.array([N,]),           # 末次回波 [-1, 1]
    'normal': np.array([N, 3]),          # 法向量
    'h_norm': np.array([N,]),            # 高度归一化
    'classification': np.array([N,]),    # 分类标签
}
```

### 2. 数据增强阶段（`transforms`）

transforms 对各个独立的字段进行操作：

```python
from pointsuite.data.transforms import (
    RandomRotate,
    RandomScale,
    RandomFlip,
    RandomIntensityScale,
    RandomIntensityNoise,
    ChromaticJitter,
    Collect,
    ToTensor
)

train_transforms = [
    # 坐标变换
    RandomRotate(angle=[-1, 1], axis='z', p=0.5),
    RandomScale(scale=[0.9, 1.1]),
    RandomFlip(p=0.5),
    
    # 强度变换（针对独立的 intensity 字段）
    RandomIntensityScale(scale=(0.8, 1.2), p=0.95),
    RandomIntensityNoise(sigma=0.01, p=0.5),
    
    # 颜色变换（针对独立的 color 字段）
    ChromaticJitter(p=0.95, std=0.005),
    
    # 最后拼接所有特征成 feature
    Collect(
        keys=['coord', 'classification'],  # 主要的键
        offset_keys_dict=dict(offset='coord'),  # 用于计算 offset
        feat_keys=['coord', 'intensity', 'color', 'is_first', 'is_last']  # 拼接成 feature
    ),
    
    # 转换为 Tensor
    ToTensor()
]
```

### 3. 特征拼接阶段（`Collect`）

`Collect` 变换负责：
1. 收集指定的键（如 coord、classification）
2. 计算 offset（用于 batch 中分隔不同样本）
3. **拼接多个特征成一个 feature 张量**

```python
# Collect 之后的 data_dict 结构：
{
    'coord': tensor([N, 3]),
    'classification': tensor([N,]),
    'offset': tensor([N]),  # 累积点数
    'feature': tensor([N, C]),  # 拼接后的特征：[coord(3) + intensity(1) + color(3) + is_first(1) + is_last(1)] = [N, 9]
}
```

## 完整使用示例

### 示例 1：基础配置

```python
from pointsuite.data import BinPklDataModule
from pointsuite.data.transforms import (
    CenterShift,
    RandomRotate,
    Collect,
    ToTensor
)

# 定义 transforms
train_transforms = [
    CenterShift(apply_z=True),
    RandomRotate(axis='z', p=0.5),
    Collect(
        keys=['coord', 'classification'],
        feat_keys=['coord', 'intensity']  # 只使用 coord 和 intensity
    ),
    ToTensor()
]

# 创建 DataModule
datamodule = BinPklDataModule(
    data_root='path/to/data',
    train_files=['train.pkl'],
    assets=['coord', 'intensity', 'class'],  # 指定要加载的资产
    train_transforms=train_transforms,
    batch_size=8,
    num_workers=4
)
```

### 示例 2：完整的数据增强

```python
from pointsuite.data.transforms import (
    # 坐标变换
    CenterShift,
    RandomRotate,
    RandomScale,
    RandomFlip,
    RandomJitter,
    
    # 强度变换
    RandomIntensityScale,
    RandomIntensityNoise,
    RandomIntensityGamma,
    
    # 颜色变换
    ChromaticJitter,
    ChromaticTranslation,
    
    # 采样和收集
    RandomDropout,
    Collect,
    ToTensor
)

train_transforms = [
    # 1. 坐标归一化和增强
    CenterShift(apply_z=True),
    RandomRotate(angle=[-1, 1], axis='z', p=0.5),
    RandomScale(scale=[0.95, 1.05], anisotropic=False),
    RandomFlip(p=0.5),
    RandomJitter(sigma=0.01, clip=0.05),
    
    # 2. 强度增强（针对 intensity 字段）
    RandomIntensityScale(scale=(0.8, 1.2), p=0.95),
    RandomIntensityNoise(sigma=0.01, p=0.5),
    RandomIntensityGamma(gamma_range=(0.8, 1.2), p=0.5),
    
    # 3. 颜色增强（针对 color 字段）
    ChromaticJitter(p=0.95, std=0.005),
    ChromaticTranslation(p=0.95, ratio=0.05),
    
    # 4. 采样
    RandomDropout(dropout_ratio=0.2, dropout_application_ratio=0.5),
    
    # 5. 收集并拼接特征
    Collect(
        keys=['coord', 'classification'],
        offset_keys_dict=dict(offset='coord'),
        feat_keys=['coord', 'intensity', 'color', 'is_first', 'is_last', 'h_norm']
    ),
    
    # 6. 转换为 Tensor
    ToTensor()
]

# 创建 DataModule，指定所有需要的资产
datamodule = BinPklDataModule(
    data_root='path/to/data',
    train_files=['train.pkl'],
    assets=['coord', 'intensity', 'color', 'echo', 'h_norm', 'class'],
    train_transforms=train_transforms,
    use_dynamic_batch=True,
    max_points=500000,
    num_workers=4
)
```

### 示例 3：验证集配置（无增强）

```python
val_transforms = [
    # 只做必要的归一化，不做数据增强
    CenterShift(apply_z=True),
    
    # 收集特征（与训练集保持一致）
    Collect(
        keys=['coord', 'classification'],
        feat_keys=['coord', 'intensity', 'color']
    ),
    
    ToTensor()
]

datamodule = BinPklDataModule(
    data_root='path/to/data',
    val_files=['val.pkl'],
    assets=['coord', 'intensity', 'color', 'class'],
    val_transforms=val_transforms,
    batch_size=8
)
```

## 常见问题

### Q1: 为什么 `_load_data` 不直接拼接 feature？

**A:** 因为 transforms 需要分别处理各个特征：
- `RandomIntensityScale` 需要访问 `intensity` 字段
- `ChromaticJitter` 需要访问 `color` 字段
- 如果预先拼接成 feature，这些变换就无法工作了

### Q2: 如何确保 feature 的维度顺序一致？

**A:** 在 `Collect` 的 `feat_keys` 参数中明确指定顺序：

```python
Collect(
    keys=['coord', 'classification'],
    feat_keys=['coord', 'intensity', 'color']  # 明确的顺序：3 + 1 + 3 = 7 维
)
```

### Q3: intensity 和 color 的值范围是什么？

**A:** 
- `intensity`: [0, 1] - 已归一化（如果原始值 > 1）
- `color`: [0, 255] - 8 位范围（transforms.py 期望这个范围）

### Q4: 如果不需要某些特征怎么办？

**A:** 只需在 `assets` 和 `feat_keys` 中不包含它们：

```python
# 只使用 coord 和 intensity
datamodule = BinPklDataModule(
    assets=['coord', 'intensity', 'class'],  # 不加载 color
    train_transforms=[
        CenterShift(),
        Collect(
            keys=['coord', 'classification'],
            feat_keys=['coord', 'intensity']  # 只拼接 coord 和 intensity
        ),
        ToTensor()
    ]
)
```

### Q5: 如何处理一维特征（如 intensity、h_norm）？

**A:** `Collect` 会自动处理维度：
- 如果是 [N,] 形状，会扩展为 [N, 1]
- 如果已经是 [N, 1]，保持不变
- 然后与其他特征拼接

```python
# intensity [N,] -> [N, 1]
# coord [N, 3] 保持不变
# 拼接后 feature: [N, 4]
```

## 迁移指南

### 从旧版本迁移

如果你的代码使用了旧版本（直接返回 feature 的版本），需要做以下修改：

#### 旧代码：
```python
# 旧版本：dataset 直接返回 feature
train_transforms = [
    RandomRotate(axis='z'),
    ToTensor()
]
```

#### 新代码：
```python
# 新版本：需要添加 Collect 来拼接 feature
train_transforms = [
    RandomRotate(axis='z'),
    # 添加数据增强
    RandomIntensityScale(p=0.95),
    # 必须添加 Collect
    Collect(
        keys=['coord', 'classification'],
        feat_keys=['coord', 'intensity', 'color']  # 指定要拼接的特征
    ),
    ToTensor()
]
```

## 最佳实践

1. **明确指定 assets**：只加载需要的数据
2. **合理使用数据增强**：训练集使用，验证集不使用
3. **保持 feat_keys 一致**：训练集和验证集应该使用相同的特征顺序
4. **测试 feature 维度**：确保模型输入维度与 feature 维度匹配

```python
# 推荐的配置模式
def get_transforms(is_train=True):
    base_transforms = [
        CenterShift(apply_z=True),
    ]
    
    if is_train:
        base_transforms.extend([
            RandomRotate(axis='z', p=0.5),
            RandomIntensityScale(p=0.95),
        ])
    
    base_transforms.extend([
        Collect(
            keys=['coord', 'classification'],
            feat_keys=['coord', 'intensity', 'color']
        ),
        ToTensor()
    ])
    
    return base_transforms

# 使用
train_transforms = get_transforms(is_train=True)
val_transforms = get_transforms(is_train=False)
```

## 参考

- `transforms.py` - 所有可用的数据增强
- `collate.py` - Batch 合并逻辑
- `dataset_base.py` - 数据集基类
- `dataset_bin.py` - BinPkl 数据集实现
