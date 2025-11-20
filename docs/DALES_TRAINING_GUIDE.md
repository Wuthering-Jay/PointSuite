# DALES 数据集训练指南

本指南展示如何使用 PointSuite 框架训练 DALES 数据集的语义分割模型。

## 数据集信息

- **数据集名称**: DALES (Dayton Annotated LiDAR Earth Scan)
- **任务**: 点云语义分割
- **类别**: 9 个类别（0-8），其中 0 为噪声需要忽略
- **实际类别**: 8 个类别（1-8 映射到 0-7）
- **类别名称**: 地面、植被、车辆、卡车、电线、篱笆、杆状物、建筑
- **特征**: coord (XYZ 坐标) + echo (回波信息，2 维)

### Echo 特征说明

Echo 特征包含 2 个维度：`[is_first_return, is_last_return]`

- `[1, 1]`: 单次回波（只有一次反射）
- `[1, 0]`: 首次回波（第一次反射）
- `[0, 1]`: 末次回波（最后一次反射）
- `[0, 0]`: 中间回波（既不是首次也不是末次）

## 快速开始

### 方法 1: 使用 Python 脚本（推荐）

```bash
python train_dales.py
```

这个脚本包含完整的配置，包括：
- 数据加载配置
- 类别映射和权重计算
- 数据增强
- 模型训练
- 测试和预测
- 结果保存到 LAS 文件

### 方法 2: 使用 YAML 配置

```bash
python main.py --config configs/experiments/dales_training.yaml
```

## 详细配置说明

### 1. 数据路径配置

```python
TRAIN_DATA = r"E:\data\DALES\dales_las\bin\train"
TEST_DATA = r"E:\data\DALES\dales_las\bin\test"
OUTPUT_DIR = r"E:\data\DALES\dales_las\bin\result"
```

### 2. 类别映射

原始标签 1-8 映射到连续标签 0-7，原始标签 0（噪声）被忽略：

```python
CLASS_MAPPING = {
    1: 0,  # 地面 -> 0
    2: 1,  # 植被 -> 1
    3: 2,  # 车辆 -> 2
    4: 3,  # 卡车 -> 3
    5: 4,  # 电线 -> 4
    6: 5,  # 篱笆 -> 5
    7: 6,  # 杆状物 -> 6
    8: 7,  # 建筑 -> 7
}
```

### 3. 类别权重计算

从训练集的 pkl 文件中自动计算类别权重：

```python
from pointsuite.utils.class_weights import calculate_class_weights_from_pkl

class_weights = calculate_class_weights_from_pkl(
    TRAIN_DATA,
    class_mapping=CLASS_MAPPING,
    method='inverse',  # 反比例权重
    smooth=1.0,
    normalize=True
)
```

支持的权重计算方法：
- `'inverse'`: 1 / count（反比例）
- `'sqrt_inverse'`: 1 / sqrt(count)（平方根反比例）
- `'log_inverse'`: 1 / log(count + 1)（对数反比例）
- `'effective_num'`: Effective Number of Samples (ENS) 方法

### 4. 数据增强配置

#### 训练数据增强（4 次循环）

```python
train_transforms = [
    # 坐标增强
    RandomRotate(angle=[-180, 180], axis='z', p=0.5),
    RandomScale(scale=[0.9, 1.1], p=0.5),
    RandomFlip(p=0.5),
    RandomJitter(sigma=0.01, clip=0.05, p=0.5),
    
    # 噪声注入：双边极端噪声
    AddExtremeOutliers(
        ratio=0.01,  # 1% 噪点
        height_range=(-10, 100),  # Z 坐标范围
        height_mode='bimodal',  # 双峰分布（高空+低空）
        intensity_range=(0, 1),
        color_value=(128, 128, 128),
        class_label='ignore',  # 噪点标记为 ignore
        p=0.5
    ),
]
```

**注意**: `AddExtremeOutliers` 已经支持 echo 特征，会为噪点自动生成 echo 值（单次回波 `[1, 1]`）

#### TTA（Test-Time Augmentation）

验证、测试、预测阶段使用 2 次循环，不做数据增强：

```python
datamodule = BinPklDataModule(
    train_loop=4,  # 训练循环 4 次
    val_loop=2,    # 验证循环 2 次
    test_loop=2,   # 测试循环 2 次
    predict_loop=2,  # 预测循环 2 次
    ...
)
```

### 5. 动态批次配置

全程使用动态批次策略，避免 OOM：

```python
datamodule = BinPklDataModule(
    use_dynamic_batch=True,  # 训练时使用动态批次
    max_points=500000,  # 每批次最大点数
    
    use_dynamic_batch_inference=True,  # 推理时也使用动态批次
    max_points_inference=500000,
    ...
)
```

### 6. 损失函数配置

使用 CrossEntropyLoss + LovaszLoss 组合：

```python
# 在 YAML 配置中
loss:
  - type: CrossEntropyLoss
    weight: 1.0
    class_weights: auto  # 自动从数据集计算
    ignore_index: -1
  
  - type: LovaszLoss
    weight: 0.2  # LAC Loss 权重
    ignore_index: -1
```

### 7. 回调函数配置

#### 保存检查点

```python
ModelCheckpoint(
    monitor='val_MeanIoU',
    mode='max',
    save_top_k=3,
    filename='dales-{epoch:02d}-{val_MeanIoU:.4f}'
)
```

#### 写回 LAS 文件

```python
SegmentationWriter(
    output_dir=OUTPUT_DIR,
    save_prob=False,  # 不保存概率
    auto_infer_reverse_mapping=True,  # 自动反向映射（7 -> 6, 6 -> 7 等）
    stages=['test', 'predict']  # 只在 test 和 predict 时保存
)
```

## 完整训练流程

### 1. 准备数据

确保数据已转换为 bin+pkl 格式：

```
E:\data\DALES\dales_las\bin\
├── train/
│   ├── scene1.pkl
│   ├── scene1_coord.bin
│   ├── scene1_echo.bin
│   ├── scene1_classification.bin
│   └── ...
└── test/
    └── ...
```

### 2. 计算类别权重（可选）

```bash
python -m pointsuite.utils.class_weights
```

### 3. 训练模型

```bash
# 使用 Python 脚本
python train_dales.py

# 或使用 YAML 配置
python main.py --config configs/experiments/dales_training.yaml
```

### 4. 查看训练日志

训练过程中会显示：
- 每个 epoch 的训练损失
- 验证集的 MeanIoU
- 每个类别的 IoU（使用类别名称）

```
Epoch 10: train_loss=0.35, val_MeanIoU=0.72
Val Metrics:
  IoU/地面: 0.89
  IoU/植被: 0.78
  IoU/车辆: 0.65
  IoU/卡车: 0.58
  IoU/电线: 0.52
  IoU/篱笆: 0.71
  IoU/杆状物: 0.63
  IoU/建筑: 0.88
```

### 5. 测试模型

```python
trainer.test(task, datamodule)
```

测试结果会自动保存到 LAS 文件（通过 `SegmentationWriter`）

### 6. 预测新数据

```python
trainer.predict(task, datamodule)
```

预测结果保存路径：`E:\data\DALES\dales_las\bin\result`

## 结果输出

### LAS 文件

预测结果会写回为 LAS 文件，保留原始坐标和属性，并添加预测标签：

```
result/
├── scene1.las  # 包含预测的分类标签（已反向映射）
├── scene2.las
└── ...
```

标签会自动反向映射：
- 预测标签 0 -> 原始标签 1（地面）
- 预测标签 1 -> 原始标签 2（植被）
- ...
- 预测标签 7 -> 原始标签 8（建筑）

### 检查点

模型检查点保存在：`./outputs/dales/`

```
outputs/dales/
├── lightning_logs/
│   └── version_0/
│       ├── checkpoints/
│       │   ├── dales-epoch=10-val_MeanIoU=0.7234.ckpt
│       │   ├── dales-epoch=15-val_MeanIoU=0.7456.ckpt
│       │   └── dales-epoch=20-val_MeanIoU=0.7589.ckpt
│       └── ...
└── ...
```

## 高级用法

### 自定义损失权重

```python
# 手动指定类别权重
custom_weights = torch.tensor([
    1.0,  # 地面
    2.0,  # 植被
    3.0,  # 车辆
    4.0,  # 卡车
    5.0,  # 电线
    3.0,  # 篱笆
    2.5,  # 杆状物
    1.5,  # 建筑
])

# 在创建任务时传入
task = SemanticSegmentationTask(
    ...,
    loss_weights=custom_weights
)
```

### 修改噪声注入策略

```python
# 只添加高空噪声（模拟飞鸟）
AddExtremeOutliers(
    ratio=0.01,
    height_mode='high',  # 只在高空
    height_range=(20, 100),
    p=0.5
)

# 只添加低空噪声（模拟反射）
AddExtremeOutliers(
    ratio=0.01,
    height_mode='low',  # 只在低空
    height_range=(-10, 5),
    p=0.5
)
```

### 调整 TTA 循环次数

```python
# 更激进的 TTA（更慢但可能更准确）
datamodule = BinPklDataModule(
    val_loop=4,
    test_loop=4,
    predict_loop=4,
    ...
)

# 不使用 TTA（更快）
datamodule = BinPklDataModule(
    val_loop=1,
    test_loop=1,
    predict_loop=1,
    ...
)
```

## 常见问题

### Q: OOM（显存不足）怎么办？

A: 减小 `max_points` 或 `batch_size`:

```python
datamodule = BinPklDataModule(
    max_points=300000,  # 减小最大点数
    batch_size=4,  # 减小批次大小（如果不使用动态批次）
    ...
)
```

### Q: 训练太慢怎么办？

A: 增加 `num_workers`:

```python
datamodule = BinPklDataModule(
    num_workers=8,  # 增加工作进程数
    ...
)
```

### Q: 如何查看类别分布？

A:

```python
from pointsuite.utils.class_weights import print_class_distribution

print_class_distribution(
    TRAIN_DATA,
    class_mapping=CLASS_MAPPING,
    class_names=CLASS_NAMES
)
```

### Q: 如何从检查点恢复训练？

A:

```python
trainer.fit(task, datamodule, ckpt_path='path/to/checkpoint.ckpt')
```

## 参考

- [数据增强指南](DATA_AUGMENTATION_GUIDE.md)
- [类别名称使用指南](CLASS_NAMES_USAGE.md)
- [动态批次采样器](DYNAMIC_BATCH_SAMPLER.md)
- [回调函数使用](SEGMENTATION_WRITER_USAGE.md)
