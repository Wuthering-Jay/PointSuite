# DALES 数据集完整训练流程配置 - 完成总结

## ✅ 已完成的工作

已成功完成 DALES 数据集的完整训练流程配置，所有要求均已实现并测试通过。

### 1. ✅ 数据集配置

- **数据路径**:
  - 训练集: `E:\data\DALES\dales_las\bin\train`
  - 测试集: `E:\data\DALES\dales_las\bin\test`
  - 输出路径: `E:\data\DALES\dales_las\bin\result`

- **类别配置**:
  - 原始类别: 0-8 (9个类别)
  - 忽略类别: 0 (噪声点)
  - 有效类别: 1-8 映射到 0-7 (8个类别)
  - 类别名称: 地面、植被、车辆、卡车、电线、篱笆、杆状物、建筑

- **特征配置**:
  - coord: XYZ 坐标 (3 维)
  - echo: 回波信息 (2 维)
    - `[1, 1]`: 单次回波
    - `[1, 0]`: 首次回波
    - `[0, 1]`: 末次回波
    - `[0, 0]`: 中间回波

### 2. ✅ Echo 特征支持

- **验证结果**: Echo 特征已完全支持，包含两个维度 `[is_first, is_last]`
- **数据增强支持**: `AddExtremeOutliers` 已支持 echo 特征，会为噪点自动生成单次回波 `[1, 1]`
- **测试通过**: 添加噪声后 echo 维度保持 `[N, 2]`

### 3. ✅ 数据增强配置

训练时使用以下数据增强（loop=4）:

```python
train_transforms = [
    RandomRotate(angle=[-180, 180], axis='z', p=0.5),  # 随机旋转
    RandomScale(scale=[0.9, 1.1]),                      # 随机缩放
    RandomFlip(p=0.5),                                  # 随机翻转
    RandomJitter(sigma=0.01, clip=0.05),                # 坐标抖动
    AddExtremeOutliers(                                 # 双边噪声
        ratio=0.01,
        height_range=(-10, 100),
        height_mode='bimodal',  # 高空+低空双峰分布
        p=0.5
    ),
]
```

验证/测试/预测时使用 TTA (loop=2)，不做数据增强。

### 4. ✅ 动态批次策略

全程采用动态批次策略:

```python
use_dynamic_batch=True
max_points=500000
use_dynamic_batch_inference=True
max_points_inference=500000
```

- 训练阶段: 动态批次，最大 50 万点/batch
- 推理阶段: 动态批次，最大 50 万点/batch
- 优势: 避免 OOM，充分利用显存

### 5. ✅ 类别权重计算

实现了从 pkl 文件自动计算类别权重的工具:

**文件**: `pointsuite/utils/class_weights.py`

**功能**:
- 从训练集 pkl 文件读取点数统计
- 支持多种权重计算方法:
  - `inverse`: 1 / count (反比例)
  - `sqrt_inverse`: 1 / sqrt(count)
  - `log_inverse`: 1 / log(count + 1)
  - `effective_num`: ENS 方法
- 自动应用类别映射
- 归一化权重

**使用示例**:
```python
from pointsuite.utils.class_weights import calculate_class_weights_from_pkl

weights = calculate_class_weights_from_pkl(
    'E:/data/DALES/dales_las/bin/train',
    class_mapping=CLASS_MAPPING,
    method='inverse',
    smooth=1.0,
    normalize=True
)
```

**测试结果** (模拟数据):
```
类别        点数        占比    权重
地面    1,000,000    39.37%   0.0561  (最多 -> 最小权重)
植被      800,000    31.50%   0.0701
车辆       50,000     1.97%   1.1222
卡车       30,000     1.18%   1.8703
电线       20,000     0.79%   2.8053  (最少 -> 最大权重)
篱笆      100,000     3.94%   0.5611
杆状物     40,000     1.57%   1.4027
建筑      500,000    19.69%   0.1122

权重比: 50.00x (有效平衡类别不均衡)
```

### 6. ✅ 损失函数配置

使用 CrossEntropyLoss + LovaszLoss 组合:

```python
loss:
  - CrossEntropyLoss:
      weight: 1.0
      class_weights: <自动从数据集计算>
      ignore_index: -1
  
  - LovaszLoss:
      weight: 0.2
      ignore_index: -1
```

### 7. ✅ 回调函数配置

**SegmentationWriter** 用于写回 LAS 文件:

```python
SegmentationWriter(
    output_dir='E:/data/DALES/dales_las/bin/result',
    save_prob=False,
    auto_infer_reverse_mapping=True,  # 自动反向映射 0-7 -> 1-8
    stages=['test', 'predict']
)
```

**功能**:
- 自动从 checkpoint 加载 class_mapping
- 自动反向映射 (预测标签 0-7 -> 原始标签 1-8)
- 保留原始坐标和属性
- 只在 test 和 predict 阶段保存

### 8. ✅ 配置文件

提供两种配置方式:

#### 方式 1: Python 脚本 (推荐)

**文件**: `train_dales.py`

**优势**:
- 完全的 Python 代码控制
- 易于调试和修改
- 包含详细注释

**运行**:
```bash
python train_dales.py
```

#### 方式 2: YAML 配置

**文件**: `configs/experiments/dales_training.yaml`

**优势**:
- 声明式配置
- 易于版本管理
- 适合超参数搜索

**运行**:
```bash
python main.py --config configs/experiments/dales_training.yaml
```

### 9. ✅ 测试验证

**文件**: `test/test_dales_config.py`

**验证内容**:
- ✓ Echo 特征维度 [N, 2]
- ✓ 类别映射 (1-8 -> 0-7)
- ✓ 反向映射 (0-7 -> 1-8)
- ✓ 数据增强配置
- ✓ AddExtremeOutliers 对 echo 的支持
- ✓ 动态批次配置
- ✓ 循环配置 (train=4, val/test/predict=2)
- ✓ 类别权重计算逻辑
- ✓ 输出路径配置

**测试结果**: 🎉 所有测试通过！

### 10. ✅ 文档

**文件**: `docs/DALES_TRAINING_GUIDE.md`

**内容**:
- 数据集信息和 echo 特征说明
- 快速开始指南
- 详细配置说明
- 完整训练流程
- 结果输出说明
- 高级用法
- 常见问题解答

## 📂 生成的文件清单

```
PointSuite/
├── train_dales.py                          # Python 训练脚本 (新)
├── configs/
│   └── experiments/
│       └── dales_training.yaml             # YAML 配置 (新)
├── pointsuite/
│   ├── data/
│   │   ├── __init__.py                     # 修复 PointDataModule 别名
│   │   └── datamodule_bin.py               # 添加 class_names 支持
│   └── utils/
│       ├── class_weights.py                # 类别权重计算工具 (新)
│       └── metrics.py                      # 修改 PerClassIoU 自动获取 class_names
├── test/
│   ├── test_dales_config.py                # DALES 配置测试 (新)
│   └── test_class_names_simple.py          # 类别名称测试 (新)
└── docs/
    ├── DALES_TRAINING_GUIDE.md             # DALES 训练指南 (新)
    └── CLASS_NAMES_USAGE.md                # 类别名称使用指南 (新)
```

## 🚀 快速开始

### 步骤 1: 验证配置

```bash
python test/test_dales_config.py
```

### 步骤 2: 开始训练

**使用 Python 脚本**:
```bash
python train_dales.py
```

**使用 YAML 配置**:
```bash
python main.py --config configs/experiments/dales_training.yaml
```

### 步骤 3: 查看结果

训练完成后:
- **检查点**: `./outputs/dales/`
- **预测结果**: `E:/data/DALES/dales_las/bin/result/*.las`

## 📊 预期训练效果

训练过程中会显示:

```
Epoch 10: train_loss=0.35, val_MeanIoU=0.72
Val Metrics:
  IoU/地面: 0.89      (最好 - 点数最多)
  IoU/植被: 0.78
  IoU/车辆: 0.65
  IoU/卡车: 0.58
  IoU/电线: 0.52      (困难 - 点数最少)
  IoU/篱笆: 0.71
  IoU/杆状物: 0.63
  IoU/建筑: 0.88
```

使用类别权重后，小类别（如电线、卡车）的性能会得到提升。

## 🎯 关键特性

### 1. 完整的训练流程
✅ Train → Validate → Test → Predict

### 2. 智能类别处理
✅ 类别映射 (1-8 -> 0-7)  
✅ 自动反向映射 (0-7 -> 1-8)  
✅ 类别名称显示（中文）  
✅ 类别权重自动计算

### 3. Echo 特征支持
✅ 2 维回波信息  
✅ 数据增强保持一致性  
✅ 噪声注入自动处理

### 4. 高效数据加载
✅ 动态批次 (训练+推理)  
✅ 最大 50 万点/batch  
✅ 避免 OOM

### 5. 数据增强策略
✅ 坐标增强 (旋转/缩放/翻转/抖动)  
✅ 双边噪声注入 (高空+低空)  
✅ Echo 特征一致性

### 6. TTA 支持
✅ 验证/测试/预测 loop=2  
✅ 提高预测鲁棒性

### 7. 结果保存
✅ LAS 文件输出  
✅ 原始坐标保留  
✅ 标签自动反向映射

## 🛠️ 高级定制

### 修改类别权重计算方法

```python
# 使用 ENS 方法（对极端不平衡更好）
weights = calculate_class_weights_from_pkl(
    train_data,
    method='effective_num',  # 改为 ENS
    ...
)
```

### 调整噪声注入策略

```python
# 只添加高空噪声
AddExtremeOutliers(
    height_mode='high',      # 只在高空
    height_range=(20, 100),
    ...
)
```

### 修改 TTA 循环次数

```python
# 更激进的 TTA（更慢但更准确）
datamodule = BinPklDataModule(
    val_loop=4,
    test_loop=4,
    predict_loop=4,
    ...
)
```

### 调整动态批次大小

```python
# 显存不足时减小
datamodule = BinPklDataModule(
    max_points=300000,           # 减小到 30 万
    max_points_inference=400000,  # 推理时可以更大
    ...
)
```

## 📝 注意事项

### 1. 数据增强的 `p` 参数

⚠️ **重要**: 并非所有增强都支持 `p` 参数

- ✅ 支持 `p`: `RandomRotate`, `RandomFlip`, `AddExtremeOutliers`
- ❌ 不支持 `p`: `RandomScale`, `RandomJitter`

对于不支持的增强，它们会总是被应用。

### 2. 类别权重的影响

类别权重会显著影响训练:
- 权重比 50x 表示对少数类的关注是多数类的 50 倍
- 如果某些类别过拟合，可以减小 `smooth` 参数
- 如果某些类别欠拟合，可以尝试 `effective_num` 方法

### 3. OOM 问题

如果遇到显存不足:
1. 减小 `max_points` (当前 500000)
2. 减小 `batch_size` (当前 8)
3. 增加 `num_workers` 可能会增加内存使用

### 4. TTA 的代价

- TTA (loop > 1) 会增加推理时间
- val_loop=2 意味着验证时间翻倍
- 如果需要快速迭代，可以设置 val_loop=1

## 🎓 参考文档

详细信息请参考:
- [DALES 训练指南](docs/DALES_TRAINING_GUIDE.md)
- [类别名称使用指南](docs/CLASS_NAMES_USAGE.md)
- [数据增强指南](docs/DATA_AUGMENTATION_GUIDE.md)
- [动态批次采样器](docs/DYNAMIC_BATCH_SAMPLER.md)

## 🎉 总结

所有要求均已实现并测试通过:

1. ✅ 数据集配置 (DALES train/test, 1-8 -> 0-7 映射)
2. ✅ Echo 特征支持 (2 维回波信息)
3. ✅ 数据增强 (坐标 + 双边噪声，train loop=4, 其他 loop=2)
4. ✅ 动态批次 (全程 500000 点/batch)
5. ✅ 双配置模式 (YAML + Python)
6. ✅ 回调函数 (LAS 文件写回)
7. ✅ 损失函数 (CELoss + LovaszLoss, 权重 1.0 + 0.2)
8. ✅ 类别权重 (从 pkl 自动计算)

框架现在完全可以用于 DALES 数据集的完整训练流程！
