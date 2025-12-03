# PointSuite 配置系统

本目录包含项目的所有配置文件，采用分层配置架构设计。

## 目录结构

```
configs/
├── experiments/          # 实验配置 (入口配置)
│   └── dales_semseg.yaml
├── data/                 # 数据配置
│   └── dales.yaml
├── model/                # 模型配置
│   └── ptv2_semseg.yaml
└── trainer/              # 训练器配置
    └── default.yaml
```

## 配置层级

实验配置 (`experiments/*.yaml`) 作为入口，通过 `defaults` 引用其他配置:

```yaml
# experiments/dales_semseg.yaml
defaults:
  - data: dales.yaml        # 引用 data/dales.yaml
  - model: ptv2_semseg.yaml # 引用 model/ptv2_semseg.yaml
  - trainer: default.yaml   # 引用 trainer/default.yaml

run:
  mode: train
  seed: 42
  output_dir: ./outputs/dales
```

## 配置变量引用

支持跨配置文件引用变量:

```yaml
# 在 model 配置中引用 data 配置的变量
head:
  class_path: pointsuite.models.SegHead
  init_args:
    num_classes: ${data.num_classes}  # 引用 data 配置中的 num_classes
```

## 配置类型

### 1. 数据配置 (`data/*.yaml`)

定义数据集路径、类别信息、数据增强等:

```yaml
train_data: /path/to/train
val_data: /path/to/val
class_mapping: [1, 2, 3, 4, 5, 6, 7, 8]
class_names: ['地面', '植被', '车辆', ...]
train_transforms:
  - class_path: pointsuite.data.transforms.CenterShift
  - class_path: pointsuite.data.transforms.RandomRotate
    init_args:
      angle: [-1, 1]
```

### 2. 模型配置 (`model/*.yaml`)

定义 backbone 和 head 结构:

```yaml
backbone:
  class_path: pointsuite.models.PointTransformerV2
  init_args:
    in_channels: 5
    enc_depths: [2, 2, 2, 2]
    ...

head:
  class_path: pointsuite.models.SegHead
  init_args:
    in_channels: 24
    num_classes: ${data.num_classes}
```

### 3. 训练器配置 (`trainer/*.yaml`)

定义 PyTorch Lightning Trainer 参数:

```yaml
max_epochs: 100
devices: 1
accelerator: auto
precision: 16-mixed
accumulate_grad_batches: 2
gradient_clip_val: 1.0
```

### 4. 实验配置 (`experiments/*.yaml`)

组合上述配置，添加任务特定设置:

```yaml
defaults:
  - data: dales.yaml
  - model: ptv2_semseg.yaml
  - trainer: default.yaml

run:
  mode: train  # train/resume/finetune/test/predict
  seed: 42
  output_dir: ./outputs/experiment

losses:
  - name: ce_loss
    class_path: pointsuite.models.losses.CrossEntropyLoss
    weight: 1.0

metrics:
  - name: seg_metrics
    class_path: pointsuite.utils.metrics.SegmentationMetrics
```

## 使用方式

### 命令行运行

```bash
# 基本运行
python main.py --config configs/experiments/dales_semseg.yaml

# 覆盖参数
python main.py --config configs/experiments/dales_semseg.yaml \
    --run.mode test \
    --run.checkpoint_path path/to/ckpt

# 修改训练参数
python main.py --config configs/experiments/dales_semseg.yaml \
    --trainer.max_epochs 50 \
    --data.batch_size 8
```

### Python API

```python
from pointsuite.engine import SemanticSegmentationEngine

# 从配置文件运行
engine = SemanticSegmentationEngine.from_config(
    'configs/experiments/dales_semseg.yaml'
)
engine.run()

# 覆盖配置
engine = SemanticSegmentationEngine.from_config(
    'configs/experiments/dales_semseg.yaml',
    cli_args=['--run.mode', 'test']
)
```

## 支持的任务类型

- **语义分割** (`semantic_segmentation`): 点级别分类
- **实例分割** (`instance_segmentation`): 点级别分类 + 实例聚类 [TODO]
- **目标检测** (`object_detection`): 3D 边界框检测 [TODO]

## 配置优先级

配置合并优先级 (从低到高):
1. defaults 中引用的子配置
2. 实验配置中的覆盖
3. 命令行参数
4. Python API 中的关键字参数
