# PointSuite

PointSuite 是一个基于 `PyTorch Lightning` 构建的点云深度学习通用工具箱，支持语义分割、目标检测、实例分割等多种任务。

## 🚀 快速开始

### 方式1: YAML 配置运行

```bash
# 训练
python main.py --config configs/experiments/dales_semseg.yaml

# 测试
python main.py --config configs/experiments/dales_semseg.yaml --run.mode test --run.checkpoint_path path/to/ckpt

# 覆盖配置
python main.py --config configs/experiments/dales_semseg.yaml --trainer.max_epochs 50 --data.batch_size 8
```

### 方式2: Python API

```python
from pointsuite.engine import SemanticSegmentationEngine

# 从 YAML 配置运行
engine = SemanticSegmentationEngine.from_config('configs/experiments/dales_semseg.yaml')
engine.run()

# 或分步执行
engine.setup()
engine.train()
engine.test()
engine.predict()
```

### 方式3: 编程式调用

```python
from pointsuite import BinPklDataModule, SemanticSegmentationTask
import pytorch_lightning as pl

# 创建 DataModule
datamodule = BinPklDataModule(
    train_data='path/to/train',
    val_data='path/to/val',
    ...
)

# 创建 Task
task = SemanticSegmentationTask(
    model_config={...},
    loss_configs=[...],
    ...
)

# 训练
trainer = pl.Trainer(...)
trainer.fit(task, datamodule)
```

## 📁 项目结构

```
PointSuite/
├── main.py                 # 统一入口
├── configs/                # 配置文件
│   ├── experiments/        # 实验配置 (入口)
│   │   └── dales_semseg.yaml
│   ├── data/               # 数据配置
│   │   └── dales.yaml
│   ├── model/              # 模型配置
│   │   └── ptv2_semseg.yaml
│   └── trainer/            # 训练器配置
│       └── default.yaml
├── pointsuite/             # 核心代码
│   ├── data/               # 数据加载
│   │   ├── datamodule_base.py
│   │   ├── datamodule_bin.py
│   │   ├── transforms.py
│   │   └── datasets/
│   ├── models/             # 模型架构
│   │   ├── backbones/      # Backbone (PTv2, ...)
│   │   ├── heads/          # Head (SegHead, ...)
│   │   └── losses/         # 损失函数
│   ├── tasks/              # Lightning 任务
│   │   ├── base_task.py
│   │   └── semantic_segmentation.py
│   ├── engine/             # 任务引擎
│   │   ├── base.py
│   │   ├── semantic_segmentation.py
│   │   ├── instance_segmentation.py  # TODO
│   │   └── object_detection.py       # TODO
│   └── utils/              # 工具函数
│       ├── config.py       # 配置管理
│       ├── callbacks.py    # 回调函数
│       └── metrics/        # 评估指标
└── examples/               # 使用示例
    └── run_experiment.py
```

## ⚙️ 配置系统

采用分层配置架构:

```yaml
# experiments/dales_semseg.yaml (入口配置)
defaults:
  - data: dales.yaml          # 数据配置
  - model: ptv2_semseg.yaml   # 模型配置
  - trainer: default.yaml     # 训练器配置

run:
  mode: train                  # train/resume/finetune/test/predict
  seed: 42
  output_dir: ./outputs/dales
```

支持变量引用:
```yaml
head:
  init_args:
    num_classes: ${data.num_classes}  # 引用 data 配置中的值
```

详细文档见 [configs/README.md](configs/README.md)

## 🎯 支持的任务

| 任务 | 状态 | Engine | Task |
|------|------|--------|------|
| 语义分割 | ✅ | `SemanticSegmentationEngine` | `SemanticSegmentationTask` |
| 实例分割 | 🚧 | `InstanceSegmentationEngine` | `InstanceSegmentationTask` |
| 目标检测 | 🚧 | `ObjectDetectionEngine` | `ObjectDetectionTask` |

## 📊 支持的模型

### Backbone
- PointTransformerV2 (PTv2)
- 更多开发中...

### Head
- SegHead (语义分割)
- 更多开发中...

## 🔧 运行模式

```yaml
run:
  mode: train      # 从头训练
  mode: resume     # 从 checkpoint 继续训练 (恢复优化器状态)
  mode: finetune   # 加载预训练权重，从头训练
  mode: test       # 仅测试
  mode: predict    # 仅预测
```

## 关键设计

1. 需要 `[B,C,N]` 和 `[C,N]+offset` 两种数据加载方式，这样可以实现大量开源工作的快速兼容
2. 采用 bin+pkl 数据格式存储分块裁剪后的点云数据，支持快速 memmap 读取
3. 通过传入 `require_labels` 手动控制有效类别，引入 `garbage_bin` 模式
4. 对于分布式训练采用 `ddp` 策略，使用 `torchmetrics` 进行指标计算

## 📝 开发日志

* **2025/12/04**: 实现基于 YAML 配置的统一框架，支持 experiment/data/model/trainer 分层配置
* **2025/11/01**: bin+pkl 的实现已经完成，支持 overlap 和 gridsample 模式
* **2025/10/29**: 设计 bin+pkl 数据格式，基于 np.memmap 进行快速读取
* **2025/10/25**: 项目初始化，确定整体架构
