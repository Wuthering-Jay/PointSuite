# PointSuite
PointSuite基于`pytorch-lightning`构建的点云深度学习通用工具箱，目标是适用于分类、语义分割、目标检测、实例分割、全景分割等多种任务，能够适配全监督、弱监督、自监督等多种策略，并形成一套可视化界面，成为一个强大、直观、易用、高扩展、贴近实际应用的点云深度学习工具。

## 关键
1.  需要`[B,C,N]`和`[C,N]+offset`两种数据加载方式，这样可以实现大量开源工作的快速兼容
2.  采用`h5`数据格式存储分块裁剪后的点云数据，设计一块`las`对应一块`h5`，数据中应该包含“原始点云头文件+属性信息+分块索引+类别权重+分块权重”，预测时利用`pytorch-lightning`的数据加载器列表的特性，实现逐块点云的预测
3.  通过传入`require_labels`手动控制有效类别，数据分块时不传入，分块时只管每块有多少点；引入`garbage_bin`模式，通过`ignore_index=0 or -1`控制是否输出一个未分类类别；因此类别权重、分块权重、类别映射都在`dataloader`中控制，而不是在分块时控制
4.  对于后续的分布式训练（虽然暂时没有硬件条件）有一些注意事项：`pytorch-lightning`可以快捷的进行分布式训练，但要采取`ddp`而非`dp`；分布式中的`batch size`指的是每张显卡的，而非全部显卡的；使用`torchmetircs`库进行指标计算，否则每个gpu一个进程将只计算自己那部分的精度；同理尽量少使用`print`语句，否则每个进程都要打印一句，使用`self.log()`，如果一定要打印确保只在主进程进行
5.  采用项目工程化设置，三个主要目录为：输入las目录、输出las目录和工程目录，工程目录应该包括分块数据、复制的完整配置文件、复制的日志文件（或者日志文件就放这里）、训练的模型权重文件等，数据分块和合并输出应该是自动化的过程，参数放在`experiments`中

## 工程结构
```
PointSuite/
├── configs/    # 所有的 YAML 配置文件
│   ├── _base_/ # 可复用的基础配置片段
│   │   ├── data/
│   │   │   └── dales_semseg.yaml   # DALES 数据集的默认配置
│   │   ├── backbones/
│   │   │   └── pt_v2m5.yaml    # PT-v2m5 骨干网络的基础配置
│   │   ├── heads/
│   │   │   └── seg_mlp.yaml    # PT-v2m5 骨干网络的基础配置
│   │   ├── losses/
│   │   │   └── focal_loss.yaml    # PT-v2m5 骨干网络的基础配置
│   │   ├── schedules/
│   │   │   └── cosine_adamw.yaml   # 默认的优化器和调度器
│   │   └── trainer/
│   │       └── default.yaml    # 默认的 trainer 配置 (epochs, accelerator)
│   │
│   └── experiments/    # 完整的实验配置
│       ├── semantic_segmentation/
│       │   └── pt-v2m5_dales_baseline.yaml # 您的第一个实验
│       │
│       ├── instance_segmentation/  # (未来扩展)
│       │   └── pt-v2m5_s3dis.yaml
│       │
│       └── weak_semantic_segmentation/   # (未来扩展)
│           └── ...
│
├── data/   # (可选) 存放原始数据集或数据连接
│   └── dales/
│
├── point_suite/    # 所有的 Python 源代码 (包名可自定, 比如 point_suite)
│   ├── __init__.py
│   │
│   ├── data/   # --- 1. 数据处理 ---
│   │   ├── __init__.py
│   │   ├── point_datamodule.py # 核心: LightningDataModule
│   │   ├── datasets/   # 原始 Dataset 类
│   │   │   └── dales_dataset.py
│   │   │   └── s3dis_dataset.py    # (未来扩展)
│   │   └── transforms/ # 所有的点云预处理/增强类
│   │       └── custom_transforms.py    # (如 CenterShift, GridSample...)
│   │
│   ├── models/ # --- 2. 网络架构 (nn.Module) ---
│   │   ├── __init__.py
│   │   ├── backbones/  # 骨干网络 (可重用)
│   │   │   └── pt_v2m5.py  # 您现在的 PT-v2m5
│   │   │   └── mink_unet.py    # (未来扩展)
│   │   ├── heads/  # 任务特定的头部 (可重用)
│   │   │   └── segmentation_head.py    # (例如一个简单的 MLP)
│   │   │   └── detection_head.py   # (未来扩展)
│   │   └── losses/ # 各种损失函数
│   │       └── focal_loss.py
│   │       └── lovasz_loss.py
│   │
│   ├── tasks/  # --- 3. 核心 "任务" (LightningModule) ---
│   │   ├── __init__.py
│   │   ├── base_task.py    # (可选) 所有任务的基类
│   │   └── semantic_segmentation.py    # 您的第一个任务
│   │   └── instance_segmentation.py    # (未来扩展)
│   │   └── object_detection.py # (未来扩展)
│   │
│   └── utils/  # --- 4. 辅助工具 ---
│       ├── __init__.py
│       ├── metrics.py  # 评估指标 (如 IoU)        
│       └── callbacks.py    # 自定义回调 (如 CudaCacheClearCallback)
│
├── test/   # (可选) 代码测试
│   └── test.py
│
├── main.py    # --- 5. 唯一的训练入口 ---
├── requirements.txt
└── README.md
```
***
## 日志
* **2025/10/25**：思考了好几天项目结构应该是怎样的，又想了好几天数据应该是怎样的，最终决定分块成h5，包含一块las的所有头信息再加上分块信息，同时还有这块las所有点个数的统计，每个分块包含的类别，h5似乎要采取blosc这类快速压缩。今天写完分块的代码吧，还得要一个分块转小块las的工具看看分块效果，再看看能不能写一个overlap分块模式，固定50%重叠率好了，省点事情。现在分块只和分块的大小有关了，样本权重、类别权重这些dataloader里面再计算。
* **2025/10/29**：经过实验，h5通过储存索引再进行分块数据读取的方式太慢了，完全不能满足要求，现在设计了一种新的bin+pkl方式，基于np.memmap进行快速读取，bin存储所有点信息，pkl存储头文件和分块信息，似乎kitti和s3dis也是这种处理方式，现在要把overlap和gridsample加入数据分块过程。
* **2025/11/01**: bin+pkl的实现已经完成，支持overlap和gridsample模式。现在要考虑的事情很多：1.garbage_bin模式的实现，开启时ignore_lable=0,num_classes+1,class_weights在0的位置要有一个占位数；关闭时ignore_label=-1,其他正常；2.类别映射关系，尤其是引入garbage_bin的类别映射关系；3.classes_weight和sample_weight，在引入garbage_bin模式后也不相同；4.除了sample_weight太大用文件保存，其他参数都写入yaml配置，sample_weight记录文件路径；5.在main.py中要实现训练和测试前的预处理，包括选定exp文件夹位置，保存总yaml，修改总yaml，后续是根据修改后的总yaml来的