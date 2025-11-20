# 🚀 PointSuite 训练就绪检查清单

## ✅ 核心组件状态

### 1. 数据模块 (DataModule) - ✅ 完整
- ✅ `DataModuleBase`: 抽象基类，支持所有阶段
- ✅ `BinPklDataModule`: 具体实现
- ✅ 支持 train/val/test/predict 四个阶段
- ✅ 独立的数据路径配置 (`train_data`, `val_data`, `test_data`, `predict_data`)
- ✅ 独立的 loop 参数 (`train_loop`, `val_loop`, `test_loop`, `predict_loop`)
- ✅ 独立的 transform 配置
- ✅ 支持 DynamicBatchSampler（训练和推理独立配置）
- ✅ 支持 WeightedRandomSampler（仅训练阶段）

### 2. 数据集 (Dataset) - ✅ 完整
- ✅ `DatasetBase`: 抽象基类
- ✅ `BinPklDataset`: 具体实现
- ✅ 支持 loop 参数（所有阶段）
- ✅ 支持 class_mapping（映射到 ignore_label）
- ✅ test/predict 阶段提供 indices 和文件信息

### 3. 任务模块 (Task) - ✅ 完整
- ✅ `BaseTask`: 抽象基类，包含所有阶段逻辑
  - ✅ `training_step`: 计算损失，返回 total_loss
  - ✅ `validation_step`: 计算损失和指标
  - ✅ `test_step`: 计算损失和指标（支持回调）
  - ✅ `on_validation_epoch_end`: 记录指标
  - ✅ `on_test_epoch_end`: 记录指标
  - ✅ `postprocess_predictions`: Mask3D 兼容钩子
- ✅ `SemanticSegmentationTask`: 语义分割实现
  - ✅ `forward`: backbone + head
  - ✅ `training_step`: 调用 BaseTask
  - ✅ `predict_step`: 返回 logits + indices + 文件信息

### 4. 损失函数 (Losses) - ✅ 完整
- ✅ `CrossEntropyLoss`: 支持 ignore_index
- ✅ 自动从配置实例化
- ✅ 支持多损失函数组合（带权重）

### 5. 指标 (Metrics) - ✅ 完整
- ✅ `OverallAccuracy`: 整体精度
- ✅ `MeanIoU`: 平均 IoU
- ✅ 支持 labels 和 logits 两种输入格式
- ✅ 自动从配置实例化
- ✅ 分别为 val/test 阶段创建独立实例

### 6. 回调函数 (Callbacks) - ✅ 完整
- ✅ `SegmentationWriter`: 保存预测结果为 .las
  - ✅ `write_on_batch_end`: 流式写入临时文件
  - ✅ `on_predict_end`: 投票并保存最终结果
  - ✅ 支持 TTA（多次预测投票）
  - ✅ 支持 reverse_class_mapping
  - ✅ 保留原始 LAS 属性

### 7. Transforms - ✅ 完整
- ✅ `CenterShift`: 中心化
- ✅ `AutoNormalizeHNorm`: 归一化高程
- ✅ `RandomRotate`: 随机旋转
- ✅ `RandomScale`: 随机缩放
- ✅ `Collect`: 收集指定字段
- ✅ `ToTensor`: 转换为 Tensor

---

## ⚠️ 缺失组件

### 1. 训练入口 (main.py) - ❌ 空文件
**这是唯一缺失的关键组件！**

需要实现：
- LightningCLI 集成
- 支持 fit/validate/test/predict 命令
- 配置文件加载
- 实验管理

---

## 🎯 四个阶段流程检查

### 1. Training (训练) - ✅ 支持
```python
# 数据流：
DataModule.train_dataloader() 
  → Dataset.__getitem__(transform=train_transforms, loop=train_loop)
  → collate_fn()
  → Task.training_step(batch)
  → Task._calculate_total_loss() → losses
  → 返回 total_loss

# 关键特性：
✅ 支持 DynamicBatchSampler (use_dynamic_batch=True)
✅ 支持 WeightedRandomSampler (use_weighted_sampler=True)
✅ 支持 loop > 1 (数据增强)
✅ shuffle=True, drop_last=True
```

### 2. Validation (验证) - ✅ 支持
```python
# 数据流：
DataModule.val_dataloader()
  → Dataset.__getitem__(transform=val_transforms, loop=val_loop)
  → collate_fn()
  → Task.validation_step(batch)
  → Task._calculate_total_loss() → losses
  → Task.postprocess_predictions() → processed_preds
  → val_metrics.update(processed_preds, batch)
  → Task.on_validation_epoch_end()
  → val_metrics.compute() → 记录到 logger

# 关键特性：
✅ 计算损失和指标
✅ 不保存预测结果
✅ 支持 DynamicBatchSampler (use_dynamic_batch_inference=True)
✅ 支持 loop > 1 (TTA)
✅ shuffle=False, drop_last=False
```

### 3. Test (测试) - ✅ 支持
```python
# 数据流：
DataModule.test_dataloader()
  → Dataset.__getitem__(transform=test_transforms, loop=test_loop)
      → 返回 {coord, feat, class, indices, bin_file, bin_path, pkl_path}
  → collate_fn()
  → Task.test_step(batch)
  → Task._calculate_total_loss() → losses
  → Task.postprocess_predictions() → processed_preds
  → test_metrics.update(processed_preds, batch)
  → Task.on_test_epoch_end()
  → test_metrics.compute() → 记录到 logger

# 可选：保存预测结果
trainer.test(model, datamodule, callbacks=[SegmentationWriter(...)])

# 关键特性：
✅ 计算损失和指标
✅ 可选保存预测结果（通过回调）
✅ 支持 DynamicBatchSampler (use_dynamic_batch_inference=True)
✅ 支持 loop > 1 (TTA)
✅ shuffle=False, drop_last=False
✅ Dataset 提供 indices 和文件信息
```

### 4. Predict (预测) - ✅ 支持
```python
# 数据流：
DataModule.predict_dataloader()
  → Dataset.__getitem__(transform=predict_transforms, loop=predict_loop)
      → 返回 {coord, feat, indices, bin_file, bin_path, pkl_path}
      → 无 'class' 字段（无真值标签）
  → collate_fn()
  → Task.predict_step(batch)
  → Task.postprocess_predictions() → processed_preds
  → 返回 {logits, indices, bin_file, bin_path, pkl_path, coord}
  → SegmentationWriter.write_on_batch_end()
      → 流式写入临时文件
  → SegmentationWriter.on_predict_end()
      → 投票并保存 .las 文件

# 必须配置回调：
trainer.predict(model, datamodule, callbacks=[SegmentationWriter(...)])

# 关键特性：
✅ 不计算损失和指标（无真值标签）
✅ 必须保存预测结果
✅ 支持 DynamicBatchSampler (use_dynamic_batch_inference=True)
✅ 支持 loop > 1 (TTA + 投票)
✅ shuffle=False, drop_last=False
✅ Dataset 提供 indices 和文件信息
```

---

## 🔧 已知限制和注意事项

### 1. DynamicBatchSampler + TTA
- ⚠️ 点数基于 transform **之前**的值预计算
- ✅ 适用于：减少点数或略微增加点数的 transform
- ❌ 不适用于：大幅增加点数的 transform（如密集采样）
- 💡 解决方案：transform 大幅增加点数时，设置 `use_dynamic_batch_inference=False`

### 2. Predict 阶段无真值标签
- ✅ Dataset 不返回 'class' 字段
- ✅ Task.predict_step 不计算损失和指标
- ✅ 只返回预测结果

### 3. Test vs Predict
- **Test**: 有标签，计算指标 + 可选保存结果
- **Predict**: 无标签，只保存结果

---

## 📝 训练前准备

### 必需步骤：

1. **创建 main.py** - ⚠️ 当前为空
   ```python
   # 需要实现 LightningCLI 入口
   from pytorch_lightning.cli import LightningCLI
   
   class PointSuiteCLI(LightningCLI):
       def add_arguments_to_parser(self, parser):
           # 添加自定义参数
           pass
   
   if __name__ == "__main__":
       cli = PointSuiteCLI()
   ```

2. **创建实验配置文件**
   - `configs/experiments/my_experiment.yaml`
   - 定义 model, data, trainer, callbacks

3. **准备数据**
   - bin + pkl 文件（通过 tile.py 生成）
   - 确保 pkl 包含 'num_points' 和文件信息

### 可选步骤：

4. **类别映射** (如果类别不连续)
   ```python
   class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
   reverse_class_mapping = {0: 0, 1: 1, 2: 2, 3: 6, 4: 9}
   ```

5. **加权采样** (如果类别不平衡)
   ```python
   # 计算权重
   weights = compute_weights(train_dataset)
   # 如果 train_loop > 1，需要重复权重
   weights = weights * train_loop
   ```

---

## ✅ 结论

**框架核心功能已完整实现！**

唯一缺失的是 `main.py` 训练入口，但这不影响你手动编写训练脚本：

```python
import pytorch_lightning as pl
from pointsuite.data import BinPklDataModule
from pointsuite.tasks import SemanticSegmentationTask
from pointsuite.models.backbones import PointTransformerV2m5
from pointsuite.models.heads import SegmentationHead

# 1. 创建 DataModule
datamodule = BinPklDataModule(
    train_data='data/train',
    val_data='data/val',
    test_data='data/test',
    batch_size=8,
    num_workers=4,
    # ... 其他参数
)

# 2. 创建 Model
backbone = PointTransformerV2m5(...)
head = SegmentationHead(...)
model = SemanticSegmentationTask(
    backbone=backbone,
    head=head,
    learning_rate=0.001,
    loss_configs=[...],
    metric_configs=[...]
)

# 3. 创建 Trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    callbacks=[...]
)

# 4. 训练
trainer.fit(model, datamodule)

# 5. 测试
trainer.test(model, datamodule)

# 6. 预测
from pointsuite.utils.callbacks import SegmentationWriter
writer = SegmentationWriter(output_dir='predictions')
trainer.predict(model, datamodule, callbacks=[writer])
```

**🎉 可以开始训练了！**
