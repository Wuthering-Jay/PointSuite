"""
DDP (分布式数据并行) 支持检查清单

检查当前代码库中的 DDP 兼容性
"""

# =============================================================================
# ✅ 已经正确支持 DDP 的部分
# =============================================================================

already_ddp_compatible = """
1. ✅ Metrics (pointsuite/utils/metrics.py)
   - 所有指标继承自 torchmetrics.Metric
   - 使用 add_state(..., dist_reduce_fx="sum") 自动聚合
   - 混淆矩阵在 DDP 进程间自动同步
   - OverallAccuracy, MeanIoU, Precision, Recall, F1Score, SegmentationMetrics
   
   原理：
   - torchmetrics 会在 compute() 前自动调用 all_gather
   - 每个 GPU 的局部混淆矩阵会被求和到主进程
   
2. ✅ Losses (pointsuite/models/losses/)
   - 所有损失函数都是标准的 nn.Module
   - 不包含需要同步的状态
   - CrossEntropyLoss, FocalLoss, LovaszLoss, DiceLoss, DiceCELoss
   
3. ✅ BaseTask 的 log_dict()
   - PyTorch Lightning 自动处理 DDP 下的 logging
   - 使用 self.log_dict(..., batch_size=batch_size) 
   - PL 会自动 reduce 损失到主进程
   
4. ✅ validation_step / test_step
   - 指标的 update() 在各 GPU 本地执行
   - on_validation_epoch_end / on_test_epoch_end 中 compute() 时自动同步
   
5. ✅ Model 的 forward()
   - PointNet++ 等模型都是标准 nn.Module
   - DDP 会自动包装并同步梯度
"""

# =============================================================================
# ⚠️ 需要注意的地方（已经是正确的，但需要理解）
# =============================================================================

ddp_considerations = """
1. ⚠️ _get_batch_size() 中的 .item()
   
   当前代码：
   ```python
   def _get_batch_size(self, batch: Dict[str, Any]) -> int:
       if 'batch_index' in batch:
           return batch['batch_index'].max().item() + 1  # ⚠️ .item()
       elif 'offset' in batch:
           return len(batch['offset'])
   ```
   
   问题：
   - .item() 会触发 GPU -> CPU 同步
   - 在 DDP 中，每个进程的 batch_size 可能不同（最后一个 batch）
   
   当前状态：✅ 已经正确
   - 这是必要的同步，因为需要传递给 self.log_dict(batch_size=...)
   - PyTorch Lightning 会正确处理不同进程间的 batch_size 差异
   - 只在 logging 时调用，对性能影响很小
   
2. ⚠️ predict_step() 中的 .cpu()
   
   当前代码：
   ```python
   def predict_step(self, batch, batch_idx):
       preds = self.forward(batch)
       results = {
           "preds": preds.cpu(),  # ⚠️ .cpu()
           "logits": logits.cpu(),
       }
       return results
   ```
   
   DDP 场景：
   - 在推理时，每个 GPU 会处理数据的不同部分
   - predict_step 的输出会被收集到主进程
   - .cpu() 是必要的，避免 GPU 内存爆炸
   
   当前状态：✅ 已经正确
   - PyTorch Lightning 的 Trainer.predict() 会自动收集所有 GPU 的结果
   - 使用 .cpu() 可以避免跨 GPU 传输大量数据
"""

# =============================================================================
# ✅ 不需要修改的地方
# =============================================================================

no_changes_needed = """
1. ✅ training_step 返回值
   
   当前代码：
   ```python
   def training_step(self, batch, batch_idx):
       loss_dict = self._calculate_total_loss(preds, batch)
       self.log_dict(loss_dict, ...)
       return loss_dict["total_loss"]  # ✅ 正确
   ```
   
   DDP 处理：
   - PyTorch Lightning 会自动对返回的 loss 调用 all_reduce
   - 梯度会在 backward() 后自动同步（DDP 的核心功能）
   - 不需要手动 synchronize
   
2. ✅ validation_step / test_step
   
   当前代码：
   ```python
   def validation_step(self, batch, batch_idx):
       preds = self.forward(batch)
       loss_dict = self._calculate_total_loss(preds, batch)
       self.log_dict(loss_dict, ...)
       for metric in self.val_metrics.values():
           metric.update(preds, batch)  # ✅ 本地更新
   
   def on_validation_epoch_end(self):
       for name, metric in self.val_metrics.items():
           metric_results[name] = metric.compute()  # ✅ 自动同步
           metric.reset()
   ```
   
   DDP 处理：
   - update() 在各 GPU 本地执行，不需要同步
   - compute() 时 torchmetrics 自动调用 all_gather 同步状态
   - reset() 在各 GPU 本地执行
   
3. ✅ 损失函数计算
   
   当前代码：
   ```python
   def _calculate_total_loss(self, preds, batch):
       total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
       for name, loss_fn in self.losses.items():
           loss = loss_fn(preds, batch)
           total_loss += self.loss_weights[name] * loss
       return {"total_loss": total_loss}
   ```
   
   DDP 处理：
   - 损失在各 GPU 本地计算
   - backward() 时梯度自动同步
   - 不需要手动同步损失值（只用于显示）
"""

# =============================================================================
# 📋 DDP 使用检查清单
# =============================================================================

ddp_checklist = """
使用 DDP 训练时需要注意的配置：

1. ✅ Trainer 配置
   ```yaml
   trainer:
     accelerator: gpu
     devices: 4              # 使用 4 个 GPU
     strategy: ddp           # 或 ddp_spawn, ddp_find_unused_parameters_false
     sync_batchnorm: true    # 如果模型使用 BatchNorm，建议开启
   ```

2. ✅ DataLoader 配置
   - 不需要手动设置 DistributedSampler
   - PyTorch Lightning 会自动处理
   - 每个 GPU 会获得 batch_size / num_gpus 的数据
   
   ```yaml
   data:
     batch_size: 16  # 每个 GPU 的 batch_size
     num_workers: 4  # 每个 GPU 的 worker 数
   ```

3. ✅ Metrics 配置
   - 使用 torchmetrics（已完成）
   - 或使用我们的 SegmentationMetrics（继承自 torchmetrics）
   
   ```yaml
   metrics:
     all:
       class_path: pointsuite.utils.metrics.SegmentationMetrics
       init_args:
         num_classes: 8
         ignore_index: -1
   ```

4. ✅ Logging
   - self.log(..., sync_dist=True) 会在所有进程间同步
   - 默认情况下，validation 的 log 会自动 sync_dist=True
   - training 的 log 默认 sync_dist=False（性能考虑）
   
   如果需要精确的 training metrics：
   ```python
   self.log_dict(metrics, sync_dist=True)  # 强制同步
   ```

5. ✅ 启动命令
   
   方法 1: torchrun (推荐)
   ```bash
   torchrun --nproc_per_node=4 train.py fit --config config.yaml
   ```
   
   方法 2: python -m torch.distributed.launch
   ```bash
   python -m torch.distributed.launch --nproc_per_node=4 train.py fit --config config.yaml
   ```
   
   方法 3: SLURM (集群)
   ```bash
   srun --nodes=2 --ntasks-per-node=4 --gres=gpu:4 python train.py fit --config config.yaml
   ```
"""

# =============================================================================
# 🚀 性能优化建议
# =============================================================================

performance_tips = """
DDP 性能优化建议：

1. 使用 SegmentationMetrics 而不是多个独立指标
   ❌ 差：
   ```yaml
   metrics:
     oa: {...}
     miou: {...}
     precision: {...}
     recall: {...}
     f1: {...}
   ```
   每个指标都会触发一次 all_gather（5次同步）
   
   ✅ 好：
   ```yaml
   metrics:
     all:
       class_path: pointsuite.utils.metrics.SegmentationMetrics
   ```
   只触发一次 all_gather

2. 适当的 log 频率
   ```yaml
   trainer:
     log_every_n_steps: 50  # 不要太频繁
   ```

3. 使用合适的 sync_batchnorm
   - 小 batch_size 时：sync_batchnorm=True
   - 大 batch_size 时：sync_batchnorm=False（性能更好）

4. 找到最佳的 num_workers
   - 通常设置为 CPU 核心数 / GPU 数
   - 例如：64 核 CPU，4 GPU -> num_workers=16

5. 使用 gradient_clip_val
   ```yaml
   trainer:
     gradient_clip_val: 1.0  # 防止梯度爆炸
   ```

6. 考虑使用混合精度训练
   ```yaml
   trainer:
     precision: 16  # 或 'bf16'
   ```
"""

# =============================================================================
# 🐛 常见 DDP 问题和解决方案
# =============================================================================

common_issues = """
常见 DDP 问题：

1. ❌ 问题：进程卡住不动
   原因：不同进程执行了不同数量的 collective 操作
   解决：确保所有进程执行相同的代码路径
   
   例如，避免：
   ```python
   if self.global_rank == 0:
       metric.compute()  # ❌ 只有主进程执行
   ```
   
   应该：
   ```python
   result = metric.compute()  # ✅ 所有进程都执行
   if self.global_rank == 0:
       print(result)  # 只在主进程打印
   ```

2. ❌ 问题：指标不准确
   原因：忘记在 epoch 结束时 reset()
   解决：已在 on_validation_epoch_end 中正确实现
   
   ```python
   def on_validation_epoch_end(self):
       for metric in self.val_metrics.values():
           result = metric.compute()
           metric.reset()  # ✅ 必须 reset
   ```

3. ❌ 问题：OOM (Out of Memory)
   原因：所有 GPU 都存储完整的 validation 结果
   解决：在 predict_step 中使用 .cpu()（已实现）
   
4. ❌ 问题：batch_size 相关的错误
   原因：最后一个 batch 可能不完整
   解决：使用 drop_last=False 和正确的 batch_size logging（已实现）

5. ❌ 问题：loss 是 NaN
   可能原因：
   - 学习率太大
   - 梯度爆炸（使用 gradient_clip_val）
   - 数据归一化问题
   - 某些 GPU 的数据有问题
   
   调试：
   ```yaml
   trainer:
     detect_anomaly: true  # 检测 NaN
     track_grad_norm: 2    # 追踪梯度范数
   ```
"""

# =============================================================================
# 总结
# =============================================================================

summary = """
✅ 总结：当前代码已经完全支持 DDP

需要做的事情：
1. ✅ Metrics 使用 torchmetrics（已完成）
2. ✅ 使用 SegmentationMetrics 减少同步次数（已实现）
3. ✅ 正确的 batch_size logging（已实现）
4. ✅ 在 predict_step 使用 .cpu()（已实现）
5. ✅ 在 epoch 结束时 reset metrics（已实现）

不需要修改的代码：
- ✅ training_step（自动同步梯度）
- ✅ validation_step（指标自动同步）
- ✅ _calculate_total_loss（本地计算）
- ✅ forward()（DDP 自动包装）

使用 DDP 的命令：
```bash
# 单机多卡
torchrun --nproc_per_node=4 train.py fit --config config.yaml

# 配置文件
trainer:
  accelerator: gpu
  devices: 4
  strategy: ddp
  sync_batchnorm: true  # 如果使用 BatchNorm
```

性能提升：
- 使用 SegmentationMetrics：同步次数从 5 次减少到 1 次
- 4 GPU 理论加速：接近 4x（取决于通信开销）
- 实际加速：通常 3-3.5x
"""

if __name__ == "__main__":
    print("=" * 80)
    print("✅ 已经正确支持 DDP 的部分")
    print("=" * 80)
    print(already_ddp_compatible)
    
    print("\n" + "=" * 80)
    print("⚠️ 需要注意的地方")
    print("=" * 80)
    print(ddp_considerations)
    
    print("\n" + "=" * 80)
    print("✅ 不需要修改的地方")
    print("=" * 80)
    print(no_changes_needed)
    
    print("\n" + "=" * 80)
    print("📋 DDP 使用检查清单")
    print("=" * 80)
    print(ddp_checklist)
    
    print("\n" + "=" * 80)
    print("🚀 性能优化建议")
    print("=" * 80)
    print(performance_tips)
    
    print("\n" + "=" * 80)
    print("🐛 常见 DDP 问题")
    print("=" * 80)
    print(common_issues)
    
    print("\n" + "=" * 80)
    print("✅ 总结")
    print("=" * 80)
    print(summary)
