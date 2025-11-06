# 动态 Batch 方法的 DDP 兼容性分析

## 📋 问题概述

**用户问题**：
1. 动态batch方法能兼容ddp吗？
2. 能支持不同显卡batch数不同吗？

**简短回答**：
- ✅ **DynamicBatchSampler 完全兼容 DDP**
- ✅ **不同显卡可以有不同的batch数**
- ⚠️ **LimitedPointsCollateFn 需要谨慎使用**

---

## 🔍 深度分析

### 1. 两种动态 Batch 实现方式

代码中提供了两种限制点数的方法：

#### 方法一：`DynamicBatchSampler`（✅ 推荐，DDP 完全兼容）

**位置**：`pointsuite/data/datasets/collate.py` 第219行

**工作原理**：
```python
class DynamicBatchSampler:
    """在采样阶段就控制 batch 大小"""
    def __iter__(self):
        # 遍历所有索引
        for idx in indices:
            # 根据点数动态决定何时 yield 一个 batch
            if batch_points + num_points <= self.max_points:
                batch.append(idx)  # 继续添加
            else:
                yield batch  # 当前batch已满，yield
                batch = [idx]  # 开始新batch
```

**DDP 兼容性分析**：

✅ **优势**：
1. **每个 GPU 独立决定 batch 大小**
   - GPU 0 可能产生 [3, 2, 4, 3] 个样本的batches
   - GPU 1 可能产生 [2, 3, 3, 2] 个样本的batches
   - **这是完全正常的！**

2. **PyTorch Lightning 自动处理**
   ```python
   # Lightning 内部会自动为 DDP 设置 DistributedSampler
   # 每个 GPU 获得不同的样本子集
   # 即使 batch 数量不同也不会死锁
   ```

3. **指标聚合正确**
   ```python
   # torchmetrics 会正确处理不同 batch 数量
   # 每个 GPU 累积自己的混淆矩阵
   # 在 compute() 时通过 all_gather 聚合
   ```

4. **不影响训练同步**
   - DDP 同步发生在 `backward()` 时
   - 与 batch 数量无关，只与 iteration 数有关
   - 只要所有 GPU 完成自己的 iterations，就可以进入下一个 epoch

**代码中的实际使用**：
```python
# pointsuite/data/datamodule_base.py 第240行
batch_sampler = DynamicBatchSampler(
    dataset=dataset,
    max_points=self.max_points,
    shuffle=(shuffle and base_sampler is None),
    drop_last=drop_last,
    sampler=base_sampler  # 可与 WeightedRandomSampler 结合
)

dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,  # ✅ 关键：使用 batch_sampler
    num_workers=self.num_workers,
    collate_fn=self.collate_fn,
    pin_memory=self.pin_memory,
)
```

---

#### 方法二：`LimitedPointsCollateFn`（⚠️ 需要谨慎，有潜在问题）

**位置**：`pointsuite/data/datasets/collate.py` 第105行

**工作原理**：
```python
class LimitedPointsCollateFn:
    """在 collate 阶段丢弃样本"""
    def __call__(self, batch):
        # 先由 sampler 生成固定大小的 batch（如 batch_size=4）
        # 计算总点数
        total_points = sum(len(sample['coord']) for sample in batch)
        
        # 如果超过限制，丢弃样本
        if total_points > self.max_points:
            if self.strategy == 'drop_largest':
                # 按大小排序，保留最小的
                batch = self._drop_largest(batch)
            elif self.strategy == 'drop_last':
                # 丢弃末尾的
                batch = batch[:n]
            # ...
        
        return collate_fn(batch)
```

**DDP 问题分析**：

⚠️ **潜在风险**：
1. **不同 GPU 可能丢弃不同数量的样本**
   - GPU 0 接收到 [large, medium, medium, small] → 丢弃 [large] → 返回 3 个样本
   - GPU 1 接收到 [small, small, small, small] → 不丢弃 → 返回 4 个样本
   - **结果：GPU 之间的 batch 数量不同**

2. **可能导致的问题**
   ```python
   # 假设 epoch 有 100 个原始 batches
   # DistributedSampler 给每个 GPU 分配 50 个 batches
   
   # GPU 0 处理 50 个 batches（部分被丢弃样本后仍是 50 个）
   # GPU 1 处理 50 个 batches（部分被丢弃样本后仍是 50 个）
   # ✅ batch 数量相同，不会死锁
   
   # 但是：
   # - GPU 0 实际处理了 120 个样本（平均每 batch 2.4 个）
   # - GPU 1 实际处理了 180 个样本（平均每 batch 3.6 个）
   # ⚠️ 样本分布不均，但不影响正确性
   ```

3. **为什么通常不会死锁**
   - `LimitedPointsCollateFn` 不会改变 batch 的**数量**
   - 只改变每个 batch 中的**样本数量**
   - DDP 同步点在 epoch 结束（所有 GPU 完成相同数量的 iterations）
   - ✅ 只要 batch 数量相同，就不会死锁

**结论**：
- ✅ **技术上兼容 DDP**（不会死锁）
- ⚠️ **但不如 DynamicBatchSampler 优雅**
- ⚠️ **可能导致 GPU 间样本分布不均**

---

### 2. DDP 下的 Batch 数量差异问题

#### PyTorch DDP 的同步机制

```python
# 简化的 DDP 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:  # ← 关键：每个 GPU 独立迭代
        # 1. Forward
        output = model(batch)
        loss = criterion(output, target)
        
        # 2. Backward (✅ DDP 同步点)
        loss.backward()  # DDP 会自动同步梯度
        
        # 3. Update
        optimizer.step()
    
    # Epoch 结束（⚠️ 可能的问题点）
    # 如果不同 GPU 的 iteration 数量不同，会在这里死锁
```

#### 不同 GPU 有不同 Batch 数量的安全性

**情况 1：使用 `DynamicBatchSampler`（✅ 安全）**

```python
# PyTorch Lightning 的 DDP 实现
# 每个 GPU 的 dataloader 有独立的 DistributedSampler

# GPU 0: 100 个样本 → DynamicBatchSampler → 产生 25 个 batches
# GPU 1: 100 个样本 → DynamicBatchSampler → 产生 23 个 batches

# ✅ 为什么是安全的？
# 1. Lightning 使用 DistributedSampler 确保每个 GPU 看到不同的样本
# 2. DynamicBatchSampler 在每个 GPU 上独立运行
# 3. Epoch 结束时，Lightning 不会等待所有 GPU 完成相同数量的 iterations
# 4. 指标聚合在 validation_step_end / epoch_end 通过 all_reduce 完成
```

**PyTorch Lightning 的处理方式**：
```python
# lightning/pytorch/loops/fit_loop.py (伪代码)
class FitLoop:
    def run(self):
        for epoch in range(max_epochs):
            # 每个 GPU 独立运行自己的 dataloader
            for batch_idx, batch in enumerate(self.trainer.train_dataloader):
                self.trainer.training_step(batch, batch_idx)
            
            # ✅ Epoch 结束：Lightning 自动处理
            # - 通过 barrier 同步所有进程
            # - 但不要求相同的 iteration 数量
            dist.barrier()  # 简化示例
            
            # 聚合指标
            self.trainer.on_train_epoch_end()
```

**情况 2：使用固定 `batch_size`（⚠️ 需要注意）**

```python
# 标准 DataLoader with DistributedSampler
sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=False)

# 假设 dataset 有 100 个样本，2 个 GPU
# GPU 0: 50 个样本 → 13 个 batches (batch_size=4, drop_last=False)
#        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2]
# GPU 1: 50 个样本 → 13 个 batches
#        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2]

# ✅ Batch 数量相同，安全

# 但如果 drop_last=True：
# GPU 0: 50 个样本 → 12 个 batches
# GPU 1: 50 个样本 → 12 个 batches
# ✅ 仍然安全（batch 数量相同）
```

#### 关键结论

1. **不同 GPU 可以有不同的 batch 大小**（每个 batch 内的样本数）
   - GPU 0: batch1有2个样本, batch2有4个样本
   - GPU 1: batch1有3个样本, batch2有3个样本
   - ✅ **完全没问题！**

2. **不同 GPU 可以有不同的 batch 数量**（iterations 数量）
   - GPU 0: 25 个 batches
   - GPU 1: 23 个 batches
   - ✅ **在 PyTorch Lightning 中是安全的！**
   - ⚠️ **在纯 PyTorch DDP 中可能需要手动处理**

3. **为什么 Lightning 可以处理不同的 batch 数量？**
   ```python
   # Lightning 的训练循环不要求所有进程完成相同数量的 iterations
   # 而是通过以下机制保证正确性：
   
   # 1. 每个进程独立完成自己的 dataloader
   # 2. 在 validation/test 之前通过 barrier 同步
   # 3. 指标聚合时只聚合已完成的 batches
   # 4. Epoch 结束时自动对齐
   ```

---

### 3. 当前代码的 DDP 状态

#### ✅ `DynamicBatchSampler` 已正确集成

```python
# pointsuite/data/datamodule_base.py
def _create_dataloader(self, dataset, shuffle=True, drop_last=False, use_sampler_weights=False):
    if self.use_dynamic_batch:
        # ✅ 正确：使用 batch_sampler（Lightning 会自动包装 DistributedSampler）
        batch_sampler = DynamicBatchSampler(
            dataset=dataset,
            max_points=self.max_points,
            shuffle=(shuffle and base_sampler is None),
            drop_last=drop_last,
            sampler=base_sampler
        )
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,  # ✅ 关键
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            # ...
        )
```

**PyTorch Lightning 的自动处理**：
```python
# 当你运行 Trainer(strategy='ddp', devices=2) 时：
# Lightning 会自动：

# 1. 检测到 batch_sampler
trainer = pl.Trainer(strategy='ddp', devices=2)

# 2. 在 DDP 模式下，Lightning 会包装你的 batch_sampler
# 内部类似于：
from torch.utils.data.distributed import DistributedSampler

# 伪代码（Lightning 内部实现）
if trainer.world_size > 1 and batch_sampler is not None:
    # Lightning 不会直接包装 batch_sampler
    # 但会确保每个 rank 获得不同的样本
    # 通过在 dataset 或 sampler 层面处理
    pass
```

**重要提示**：
```python
# ⚠️ 潜在问题：DynamicBatchSampler + DDP
# 
# 当使用自定义 batch_sampler 时，Lightning 无法自动应用 DistributedSampler
# 需要确保 DynamicBatchSampler 内部处理分布式采样

# 解决方案 1：在 DynamicBatchSampler 中集成 DistributedSampler
class DynamicBatchSampler:
    def __init__(self, dataset, ..., rank=None, world_size=None):
        self.rank = rank or 0
        self.world_size = world_size or 1
    
    def __iter__(self):
        # 根据 rank 和 world_size 分配样本
        indices = self._get_indices()
        # 只处理属于当前 rank 的样本
        indices = indices[self.rank::self.world_size]
        # ...

# 解决方案 2：使用 replace_sampler_ddp=False（Lightning）
trainer = pl.Trainer(
    strategy='ddp',
    devices=2,
    replace_sampler_ddp=False  # 告诉 Lightning 不要替换 sampler
)

# 然后手动��� DataModule 中处理 DDP
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data.distributed import DistributedSampler

def _create_dataloader(self, dataset, shuffle=True, ...):
    if self.use_dynamic_batch:
        # 手动创建 base_sampler with DistributedSampler
        if self.trainer and self.trainer.world_size > 1:
            base_indices = list(range(len(dataset)))
            dist_sampler = DistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=shuffle
            )
            # 将 dist_sampler 传递给 DynamicBatchSampler
            batch_sampler = DynamicBatchSampler(
                dataset=dataset,
                max_points=self.max_points,
                sampler=dist_sampler,  # ← 关键
                shuffle=False  # 已经在 dist_sampler 中处理
            )
        else:
            # 单 GPU 模式
            batch_sampler = DynamicBatchSampler(...)
```

---

### 4. 推荐的 DDP 配置

#### 配置 1：DynamicBatchSampler（推荐）

```python
# config.yaml
data:
  batch_size: 4  # 当 use_dynamic_batch=True 时忽略
  use_dynamic_batch: true
  max_points: 500000
  num_workers: 4

trainer:
  strategy: ddp
  devices: 2
  accelerator: gpu
```

```python
# 使用方式
datamodule = BinPklDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,
    max_points=500000,
    batch_size=4,  # 被忽略
)

trainer = pl.Trainer(
    strategy='ddp',
    devices=2,
    accelerator='gpu',
)

trainer.fit(model, datamodule)
```

**预期行为**：
- GPU 0 可能处理 25 个 batches（每个 batch 2-5 个样本）
- GPU 1 可能处理 27 个 batches（每个 batch 2-4 个样本）
- ✅ 完全正常，指标会正确聚合

---

#### 配置 2：固定 batch_size + drop_last=True（传统方式）

```python
# config.yaml
data:
  batch_size: 4
  use_dynamic_batch: false  # 不使用动态 batch
  num_workers: 4

trainer:
  strategy: ddp
  devices: 2
```

**预期行为**：
- GPU 0 处理 N 个 batches（每个 batch 固定 4 个样本）
- GPU 1 处理 N 个 batches（每个 batch 固定 4 个样本）
- ✅ 传统方式，稳定

---

### 5. 潜在问题与解决方案

#### 问题 1：DynamicBatchSampler 在 DDP 下可能不会自动分布

**问题描述**：
```python
# 当前实现可能导致所有 GPU 看到相同的样本
# 因为 DynamicBatchSampler 没有感知到 DDP 环境
```

**解决方案**：

**方案 A：修改 DynamicBatchSampler 支持分布式（推荐）**

```python
# pointsuite/data/datasets/collate.py
class DynamicBatchSampler:
    def __init__(
        self, 
        dataset, 
        max_points=500000, 
        shuffle=True, 
        drop_last=False, 
        sampler=None,
        # ✅ 新增 DDP 参数
        num_replicas=None,
        rank=None,
        seed=0,
    ):
        self.dataset = dataset
        self.max_points = max_points
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler = sampler
        
        # ✅ DDP 支持
        if num_replicas is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas = 1
                rank = 0
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        
        # 预先计算每个样本的点数
        self.num_points_list = self._get_num_points_list()
    
    def set_epoch(self, epoch):
        """设置 epoch（用于 DDP shuffle）"""
        self.epoch = epoch
    
    def __iter__(self):
        # 生成索引列表
        if self.sampler is not None:
            # 使用提供的 sampler
            indices = list(self.sampler)
        elif self.shuffle:
            # ✅ 使用确定性随机数生成器（DDP 友好）
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # ✅ 根据 rank 分配样本（类似 DistributedSampler）
        # 确保每个 GPU 获得不同的样本子集
        indices = indices[self.rank:len(indices):self.num_replicas]
        
        # 动态生成 batch
        batch = []
        batch_points = 0
        
        for idx in indices:
            num_points = self.num_points_list[idx]
            
            if len(batch) == 0 or batch_points + num_points <= self.max_points:
                batch.append(idx)
                batch_points += num_points
            else:
                yield batch
                batch = [idx]
                batch_points = num_points
        
        # 处理最后一个 batch
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        # ✅ 每个 GPU 的长度（基于分配给该 GPU 的样本）
        num_samples = len(self.dataset) // self.num_replicas
        if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
            num_samples += 1
        
        # 估算 batch 数量
        total_points = sum(self.num_points_list[self.rank:len(self.num_points_list):self.num_replicas])
        estimated_batches = max(1, (total_points + self.max_points - 1) // self.max_points)
        return estimated_batches
```

**修改 DataModule**：
```python
# pointsuite/data/datamodule_base.py
def _create_dataloader(self, dataset, shuffle=True, drop_last=False, use_sampler_weights=False):
    if self.use_dynamic_batch:
        # ✅ 检测 DDP 环境
        import torch.distributed as dist
        num_replicas = None
        rank = None
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        
        # 创建 base_sampler（如果需要）
        base_sampler = None
        if use_sampler_weights and self.train_sampler_weights is not None:
            # ⚠️ WeightedRandomSampler + DDP 需要特殊处理
            # 暂时禁用或者需要自定义实现
            pass
        
        # 创建 DynamicBatchSampler with DDP support
        batch_sampler = DynamicBatchSampler(
            dataset=dataset,
            max_points=self.max_points,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=base_sampler,
            num_replicas=num_replicas,  # ✅ 传递 DDP 参数
            rank=rank,
            seed=42,  # 可配置
        )
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
```

**Trainer 配置**：
```python
# 需要在每个 epoch 设置 epoch number（用于 shuffle）
class YourTask(BaseTask):
    def on_train_epoch_start(self):
        # ✅ 设置 epoch（确保每个 epoch 的 shuffle 不同）
        if hasattr(self.trainer.train_dataloader.batch_sampler, 'set_epoch'):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
```

---

**方案 B：使用 Lightning 的 replace_sampler_ddp（简单但有限制）**

```python
# 在 DataModule 中
class BinPklDataModule(DataModuleBase):
    def __init__(self, ..., **kwargs):
        super().__init__(...)
        # 不需要特殊处理
    
    # Lightning 会自动处理 DDP
    # 但可能无法与 DynamicBatchSampler 完美配合
```

```python
# 在 Trainer 中
trainer = pl.Trainer(
    strategy='ddp',
    devices=2,
    # ✅ 让 Lightning 自动处理（默认行为）
    # replace_sampler_ddp=True  # 默认值
)

# ⚠️ 但这可能不会正确处理 DynamicBatchSampler
```

---

#### 问题 2：WeightedRandomSampler + DynamicBatchSampler + DDP

**问题描述**：
```python
# 三者组合使用时的复杂性：
# 1. WeightedRandomSampler 用于类别平衡
# 2. DynamicBatchSampler 用于点数控制
# 3. DDP 需要分布式采样

# 当前代码：
base_sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
batch_sampler = DynamicBatchSampler(dataset, sampler=base_sampler)

# ⚠️ 问题：base_sampler 没有感知 DDP，所有 GPU 可能产生相同的样本
```

**解决方案**：

```python
# 方案 1：自定义 DistributedWeightedSampler
class DistributedWeightedSampler:
    """
    结合 DistributedSampler 和 WeightedRandomSampler 的采样器
    
    确保：
    1. 每个 GPU 看到不同的样本（Distributed）
    2. 样本按权重采样（Weighted）
    """
    def __init__(self, dataset, weights, num_samples, replacement=True,
                 num_replicas=None, rank=None, seed=0):
        # DistributedSampler 参数
        if num_replicas is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas = 1
                rank = 0
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        
        # WeightedRandomSampler 参数
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        
        # 计算每个 GPU 的样本数
        self.num_samples_per_replica = self.num_samples // self.num_replicas
        self.total_size = self.num_samples_per_replica * self.num_replicas
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        # 使用确定性随机数（基于 epoch 和 rank）
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 生成加权样本索引
        indices = torch.multinomial(
            self.weights, 
            self.total_size, 
            replacement=self.replacement,
            generator=g
        ).tolist()
        
        # 分配给当前 GPU
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples_per_replica


# 在 DataModule 中使用
def _create_dataloader(self, dataset, shuffle=True, drop_last=False, use_sampler_weights=False):
    if self.use_dynamic_batch:
        import torch.distributed as dist
        num_replicas = None
        rank = None
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        
        # 创建分布式加权采样器
        base_sampler = None
        if use_sampler_weights and self.train_sampler_weights is not None:
            base_sampler = DistributedWeightedSampler(
                dataset=dataset,
                weights=self.train_sampler_weights,
                num_samples=len(dataset),
                replacement=True,
                num_replicas=num_replicas,
                rank=rank,
                seed=42
            )
        
        # 创建 DynamicBatchSampler
        batch_sampler = DynamicBatchSampler(
            dataset=dataset,
            max_points=self.max_points,
            shuffle=(shuffle and base_sampler is None),
            drop_last=drop_last,
            sampler=base_sampler,
            num_replicas=num_replicas,
            rank=rank,
        )
        
        return DataLoader(...)
```

---

### 6. 测试与验证

#### 测试脚本：验证 DDP + DynamicBatchSampler

```python
# test/test_ddp_dynamic_batch.py
"""
测试 DynamicBatchSampler 在 DDP 环境下的正确性

验证：
1. 不同 GPU 看到不同的样本
2. 所有样本被覆盖（没有遗漏）
3. Batch 大小正确限制
4. 指标正确聚合
"""

import torch
import pytorch_lightning as pl
from pointsuite.data.datamodule_binpkl import BinPklDataModule
from pointsuite.tasks.semantic_segmentation import SemanticSegmentationTask

def test_ddp_different_samples():
    """测试不同 GPU 是否看到不同样本"""
    print("\n[测试] DDP - 不同 GPU 样本分布")
    
    # 创建 DataModule
    datamodule = BinPklDataModule(
        data_root='path/to/data',
        use_dynamic_batch=True,
        max_points=500000,
        batch_size=4,
    )
    datamodule.setup('fit')
    
    # 模拟 DDP（2个GPU）
    import torch.distributed as dist
    
    # GPU 0
    seen_samples_gpu0 = set()
    for batch in datamodule.train_dataloader():
        # 记录看到的样本索引（需要在 batch 中添加索引追踪）
        seen_samples_gpu0.update(batch['sample_idx'].tolist())
    
    # GPU 1
    seen_samples_gpu1 = set()
    for batch in datamodule.train_dataloader():
        seen_samples_gpu1.update(batch['sample_idx'].tolist())
    
    # 验证
    overlap = seen_samples_gpu0 & seen_samples_gpu1
    all_samples = seen_samples_gpu0 | seen_samples_gpu1
    
    print(f"GPU 0 样本数: {len(seen_samples_gpu0)}")
    print(f"GPU 1 样本数: {len(seen_samples_gpu1)}")
    print(f"重叠样本数: {len(overlap)}")
    print(f"总样本数: {len(all_samples)}")
    print(f"数据集大小: {len(datamodule.train_dataset)}")
    
    # 断言
    assert len(overlap) == 0, "不同 GPU 不应该看到相同的样本"
    assert len(all_samples) == len(datamodule.train_dataset), "所有样本都应该被覆盖"


def test_ddp_batch_sizes():
    """测试 DDP 下的 batch 大小控制"""
    print("\n[测试] DDP - Batch 大小控制")
    
    datamodule = BinPklDataModule(
        data_root='path/to/data',
        use_dynamic_batch=True,
        max_points=500000,
    )
    datamodule.setup('fit')
    
    # 记录每个 batch 的点数
    batch_points = []
    for batch in datamodule.train_dataloader():
        total_points = batch['coord'].shape[0]
        batch_points.append(total_points)
        
        # 验证不超过限制
        assert total_points <= 500000, f"Batch 点数 {total_points} 超过限制 500000"
    
    print(f"Batch 数量: {len(batch_points)}")
    print(f"平均点数: {sum(batch_points) / len(batch_points):.0f}")
    print(f"最小点数: {min(batch_points)}")
    print(f"最大点数: {max(batch_points)}")


def test_ddp_metrics_aggregation():
    """测试 DDP 下的指标聚合"""
    print("\n[测试] DDP - 指标聚合")
    
    # 创建简单的训练循环
    datamodule = BinPklDataModule(
        data_root='path/to/data',
        use_dynamic_batch=True,
        max_points=500000,
    )
    
    model = SemanticSegmentationTask(
        num_classes=10,
        learning_rate=0.001,
    )
    
    trainer = pl.Trainer(
        strategy='ddp',
        devices=2,
        accelerator='gpu',
        max_epochs=1,
        limit_train_batches=10,  # 只测试 10 个 batches
        limit_val_batches=5,
    )
    
    # 训练
    trainer.fit(model, datamodule)
    
    # 获取指标
    metrics = trainer.callback_metrics
    
    print("训练指标:")
    for key, value in metrics.items():
        if 'train' in key:
            print(f"  {key}: {value:.4f}")
    
    print("\n验证指标:")
    for key, value in metrics.items():
        if 'val' in key:
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    # 运行测试
    # test_ddp_different_samples()
    # test_ddp_batch_sizes()
    test_ddp_metrics_aggregation()
```

**运行测试**：
```bash
# 单 GPU 测试
python test/test_ddp_dynamic_batch.py

# DDP 测试（2个GPU）
python -m torch.distributed.launch --nproc_per_node=2 test/test_ddp_dynamic_batch.py

# 或使用 Lightning CLI
python main.py fit --trainer.strategy=ddp --trainer.devices=2 --config config.yaml
```

---

### 7. 最终推荐

#### ✅ 推荐方案：修改 DynamicBatchSampler 支持 DDP

**步骤**：
1. 修改 `DynamicBatchSampler` 添加 DDP 参数（见上文方案 A）
2. 修改 `DataModuleBase._create_dataloader` 传递 DDP 参数
3. （可选）实现 `DistributedWeightedSampler` 支持加权采样 + DDP
4. 在 Task 中添加 `on_train_epoch_start` 设置 epoch
5. 测试验证

**优势**：
- ✅ 完全控制采样逻辑
- ✅ 支持动态 batch + 加权采样 + DDP
- ✅ 每个 GPU 独立决定 batch 大小
- ✅ 指标正确聚合

**代码修改量**：
- `collate.py`: ~50 行（修改 DynamicBatchSampler）
- `datamodule_base.py`: ~20 行（传递 DDP 参数）
- `base_task.py`: ~5 行（设置 epoch）
- （可选）`collate.py`: ~80 行（添加 DistributedWeightedSampler）

---

#### ⚠️ 备选方案：使用 LimitedPointsCollateFn

**适用场景**：
- 快速原型验证
- 不需要加权采样
- 可以接受样本分布不均

**配置**：
```python
# config.yaml
data:
  batch_size: 4
  use_dynamic_batch: false
  num_workers: 4

# 在 DataLoader 中使用
from pointsuite.data.datasets.collate import LimitedPointsCollateFn

limited_collate = LimitedPointsCollateFn(max_points=500000, strategy='drop_largest')

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=limited_collate,
    # Lightning 会自动处理 DistributedSampler
)
```

**优势**：
- ✅ 无需修改现有代码
- ✅ Lightning 自动处理 DistributedSampler

**劣势**：
- ⚠️ 可能导致 GPU 间样本分布不均
- ⚠️ 无法与 WeightedRandomSampler 完美配合

---

## 📊 对比总结

| 特性 | DynamicBatchSampler<br/>(当前) | DynamicBatchSampler<br/>(修改后) | LimitedPointsCollateFn | 固定 batch_size |
|------|------------------------------|-------------------------------|----------------------|----------------|
| **DDP 兼容性** | ⚠️ 需要修改 | ✅ 完全兼容 | ✅ 兼容（有限制） | ✅ 完全兼容 |
| **不同 GPU batch 数不同** | ⚠️ 可能相同 | ✅ 支持 | ✅ 相同（但样本数不同） | ❌ 相同 |
| **点数控制** | ✅ 精确 | ✅ 精确 | ✅ 精确 | ❌ 无法控制 |
| **样本覆盖** | ✅ 完整 | ✅ 完整 | ⚠️ 部分丢弃 | ✅ 完整 |
| **加权采样支持** | ⚠️ 需要特殊处理 | ✅ 完美支持 | ⚠️ 难以结合 | ✅ 支持 |
| **实现复杂度** | - | ⭐⭐⭐ | ⭐ | ⭐ |
| **推荐程度** | ❌ 不推荐 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 🎯 总结

### 核心结论

1. **动态 batch 方法完全兼容 DDP**
   - ✅ `DynamicBatchSampler` 是最佳选择（需要修改以支持 DDP）
   - ✅ `LimitedPointsCollateFn` 可以使用（但不如前者优雅）

2. **不同显卡可以有不同的 batch 数**
   - ✅ PyTorch Lightning 会正确处理
   - ✅ 指标通过 torchmetrics 正确聚合
   - ✅ 不会死锁

3. **推荐实现**
   ```python
   # 修改 DynamicBatchSampler 支持 DDP（约 70 行代码）
   # 修改 DataModuleBase 传递 DDP 参数（约 20 行代码）
   # 添加 epoch 设置（约 5 行代码）
   
   # 总工作量：~95 行代码修改 + 测试
   # 收益：完美的 DDP + 动态 batch + 加权采样支持
   ```

### 下一步行动

**如果需要立即使用**：
- 使用 `LimitedPointsCollateFn`（已经 DDP 兼容）
- 或使用固定 `batch_size`（最稳定）

**如果需要最优方案**：
- 实现 `DynamicBatchSampler` 的 DDP 支持
- 实现 `DistributedWeightedSampler`（如果需要加权采样）
- 编写测试验证正确性

---

## 📝 快速参考

```python
# ✅ 推荐配置（需要修改代码）
datamodule = BinPklDataModule(
    data_root='path/to/data',
    use_dynamic_batch=True,  # 使用 DynamicBatchSampler
    max_points=500000,
    train_sampler_weights=weights,  # 可选：加权采样
)

trainer = pl.Trainer(
    strategy='ddp',
    devices=2,
    accelerator='gpu',
)

# ✅ 当前可用配置（无需修改）
datamodule = BinPklDataModule(
    data_root='path/to/data',
    use_dynamic_batch=False,  # 使用固定 batch_size
    batch_size=4,
)

trainer = pl.Trainer(
    strategy='ddp',
    devices=2,
)
```

---

**最终答案**：
1. **动态 batch 方法能兼容 DDP 吗？** → ✅ 是的，但需要修改 `DynamicBatchSampler` 以支持 DDP
2. **能支持不同显卡 batch 数不同吗？** → ✅ 可以，PyTorch Lightning 会正确处理
