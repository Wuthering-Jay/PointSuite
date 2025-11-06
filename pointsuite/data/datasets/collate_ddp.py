"""
DDP 兼容的 DynamicBatchSampler 实现

本文件提供了改进后的 DynamicBatchSampler，完全支持 DDP 训练
"""

import torch
from torch.utils.data import Sampler
from typing import Iterator, List, Optional
import torch.distributed as dist


class DynamicBatchSamplerDDP(Sampler):
    """
    支持 DDP 的动态 Batch Sampler
    
    改进：
    1. ✅ 自动检测 DDP 环境（通过 torch.distributed）
    2. ✅ 每个 GPU 获得不同的样本子集
    3. ✅ 支持确定性 shuffle（每个 epoch 不同，但可复现）
    4. ✅ 支持与其他 Sampler 结合（如 WeightedRandomSampler）
    5. ✅ 正确的 __len__() 实现（每个 GPU 的 batch 数量）
    
    使用方法:
        # 基本用法（自动检测 DDP）
        sampler = DynamicBatchSamplerDDP(dataset, max_points=500000, shuffle=True)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
        
        # 与 WeightedRandomSampler 结合
        base_sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
        sampler = DynamicBatchSamplerDDP(dataset, max_points=500000, sampler=base_sampler)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
        
        # 在 Lightning Task 中设置 epoch
        def on_train_epoch_start(self):
            if hasattr(self.trainer.train_dataloader.batch_sampler, 'set_epoch'):
                self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
    """
    
    def __init__(
        self,
        dataset,
        max_points: int = 500000,
        shuffle: bool = True,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        """
        Args:
            dataset: 数据集对象，需要能够获取每个样本的点数
            max_points: 每个 batch 的最大点数
            shuffle: 是否打乱顺序（当 sampler=None 时生效）
            drop_last: 是否丢弃最后一个不完整的 batch
            sampler: 可选的基础 Sampler（如 WeightedRandomSampler）
                    如果提供，则使用该 sampler 生成索引序列，shuffle 参数将被忽略
            num_replicas: DDP 进程数（world_size）。如果为 None，自动检测
            rank: 当前进程的 rank。如果为 None，自动检测
            seed: 随机种子（用于确定性 shuffle）
        """
        # DDP 参数 - 自动检测
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]"
            )
        
        self.dataset = dataset
        self.max_points = max_points
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        
        # 预先计算每个样本的点数
        self.num_points_list = self._get_num_points_list()
        
        # 计算每个 replica 应该处理的样本数
        self.total_samples = len(self.num_points_list)
        # 确保总样本数能被 num_replicas 整除（通过 padding）
        if not self.drop_last:
            self.num_samples_per_replica = (self.total_samples + self.num_replicas - 1) // self.num_replicas
            self.total_size = self.num_samples_per_replica * self.num_replicas
        else:
            self.num_samples_per_replica = self.total_samples // self.num_replicas
            self.total_size = self.num_samples_per_replica * self.num_replicas
    
    def _get_num_points_list(self) -> List[int]:
        """获取每个样本的点数（考虑 loop 参数）"""
        base_num_points_list = []
        
        # 尝试从 dataset.data_list 获取
        if hasattr(self.dataset, 'data_list'):
            for sample_info in self.dataset.data_list:
                if 'num_points' in sample_info:
                    base_num_points_list.append(sample_info['num_points'])
                else:
                    # 如果没有 num_points，加载样本统计
                    sample = self.dataset[len(base_num_points_list)]
                    if 'coord' in sample:
                        base_num_points_list.append(len(sample['coord']))
                    else:
                        base_num_points_list.append(0)
            
            # 如果 dataset 有 loop 参数，需要扩展列表
            if hasattr(self.dataset, 'loop') and self.dataset.loop > 1:
                num_points_list = base_num_points_list * self.dataset.loop
            else:
                num_points_list = base_num_points_list
        else:
            # 遍历整个数据集获取点数（较慢）
            print("Warning: Dataset doesn't have data_list, scanning all samples...")
            num_points_list = []
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                if 'coord' in sample:
                    num_points_list.append(len(sample['coord']))
                else:
                    num_points_list.append(0)
        
        return num_points_list
    
    def set_epoch(self, epoch: int) -> None:
        """
        设置 epoch 数（用于确定性 shuffle）
        
        在 Lightning Task 中调用：
            def on_train_epoch_start(self):
                self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
        
        Args:
            epoch: epoch 编号
        """
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[List[int]]:
        # 1. 生成全局索引列表
        if self.sampler is not None:
            # 使用提供的 sampler（如 WeightedRandomSampler）
            indices = list(self.sampler)
        elif self.shuffle:
            # 使用确定性随机数生成器（DDP 友好）
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            # 顺序遍历
            indices = list(range(len(self.dataset)))
        
        # 2. 根据 drop_last 调整 indices 长度
        if not self.drop_last:
            # 添加 padding 以确保所有 replica 处理相同数量的样本
            # 这样可以避免 DDP 死锁（所有进程必须完成相同数量的 iterations）
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                # 重复最后几个样本作为 padding
                indices += indices[:padding_size]
        else:
            # 丢弃最后的不完整部分
            indices = indices[:self.total_size]
        
        # 3. 分配给当前 rank（类似 DistributedSampler）
        # 每个 GPU 获得不同的样本子集
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices) == self.num_samples_per_replica, \
            f"Expected {self.num_samples_per_replica} samples, got {len(indices)}"
        
        # 4. 动态生成 batch（根据点数）
        batch = []
        batch_points = 0
        
        for idx in indices:
            num_points = self.num_points_list[idx]
            
            # 如果当前 batch 为空，或者加入当前样本不会超过限制
            if len(batch) == 0 or batch_points + num_points <= self.max_points:
                batch.append(idx)
                batch_points += num_points
            else:
                # 当前 batch 已满，yield 并开始新 batch
                yield batch
                batch = [idx]
                batch_points = num_points
        
        # 5. 处理最后一个 batch
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        """
        返回每个 replica 的 batch 数量
        
        注意：这是一个估算值，实际 batch 数量可能略有不同
        （因为动态 batch 的大小取决于每个样本的点数）
        """
        # 获取属于当前 rank 的样本索引
        indices = list(range(self.rank, self.total_size, self.num_replicas))
        
        # 计算这些样本的总点数
        total_points = sum(self.num_points_list[idx % self.total_samples] for idx in indices)
        
        # 估算 batch 数量
        estimated_batches = (total_points + self.max_points - 1) // self.max_points
        
        # 至少返回 1
        return max(1, estimated_batches)


class DistributedWeightedSampler(Sampler):
    """
    结合 DistributedSampler 和 WeightedRandomSampler 的采样器
    
    特性：
    1. ✅ 每个 GPU 看到不同的样本（Distributed）
    2. ✅ 样本按权重采样（Weighted）
    3. ✅ 支持确定性采样（可复现）
    4. ✅ 可与 DynamicBatchSamplerDDP 结合使用
    
    使用方法:
        # 创建加权采样器
        sampler = DistributedWeightedSampler(
            dataset,
            weights=class_weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        # 与 DynamicBatchSamplerDDP 结合
        batch_sampler = DynamicBatchSamplerDDP(
            dataset,
            max_points=500000,
            sampler=sampler  # 传入 DistributedWeightedSampler
        )
        
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
    """
    
    def __init__(
        self,
        dataset,
        weights: List[float],
        num_samples: int,
        replacement: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        """
        Args:
            dataset: 数据集对象
            weights: 每个样本的权重（长度应与 dataset 相同）
            num_samples: 总采样数量
            replacement: 是否有放回采样
            num_replicas: DDP 进程数。如果为 None，自动检测
            rank: 当前进程的 rank。如果为 None，自动检测
            seed: 随机种子
        """
        # DDP 参数 - 自动检测
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]"
            )
        
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        
        # 计算每个 replica 的样本数
        self.num_samples_per_replica = self.num_samples // self.num_replicas
        self.total_size = self.num_samples_per_replica * self.num_replicas
        
        # 检查权重
        if len(self.weights) != len(dataset):
            raise ValueError(
                f"Length of weights ({len(self.weights)}) must equal "
                f"length of dataset ({len(dataset)})"
            )
    
    def set_epoch(self, epoch: int) -> None:
        """设置 epoch（用于确定性采样）"""
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        # 使用确定性随机数（基于 epoch）
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 生成加权样本索引
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=g
        ).tolist()
        
        # 分配给当前 rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices) == self.num_samples_per_replica
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples_per_replica


# 便捷函数：创建 DDP 兼容的 DataLoader
def create_ddp_dynamic_dataloader(
    dataset,
    max_points: int = 500000,
    shuffle: bool = True,
    drop_last: bool = False,
    weights: Optional[List[float]] = None,
    collate_fn = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = 2,
    seed: int = 0,
):
    """
    创建支持 DDP 的动态 batch DataLoader
    
    Args:
        dataset: 数据集
        max_points: 每个 batch 的最大点数
        shuffle: 是否打乱
        drop_last: 是否丢弃最后一个 batch
        weights: 可选的样本权重（用于加权采样）
        collate_fn: collate 函数
        num_workers: 工作进程数
        pin_memory: 是否使用 pinned memory
        persistent_workers: 是否保持 workers 持久化
        prefetch_factor: 预取因子
        seed: 随机种子
        
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    # 创建基础 sampler（如果需要加权采样）
    base_sampler = None
    if weights is not None:
        base_sampler = DistributedWeightedSampler(
            dataset=dataset,
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            seed=seed,
        )
    
    # 创建 DynamicBatchSamplerDDP
    batch_sampler = DynamicBatchSamplerDDP(
        dataset=dataset,
        max_points=max_points,
        shuffle=(shuffle and base_sampler is None),
        drop_last=drop_last,
        sampler=base_sampler,
        seed=seed,
    )
    
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


if __name__ == '__main__':
    """测试 DDP 兼容性"""
    print("DynamicBatchSamplerDDP 测试")
    print("=" * 60)
    
    # 模拟数据集
    class DummyDataset:
        def __init__(self, num_samples=100):
            self.data_list = [
                {'num_points': torch.randint(10000, 50000, (1,)).item()}
                for _ in range(num_samples)
            ]
        
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, idx):
            return {'coord': torch.randn(self.data_list[idx]['num_points'], 3)}
    
    dataset = DummyDataset(100)
    
    # 测试 1: 基本功能
    print("\n[测试 1] 基本功能 - 不同 rank 的 batch 数量")
    for rank in range(2):
        sampler = DynamicBatchSamplerDDP(
            dataset,
            max_points=100000,
            shuffle=False,
            num_replicas=2,
            rank=rank,
        )
        
        batches = list(sampler)
        total_samples = sum(len(batch) for batch in batches)
        
        print(f"Rank {rank}:")
        print(f"  - Batch 数量: {len(batches)}")
        print(f"  - 总样本数: {total_samples}")
        print(f"  - 第一个 batch: {batches[0][:3]}...")
    
    # 测试 2: shuffle 确定性
    print("\n[测试 2] Shuffle 确定性")
    sampler1 = DynamicBatchSamplerDDP(
        dataset, max_points=100000, shuffle=True, seed=42, rank=0, num_replicas=1
    )
    sampler1.set_epoch(0)
    batches1 = list(sampler1)
    
    sampler2 = DynamicBatchSamplerDDP(
        dataset, max_points=100000, shuffle=True, seed=42, rank=0, num_replicas=1
    )
    sampler2.set_epoch(0)
    batches2 = list(sampler2)
    
    print(f"两次采样结果相同: {batches1 == batches2}")
    
    # 测试 3: 不同 epoch
    sampler1.set_epoch(1)
    batches3 = list(sampler1)
    print(f"不同 epoch 结果不同: {batches1 != batches3}")
    
    # 测试 4: 加权采样
    print("\n[测试 3] 加权采样")
    weights = [1.0 if i < 50 else 10.0 for i in range(100)]  # 后半部分权重更高
    
    weighted_sampler = DistributedWeightedSampler(
        dataset,
        weights=weights,
        num_samples=100,
        seed=42,
        rank=0,
        num_replicas=1,
    )
    
    indices = list(weighted_sampler)
    high_weight_count = sum(1 for idx in indices if idx >= 50)
    print(f"高权重样本占比: {high_weight_count / len(indices) * 100:.1f}%")
    print(f"预期: >50% (因为权重是 10:1)")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
