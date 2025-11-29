"""
自定义的 DataLoader Collate 函数

包含:
1. 基础的 collate_fn：拼接点云数据
2. 带点数限制的 collate_fn：动态调整 batch 大小
"""
import numpy as np
import torch
from collections.abc import Mapping, Sequence


def collate_fn(batch):
    """
    基础 collate function for point cloud which support dict and list
    
    该函数将多个不同点数的样本合并成一个batch：
    - 点云数据（coord, feat 等）会被拼接成一个大的点云
    - 分类标签（class）保持为列表（每个样本一个 tensor）
    - 自动添加 'offset' 字段，格式为 [n1, n1+n2, ...]（不包含起始0），长度为 batch_size
    - test模式下会拼接 indices 用于投票机制
    
    统一的输出格式：
    - coord: [total_points, 3] - 拼接所有点
    - feat: [total_points, C] - 拼接所有特征
    - class: [total_points] - 拼接所有点的标签（点级标签）
    - offset: [batch_size] - 累积偏移（不包含起始0），格式为 [n1, n1+n2, ...]
    
    注意：如果需要样本级标签（如整个场景的分类），请使用不同的键名（如 'scene_label'）
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{type(batch)} is not supported.")

    # 处理 dict 类型（我们的数据集返回的是 dict）
    if isinstance(batch[0], Mapping):
        # 获取所有keys
        keys = batch[0].keys()
        
        # 合并后的结果
        result = {}
        
        # 计算每个样本的点数（用于offset）
        num_points_per_sample = []
        
        # 需要拼接的字段（点级数据）
        concat_keys = ['coord', 'feat', 'feature', 'indices', 'normal', 'class', 'label', 'classification', 
                       'echo', 'intensity', 'color', 'h_norm']
        
        # 需要保持为列表的字段（样本级数据）
        # 注意：如果有真正的样本级标签（如整个场景的分类），应该使用不同的键名（如 'scene_label'）
        list_keys = []
        
        for key in keys:
            # 收集所有样本的该字段
            values = [torch.from_numpy(d[key]) if isinstance(d[key], np.ndarray) else d[key] for d in batch]
            
            # 跳过 offset（如果样本中已有，会被覆盖）
            if key == 'offset':
                continue
            
            # 拼接点级数据
            if key in concat_keys:
                result[key] = torch.cat(values, dim=0)
                
                # 记录点数（从 coord 优先，否则从任何拼接字段获取）
                if len(num_points_per_sample) == 0:
                    num_points_per_sample = [v.shape[0] for v in values]
            
            # 保持样本级数据为列表
            elif key in list_keys:
                result[key] = values  # 保持为列表
            
            # 其他字段尝试 stack，失败则保持为列表
            else:
                try:
                    result[key] = torch.stack(values, dim=0)
                except:
                    result[key] = values
        
        # 添加 offset 字段（格式：[n1, n1+n2, ...]，长度为 batch_size）
        if len(num_points_per_sample) > 0:
            # 生成累积和，但不包含起始 0：offset[i] 表示前 i+1 个样本的累计点数
            offset = torch.cumsum(torch.tensor(num_points_per_sample), dim=0).int()
            result['offset'] = offset  # 不包含起始 0，长度为 batch_size
        
        return result
    
    # 处理其他类型（保持原有逻辑）
    elif isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            # data[0] 是点级字段（如 coord），append 每个样本的点数
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        # 此处 batch[-1] 包含每个样本的点数，直接取累积和（不添加起始0）
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)


class DynamicBatchSampler:
    """
    动态 Batch Sampler，根据点数动态调整 batch 大小
    
    这是一个更优雅的解决方案，在采样阶段就控制 batch 大小，
    而不是在 collate 阶段丢弃样本。
    
    特性：
    - 确保覆盖所有样本（每个 epoch）
    - 支持与 WeightedRandomSampler 等其他 Sampler 结合使用
    - 动态调整 batch 大小以满足点数限制
    
    使用方法:
        # 基础用法
        sampler = DynamicBatchSampler(dataset, max_points=500000, shuffle=True)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
        
        # 与 WeightedRandomSampler 结合
        from torch.utils.data import WeightedRandomSampler
        base_sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
        sampler = DynamicBatchSampler(dataset, max_points=500000, sampler=base_sampler)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    """
    
    def __init__(self, dataset, max_points=500000, shuffle=True, drop_last=False, sampler=None):
        """
        Args:
            dataset: 数据集对象，需要能够获取每个样本的点数
            max_points: 每个 batch 的最大点数
            shuffle: 是否打乱顺序（当 sampler=None 时生效）
            drop_last: 是否丢弃最后一个不完整的 batch
            sampler: 可选的基础 Sampler（如 WeightedRandomSampler）
                    如果提供，则使用该 sampler 生成索引序列，shuffle 参数将被忽略
        """
        self.dataset = dataset
        self.max_points = max_points
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler = sampler
        
        # 预先计算每个样本的点数
        self.num_points_list = self._get_num_points_list()
        
    def _get_num_points_list(self):
        """获取每个样本的点数（考虑 loop 参数和采样模式）"""
        
        # 🔥 优先使用 get_sample_num_points() 方法（支持体素模式的正确点数）
        if hasattr(self.dataset, 'get_sample_num_points'):
            num_points_list = self.dataset.get_sample_num_points()
            # 如果 dataset 有 loop 参数，需要扩展列表
            if hasattr(self.dataset, 'loop') and self.dataset.loop > 1:
                num_points_list = num_points_list * self.dataset.loop
            return num_points_list
        
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
    
    def __iter__(self):
        # 生成索引列表
        if self.sampler is not None:
            # 使用提供的 sampler（如 WeightedRandomSampler）
            indices = list(self.sampler)
        elif self.shuffle:
            # 随机打乱
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            # 顺序遍历
            indices = list(range(len(self.dataset)))
        
        # 动态生成 batch
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
        
        # 处理最后一个 batch
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        # 估算 batch 数量（不完全准确，但足够用）
        total_points = sum(self.num_points_list)
        estimated_batches = (total_points + self.max_points - 1) // self.max_points
        return max(1, estimated_batches)

