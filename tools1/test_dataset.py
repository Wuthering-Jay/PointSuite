import numpy as np
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch


class BinPklPointCloudDataset(Dataset):
    """
    高效的点云数据集，基于bin+pkl格式。
    支持跨文件随机访问和shuffle。
    """
    
    def __init__(self, 
                 data_dir: str,
                 fields_to_load: List[str] = ['X', 'Y', 'Z', 'classification'],
                 transform=None,
                 cache_metadata: bool = True):
        """
        初始化数据集。
        
        Args:
            data_dir: 包含bin和pkl文件的目录
            fields_to_load: 要加载的字段列表
            transform: 数据变换函数
            cache_metadata: 是否缓存元数据到内存
        """
        self.data_dir = Path(data_dir)
        self.fields_to_load = fields_to_load
        self.transform = transform
        self.cache_metadata = cache_metadata
        
        # 查找所有bin文件
        self.bin_files = sorted(list(self.data_dir.glob('*.bin')))
        if not self.bin_files:
            raise ValueError(f"No bin files found in {data_dir}")
        
        print(f"Found {len(self.bin_files)} bin files")
        
        # 加载所有元数据
        self._load_all_metadata()
        
        # 创建全局分块索引
        self._create_global_index()
        
        print(f"Total segments: {len(self.segment_index)}")
        print(f"Total points: {self.total_points:,}")
    
    def _load_all_metadata(self):
        """加载所有pkl元数据文件。"""
        self.metadata_list = []
        self.total_points = 0
        
        print("Loading metadata...")
        start_time = time.time()
        
        for bin_file in self.bin_files:
            pkl_file = bin_file.with_suffix('.pkl')
            if not pkl_file.exists():
                print(f"Warning: {pkl_file.name} not found, skipping {bin_file.name}")
                continue
            
            with open(pkl_file, 'rb') as f:
                metadata = pickle.load(f)
            
            if self.cache_metadata:
                self.metadata_list.append(metadata)
            else:
                # 只保存文件路径
                self.metadata_list.append(pkl_file)
            
            self.total_points += metadata['num_points']
        
        elapsed = time.time() - start_time
        print(f"Metadata loaded in {elapsed:.3f}s")
    
    def _create_global_index(self):
        """创建全局分块索引：[(file_idx, segment_id), ...]"""
        self.segment_index = []
        
        for file_idx, metadata in enumerate(self.metadata_list):
            if isinstance(metadata, Path):
                # 如果没有缓存，临时加载
                with open(metadata, 'rb') as f:
                    meta = pickle.load(f)
                num_segments = meta['num_segments']
            else:
                num_segments = metadata['num_segments']
            
            for segment_id in range(num_segments):
                self.segment_index.append((file_idx, segment_id))
    
    def __len__(self):
        """返回数据集中的分块总数。"""
        return len(self.segment_index)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        获取第idx个分块的数据。
        
        Args:
            idx: 分块索引
            
        Returns:
            包含点云数据的字典
        """
        file_idx, segment_id = self.segment_index[idx]
        
        # 获取元数据
        if isinstance(self.metadata_list[file_idx], Path):
            with open(self.metadata_list[file_idx], 'rb') as f:
                metadata = pickle.load(f)
        else:
            metadata = self.metadata_list[file_idx]
        
        # 获取bin文件路径
        bin_file = self.bin_files[file_idx]
        
        # 获取分块信息
        segment_info = metadata['segments'][segment_id]
        indices = segment_info['indices']
        
        # 使用memmap加载数据
        dtype = np.dtype(metadata['dtype'])
        mmap_data = np.memmap(bin_file, dtype=dtype, mode='r')
        
        # 读取指定分块的数据
        segment_data = mmap_data[indices].copy()  # copy to memory
        
        # 提取需要的字段 - 使用 np.ascontiguousarray 确保内存连续
        result = {}
        for field in self.fields_to_load:
            if field in segment_data.dtype.names:
                result[field] = np.ascontiguousarray(segment_data[field])
            else:
                print(f"Warning: Field '{field}' not found in data")
        
        # 添加元信息
        result['segment_id'] = idx
        result['file_name'] = bin_file.stem
        result['num_points'] = len(segment_data)
        
        # 应用变换
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def get_segment_info(self, idx: int) -> Dict:
        """获取分块的元信息（不加载实际数据）。"""
        file_idx, segment_id = self.segment_index[idx]
        
        if isinstance(self.metadata_list[file_idx], Path):
            with open(self.metadata_list[file_idx], 'rb') as f:
                metadata = pickle.load(f)
        else:
            metadata = self.metadata_list[file_idx]
        
        return metadata['segments'][segment_id]


def custom_collate_fn(batch):
    """
    自定义collate函数，处理不同大小的点云。
    将batch中的数据组织成列表而不是堆叠成tensor。
    """
    if len(batch) == 1:
        return batch[0]
    
    # 组织成字典的列表
    result = {}
    keys = batch[0].keys()
    
    for key in keys:
        result[key] = [item[key] for item in batch]
    
    return result


def benchmark_dataset(data_dir: str, 
                     num_samples: int = 100,
                     batch_size: int = 1,
                     num_workers: int = 0,
                     shuffle: bool = True):
    """
    测试数据集的加载性能。
    
    Args:
        data_dir: 数据目录
        num_samples: 测试样本数量
        batch_size: 批次大小
        num_workers: DataLoader的worker数量
        shuffle: 是否shuffle
    """
    print("\n" + "="*70)
    print("Dataset Benchmark")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Shuffle: {shuffle}")
    print("-"*70)
    
    # 创建数据集
    dataset = BinPklPointCloudDataset(
        data_dir=data_dir,
        fields_to_load=['X', 'Y', 'Z', 'classification'],
        cache_metadata=True
    )
    
    print(f"\nDataset size: {len(dataset)} segments")
    
    # 测试1: 单个样本随机访问
    print("\n" + "="*70)
    print("Test 1: Random Access Performance")
    print("="*70)
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    start_time = time.time()
    for idx in indices:
        data = dataset[idx]
    elapsed = time.time() - start_time
    
    print(f"Loaded {len(indices)} random samples")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Average: {elapsed/len(indices)*1000:.2f}ms per sample")
    print(f"Throughput: {len(indices)/elapsed:.2f} samples/s")
    
    # 测试2: 顺序遍历整个数据集
    print("\n" + "="*70)
    print("Test 2: Sequential Full Dataset Loading")
    print("="*70)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate_fn if batch_size > 1 else None
    )
    
    start_time = time.time()
    total_points = 0
    for i, batch in enumerate(dataloader):
        if batch_size == 1:
            total_points += int(batch['num_points'])
        else:
            total_points += sum(int(x) for x in batch['num_points'])
    
    elapsed = time.time() - start_time
    
    print(f"Loaded {len(dataset)} segments ({total_points:,} points)")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Average: {elapsed/len(dataset)*1000:.2f}ms per segment")
    print(f"Throughput: {len(dataset)/elapsed:.2f} segments/s")
    print(f"Point throughput: {total_points/elapsed:,.0f} points/s")
    
    # 测试3: Shuffle加载整个数据集
    if shuffle:
        print("\n" + "="*70)
        print("Test 3: Shuffled Full Dataset Loading")
        print("="*70)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=custom_collate_fn if batch_size > 1 else None
        )
        
        start_time = time.time()
        total_points = 0
        for i, batch in enumerate(dataloader):
            if batch_size == 1:
                total_points += int(batch['num_points'])
            else:
                total_points += sum(int(x) for x in batch['num_points'])
        
        elapsed = time.time() - start_time
        
        print(f"Loaded {len(dataset)} segments ({total_points:,} points) with shuffle")
        print(f"Total time: {elapsed:.3f}s")
        print(f"Average: {elapsed/len(dataset)*1000:.2f}ms per segment")
        print(f"Throughput: {len(dataset)/elapsed:.2f} segments/s")
        print(f"Point throughput: {total_points/elapsed:,.0f} points/s")
    
    # 测试4: 统计信息
    print("\n" + "="*70)
    print("Test 4: Dataset Statistics")
    print("="*70)
    
    # 采样一些数据查看统计信息
    sample_indices = np.random.choice(len(dataset), size=min(10, len(dataset)), replace=False)
    
    point_counts = []
    label_distributions = []
    
    for idx in sample_indices:
        info = dataset.get_segment_info(idx)
        point_counts.append(info['num_points'])
        if 'label_counts' in info:
            label_distributions.append(info['label_counts'])
    
    print(f"Sample size: {len(sample_indices)} segments")
    print(f"Points per segment:")
    print(f"  Min: {min(point_counts):,}")
    print(f"  Max: {max(point_counts):,}")
    print(f"  Mean: {np.mean(point_counts):,.0f}")
    print(f"  Std: {np.std(point_counts):,.0f}")
    
    if label_distributions:
        print(f"\nLabel distribution in sampled segments:")
        all_labels = set()
        for ld in label_distributions:
            all_labels.update(ld.keys())
        
        for label in sorted(all_labels):
            counts = [ld.get(label, 0) for ld in label_distributions]
            total = sum(counts)
            print(f"  Class {label}: {total:,} points ({total/sum(point_counts)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("Benchmark completed!")
    print("="*70)


if __name__ == "__main__":
    # 配置
    data_dir = r"E:\data\Dales\dales_las\tile_bin\test"
    
    # 运行基准测试
    benchmark_dataset(
        data_dir=data_dir,
        num_samples=100,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )
    
    print("\n" + "="*70)
    print("Testing with different configurations...")
    print("="*70)
    
    # 测试不同的batch size
    for batch_size in [1, 4, 8]:
        print(f"\n--- Batch size: {batch_size} ---")
        dataset = BinPklPointCloudDataset(
            data_dir=data_dir,
            fields_to_load=['X', 'Y', 'Z', 'classification'],
            cache_metadata=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn if batch_size > 1 else None
        )
        
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            pass
        elapsed = time.time() - start_time
        
        print(f"Time: {elapsed:.3f}s, Throughput: {len(dataset)/elapsed:.2f} segments/s")
