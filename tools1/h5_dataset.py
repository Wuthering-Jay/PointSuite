"""
高效的H5点云数据集读取类

特性：
- 支持PyTorch DataLoader with multiprocessing
- 智能处理indices排序
- 支持数据增强
- 内存高效
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable, Tuple
import time


class H5PointCloudDataset(Dataset):
    """
    H5格式点云数据集
    
    参数:
        h5_path: H5文件路径
        transform: 数据增强函数 (可选)
        preload: 是否预加载所有数据到内存 (默认True，推荐)
        cache_indices: 如果不预加载，是否缓存indices信息 (默认True)
    """
    
    def __init__(
        self, 
        h5_path: str,
        transform: Optional[Callable] = None,
        preload: bool = True,
        cache_indices: bool = True
    ):
        self.h5_path = Path(h5_path)
        self.transform = transform
        self.preload = preload
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5文件不存在: {h5_path}")
        
        # 读取基本信息
        with h5py.File(self.h5_path, 'r') as f:
            self.num_segments = f['segments'].attrs['num_segments']
            
            if preload:
                # 预加载所有数据到内存（推荐，速度最快）
                print(f"预加载数据到内存...")
                start = time.time()
                self.data = {
                    'x': f['data']['x'][:],
                    'y': f['data']['y'][:],
                    'z': f['data']['z'][:],
                    'labels': f['data']['classification'][:]
                }
                
                # 读取所有indices
                self.segments_info = []
                for i in range(self.num_segments):
                    indices = f['segments'][f'segment_{i:04d}']['indices'][:]
                    self.segments_info.append(indices)
                
                elapsed = time.time() - start
                memory_mb = sum(arr.nbytes for arr in self.data.values()) / (1024**2)
                print(f"✅ 预加载完成: {self.num_segments} segments, "
                      f"{memory_mb:.1f}MB, 耗时{elapsed:.2f}秒")
                self.indices_cache = None
            elif cache_indices:
                # 缓存indices排序信息（内存占用小，速度较快）
                print(f"缓存indices信息...")
                self.data = None
                self.segments_info = None
                self.indices_cache = {}
                for i in range(self.num_segments):
                    indices = f['segments'][f'segment_{i:04d}']['indices'][:]
                    need_sort = not np.all(indices[:-1] <= indices[1:])
                    self.indices_cache[i] = {
                        'indices': indices,
                        'need_sort': need_sort
                    }
                    if need_sort:
                        # 预计算排序
                        sort_order = np.argsort(indices)
                        self.indices_cache[i]['sort_order'] = sort_order
                        self.indices_cache[i]['unsort_order'] = np.argsort(sort_order)
                        self.indices_cache[i]['sorted_indices'] = indices[sort_order]
                print(f"✅ 缓存完成: {self.num_segments} segments")
            else:
                self.data = None
                self.segments_info = None
                self.indices_cache = None
    
    def __len__(self) -> int:
        return self.num_segments
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        读取单个segment
        
        返回:
            xyz: [N, 3] 点云坐标
            labels: [N] 点标签
        """
        if self.preload:
            # 从内存读取（最快）
            indices = self.segments_info[idx]
            xyz = np.stack([
                self.data['x'][indices],
                self.data['y'][indices],
                self.data['z'][indices]
            ], axis=1)
            labels = self.data['labels'][indices]
        else:
            # 从文件读取
            with h5py.File(self.h5_path, 'r') as f:
                if self.indices_cache is not None:
                    # 使用缓存（快速）
                    cache = self.indices_cache[idx]
                    
                    if cache['need_sort']:
                        # 使用预计算的排序
                        sorted_indices = cache['sorted_indices']
                        xyz = np.stack([
                            f['data']['x'][sorted_indices],
                            f['data']['y'][sorted_indices],
                            f['data']['z'][sorted_indices]
                        ], axis=1)
                        labels = f['data']['classification'][sorted_indices]
                        
                        # 恢复原始顺序
                        unsort_order = cache['unsort_order']
                        xyz = xyz[unsort_order]
                        labels = labels[unsort_order]
                    else:
                        # 直接读取
                        indices = cache['indices']
                        xyz = np.stack([
                            f['data']['x'][indices],
                            f['data']['y'][indices],
                            f['data']['z'][indices]
                        ], axis=1)
                        labels = f['data']['classification'][indices]
                else:
                    # 不使用缓存（实时计算）
                    indices = f['segments'][f'segment_{idx:04d}']['indices'][:]
                    
                    if not np.all(indices[:-1] <= indices[1:]):
                        sort_order = np.argsort(indices)
                        sorted_indices = indices[sort_order]
                        xyz = np.stack([
                            f['data']['x'][sorted_indices],
                            f['data']['y'][sorted_indices],
                            f['data']['z'][sorted_indices]
                        ], axis=1)
                        labels = f['data']['classification'][sorted_indices]
                        unsort_order = np.argsort(sort_order)
                        xyz = xyz[unsort_order]
                        labels = labels[unsort_order]
                    else:
                        xyz = np.stack([
                            f['data']['x'][indices],
                            f['data']['y'][indices],
                            f['data']['z'][indices]
                        ], axis=1)
                        labels = f['data']['classification'][indices]
        
        # 数据增强
        if self.transform is not None:
            xyz, labels = self.transform(xyz, labels)
        
        return torch.from_numpy(xyz).float(), torch.from_numpy(labels).long()


# ========== 数据增强示例 ==========

def collate_fn(batch):
    """
    自定义collate函数，处理不同点数的segments
    
    返回:
        xyz_list: List[Tensor] - 每个segment的xyz
        labels_list: List[Tensor] - 每个segment的labels
        或者返回batch list直接给模型
    """
    xyz_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    return xyz_list, labels_list


def random_rotate_z(xyz: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """绕Z轴随机旋转"""
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    xyz = xyz @ rotation_matrix.T
    return xyz, labels


def random_scale(xyz: np.ndarray, labels: np.ndarray, scale_range=(0.8, 1.2)) -> Tuple[np.ndarray, np.ndarray]:
    """随机缩放"""
    scale = np.random.uniform(*scale_range)
    xyz *= scale
    return xyz, labels


def random_jitter(xyz: np.ndarray, labels: np.ndarray, sigma=0.01) -> Tuple[np.ndarray, np.ndarray]:
    """添加随机噪声"""
    noise = np.random.normal(0, sigma, xyz.shape)
    xyz += noise
    return xyz, labels


def compose_transforms(*transforms):
    """组合多个数据增强"""
    def combined(xyz, labels):
        for transform in transforms:
            xyz, labels = transform(xyz, labels)
        return xyz, labels
    return combined


# ========== 使用示例 ==========

def demo_usage():
    """演示如何使用"""
    
    print("="*70)
    print("H5PointCloudDataset 使用示例")
    print("="*70)
    
    # 1. 创建数据集（带缓存）
    print("\n1. 创建数据集...")
    dataset = H5PointCloudDataset(
        h5_path=r"E:\data\云南遥感中心\第一批\h5\train\processed_02.h5",
        transform=compose_transforms(
            random_rotate_z,
            random_scale,
            random_jitter
        ),
        preload=True  # 预加载到内存，速度最快
    )
    
    print(f"数据集大小: {len(dataset)} segments")
    
    # 2. 单个读取测试
    print("\n2. 单个读取测试...")
    start = time.time()
    xyz, labels = dataset[0]
    print(f"Segment 0: {xyz.shape}, 标签: {labels.shape}")
    print(f"读取耗时: {(time.time() - start)*1000:.2f}ms")
    
    # 3. 使用DataLoader（多进程并行）
    print("\n3. 使用DataLoader并行读取...")
    dataloader = DataLoader(
        dataset,
        batch_size=8,        # 每批8个segments
        shuffle=True,        # 随机打乱
        num_workers=4,       # 4个进程并行
        pin_memory=True,     # 加速GPU传输
        persistent_workers=True,  # 保持worker进程
        collate_fn=collate_fn  # 自定义collate
    )
    
    # 测试速度
    print("测试前10个batch的速度...")
    start = time.time()
    for i, (batch_xyz, batch_labels) in enumerate(dataloader):
        if i >= 10:
            break
        print(f"  Batch {i}: xyz={batch_xyz.shape}, labels={batch_labels.shape}")
    
    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.2f}秒")
    print(f"速度: {10 * 8 / elapsed:.2f} segments/秒")
    
    # 4. 训练循环示例
    print("\n4. 训练循环示例:")
    print("""
    # 完整训练代码
    for epoch in range(100):
        for batch_xyz, batch_labels in dataloader:
            # batch_xyz: [B, N, 3]
            # batch_labels: [B, N]
            
            # 前向传播
            output = model(batch_xyz)
            loss = criterion(output, batch_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """)


def benchmark_reading_speed():
    """测试不同配置的读取速度"""
    
    print("\n" + "="*70)
    print("读取速度测试")
    print("="*70)
    
    h5_path = r"E:\data\云南遥感中心\第一批\h5\train\processed_02.h5"
    test_segments = 100
    
    configs = [
        {"preload": True, "workers": 0, "desc": "预加载 + 单线程"},
        {"preload": True, "workers": 2, "desc": "预加载 + 2 workers"},
        {"preload": True, "workers": 4, "desc": "预加载 + 4 workers"},
        {"preload": True, "workers": 8, "desc": "预加载 + 8 workers"},
        {"preload": False, "workers": 4, "desc": "文件读取 + 4 workers"},
    ]
    
    for config in configs:
        print(f"\n配置: {config['desc']}")
        
        # 创建数据集
        start = time.time()
        dataset = H5PointCloudDataset(
            h5_path=h5_path,
            preload=config['preload'],
            cache_indices=not config['preload']  # 如果不预加载，则缓存indices
        )
        init_time = time.time() - start
        print(f"  初始化耗时: {init_time:.2f}秒")
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=config['workers'],
            pin_memory=False,
            collate_fn=collate_fn
        )
        
        # 测试读取
        start = time.time()
        total_points = 0
        for i, (batch_xyz, batch_labels) in enumerate(dataloader):
            if i * 8 >= test_segments:
                break
            # batch_xyz是list，计算总点数
            for xyz in batch_xyz:
                total_points += xyz.shape[0]
        
        elapsed = time.time() - start
        actual_segs = min(test_segments, len(dataset))
        
        print(f"  读取耗时: {elapsed:.2f}秒")
        print(f"  速度: {actual_segs/elapsed:.2f} segments/秒")
        print(f"  速度: {total_points/elapsed:,.0f} 点/秒")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_reading_speed()
    else:
        demo_usage()
