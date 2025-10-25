"""
多H5文件高效数据集类 - 支持LRU缓存

特性：
- 支持多个H5文件
- LRU缓存：智能管理内存
- 跨文件随机采样
- 自动预加载热点文件
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict
from collections import OrderedDict
import time


class MultiH5Dataset(Dataset):
    """
    多H5文件数据集，支持LRU缓存
    
    参数:
        h5_paths: H5文件路径列表
        transform: 数据增强函数
        cache_size: 缓存的H5文件数量（默认5，每个500MB）
        preload_all: 是否预加载所有文件（如果内存充足）
    """
    
    def __init__(
        self,
        h5_paths: List[str],
        transform: Optional[Callable] = None,
        cache_size: int = 5,
        preload_all: bool = False
    ):
        self.h5_paths = [Path(p) for p in h5_paths]
        self.transform = transform
        self.cache_size = cache_size
        self.preload_all = preload_all
        
        # 验证文件存在
        for p in self.h5_paths:
            if not p.exists():
                raise FileNotFoundError(f"H5文件不存在: {p}")
        
        print(f"初始化多H5数据集: {len(self.h5_paths)}个文件")
        
        # 构建segment索引映射
        self.file_segment_map = []  # [(file_idx, segment_idx_in_file), ...]
        self.file_info = []  # [{'num_segments': N, 'path': path}, ...]
        
        total_segments = 0
        for file_idx, h5_path in enumerate(self.h5_paths):
            with h5py.File(h5_path, 'r') as f:
                num_segs = f['segments'].attrs['num_segments']
                self.file_info.append({
                    'num_segments': num_segs,
                    'path': h5_path
                })
                
                # 添加映射
                for seg_idx in range(num_segs):
                    self.file_segment_map.append((file_idx, seg_idx))
                
                total_segments += num_segs
        
        self.total_segments = total_segments
        print(f"  总segments: {self.total_segments}")
        print(f"  平均每文件: {self.total_segments / len(self.h5_paths):.0f} segments")
        
        # 初始化缓存
        if preload_all:
            print(f"预加载所有{len(self.h5_paths)}个H5文件...")
            start = time.time()
            self.cache = {}
            for file_idx in range(len(self.h5_paths)):
                self.cache[file_idx] = self._load_file(file_idx)
            elapsed = time.time() - start
            total_mb = sum(self._estimate_memory(data) for data in self.cache.values())
            print(f"✅ 预加载完成: {total_mb:.1f}MB, 耗时{elapsed:.2f}秒")
            self.lru_cache = None
        else:
            print(f"使用LRU缓存: 最多缓存{cache_size}个H5文件")
            self.cache = {}
            self.lru_cache = OrderedDict()  # file_idx -> last_access_time
            print(f"✅ 初始化完成")
    
    def _load_file(self, file_idx: int) -> Dict:
        """加载单个H5文件到内存"""
        h5_path = self.file_info[file_idx]['path']
        
        with h5py.File(h5_path, 'r') as f:
            data = {
                'x': f['data']['x'][:],
                'y': f['data']['y'][:],
                'z': f['data']['z'][:],
                'labels': f['data']['classification'][:]
            }
            
            # 读取所有segment的indices
            num_segs = self.file_info[file_idx]['num_segments']
            segments_info = []
            for i in range(num_segs):
                indices = f['segments'][f'segment_{i:04d}']['indices'][:]
                segments_info.append(indices)
            
            data['segments_info'] = segments_info
        
        return data
    
    def _estimate_memory(self, data: Dict) -> float:
        """估算数据内存占用（MB）"""
        total_bytes = sum(
            arr.nbytes for key, arr in data.items() 
            if key != 'segments_info'
        )
        return total_bytes / (1024**2)
    
    def _get_cached_data(self, file_idx: int) -> Dict:
        """获取缓存的文件数据，使用LRU策略"""
        
        if self.preload_all:
            # 全预加载模式，直接返回
            return self.cache[file_idx]
        
        # LRU缓存模式
        if file_idx in self.cache:
            # 缓存命中，更新访问时间
            self.lru_cache.move_to_end(file_idx)
            return self.cache[file_idx]
        
        # 缓存未命中，需要加载
        if len(self.cache) >= self.cache_size:
            # 缓存满了，移除最久未使用的
            oldest_file_idx = next(iter(self.lru_cache))
            del self.cache[oldest_file_idx]
            del self.lru_cache[oldest_file_idx]
        
        # 加载新文件
        data = self._load_file(file_idx)
        self.cache[file_idx] = data
        self.lru_cache[file_idx] = time.time()
        
        return data
    
    def __len__(self) -> int:
        return self.total_segments
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        读取单个segment
        
        参数:
            idx: 全局segment索引（0到total_segments-1）
        
        返回:
            xyz: [N, 3] 点云坐标
            labels: [N] 点标签
        """
        # 获取对应的文件和segment索引
        file_idx, seg_idx = self.file_segment_map[idx]
        
        # 获取缓存数据
        data = self._get_cached_data(file_idx)
        
        # 读取segment
        indices = data['segments_info'][seg_idx]
        xyz = np.stack([
            data['x'][indices],
            data['y'][indices],
            data['z'][indices]
        ], axis=1)
        labels = data['labels'][indices]
        
        # 数据增强
        if self.transform is not None:
            xyz, labels = self.transform(xyz, labels)
        
        return torch.from_numpy(xyz).float(), torch.from_numpy(labels).long()
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        if self.preload_all:
            return {
                'mode': 'preload_all',
                'cached_files': len(self.cache),
                'total_files': len(self.h5_paths)
            }
        else:
            return {
                'mode': 'lru_cache',
                'cached_files': len(self.cache),
                'cache_size': self.cache_size,
                'total_files': len(self.h5_paths)
            }


def collate_fn(batch):
    """处理不同点数的segments"""
    xyz_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    return xyz_list, labels_list


# ========== 使用示例 ==========

def demo_multi_h5():
    """演示多H5文件的使用"""
    
    print("="*70)
    print("多H5文件数据集使用示例")
    print("="*70)
    
    # 获取所有H5文件
    h5_dir = Path(r"E:\data\云南遥感中心\第一批\h5\train")
    h5_files = sorted(h5_dir.glob("*.h5"))[:3]  # 先用3个测试
    
    print(f"\n找到{len(h5_files)}个H5文件:")
    for f in h5_files:
        print(f"  {f.name}")
    
    # 方案1: LRU缓存（推荐）
    print("\n" + "="*70)
    print("方案1: LRU缓存（缓存2个文件）")
    print("="*70)
    
    dataset = MultiH5Dataset(
        h5_paths=[str(f) for f in h5_files],
        cache_size=2,  # 只缓存2个文件
        preload_all=False
    )
    
    print(f"\n数据集大小: {len(dataset)} segments")
    print(f"缓存统计: {dataset.get_cache_stats()}")
    
    # 测试读取
    print("\n测试随机读取:")
    test_indices = [0, 100, 200, 50, 150]  # 跨文件访问
    for idx in test_indices:
        xyz, labels = dataset[idx]
        file_idx, seg_idx = dataset.file_segment_map[idx]
        print(f"  Global idx {idx} -> File {file_idx}, Seg {seg_idx}: {xyz.shape}")
    
    print(f"\n缓存统计: {dataset.get_cache_stats()}")
    
    # 方案2: 全预加载（如果内存充足）
    print("\n" + "="*70)
    print("方案2: 全预加载")
    print("="*70)
    
    dataset2 = MultiH5Dataset(
        h5_paths=[str(f) for f in h5_files],
        preload_all=True
    )
    
    print(f"\n数据集大小: {len(dataset2)} segments")
    print(f"缓存统计: {dataset2.get_cache_stats()}")
    
    # 使用DataLoader
    print("\n" + "="*70)
    print("使用DataLoader")
    print("="*70)
    
    dataloader = DataLoader(
        dataset2,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # 预加载模式用0
        collate_fn=collate_fn
    )
    
    print("测试前3个batch:")
    for i, (batch_xyz, batch_labels) in enumerate(dataloader):
        if i >= 3:
            break
        print(f"  Batch {i}: {len(batch_xyz)} segments")
        for j, xyz in enumerate(batch_xyz):
            print(f"    Segment {j}: {xyz.shape}")


def benchmark_multi_h5():
    """测试多H5文件的性能"""
    
    print("\n" + "="*70)
    print("多H5文件性能测试")
    print("="*70)
    
    h5_dir = Path(r"E:\data\云南遥感中心\第一批\h5\train")
    h5_files = sorted(h5_dir.glob("*.h5"))[:5]  # 测试5个文件
    
    print(f"测试文件: {len(h5_files)}个H5")
    
    configs = [
        {"preload": True, "desc": "全预加载"},
        {"preload": False, "cache": 2, "desc": "LRU缓存(size=2)"},
        {"preload": False, "cache": 3, "desc": "LRU缓存(size=3)"},
        {"preload": False, "cache": 5, "desc": "LRU缓存(size=5)"},
    ]
    
    for config in configs:
        print(f"\n配置: {config['desc']}")
        
        # 创建数据集
        start = time.time()
        dataset = MultiH5Dataset(
            h5_paths=[str(f) for f in h5_files],
            preload_all=config['preload'],
            cache_size=config.get('cache', 5)
        )
        init_time = time.time() - start
        print(f"  初始化: {init_time:.2f}秒")
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # 测试读取
        start = time.time()
        total_segs = 0
        for i, (batch_xyz, batch_labels) in enumerate(dataloader):
            if i >= 20:  # 测试20个batch
                break
            total_segs += len(batch_xyz)
        
        elapsed = time.time() - start
        print(f"  读取20 batches: {elapsed:.2f}秒")
        print(f"  速度: {total_segs/elapsed:.2f} segments/秒")
        print(f"  缓存状态: {dataset.get_cache_stats()}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_multi_h5()
    else:
        demo_multi_h5()
