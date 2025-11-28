"""
用于加载 bin+pkl 逻辑索引格式点云数据的数据集 (对应 tile_las1.py)

本模块实现了新的 bin+pkl 数据格式的数据集类，支持：
1. 全量模式 (full): 加载所有原始点
2. 体素模式 (voxel): 使用体素化索引进行采样
   - train/val: 从每个体素随机取 1 个点
   - test/predict: 使用模运算采样确保全覆盖

数据结构 (tile_las1.py 生成):
- .bin 文件：以结构化 numpy 数组格式包含所有点数据
- .pkl 文件：包含元数据，包括：
    - segments: 分块信息列表，每个分块包含：
        - indices: 点索引
        - sort_idx: 体素化排序索引
        - voxel_counts: 每个体素的点数
        - num_voxels: 体素数量
        - max_voxel_density: 最大体素点数
    - header_info: 原始 LAS 文件头
    - grid_size: 体素化网格大小
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from numba import jit, prange

from .dataset_base import DatasetBase


# ============================================================================
# Numba 加速采样函数
# ============================================================================

@jit(nopython=True, cache=True)
def _voxel_random_sample_numba(sort_idx: np.ndarray, 
                                voxel_counts: np.ndarray,
                                cumsum: np.ndarray) -> np.ndarray:
    """
    Numba 加速的随机体素采样
    从每个体素中随机采样 1 个点
    """
    n_voxels = len(voxel_counts)
    sampled = np.empty(n_voxels, dtype=np.int32)
    
    for i in range(n_voxels):
        voxel_count = voxel_counts[i]
        start_pos = cumsum[i]
        # 使用 numpy 随机数
        random_offset = np.random.randint(0, voxel_count)
        sampled[i] = sort_idx[start_pos + random_offset]
    
    return sampled


@jit(nopython=True, cache=True)
def _voxel_modulo_sample_numba(sort_idx: np.ndarray,
                                voxel_counts: np.ndarray,
                                cumsum: np.ndarray,
                                loop_idx: int,
                                points_per_loop: int) -> np.ndarray:
    """
    Numba 加速的模运算体素采样
    """
    n_voxels = len(voxel_counts)
    total_points = n_voxels * points_per_loop
    sampled = np.empty(total_points, dtype=np.int32)
    
    idx = 0
    for i in range(n_voxels):
        voxel_count = voxel_counts[i]
        start_pos = cumsum[i]
        
        for p in range(points_per_loop):
            logical_idx = loop_idx * points_per_loop + p
            local_idx = logical_idx % voxel_count
            sampled[idx] = sort_idx[start_pos + local_idx]
            idx += 1
    
    return sampled


@jit(nopython=True, parallel=True, cache=True)
def _voxel_random_sample_parallel(sort_idx: np.ndarray, 
                                   voxel_counts: np.ndarray,
                                   cumsum: np.ndarray,
                                   random_offsets: np.ndarray) -> np.ndarray:
    """
    Numba 并行加速的随机体素采样
    注意：需要预生成随机数以避免并行随机数问题
    """
    n_voxels = len(voxel_counts)
    sampled = np.empty(n_voxels, dtype=np.int32)
    
    for i in prange(n_voxels):
        voxel_count = voxel_counts[i]
        start_pos = cumsum[i]
        random_offset = random_offsets[i] % voxel_count
        sampled[i] = sort_idx[start_pos + random_offset]
    
    return sampled


@jit(nopython=True, parallel=True, cache=True)
def _voxel_modulo_sample_parallel(sort_idx: np.ndarray,
                                   voxel_counts: np.ndarray,
                                   cumsum: np.ndarray,
                                   loop_idx: int,
                                   points_per_loop: int) -> np.ndarray:
    """
    Numba 并行加速的模运算体素采样
    """
    n_voxels = len(voxel_counts)
    total_points = n_voxels * points_per_loop
    sampled = np.empty(total_points, dtype=np.int32)
    
    for i in prange(n_voxels):
        voxel_count = voxel_counts[i]
        start_pos = cumsum[i]
        base_idx = i * points_per_loop
        
        for p in range(points_per_loop):
            logical_idx = loop_idx * points_per_loop + p
            local_idx = logical_idx % voxel_count
            sampled[base_idx + p] = sort_idx[start_pos + local_idx]
    
    return sampled


class BinPklDataset1(DatasetBase):
    """
    bin+pkl 逻辑索引格式点云数据的数据集类 (对应 tile_las1.py)
    
    支持两种模式：
    - full: 全量模式，加载所有原始点
    - voxel: 体素模式，基于体素化索引采样
      - train/val: 每个体素随机取 1 个点，数据集长度 = 分块数
      - test/predict: 模运算采样确保全覆盖，数据集长度 = sum(actual_loops)
    """
    
    def __init__(
        self,
        data_root,
        split='train',
        assets=None,
        transform=None,
        ignore_label=-1,
        loop=1,
        cache_data=False,
        class_mapping=None,
        h_norm_grid=1.0,
        mode='voxel',
        max_loops: Optional[int] = None,
    ):
        """
        初始化 BinPklDataset1
        
        参数：
            data_root: 包含 bin+pkl 文件的根目录，或单个 pkl 文件路径，
                      或 pkl 文件路径列表
            split: 数据集划分（'train'、'val'、'test'、'predict'）
            assets: 要加载的数据属性列表（默认：['coord', 'intensity', 'classification']）
            transform: 要应用的数据变换
            ignore_label: 在训练中忽略的标签
            loop: 遍历数据集的次数（用于训练）
            cache_data: 是否在内存中缓存加载的数据
            class_mapping: 将原始类别标签映射到连续标签的字典
            h_norm_grid: 计算归一化高程时使用的栅格分辨率（米）
            mode: 采样模式
                - 'full': 全量模式，加载所有原始点
                - 'voxel': 体素模式，基于体素化索引采样
            max_loops: 体素模式下的最大采样轮次 (仅 test/predict 生效)
                - None: 按体素内最大点数进行采样
                - 设置值: 限制最大轮数，确保在 max_loops 轮内采完所有点
        """
        # 如果未指定，则设置默认资产
        if assets is None:
            assets = ['coord', 'classification']
        
        # 初始化元数据缓存
        self._metadata_cache = {}
        self._mmap_cache = {}  # memmap 缓存
        self.h_norm_grid = h_norm_grid
        self.mode = mode
        self.max_loops = max_loops
        
        # 调用父类初始化
        super().__init__(
            data_root=data_root,
            split=split,
            assets=assets,
            transform=transform,
            ignore_label=ignore_label,
            loop=loop,
            cache_data=cache_data,
            class_mapping=class_mapping
        )
    
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """
        加载所有数据样本的列表
        
        根据模式和 split 生成不同的数据列表：
        - full 模式: 每个 segment 一个样本
        - voxel 模式:
          - train/val: 每个 segment 一个样本（随机采样）
          - test/predict: 每个 segment 的每个 loop 一个样本
        """
        data_list = []
        
        # 处理不同的 data_root 类型
        pkl_files = []
        
        if isinstance(self.data_root, (list, tuple)):
            pkl_files = [Path(p) for p in self.data_root]
            print(f"从 {len(pkl_files)} 个指定的 pkl 文件加载")
        elif self.data_root.is_file() and self.data_root.suffix == '.pkl':
            pkl_files = [self.data_root]
            print(f"从单个 pkl 文件加载: {self.data_root.name}")
        else:
            pkl_files = sorted(self.data_root.glob('*.pkl'))
            if len(pkl_files) == 0:
                raise ValueError(f"在 {self.data_root} 中未找到 pkl 文件")
            print(f"在目录中找到 {len(pkl_files)} 个 pkl 文件")
        
        total_segments = 0
        total_samples = 0
        
        for pkl_path in pkl_files:
            if not pkl_path.exists():
                print(f"警告: {pkl_path} 未找到，跳过")
                continue
                
            bin_path = pkl_path.with_suffix('.bin')
            
            if not bin_path.exists():
                print(f"警告: {bin_path.name} 未找到，跳过 {pkl_path.name}")
                continue
            
            # 加载 pkl 元数据并缓存
            with open(pkl_path, 'rb') as f:
                metadata = pickle.load(f)
            
            pkl_key = str(pkl_path)
            self._metadata_cache[pkl_key] = metadata
            
            grid_size = metadata.get('grid_size', None)
            
            # 处理每个 segment
            for segment_info in metadata['segments']:
                segment_id = segment_info['segment_id']
                num_points = segment_info['num_points']
                total_segments += 1
                
                # 获取体素信息
                voxel_counts = segment_info.get('voxel_counts', None)
                max_voxel_count = int(voxel_counts.max()) if voxel_counts is not None and len(voxel_counts) > 0 else 1
                num_voxels = len(voxel_counts) if voxel_counts is not None else 0
                
                # 计算边界信息
                bounds = segment_info.get('bounds', {})
                if not bounds:
                    bounds = {
                        'x_min': segment_info.get('x_min', 0),
                        'x_max': segment_info.get('x_max', 0),
                        'y_min': segment_info.get('y_min', 0),
                        'y_max': segment_info.get('y_max', 0),
                        'z_min': segment_info.get('z_min', 0),
                        'z_max': segment_info.get('z_max', 0),
                    }
                
                # 根据模式和 split 决定如何生成样本
                if self.mode == 'full':
                    # 全量模式：每个 segment 一个样本
                    data_list.append({
                        'bin_path': str(bin_path),
                        'pkl_path': str(pkl_path),
                        'segment_id': segment_id,
                        'num_points': num_points,
                        'num_voxels': num_voxels,
                        'max_voxel_count': max_voxel_count,
                        'file_name': bin_path.stem,
                        'bounds': bounds,
                        'loop_idx': None,  # 全量模式无 loop
                        'points_per_loop': None,
                    })
                    total_samples += 1
                    
                elif self.mode == 'voxel':
                    if self.split in ['train', 'val']:
                        # train/val: 每个 segment 一个样本，随机采样
                        data_list.append({
                            'bin_path': str(bin_path),
                            'pkl_path': str(pkl_path),
                            'segment_id': segment_id,
                            'num_points': num_points,
                            'num_voxels': num_voxels,
                            'max_voxel_count': max_voxel_count,
                            'file_name': bin_path.stem,
                            'bounds': bounds,
                            'loop_idx': None,  # train/val 时为 None，表示随机采样
                            'points_per_loop': 1,
                        })
                        total_samples += 1
                        
                    else:
                        # test/predict: 每个 loop 一个样本，确保全覆盖
                        if voxel_counts is None or num_voxels == 0:
                            # 无体素化信息，单个样本
                            actual_loops = 1
                            points_per_loop = num_points
                        else:
                            # 计算实际轮数和每轮采样点数
                            actual_loops, points_per_loop = self._compute_sampling_params(
                                max_voxel_count, self.max_loops
                            )
                        
                        for loop_idx in range(actual_loops):
                            data_list.append({
                                'bin_path': str(bin_path),
                                'pkl_path': str(pkl_path),
                                'segment_id': segment_id,
                                'num_points': num_points,
                                'num_voxels': num_voxels,
                                'max_voxel_count': max_voxel_count,
                                'file_name': bin_path.stem,
                                'bounds': bounds,
                                'loop_idx': loop_idx,
                                'points_per_loop': points_per_loop,
                                'actual_loops': actual_loops,
                            })
                            total_samples += 1
                else:
                    raise ValueError(f"未知模式: {self.mode}")
        
        print(f"从 {len(pkl_files)} 个文件加载了 {total_segments} 个 segments, "
              f"共 {total_samples} 个样本 (mode={self.mode}, split={self.split})")
        
        return data_list
    
    def _compute_sampling_params(self, max_voxel_count: int, max_loops: Optional[int]) -> Tuple[int, int]:
        """
        计算采样参数：实际轮数和每轮采样点数
        
        Args:
            max_voxel_count: 体素内最大点数
            max_loops: 最大采样轮次限制
            
        Returns:
            (actual_loops, points_per_loop)
        """
        if max_loops is None:
            # 未设置 max_loops：按最大体素点数采样，每轮采 1 个点
            return max_voxel_count, 1
        elif max_voxel_count <= max_loops:
            # 最大点数 <= max_loops：按实际最大点数采样，每轮采 1 个点
            return max_voxel_count, 1
        else:
            # 最大点数 > max_loops：限制轮数，每轮采多个点
            points_per_loop = int(np.ceil(max_voxel_count / max_loops))
            return max_loops, points_per_loop
    
    def _get_metadata(self, pkl_path: str) -> dict:
        """获取缓存的元数据"""
        if pkl_path not in self._metadata_cache:
            with open(pkl_path, 'rb') as f:
                self._metadata_cache[pkl_path] = pickle.load(f)
        return self._metadata_cache[pkl_path]
    
    def _get_mmap(self, bin_path: str, dtype) -> np.ndarray:
        """获取缓存的 memmap"""
        if bin_path not in self._mmap_cache:
            self._mmap_cache[bin_path] = np.memmap(bin_path, dtype=dtype, mode='r')
        return self._mmap_cache[bin_path]
    
    def _voxel_random_sample(self, segment_info: dict, mmap_data: np.ndarray) -> np.ndarray:
        """
        从每个体素中随机采样 1 个点 (用于 train/val)
        使用 Numba 加速
        
        注意：为确保每次调用的随机性（不同 epoch、不同 worker），
        使用基于时间和对象 id 的熵源生成随机数，避免受全局种子影响。
        
        Args:
            segment_info: segment 元数据
            mmap_data: 内存映射的 bin 数据
            
        Returns:
            采样后的结构化数组
        """
        indices = segment_info['indices']
        sort_idx = segment_info.get('sort_idx', None)
        voxel_counts = segment_info.get('voxel_counts', None)
        
        # 如果没有体素化信息，返回全部数据
        if sort_idx is None or voxel_counts is None:
            return mmap_data[indices]
        
        # 计算每个体素的起始位置
        cumsum = np.cumsum(np.insert(voxel_counts, 0, 0)).astype(np.int64)
        
        # 🔥 关键修复：使用独立的随机数生成器，不受全局种子影响
        # 结合多种熵源确保真正的随机性：
        # - 当前时间纳秒级
        # - segment_id (不同 segment 不同)
        # - Python 对象 id (不同调用不同)
        import time
        import os
        
        # 使用多种熵源创建种子
        entropy = (
            int(time.time_ns()) ^  # 纳秒级时间戳
            id(segment_info) ^     # 对象地址
            segment_info.get('segment_id', 0) ^  # segment id
            os.getpid()            # 进程 id (多 worker 时不同)
        )
        
        # 创建独立的随机数生成器
        rng = np.random.Generator(np.random.PCG64(entropy & 0xFFFFFFFFFFFFFFFF))
        
        n_voxels = len(voxel_counts)
        random_offsets = rng.integers(0, 2**31, size=n_voxels, dtype=np.int32)
        
        # 使用 Numba 并行加速采样
        sampled_local_indices = _voxel_random_sample_parallel(
            sort_idx.astype(np.int32), 
            voxel_counts.astype(np.int32), 
            cumsum,
            random_offsets
        )
        
        global_indices = indices[sampled_local_indices]
        return mmap_data[global_indices]
    
    def _voxel_modulo_sample(self, segment_info: dict, mmap_data: np.ndarray,
                             loop_idx: int, points_per_loop: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        对 segment 进行体素模运算采样 (用于 test/predict)
        使用 Numba 加速
        
        Args:
            segment_info: segment 元数据
            mmap_data: 内存映射的 bin 数据
            loop_idx: 当前采样轮次
            points_per_loop: 每轮每体素采样点数
            
        Returns:
            (采样后的结构化数组, 原始点索引)
        """
        indices = segment_info['indices']
        sort_idx = segment_info.get('sort_idx', None)
        voxel_counts = segment_info.get('voxel_counts', None)
        
        # 如果没有体素化信息，返回全部数据
        if sort_idx is None or voxel_counts is None:
            return mmap_data[indices], indices.copy()
        
        # 计算每个体素的起始位置
        cumsum = np.cumsum(np.insert(voxel_counts, 0, 0)).astype(np.int64)
        
        # 使用 Numba 并行加速采样
        sampled_local_indices = _voxel_modulo_sample_parallel(
            sort_idx.astype(np.int32),
            voxel_counts.astype(np.int32),
            cumsum,
            loop_idx,
            points_per_loop
        )
        
        global_indices = indices[sampled_local_indices]
        return mmap_data[global_indices], global_indices
    
    def _compute_h_norm(self, coord: np.ndarray, is_ground: np.ndarray, 
                       grid_resolution: float = 1.0) -> np.ndarray:
        """
        基于地面点标记计算归一化高程（地上高程）
        """
        ground_mask = (is_ground == 1)
        
        if not np.any(ground_mask):
            z_min = coord[:, 2].min()
            return (coord[:, 2] - z_min).astype(np.float32)
        
        ground_points = coord[ground_mask]
        ground_xy = ground_points[:, :2]
        ground_z = ground_points[:, 2]
        n_ground = len(ground_points)
        
        if n_ground < 10:
            ground_z_base = ground_z.min()
            h_norm = coord[:, 2] - ground_z_base
        elif n_ground < 50:
            from scipy.spatial import cKDTree
            tree = cKDTree(ground_xy)
            k = min(3, n_ground)
            distances, indices = tree.query(coord[:, :2], k=k)
            
            if k == 1:
                local_ground_z = ground_z[indices]
            else:
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                local_ground_z = (ground_z[indices] * weights).sum(axis=1)
            
            h_norm = coord[:, 2] - local_ground_z
        else:
            h_norm = self._compute_h_norm_tin_raster(
                coord, ground_xy, ground_z, grid_resolution
            )
        
        return h_norm.astype(np.float32)
    
    def _compute_h_norm_tin_raster(self, coord: np.ndarray, ground_xy: np.ndarray, 
                                   ground_z: np.ndarray, grid_resolution: float) -> np.ndarray:
        """使用 TIN + Raster 混合方法计算 h_norm"""
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree
        
        x_min, y_min = coord[:, :2].min(axis=0)
        x_max, y_max = coord[:, :2].max(axis=0)
        
        n_x = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
        n_y = int(np.ceil((y_max - y_min) / grid_resolution)) + 1
        
        MAX_GRID_SIZE = 2000
        if n_x > MAX_GRID_SIZE or n_y > MAX_GRID_SIZE:
            grid_resolution = max(
                (x_max - x_min) / MAX_GRID_SIZE,
                (y_max - y_min) / MAX_GRID_SIZE
            )
            n_x = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
            n_y = int(np.ceil((y_max - y_min) / grid_resolution)) + 1
        
        grid_x = np.linspace(x_min, x_max, n_x)
        grid_y = np.linspace(y_min, y_max, n_y)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        
        dtm_grid = griddata(
            ground_xy, ground_z, (grid_xx, grid_yy),
            method='linear', fill_value=np.nan
        )
        
        indices_x = ((coord[:, 0] - x_min) / grid_resolution).astype(int)
        indices_y = ((coord[:, 1] - y_min) / grid_resolution).astype(int)
        indices_x = np.clip(indices_x, 0, dtm_grid.shape[1] - 1)
        indices_y = np.clip(indices_y, 0, dtm_grid.shape[0] - 1)
        
        z_ground = dtm_grid[indices_y, indices_x]
        
        nan_mask = np.isnan(z_ground)
        if np.any(nan_mask):
            tree = cKDTree(ground_xy)
            k = min(3, len(ground_xy))
            nan_points = coord[nan_mask, :2]
            
            if k == 1:
                _, indices = tree.query(nan_points, k=1)
                z_ground[nan_mask] = ground_z[indices]
            else:
                distances, indices = tree.query(nan_points, k=k)
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                z_ground[nan_mask] = (ground_z[indices] * weights).sum(axis=1)
        
        return coord[:, 2] - z_ground
    
    def _load_data(self, idx: int) -> Dict[str, Any]:
        """
        加载特定的数据样本
        """
        sample_info = self.data_list[idx]
        
        bin_path = sample_info['bin_path']
        pkl_path = sample_info['pkl_path']
        segment_id = sample_info['segment_id']
        loop_idx = sample_info.get('loop_idx', None)
        points_per_loop = sample_info.get('points_per_loop', 1)
        
        # 获取元数据
        metadata = self._get_metadata(pkl_path)
        
        # 查找 segment 信息
        segment_info = None
        for seg in metadata['segments']:
            if seg['segment_id'] == segment_id:
                segment_info = seg
                break
        
        if segment_info is None:
            raise ValueError(f"在 {pkl_path} 中未找到 segment {segment_id}")
        
        # 获取 memmap 数据
        mmap_data = self._get_mmap(bin_path, metadata['dtype'])
        
        # 根据模式和 split 采样数据
        original_indices = None
        
        if self.mode == 'full':
            # 全量模式
            indices = segment_info['indices']
            segment_points = mmap_data[indices]
            if self.split in ['test', 'predict']:
                original_indices = indices.copy()
                
        elif self.mode == 'voxel':
            if self.split in ['train', 'val']:
                # 随机采样
                segment_points = self._voxel_random_sample(segment_info, mmap_data)
            else:
                # 模运算采样
                segment_points, original_indices = self._voxel_modulo_sample(
                    segment_info, mmap_data, loop_idx, points_per_loop
                )
        else:
            raise ValueError(f"未知模式: {self.mode}")
        
        # 提取请求的资产
        data = {}
        
        # 🔥 坐标：转换为局部坐标以保持 float32 精度
        # 原始坐标为 float64（如 508000.0, 5443500.0），直接转 float32 会丢失精度
        # 使用局部坐标（减去 local_min）后，坐标范围通常在 0~50m 内，float32 足够精确
        local_min = segment_info.get('local_min', None)
        
        coord = np.stack([
            segment_points['X'],
            segment_points['Y'],
            segment_points['Z']
        ], axis=1)  # 保持 float64
        
        if local_min is not None:
            # 转换为局部坐标
            coord = coord - local_min.astype(np.float64)
        
        coord = coord.astype(np.float32)
        data['coord'] = coord
        
        # 保存原始坐标偏移量（用于预测时恢复全局坐标）
        if self.split in ['test', 'predict'] and local_min is not None:
            data['coord_offset'] = local_min.astype(np.float64)
        
        # 其他资产
        for asset in self.assets:
            if asset == 'coord':
                continue
                
            elif asset == 'intensity':
                if 'intensity' not in segment_points.dtype.names:
                    raise ValueError(
                        f"请求的属性 'intensity' 在数据中不存在。\n"
                        f"可用字段: {list(segment_points.dtype.names)}\n"
                        f"请检查 assets 配置或数据文件。"
                    )
                intensity = segment_points['intensity'].astype(np.float32)
                # 归一化到 [0, 1]
                intensity = intensity / 65535.0
                data['intensity'] = intensity
                    
            elif asset == 'color':
                required_fields = ['red', 'green', 'blue']
                missing = [f for f in required_fields if f not in segment_points.dtype.names]
                if missing:
                    raise ValueError(
                        f"请求的属性 'color' 所需字段 {missing} 在数据中不存在。\n"
                        f"可用字段: {list(segment_points.dtype.names)}\n"
                        f"请检查 assets 配置或数据文件。"
                    )
                color = np.stack([
                    segment_points['red'],
                    segment_points['green'],
                    segment_points['blue']
                ], axis=1).astype(np.float32)
                data['color'] = color

            elif asset == 'echo':
                required_fields = ['return_number', 'number_of_returns']
                missing = [f for f in required_fields if f not in segment_points.dtype.names]
                if missing:
                    raise ValueError(
                        f"请求的属性 'echo' 所需字段 {missing} 在数据中不存在。\n"
                        f"可用字段: {list(segment_points.dtype.names)}\n"
                        f"请检查 assets 配置或数据文件。"
                    )
                return_number = segment_points['return_number'].astype(np.float32)
                number_of_returns = segment_points['number_of_returns'].astype(np.float32)
                echo = np.stack([
                    (return_number == 1).astype(np.float32) * 2 - 1,
                    (return_number == number_of_returns).astype(np.float32) * 2 - 1,
                ], axis=1)
                data['echo'] = echo

            elif asset == 'normal':
                required_fields = ['normal_x', 'normal_y', 'normal_z']
                missing = [f for f in required_fields if f not in segment_points.dtype.names]
                if missing:
                    raise ValueError(
                        f"请求的属性 'normal' 所需字段 {missing} 在数据中不存在。\n"
                        f"可用字段: {list(segment_points.dtype.names)}\n"
                        f"请检查 assets 配置或数据文件。"
                    )
                normal = np.stack([
                    segment_points['normal_x'],
                    segment_points['normal_y'],
                    segment_points['normal_z']
                ], axis=1).astype(np.float32)
                data['normal'] = normal

            elif asset == 'h_norm':
                if 'is_ground' in segment_points.dtype.names:
                    is_ground = segment_points['is_ground']
                    h_norm = self._compute_h_norm(coord, is_ground, self.h_norm_grid)
                    data['h_norm'] = h_norm
                else:
                    raise ValueError(
                        f"请求的属性 'h_norm' 所需字段 'is_ground' 在数据中不存在。\n"
                        f"可用字段: {list(segment_points.dtype.names)}\n"
                        f"请检查 assets 配置或数据文件。"
                    )

            elif asset == 'class':
                if 'classification' not in segment_points.dtype.names:
                    raise ValueError(
                        f"请求的属性 'class' 所需字段 'classification' 在数据中不存在。\n"
                        f"可用字段: {list(segment_points.dtype.names)}\n"
                        f"请检查 assets 配置或数据文件。"
                    )
                labels = segment_points['classification'].astype(np.int64)
                # 应用类别映射
                if self.class_mapping is not None:
                    mapped_labels = np.full_like(labels, self.ignore_label)
                    for orig_label, new_label in self.class_mapping.items():
                        mapped_labels[labels == orig_label] = new_label
                    labels = mapped_labels
                data['class'] = labels

        # test/predict 时存储索引信息
        if self.split in ['test', 'predict']:
            if original_indices is not None:
                data['indices'] = original_indices
            data['bin_file'] = sample_info.get('file_name', Path(bin_path).stem)
            data['bin_path'] = bin_path
            data['pkl_path'] = pkl_path
            data['segment_id'] = segment_id
            if loop_idx is not None:
                data['loop_idx'] = loop_idx
        
        return data
    
    def get_segment_info(self, idx: int) -> Dict[str, Any]:
        """获取特定片段的元数据"""
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.data_list)})")
        
        sample_info = self.data_list[idx]
        pkl_path = sample_info['pkl_path']
        segment_id = sample_info['segment_id']
        
        metadata = self._get_metadata(pkl_path)
        
        for seg in metadata['segments']:
            if seg['segment_id'] == segment_id:
                return seg
        
        raise ValueError(f"未找到 segment {segment_id}")
    
    def get_file_metadata(self, idx: int) -> Dict[str, Any]:
        """获取包含特定片段的文件的元数据"""
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.data_list)})")
        
        sample_info = self.data_list[idx]
        pkl_path = sample_info['pkl_path']
        
        metadata = self._get_metadata(pkl_path)
        
        return {k: v for k, v in metadata.items() if k != 'segments'}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if len(self.data_list) == 0:
            return {}
        
        num_points_list = [s['num_points'] for s in self.data_list]
        num_voxels_list = [s.get('num_voxels', 0) for s in self.data_list]
        
        stats = {
            'num_samples': len(self.data_list),
            'mode': self.mode,
            'split': self.split,
            'num_points': {
                'total': sum(num_points_list),
                'mean': np.mean(num_points_list),
                'median': np.median(num_points_list),
                'min': np.min(num_points_list),
                'max': np.max(num_points_list),
                'std': np.std(num_points_list),
            },
            'num_voxels': {
                'mean': np.mean(num_voxels_list) if num_voxels_list else 0,
                'min': np.min(num_voxels_list) if num_voxels_list else 0,
                'max': np.max(num_voxels_list) if num_voxels_list else 0,
            }
        }
        
        return stats
    
    def print_stats(self):
        """打印数据集统计信息"""
        stats = self.get_stats()
        
        print("="*70)
        print(f"数据集统计信息 ({self.__class__.__name__})")
        print("="*70)
        print(f"划分: {self.split}")
        print(f"模式: {self.mode}")
        print(f"样本数: {stats['num_samples']:,}")
        print(f"\n每样本点数:")
        print(f"  - 总计: {stats['num_points']['total']:,}")
        print(f"  - 平均: {stats['num_points']['mean']:,.1f}")
        print(f"  - 中位数: {stats['num_points']['median']:,.0f}")
        print(f"  - 最小: {stats['num_points']['min']:,}")
        print(f"  - 最大: {stats['num_points']['max']:,}")
        if self.mode == 'voxel':
            print(f"\n体素数:")
            print(f"  - 平均: {stats['num_voxels']['mean']:,.1f}")
            print(f"  - 最小: {stats['num_voxels']['min']:,}")
            print(f"  - 最大: {stats['num_voxels']['max']:,}")
        print("="*70)
    
    def get_class_distribution(self) -> Optional[Dict[int, int]]:
        """获取数据集的类别分布"""
        if len(self.data_list) == 0:
            return {}
        
        pkl_path = self.data_list[0]['pkl_path']
        metadata = self._get_metadata(pkl_path)
        
        if 'label_counts' in metadata:
            if self.class_mapping is not None:
                mapped_counts = {}
                for orig_label, count in metadata['label_counts'].items():
                    if orig_label in self.class_mapping:
                        new_label = self.class_mapping[orig_label]
                        mapped_counts[new_label] = mapped_counts.get(new_label, 0) + count
                return mapped_counts
            else:
                return metadata['label_counts']
        
        return {}
    
    def get_sample_num_points(self) -> List[int]:
        """
        获取每个样本的点数列表（用于 DynamicBatchSampler）
        
        注意：在 voxel 模式下，返回的是体素数（采样后的点数）
        """
        if self.mode == 'voxel':
            # 体素模式：采样后的点数 = 体素数 × points_per_loop
            return [
                s.get('num_voxels', s['num_points']) * s.get('points_per_loop', 1)
                for s in self.data_list
            ]
        else:
            # 全量模式：原始点数
            return [s['num_points'] for s in self.data_list]


def create_dataset(
    data_root,
    split='train',
    assets=None,
    transform=None,
    ignore_label=-1,
    loop=1,
    cache_data=False,
    mode='voxel',
    max_loops=None,
    **kwargs
):
    """
    创建 BinPklDataset1 的工厂函数
    """
    return BinPklDataset1(
        data_root=data_root,
        split=split,
        assets=assets,
        transform=transform,
        ignore_label=ignore_label,
        loop=loop,
        cache_data=cache_data,
        mode=mode,
        max_loops=max_loops,
        **kwargs
    )
