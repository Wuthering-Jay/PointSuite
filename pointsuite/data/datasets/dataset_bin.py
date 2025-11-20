"""
用于加载 bin+pkl 格式点云数据的数据集

本模块实现了我们自定义 bin+pkl 数据格式的数据集类，
其中点云数据存储在二进制文件（.bin）中，元数据存储在 pickle 文件（.pkl）中
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional

from .dataset_base import DatasetBase


class BinPklDataset(DatasetBase):
    """
    bin+pkl 格式点云数据的数据集类
    
    此数据集加载以二进制格式（.bin）存储的预处理点云片段，
    元数据以 pickle 格式（.pkl）存储
    
    数据结构：
    - .bin 文件：以结构化 numpy 数组格式包含所有点数据
    - .pkl 文件：包含元数据，包括：
        - 片段信息（索引、边界、标签计数）
        - 原始 LAS 文件头
        - 处理参数
    
    每个片段成为一个训练样本
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
        h_norm_grid=1.0
    ):
        """
        初始化 BinPklDataset
        
        参数：
            data_root: 包含 bin+pkl 文件的根目录，或单个 pkl 文件路径，
                      或 pkl 文件路径列表
            split: 数据集划分（'train'、'val'、'test'、'predict'）
                  - train/val: 不存储点索引
                  - test/predict: 存储点索引用于预测投票机制
            assets: 要加载的数据属性列表（默认：['coord', 'intensity', 'classification']）
            transform: 要应用的数据变换
            ignore_label: 在训练中忽略的标签
            loop: 遍历数据集的次数（用于训练）
            cache_data: 是否在内存中缓存加载的数据
                       - 如果为 True：所有加载的样本都缓存在内存中以加快重复访问
                                     适用于能放入 RAM 的小型数据集
                       - 如果为 False：每次从磁盘加载数据（使用 memmap 提高效率）
                                      适用于大型数据集
            class_mapping: 将原始类别标签映射到连续标签的字典
                          示例：{0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
                          如果为 None，则不应用映射
            h_norm_grid: 计算归一化高程时使用的栅格分辨率（米）
        """
        # 如果未指定，则设置默认资产
        if assets is None:
            assets = ['coord', 'intensity', 'classification']
        
        # 初始化元数据缓存（显著加快数据加载速度）
        self._metadata_cache = {}
        self.h_norm_grid = h_norm_grid
        
        # 调用父类初始化（传递 class_mapping 参数）
        super().__init__(
            data_root=data_root,
            split=split,
            assets=assets,
            transform=transform,
            ignore_label=ignore_label,
            loop=loop,
            cache_data=cache_data,
            class_mapping=class_mapping  # 传递 class_mapping 给父类
        )
    
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """
        加载所有数据样本的列表
        
        返回：
            包含样本信息的字典列表
        """
        data_list = []
        
        # 处理不同的 data_root 类型
        pkl_files = []
        
        if isinstance(self.data_root, (list, tuple)):
            # pkl 文件路径列表
            pkl_files = [Path(p) for p in self.data_root]
            print(f"从 {len(pkl_files)} 个指定的 pkl 文件加载")
        elif self.data_root.is_file() and self.data_root.suffix == '.pkl':
            # 单个 pkl 文件
            pkl_files = [self.data_root]
            print(f"从单个 pkl 文件加载: {self.data_root.name}")
        else:
            # 包含 pkl 文件的目录
            pkl_files = sorted(self.data_root.glob('*.pkl'))
            if len(pkl_files) == 0:
                raise ValueError(f"在 {self.data_root} 中未找到 pkl 文件")
            print(f"在目录中找到 {len(pkl_files)} 个 pkl 文件")
        
        # 从每个 pkl 文件加载元数据
        total_segments = 0
        
        for pkl_path in pkl_files:
            if not pkl_path.exists():
                print(f"警告: {pkl_path} 未找到，跳过")
                continue
                
            bin_path = pkl_path.with_suffix('.bin')
            
            if not bin_path.exists():
                print(f"警告: {bin_path.name} 未找到，跳过 {pkl_path.name}")
                continue
            
            # 加载 pkl 元数据
            with open(pkl_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # 将每个片段添加为单独的数据样本
            for segment_info in metadata['segments']:
                total_segments += 1
                
                data_list.append({
                    'bin_path': str(bin_path),
                    'pkl_path': str(pkl_path),
                    'segment_id': segment_info['segment_id'],
                    'num_points': segment_info['num_points'],
                    'file_name': bin_path.stem,
                    'bounds': {
                        'x_min': segment_info.get('x_min', 0),
                        'x_max': segment_info.get('x_max', 0),
                        'y_min': segment_info.get('y_min', 0),
                        'y_max': segment_info.get('y_max', 0),
                        'z_min': segment_info.get('z_min', 0),
                        'z_max': segment_info.get('z_max', 0),
                    }
                })
        
        print(f"从 {len(pkl_files)} 个文件加载了 {total_segments} 个片段")
        
        return data_list
    
    def _compute_h_norm(self, coord: np.ndarray, is_ground: np.ndarray, 
                       grid_resolution: float = 1.0) -> np.ndarray:
        """
        基于地面点标记计算归一化高程（地上高程）
        
        采用 TIN + Raster 混合方法（工业界标准）：
        1. 使用 TIN 插值生成 DTM（数字地形模型）栅格
        2. 通过快速栅格查询计算所有点的地面高程
        3. 对 DTM 未覆盖区域使用 KNN 回退策略
        
        优势：
        - 速度：栅格查询 O(1)，比 KNN 快得多
        - 精度：TIN 插值保持地面点的几何精度
        - 内存友好：栅格大小可控
        
        参数：
            coord: [N, 3] 点云坐标 (X, Y, Z)
            is_ground: [N,] 地面点标记，1 表示地面点，0 表示非地面点
            grid_resolution: DTM 栅格分辨率（米），默认 0.5m
                            更小的值 = 更精确但更慢、占用更多内存
                            更大的值 = 更快但精度稍低
            
        返回：
            h_norm: [N,] 归一化高程（地上高程），单位与输入坐标相同
        """
        # 提取地面点
        ground_mask = (is_ground == 1)
        
        # 如果没有地面点，返回相对于最低点的高度
        if not np.any(ground_mask):
            z_min = coord[:, 2].min()
            return (coord[:, 2] - z_min).astype(np.float32)
        
        ground_points = coord[ground_mask]
        ground_xy = ground_points[:, :2]
        ground_z = ground_points[:, 2]
        n_ground = len(ground_points)
        
        # 策略选择：根据地面点数量选择最优方法
        
        if n_ground < 10:
            # 地面点太少：使用全局最小值方法
            ground_z_base = ground_z.min()
            h_norm = coord[:, 2] - ground_z_base
            
        elif n_ground < 50:
            # 地面点很少：使用简单 KNN（不值得构建栅格）
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
            # 地面点足够：使用 TIN + Raster 混合方法（推荐）
            h_norm = self._compute_h_norm_tin_raster(
                coord, ground_xy, ground_z, grid_resolution
            )
        
        return h_norm.astype(np.float32)
    
    def _compute_h_norm_tin_raster(self, coord: np.ndarray, ground_xy: np.ndarray, 
                                   ground_z: np.ndarray, grid_resolution: float) -> np.ndarray:
        """
        使用 TIN + Raster 混合方法计算 h_norm（核心算法）
        
        步骤：
        1. 用 scipy.interpolate.griddata 构建 TIN 并插值到规则栅格
        2. 将所有点坐标映射到栅格索引
        3. 快速查询栅格得到地面高程
        4. 对 DTM 未覆盖区域（NaN）使用 KNN 回退
        
        参数：
            coord: [N, 3] 所有点坐标
            ground_xy: [M, 2] 地面点 XY 坐标
            ground_z: [M,] 地面点 Z 坐标
            grid_resolution: DTM 栅格分辨率
            
        返回：
            h_norm: [N,] 归一化高程
        """
        from scipy.interpolate import griddata
        
        # ===== 步骤 1: 定义 DTM 栅格 =====
        x_min, y_min = coord[:, :2].min(axis=0)
        x_max, y_max = coord[:, :2].max(axis=0)
        
        # 计算栅格大小（向上取整）
        n_x = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
        n_y = int(np.ceil((y_max - y_min) / grid_resolution)) + 1
        
        # 限制栅格大小（防止内存爆炸）
        MAX_GRID_SIZE = 2000  # 最大 2000x2000 = 400 万格子
        if n_x > MAX_GRID_SIZE or n_y > MAX_GRID_SIZE:
            # 动态调整分辨率
            grid_resolution = max(
                (x_max - x_min) / MAX_GRID_SIZE,
                (y_max - y_min) / MAX_GRID_SIZE
            )
            n_x = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
            n_y = int(np.ceil((y_max - y_min) / grid_resolution)) + 1
        
        # 创建规则栅格
        grid_x = np.linspace(x_min, x_max, n_x)
        grid_y = np.linspace(y_min, y_max, n_y)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        
        # ===== 步骤 2: TIN 插值生成 DTM =====
        # 使用 'linear' 方法（Delaunay 三角网）
        # 'cubic' 更平滑但更慢，'nearest' 最快但质量差
        dtm_grid = griddata(
            ground_xy,           # 稀疏地面点 XY
            ground_z,            # 稀疏地面点 Z
            (grid_xx, grid_yy),  # 目标栅格
            method='linear',     # TIN 方法
            fill_value=np.nan    # 无法插值区域填充 NaN
        )
        
        # ===== 步骤 3: 计算所有点的栅格索引 =====
        # 将真实坐标映射到栅格索引
        indices_x = ((coord[:, 0] - x_min) / grid_resolution).astype(int)
        indices_y = ((coord[:, 1] - y_min) / grid_resolution).astype(int)
        
        # 防止索引越界（边界点可能超出）
        indices_x = np.clip(indices_x, 0, dtm_grid.shape[1] - 1)
        indices_y = np.clip(indices_y, 0, dtm_grid.shape[0] - 1)
        
        # ===== 步骤 4: 快速栅格查询 =====
        # 注意：meshgrid 创建的数组是 (n_y, n_x) 形状
        z_ground = dtm_grid[indices_y, indices_x]
        
        # ===== 步骤 5: 处理 DTM 未覆盖区域（NaN） =====
        nan_mask = np.isnan(z_ground)
        
        if np.any(nan_mask):
            # DTM 未覆盖的点（通常在边界或地面点稀疏区域）
            # 使用 KNN 回退策略：查询最近的地面点
            from scipy.spatial import cKDTree
            
            tree = cKDTree(ground_xy)
            # 对 NaN 点查询最近的 3 个地面点
            k = min(3, len(ground_xy))
            nan_points = coord[nan_mask, :2]
            
            if k == 1:
                _, indices = tree.query(nan_points, k=1)
                z_ground[nan_mask] = ground_z[indices]
            else:
                distances, indices = tree.query(nan_points, k=k)
                # 距离加权平均
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                z_ground[nan_mask] = (ground_z[indices] * weights).sum(axis=1)
        
        # ===== 步骤 6: 计算归一化高程 =====
        h_norm = coord[:, 2] - z_ground
        
        return h_norm
    
    def _load_data(self, idx: int) -> Dict[str, Any]:
        """
        加载特定的数据样本
        
        重要：此方法返回的数据字典中各个特征（coord、intensity、color 等）是独立的，
        不会预先拼接成 feature。这样 transforms.py 中的数据增强才能正确处理各个特征。
        最终的 feature 拼接应该在 transforms 之后通过 Collect 变换完成。
        
        参数：
            idx: 要加载的样本索引
            
        返回：
            包含加载数据的字典，包括：
            - coord: [N, 3] 坐标（必需）
            - intensity: [N,] 强度值（如果在 assets 中）
            - color: [N, 3] RGB 颜色，范围 [0, 255]（如果在 assets 中）
            - echo: [N, 2] 回波信息，范围 [-1, 1]（如果在 assets 中）
                - 第 0 列：是否首次回波
                - 第 1 列：是否末次回波
            - normal: [N, 3] 法向量（如果在 assets 中）
            - h_norm: [N,] 高度归一化值（如果在 assets 中）
            - class: [N,] 分类标签（如果在 assets 中）
            - indices: [N,] 原始点索引（仅在 test/predict split 中）
        """
        sample_info = self.data_list[idx]
        
        # 获取路径
        bin_path = Path(sample_info['bin_path'])
        segment_id = sample_info['segment_id']
        
        # 加载 pkl 元数据（使用缓存避免重复磁盘 I/O）
        pkl_path = Path(sample_info['pkl_path'])
        pkl_key = str(pkl_path)
        
        if pkl_key not in self._metadata_cache:
            with open(pkl_path, 'rb') as f:
                self._metadata_cache[pkl_key] = pickle.load(f)
        
        metadata = self._metadata_cache[pkl_key]
        
        # 查找片段信息
        segment_info = None
        for seg in metadata['segments']:
            if seg['segment_id'] == segment_id:
                segment_info = seg
                break
        
        if segment_info is None:
            raise ValueError(f"在 {pkl_path} 中未找到片段 {segment_id}")
        
        # 使用 memmap 从 bin 文件加载点数据
        point_data = np.memmap(bin_path, dtype=metadata['dtype'], mode='r')
        
        # 使用离散索引提取片段点
        # 点云数据始终使用离散索引（非连续）
        if 'indices' not in segment_info:
            raise ValueError(f"片段信息必须包含 'indices' 字段")
        
        indices = segment_info['indices']
        segment_points = point_data[indices]
        
        # 提取请求的资产
        # 注意：不在这里拼接 feature，而是保持各个特征独立
        # 这样 transforms.py 中的数据增强可以分别处理 intensity、color 等
        # 最后通过 Collect 变换来拼接所有特征
        data = {}
        
        # 总是首先提取 coord
        coord = np.stack([
            segment_points['X'],
            segment_points['Y'],
            segment_points['Z']
        ], axis=1).astype(np.float32)
        data['coord'] = coord
        
        # 根据资产顺序提取其他特征（保持独立，不拼接）
        for asset in self.assets:
            if asset == 'coord':
                continue  # 已处理
                
            elif asset == 'intensity':
                # 提取原始强度值（保持原始位数，不归一化）
                # 归一化应在 transforms 中完成，如 AutoNormalizeIntensity
                intensity = segment_points['intensity'].astype(np.float32)
                data['intensity'] = intensity  # [N,]
                
            elif asset == 'color' and all(c in segment_points.dtype.names for c in ['red', 'green', 'blue']):
                # 提取原始 RGB 颜色值（保持原始位数，不归一化）
                # 归一化应在 transforms 中完成，如 AutoNormalizeColor
                color = np.stack([
                    segment_points['red'],
                    segment_points['green'],
                    segment_points['blue']
                ], axis=1).astype(np.float32)
                data['color'] = color  # [N, 3]

            elif asset == 'echo' and all(c in segment_points.dtype.names for c in ['return_number', 'number_of_returns']):
                # 提取回波信息
                is_first = (segment_points['return_number'] == 1).astype(np.float32)
                is_last = (segment_points['return_number'] == segment_points['number_of_returns']).astype(np.float32)
                # 转换为 [-1, 1] 范围：True -> 1, False -> -1
                is_first = is_first * 2.0 - 1.0
                is_last = is_last * 2.0 - 1.0
                echo = np.stack([is_first, is_last], axis=1)  # [N, 2]
                data['echo'] = echo  # [N, 2]

            elif asset == 'normal' and all(c in segment_points.dtype.names for c in ['normal_x', 'normal_y', 'normal_z']):
                # 提取法向量
                normal = np.stack([
                    segment_points['normal_x'],
                    segment_points['normal_y'],
                    segment_points['normal_z']
                ], axis=1).astype(np.float32)
                data['normal'] = normal  # [N, 3]

            elif asset == 'h_norm':
                # 计算归一化高程（地上高程）
                # 如果 bin 文件中已有预计算的 h_norm，直接使用
                if 'h_norm' in segment_points.dtype.names:
                    h_norm = segment_points['h_norm'].astype(np.float32)
                # 否则，基于 is_ground 字段动态计算
                elif 'is_ground' in segment_points.dtype.names:
                    h_norm = self._compute_h_norm(coord, segment_points['is_ground'], self.h_norm_grid)
                else:
                    raise ValueError("既没有 'h_norm' 也没有 'is_ground' 字段，无法计算归一化高程")
                data['h_norm'] = h_norm

            elif asset == 'class':
                # 单独存储分类标签为目标
                classification = segment_points['classification'].astype(np.int64)
                
                # 如果提供了类别映射则应用
                if self.class_mapping is not None:
                    # 🔥 新策略：不在 class_mapping 中的类别设为 ignore_label
                    # 这些点会参与网络前向传播（保持数据连续性），
                    # 但不参与损失计算和精度评估（通过 ignore_index 机制）
                    
                    # 初始化所有标签为 ignore_label
                    mapped_classification = np.full_like(classification, self.ignore_label, dtype=np.int64)
                    
                    # 只映射 class_mapping 中定义的类别
                    for original_label, new_label in self.class_mapping.items():
                        mask = (classification == original_label)
                        mapped_classification[mask] = new_label
                    
                    data['class'] = mapped_classification
                else:
                    data['class'] = classification

        # 在 test 和 predict 划分中，存储点索引用于投票机制
        if self.split in ['test', 'predict']:
            data['indices'] = indices.copy()  # 存储原始点索引
            
            # 🔥 新增：直接传递文件信息，避免在 callback 中推断
            # 这些信息在 tile.py 中已经保存到 segment_info 中
            data['bin_file'] = sample_info.get('bin_file', Path(sample_info['bin_path']).stem)
            data['bin_path'] = sample_info['bin_path']
            data['pkl_path'] = sample_info['pkl_path']
        
        return data
    
    def get_segment_info(self, idx: int) -> Dict[str, Any]:
        """
        获取特定片段的元数据
        
        参数：
            idx: 片段的索引
            
        返回：
            包含片段元数据的字典
        """
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.data_list)})")
        
        sample_info = self.data_list[idx]
        pkl_path = Path(sample_info['pkl_path'])
        
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 查找片段信息
        segment_id = sample_info['segment_id']
        for seg in metadata['segments']:
            if seg['segment_id'] == segment_id:
                return seg
        
        raise ValueError(f"未找到片段 {segment_id}")
    
    def get_file_metadata(self, idx: int) -> Dict[str, Any]:
        """
        获取包含特定片段的文件的元数据
        
        参数：
            idx: 片段的索引
            
        返回：
            包含文件级元数据的字典
        """
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.data_list)})")
        
        sample_info = self.data_list[idx]
        pkl_path = Path(sample_info['pkl_path'])
        
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 返回元数据，排除片段列表（可能很大）
        file_metadata = {k: v for k, v in metadata.items() if k != 'segments'}
        return file_metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        返回：
            包含数据集统计信息的字典
        """
        if len(self.data_list) == 0:
            return {}
        
        # 收集统计信息
        num_points_list = [s['num_points'] for s in self.data_list]
        
        stats = {
            'num_samples': len(self.data_list),
            'num_points': {
                'total': sum(num_points_list),
                'mean': np.mean(num_points_list),
                'median': np.median(num_points_list),
                'min': np.min(num_points_list),
                'max': np.max(num_points_list),
                'std': np.std(num_points_list),
            }
        }
        
        # 从第一个文件获取标签分布
        if len(self.data_list) > 0:
            pkl_path = Path(self.data_list[0]['pkl_path'])
            with open(pkl_path, 'rb') as f:
                metadata = pickle.load(f)
            
            if 'label_counts' in metadata:
                stats['label_distribution'] = metadata['label_counts']
        
        return stats
    
    def print_stats(self):
        """打印数据集统计信息"""
        stats = self.get_stats()
        
        print("="*70)
        print("数据集统计信息")
        print("="*70)
        print(f"划分: {self.split}")
        print(f"样本数: {stats['num_samples']:,}")
        print(f"\n每样本点数:")
        print(f"  - 总计: {stats['num_points']['total']:,}")
        print(f"  - 平均: {stats['num_points']['mean']:,.1f}")
        print(f"  - 中位数: {stats['num_points']['median']:,.0f}")
        print(f"  - 最小: {stats['num_points']['min']:,}")
        print(f"  - 最大: {stats['num_points']['max']:,}")
        print(f"  - 标准差: {stats['num_points']['std']:,.1f}")
        
        if 'label_distribution' in stats:
            print(f"\n标签分布（整体）:")
            for label, count in sorted(stats['label_distribution'].items()):
                print(f"  类别 {label}: {count:,}")
        
        print("="*70)
    
    def get_class_distribution(self) -> Optional[Dict[int, int]]:
        """
        获取数据集的类别分布
        
        返回：
            类别分布字典 {class_id: count}
        """
        if len(self.data_list) == 0:
            return {}
        
        # 从第一个 pkl 文件获取整体类别分布
        pkl_path = Path(self.data_list[0]['pkl_path'])
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        if 'label_counts' in metadata:
            # 如果有 class_mapping，转换类别标签
            if self.class_mapping is not None:
                mapped_counts = {}
                for original_label, count in metadata['label_counts'].items():
                    if original_label in self.class_mapping:
                        new_label = self.class_mapping[original_label]
                        mapped_counts[new_label] = mapped_counts.get(new_label, 0) + count
                return mapped_counts
            else:
                return dict(metadata['label_counts'])
        
        return {}
    
    def get_sample_weights(self, class_weights: Optional[Dict[int, float]] = None) -> Optional[np.ndarray]:
        """
        计算每个样本（segment）的权重
        
        权重计算策略：
        - 样本权重 = Σ(样本中包含的每个类别的类别权重)
        - 包含稀有类别的样本获得更高权重
        - 包含多个不同类别的样本获得更高权重
        
        参数：
            class_weights: 类别权重字典 {class_id: weight}
        
        返回：
            样本权重数组 [num_samples]
        """
        if class_weights is None or len(self.data_list) == 0:
            return None
        
        # 加载 pkl 元数据获取每个 segment 的类别信息
        sample_weights = []
        
        # 按 pkl 文件分组处理（避免重复加载）
        pkl_to_samples = {}
        for idx, sample_info in enumerate(self.data_list):
            pkl_path = sample_info['pkl_path']
            if pkl_path not in pkl_to_samples:
                pkl_to_samples[pkl_path] = []
            pkl_to_samples[pkl_path].append((idx, sample_info['segment_id']))
        
        # 为每个 pkl 文件计算其 segments 的权重
        weights_dict = {}
        for pkl_path, samples in pkl_to_samples.items():
            with open(pkl_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # 为每个 segment 计算权重
            for idx, segment_id in samples:
                segment_info = None
                for seg in metadata['segments']:
                    if seg['segment_id'] == segment_id:
                        segment_info = seg
                        break
                
                if segment_info is None or 'unique_labels' not in segment_info:
                    # 如果没有类别信息，使用默认权重 1.0
                    weights_dict[idx] = 1.0
                    continue
                
                # 计算权重：包含的所有类别的类别权重之和
                unique_labels = segment_info['unique_labels']
                segment_weight = 0.0
                
                for label in unique_labels:
                    # 如果有 class_mapping，先映射标签
                    if self.class_mapping is not None:
                        if label in self.class_mapping:
                            mapped_label = self.class_mapping[label]
                            segment_weight += class_weights.get(mapped_label, 0.0)
                    else:
                        segment_weight += class_weights.get(label, 0.0)
                
                weights_dict[idx] = max(segment_weight, 1e-6)  # 避免零权重
        
        # 按顺序构建权重数组
        sample_weights = np.array([weights_dict.get(i, 1.0) for i in range(len(self.data_list))], dtype=np.float32)
        
        return sample_weights


def create_dataset(
    data_root,
    split='train',
    assets=None,
    transform=None,
    ignore_label=-1,
    loop=1,
    cache_data=False,
    **kwargs
):
    """
    创建 BinPklDataset 的工厂函数
    
    参数：
        data_root: 根目录、单个 pkl 文件或 pkl 文件列表
        split: 数据集划分（'train'、'val'、'test'、'predict'）
        assets: 要加载的数据属性列表
        transform: 数据变换
        ignore_label: 要忽略的标签
        loop: 数据集循环因子
        cache_data: 是否缓存数据
        **kwargs: 其他参数
        
    返回：
        BinPklDataset 实例
    """
    return BinPklDataset(
        data_root=data_root,
        split=split,
        assets=assets,
        transform=transform,
        ignore_label=ignore_label,
        loop=loop,
        cache_data=cache_data,
        **kwargs
    )
