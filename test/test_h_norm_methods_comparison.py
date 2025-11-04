"""
对比不同 h_norm 计算方法的性能和精度

比较：
1. 简单最小值法
2. KNN 方法
3. TIN + Raster 混合法（新实现）
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def method_1_simple_min(coord, ground_mask):
    """方法 1: 简单最小值"""
    ground_z = coord[ground_mask, 2]
    z_min = ground_z.min()
    h_norm = coord[:, 2] - z_min
    return h_norm


def method_2_knn(coord, ground_mask, k=5):
    """方法 2: KNN 局部地面高程"""
    from scipy.spatial import cKDTree
    
    ground_points = coord[ground_mask]
    ground_xy = ground_points[:, :2]
    ground_z = ground_points[:, 2]
    
    tree = cKDTree(ground_xy)
    k = min(k, len(ground_points))
    distances, indices = tree.query(coord[:, :2], k=k)
    
    if k == 1:
        local_ground_z = ground_z[indices]
    else:
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        local_ground_z = (ground_z[indices] * weights).sum(axis=1)
    
    h_norm = coord[:, 2] - local_ground_z
    return h_norm


def method_3_tin_raster(coord, ground_mask, grid_resolution=0.5):
    """方法 3: TIN + Raster 混合法"""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    
    ground_points = coord[ground_mask]
    ground_xy = ground_points[:, :2]
    ground_z = ground_points[:, 2]
    
    # 定义栅格
    x_min, y_min = coord[:, :2].min(axis=0)
    x_max, y_max = coord[:, :2].max(axis=0)
    
    n_x = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
    n_y = int(np.ceil((y_max - y_min) / grid_resolution)) + 1
    
    # 限制栅格大小
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
    
    # TIN 插值生成 DTM
    dtm_grid = griddata(
        ground_xy,
        ground_z,
        (grid_xx, grid_yy),
        method='linear',
        fill_value=np.nan
    )
    
    # 栅格查询
    indices_x = ((coord[:, 0] - x_min) / grid_resolution).astype(int)
    indices_y = ((coord[:, 1] - y_min) / grid_resolution).astype(int)
    indices_x = np.clip(indices_x, 0, dtm_grid.shape[1] - 1)
    indices_y = np.clip(indices_y, 0, dtm_grid.shape[0] - 1)
    
    z_ground = dtm_grid[indices_y, indices_x]
    
    # 处理 NaN（KNN 回退）
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
    
    h_norm = coord[:, 2] - z_ground
    return h_norm, dtm_grid


def generate_synthetic_data(n_points=10000, terrain_type='hilly'):
    """
    生成合成点云数据用于测试
    
    参数：
        n_points: 点数
        terrain_type: 地形类型
            - 'flat': 平坦地形
            - 'hilly': 起伏地形
            - 'complex': 复杂地形（山丘、坡度）
    """
    np.random.seed(42)
    
    # 生成 XY 坐标（均匀分布）
    xy = np.random.rand(n_points, 2) * 100  # 100m x 100m 区域
    
    # 生成地形高程
    if terrain_type == 'flat':
        # 平坦地形 + 噪声
        z_ground = np.ones(n_points) * 10.0 + np.random.randn(n_points) * 0.1
    elif terrain_type == 'hilly':
        # 起伏地形（正弦波）
        z_ground = 10.0 + 3.0 * np.sin(xy[:, 0] / 10) + 2.0 * np.cos(xy[:, 1] / 8)
    else:  # complex
        # 复杂地形（多个频率叠加）
        z_ground = (10.0 + 
                   3.0 * np.sin(xy[:, 0] / 10) + 
                   2.0 * np.cos(xy[:, 1] / 8) +
                   1.5 * np.sin(xy[:, 0] / 5) * np.cos(xy[:, 1] / 5))
    
    # 添加地上物体（建筑、植被等）
    # 50% 地面点，50% 地上点
    is_ground = np.random.rand(n_points) < 0.5
    
    # 地上点的高度偏移
    h_offset = np.zeros(n_points)
    h_offset[~is_ground] = np.random.exponential(scale=5.0, size=(~is_ground).sum())  # 0-20m
    
    # 真实 Z 坐标 = 地面高程 + 地上高度
    z = z_ground + h_offset
    
    coord = np.column_stack([xy, z])
    
    # 真实的 h_norm（用于验证精度）
    true_h_norm = h_offset
    
    return coord, is_ground, true_h_norm


def benchmark_methods(n_points_list=[1000, 5000, 10000, 50000], 
                     ground_ratio_list=[0.05, 0.1, 0.2, 0.5],
                     terrain_type='hilly'):
    """
    性能基准测试
    """
    print("=" * 80)
    print("h_norm 计算方法对比基准测试")
    print("=" * 80)
    
    results = []
    
    for n_points in n_points_list:
        for ground_ratio in ground_ratio_list:
            print(f"\n测试配置: {n_points} 点, {ground_ratio*100:.0f}% 地面点, {terrain_type} 地形")
            print("-" * 80)
            
            # 生成数据
            coord, is_ground_full, true_h_norm = generate_synthetic_data(n_points, terrain_type)
            
            # 模拟指定的地面点比例
            n_ground_target = int(n_points * ground_ratio)
            ground_indices = np.where(is_ground_full)[0]
            if len(ground_indices) > n_ground_target:
                ground_indices = np.random.choice(ground_indices, n_ground_target, replace=False)
            is_ground = np.zeros(n_points, dtype=bool)
            is_ground[ground_indices] = True
            
            print(f"实际地面点数: {is_ground.sum()} ({is_ground.sum()/n_points*100:.1f}%)")
            
            # 测试方法 1: 简单最小值
            start = time.time()
            h_norm_1 = method_1_simple_min(coord, is_ground)
            time_1 = (time.time() - start) * 1000
            error_1 = np.abs(h_norm_1 - true_h_norm).mean()
            
            # 测试方法 2: KNN
            start = time.time()
            h_norm_2 = method_2_knn(coord, is_ground, k=5)
            time_2 = (time.time() - start) * 1000
            error_2 = np.abs(h_norm_2 - true_h_norm).mean()
            
            # 测试方法 3: TIN + Raster
            start = time.time()
            h_norm_3, dtm = method_3_tin_raster(coord, is_ground, grid_resolution=0.5)
            time_3 = (time.time() - start) * 1000
            error_3 = np.abs(h_norm_3 - true_h_norm).mean()
            
            # 打印结果
            print(f"\n{'方法':<25} {'时间 (ms)':<12} {'平均误差':<12} {'相对速度'}")
            print("-" * 80)
            print(f"{'1. 简单最小值':<25} {time_1:>8.2f}    {error_1:>8.4f}     {time_1/time_1:.2f}x (基准)")
            print(f"{'2. KNN (k=5)':<25} {time_2:>8.2f}    {error_2:>8.4f}     {time_2/time_1:.2f}x")
            print(f"{'3. TIN+Raster (0.5m)':<25} {time_3:>8.2f}    {error_3:>8.4f}     {time_3/time_1:.2f}x")
            
            # 速度对比（vs KNN）
            if time_3 < time_2:
                speedup = time_2 / time_3
                print(f"\n✅ TIN+Raster 比 KNN 快 {speedup:.1f}x")
            else:
                slowdown = time_3 / time_2
                print(f"\n⚠️  TIN+Raster 比 KNN 慢 {slowdown:.1f}x")
            
            # 精度对比
            if error_3 < error_2:
                accuracy_gain = (error_2 - error_3) / error_2 * 100
                print(f"✅ TIN+Raster 精度提升 {accuracy_gain:.1f}%")
            else:
                accuracy_loss = (error_3 - error_2) / error_2 * 100
                print(f"⚠️  TIN+Raster 精度下降 {accuracy_loss:.1f}%")
            
            results.append({
                'n_points': n_points,
                'ground_ratio': ground_ratio,
                'n_ground': is_ground.sum(),
                'time_simple': time_1,
                'time_knn': time_2,
                'time_tin_raster': time_3,
                'error_simple': error_1,
                'error_knn': error_2,
                'error_tin_raster': error_3,
            })
    
    return results


def visualize_comparison(coord, is_ground, h_norm_methods, dtm_grid=None):
    """
    可视化不同方法的结果对比
    """
    fig = plt.figure(figsize=(18, 12))
    
    n_methods = len(h_norm_methods)
    
    for i, (method_name, h_norm) in enumerate(h_norm_methods.items()):
        # 3D 视图
        ax = fig.add_subplot(2, n_methods, i + 1, projection='3d')
        scatter = ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2],
                           c=h_norm, cmap='jet', s=1, alpha=0.6,
                           vmin=0, vmax=np.percentile(h_norm, 95))
        ax.set_title(f'{method_name}\n(3D 视图)', fontsize=11, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax, label='h_norm', shrink=0.6, pad=0.1)
        
        # 俯视图
        ax2 = fig.add_subplot(2, n_methods, i + 1 + n_methods)
        scatter2 = ax2.scatter(coord[:, 0], coord[:, 1],
                             c=h_norm, cmap='jet', s=1, alpha=0.6,
                             vmin=0, vmax=np.percentile(h_norm, 95))
        
        # 叠加地面点
        ground_points = coord[is_ground]
        ax2.scatter(ground_points[:, 0], ground_points[:, 1],
                   c='red', marker='x', s=5, alpha=0.3, label='地面点')
        
        ax2.set_title(f'{method_name}\n(俯视图 + 地面点)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        ax2.legend(loc='upper right', fontsize=8)
        plt.colorbar(scatter2, ax=ax2, label='h_norm', shrink=0.8, pad=0.02)
    
    plt.suptitle('不同 h_norm 计算方法对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 如果有 DTM，单独可视化
    if dtm_grid is not None:
        fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(dtm_grid, cmap='terrain', origin='lower', aspect='auto')
        ax.set_title('DTM 栅格（TIN 插值）', fontsize=14, fontweight='bold')
        ax.set_xlabel('X 栅格索引')
        ax.set_ylabel('Y 栅格索引')
        plt.colorbar(im, ax=ax, label='地面高程 (Z)')
        
        # 标记 NaN 区域
        nan_mask = np.isnan(dtm_grid)
        if np.any(nan_mask):
            ax.contour(nan_mask, levels=[0.5], colors='red', linewidths=2)
            ax.text(0.02, 0.98, f'NaN 区域: {nan_mask.sum()/nan_mask.size*100:.1f}%',
                   transform=ax.transAxes, color='red', fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='对比 h_norm 计算方法')
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')
    parser.add_argument('--visualize', action='store_true', help='可视化对比')
    parser.add_argument('--n-points', type=int, default=10000, help='点数（用于可视化）')
    parser.add_argument('--ground-ratio', type=float, default=0.2, help='地面点比例')
    parser.add_argument('--terrain', type=str, default='hilly', 
                       choices=['flat', 'hilly', 'complex'], help='地形类型')
    
    args = parser.parse_args()
    
    if args.benchmark:
        # 性能基准测试
        results = benchmark_methods(
            n_points_list=[1000, 5000, 10000, 50000],
            ground_ratio_list=[0.05, 0.1, 0.2, 0.5],
            terrain_type=args.terrain
        )
        
        print("\n" + "=" * 80)
        print("基准测试完成！")
        print("=" * 80)
        
        # 总结
        print("\n总结:")
        for result in results:
            if result['n_ground'] >= 50:  # TIN+Raster 适用范围
                speedup = result['time_knn'] / result['time_tin_raster']
                if speedup > 1.0:
                    print(f"  {result['n_points']:>6} 点, {result['ground_ratio']*100:>4.0f}% 地面点: "
                          f"TIN+Raster 快 {speedup:.1f}x")
    
    elif args.visualize:
        # 可视化对比
        print(f"生成合成数据: {args.n_points} 点, {args.ground_ratio*100:.0f}% 地面点, {args.terrain} 地形")
        coord, is_ground_full, true_h_norm = generate_synthetic_data(args.n_points, args.terrain)
        
        # 模拟指定的地面点比例
        n_ground_target = int(args.n_points * args.ground_ratio)
        ground_indices = np.where(is_ground_full)[0]
        if len(ground_indices) > n_ground_target:
            ground_indices = np.random.choice(ground_indices, n_ground_target, replace=False)
        is_ground = np.zeros(args.n_points, dtype=bool)
        is_ground[ground_indices] = True
        
        print(f"计算 h_norm...")
        h_norm_1 = method_1_simple_min(coord, is_ground)
        h_norm_2 = method_2_knn(coord, is_ground, k=5)
        h_norm_3, dtm = method_3_tin_raster(coord, is_ground, grid_resolution=0.5)
        
        h_norm_methods = {
            '真实值': true_h_norm,
            '简单最小值': h_norm_1,
            'KNN (k=5)': h_norm_2,
            'TIN+Raster (0.5m)': h_norm_3,
        }
        
        print(f"生成可视化...")
        visualize_comparison(coord, is_ground, h_norm_methods, dtm)
    
    else:
        # 快速演示
        print("快速演示: 对比三种方法")
        print("使用 --benchmark 运行完整基准测试")
        print("使用 --visualize 查看可视化对比")
        print()
        
        coord, is_ground, true_h_norm = generate_synthetic_data(10000, 'hilly')
        
        # 只保留 20% 地面点（模拟稀疏地面点）
        ground_indices = np.where(is_ground)[0]
        ground_indices = np.random.choice(ground_indices, int(len(ground_indices) * 0.2), replace=False)
        is_ground_sparse = np.zeros(len(coord), dtype=bool)
        is_ground_sparse[ground_indices] = True
        
        print(f"测试数据: 10,000 点, {is_ground_sparse.sum()} 地面点")
        print("-" * 60)
        
        # 方法 1
        start = time.time()
        h_norm_1 = method_1_simple_min(coord, is_ground_sparse)
        time_1 = (time.time() - start) * 1000
        error_1 = np.abs(h_norm_1 - true_h_norm).mean()
        
        # 方法 2
        start = time.time()
        h_norm_2 = method_2_knn(coord, is_ground_sparse, k=5)
        time_2 = (time.time() - start) * 1000
        error_2 = np.abs(h_norm_2 - true_h_norm).mean()
        
        # 方法 3
        start = time.time()
        h_norm_3, _ = method_3_tin_raster(coord, is_ground_sparse, grid_resolution=0.5)
        time_3 = (time.time() - start) * 1000
        error_3 = np.abs(h_norm_3 - true_h_norm).mean()
        
        print(f"\n{'方法':<25} {'时间':<12} {'平均误差'}")
        print("-" * 60)
        print(f"{'简单最小值':<25} {time_1:>8.2f} ms {error_1:>8.4f}")
        print(f"{'KNN (k=5)':<25} {time_2:>8.2f} ms {error_2:>8.4f}")
        print(f"{'TIN+Raster (0.5m)':<25} {time_3:>8.2f} ms {error_3:>8.4f}")
        
        speedup = time_2 / time_3
        print(f"\n{'✅ TIN+Raster 比 KNN 快 ' + f'{speedup:.1f}x'}")
