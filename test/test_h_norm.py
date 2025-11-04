"""
测试和验证 h_norm（归一化高程）计算

此脚本用于：
1. 验证 h_norm 计算的正确性
2. 可视化地面点分布和 h_norm 结果
3. 性能测试
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pointsuite.data import BinPklDataModule


def test_h_norm_computation(data_path, assets=['coord', 'h_norm', 'class']):
    """
    测试 h_norm 计算
    
    参数：
        data_path: pkl 文件路径或包含 pkl 文件的目录
        assets: 要加载的资产列表
    """
    print("=" * 60)
    print("h_norm 计算测试")
    print("=" * 60)
    
    # 创建 DataModule
    print(f"\n1. 加载数据集: {data_path}")
    datamodule = BinPklDataModule(
        data_root=data_path,
        assets=assets,
        cache_data=False,  # 不缓存，测试每次计算的性能
        transform=None,  # 不使用变换，直接测试原始数据
    )
    
    datamodule.setup()
    dataset = datamodule.train_dataset
    print(f"   数据集大小: {len(dataset)} 个片段")
    
    # 测试单个样本
    print(f"\n2. 加载第一个样本")
    start_time = time.time()
    sample = dataset[0]
    load_time = (time.time() - start_time) * 1000
    print(f"   加载时间: {load_time:.2f} ms")
    
    # 检查数据结构
    print(f"\n3. 数据结构检查")
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
    
    # h_norm 统计
    if 'h_norm' in sample:
        h_norm = sample['h_norm']
        coord = sample['coord']
        
        print(f"\n4. h_norm 统计信息")
        print(f"   点数: {len(h_norm)}")
        print(f"   最小值: {h_norm.min():.3f}")
        print(f"   最大值: {h_norm.max():.3f}")
        print(f"   均值: {h_norm.mean():.3f}")
        print(f"   中位数: {np.median(h_norm):.3f}")
        print(f"   标准差: {h_norm.std():.3f}")
        
        # 分布统计
        print(f"\n5. h_norm 分布")
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        values = np.percentile(h_norm, percentiles)
        for p, v in zip(percentiles, values):
            print(f"   {p:3d}%: {v:7.3f}")
        
        # 负值检查（理论上不应该有负值，除非地面点识别有误）
        negative_ratio = (h_norm < 0).sum() / len(h_norm)
        if negative_ratio > 0:
            print(f"\n   ⚠️  警告: {negative_ratio*100:.2f}% 的点 h_norm < 0")
            print(f"       这可能表示地面点识别不准确")
        
        # Z 坐标范围
        z = coord[:, 2]
        print(f"\n6. Z 坐标信息（对比）")
        print(f"   Z 最小值: {z.min():.3f}")
        print(f"   Z 最大值: {z.max():.3f}")
        print(f"   Z 范围: {z.max() - z.min():.3f}")
        print(f"   h_norm 范围: {h_norm.max() - h_norm.min():.3f}")
        
    else:
        print(f"\n   ⚠️  样本中没有 h_norm 字段")
        return
    
    # 性能测试
    print(f"\n7. 性能测试（加载前 10 个样本）")
    times = []
    for i in range(min(10, len(dataset))):
        start = time.time()
        _ = dataset[i]
        times.append((time.time() - start) * 1000)
    
    print(f"   平均加载时间: {np.mean(times):.2f} ms")
    print(f"   最小加载时间: {np.min(times):.2f} ms")
    print(f"   最大加载时间: {np.max(times):.2f} ms")
    
    return sample


def visualize_h_norm(sample, save_path=None):
    """
    可视化 h_norm
    
    参数：
        sample: 数据样本字典
        save_path: 保存图像的路径（可选）
    """
    if 'coord' not in sample or 'h_norm' not in sample:
        print("样本缺少 coord 或 h_norm 字段")
        return
    
    coord = sample['coord']
    h_norm = sample['h_norm']
    
    print(f"\n8. 生成可视化")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 原始 Z 坐标的 3D 视图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(coord[:, 0], coord[:, 1], coord[:, 2],
                          c=coord[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax1.set_title('原始点云 (按 Z 坐标着色)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, label='Z 坐标', pad=0.1)
    
    # 2. h_norm 的 3D 视图
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter2 = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2],
                          c=h_norm, cmap='jet', s=1, alpha=0.6,
                          vmin=0, vmax=np.percentile(h_norm, 95))
    ax2.set_title('点云 (按 h_norm 着色)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='h_norm (地上高程)', pad=0.1)
    
    # 3. h_norm 俯视图
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(coord[:, 0], coord[:, 1],
                          c=h_norm, cmap='jet', s=1, alpha=0.6,
                          vmin=0, vmax=np.percentile(h_norm, 95))
    ax3.set_title('俯视图 (按 h_norm 着色)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3, label='h_norm', pad=0.02)
    
    # 4. h_norm 直方图
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(h_norm, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(h_norm.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {h_norm.mean():.2f}')
    ax4.axvline(np.median(h_norm), color='green', linestyle='--', linewidth=2, label=f'中位数: {np.median(h_norm):.2f}')
    ax4.set_title('h_norm 分布', fontsize=12, fontweight='bold')
    ax4.set_xlabel('h_norm (地上高程)')
    ax4.set_ylabel('点数')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. h_norm vs Z 散点图
    ax5 = fig.add_subplot(2, 3, 5)
    # 采样以避免绘制太多点
    n_sample = min(10000, len(coord))
    idx_sample = np.random.choice(len(coord), n_sample, replace=False)
    ax5.scatter(coord[idx_sample, 2], h_norm[idx_sample], 
               s=1, alpha=0.3, color='blue')
    ax5.set_title('h_norm vs Z 坐标', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Z 坐标')
    ax5.set_ylabel('h_norm (地上高程)')
    ax5.grid(alpha=0.3)
    
    # 添加 y=x-min(z) 参考线
    z_min = coord[:, 2].min()
    z_range = [coord[:, 2].min(), coord[:, 2].max()]
    ax5.plot(z_range, [z_range[0]-z_min, z_range[1]-z_min], 
            'r--', linewidth=2, label='简单方法 (Z - Z_min)')
    ax5.legend()
    
    # 6. 分类信息（如果有）
    ax6 = fig.add_subplot(2, 3, 6)
    if 'class' in sample:
        classification = sample['class']
        unique_classes = np.unique(classification)
        
        # 计算每个类别的 h_norm 统计
        class_stats = []
        for cls in unique_classes:
            mask = classification == cls
            class_stats.append({
                'class': int(cls),
                'count': mask.sum(),
                'mean_h': h_norm[mask].mean(),
                'std_h': h_norm[mask].std()
            })
        
        # 绘制柱状图
        classes = [s['class'] for s in class_stats]
        means = [s['mean_h'] for s in class_stats]
        stds = [s['std_h'] for s in class_stats]
        counts = [s['count'] for s in class_stats]
        
        bars = ax6.bar(classes, means, yerr=stds, capsize=5, alpha=0.7)
        
        # 在柱子上标注点数
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}',
                    ha='center', va='bottom', fontsize=8)
        
        ax6.set_title('各类别的平均 h_norm', fontsize=12, fontweight='bold')
        ax6.set_xlabel('类别')
        ax6.set_ylabel('平均 h_norm ± std')
        ax6.grid(axis='y', alpha=0.3)
    else:
        ax6.text(0.5, 0.5, '无分类信息', ha='center', va='center',
                fontsize=14, transform=ax6.transAxes)
        ax6.axis('off')
    
    plt.suptitle('h_norm (归一化高程) 分析', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   图像已保存到: {save_path}")
    
    plt.show()


def compare_methods(coord, is_ground):
    """
    比较不同 h_norm 计算方法的结果
    
    参数：
        coord: [N, 3] 点云坐标
        is_ground: [N,] 地面点标记
    """
    from scipy.spatial import cKDTree
    from scipy.interpolate import LinearNDInterpolator
    
    print(f"\n9. 比较不同计算方法")
    
    ground_mask = (is_ground == 1)
    ground_points = coord[ground_mask]
    ground_xy = ground_points[:, :2]
    ground_z = ground_points[:, 2]
    
    print(f"   地面点数: {len(ground_points)} / {len(coord)} ({len(ground_points)/len(coord)*100:.1f}%)")
    
    # 方法 1: 简单最小值
    start = time.time()
    h_norm_simple = coord[:, 2] - ground_z.min()
    time_simple = (time.time() - start) * 1000
    
    # 方法 2: KNN (k=5)
    start = time.time()
    tree = cKDTree(ground_xy)
    k = min(5, len(ground_points))
    distances, indices = tree.query(coord[:, :2], k=k)
    if k == 1:
        local_ground_z = ground_z[indices]
    else:
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        local_ground_z = (ground_z[indices] * weights).sum(axis=1)
    h_norm_knn = coord[:, 2] - local_ground_z
    time_knn = (time.time() - start) * 1000
    
    # 方法 3: 线性插值（如果地面点足够多）
    if len(ground_points) >= 100:
        start = time.time()
        interpolator = LinearNDInterpolator(ground_xy, ground_z, fill_value=ground_z.min())
        local_ground_z_interp = interpolator(coord[:, :2])
        nan_mask = np.isnan(local_ground_z_interp)
        if np.any(nan_mask):
            _, indices_nan = tree.query(coord[nan_mask, :2], k=1)
            local_ground_z_interp[nan_mask] = ground_z[indices_nan]
        h_norm_interp = coord[:, 2] - local_ground_z_interp
        time_interp = (time.time() - start) * 1000
    else:
        h_norm_interp = None
        time_interp = 0
    
    # 结果比较
    print(f"\n   方法对比:")
    print(f"   {'方法':<20} {'计算时间':<12} {'均值':<10} {'标准差':<10} {'最小值':<10}")
    print(f"   {'-'*60}")
    print(f"   {'简单最小值':<20} {time_simple:>8.2f} ms {h_norm_simple.mean():>8.3f} {h_norm_simple.std():>8.3f} {h_norm_simple.min():>8.3f}")
    print(f"   {'KNN (k=5)':<20} {time_knn:>8.2f} ms {h_norm_knn.mean():>8.3f} {h_norm_knn.std():>8.3f} {h_norm_knn.min():>8.3f}")
    if h_norm_interp is not None:
        print(f"   {'线性插值':<20} {time_interp:>8.2f} ms {h_norm_interp.mean():>8.3f} {h_norm_interp.std():>8.3f} {h_norm_interp.min():>8.3f}")
    
    # 方法差异
    if h_norm_interp is not None:
        diff_knn_simple = np.abs(h_norm_knn - h_norm_simple).mean()
        diff_interp_knn = np.abs(h_norm_interp - h_norm_knn).mean()
        print(f"\n   平均差异:")
        print(f"   KNN vs 简单方法: {diff_knn_simple:.3f}")
        print(f"   插值 vs KNN: {diff_interp_knn:.3f}")
        
        if diff_interp_knn < 0.1:
            print(f"   ✓ KNN 和插值结果非常接近，KNN 更快，推荐使用")
        elif diff_knn_simple < 0.1:
            print(f"   ✓ 地形平坦，简单方法即可")
        else:
            print(f"   ✓ 地形复杂，建议使用插值方法")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试 h_norm 计算')
    parser.add_argument('data_path', type=str, help='pkl 文件路径或包含 pkl 文件的目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化')
    parser.add_argument('--save', type=str, default=None, help='保存可视化图像的路径')
    parser.add_argument('--compare', action='store_true', help='比较不同计算方法')
    
    args = parser.parse_args()
    
    try:
        # 测试 h_norm 计算
        sample = test_h_norm_computation(args.data_path)
        
        # 可视化
        if args.visualize or args.save:
            visualize_h_norm(sample, save_path=args.save)
        
        # 比较方法（需要访问原始 is_ground 数据）
        if args.compare and 'coord' in sample:
            print("\n⚠️  比较功能需要访问原始 is_ground 字段")
            print("   这需要修改代码直接加载 bin 文件")
            # compare_methods(sample['coord'], is_ground)  # 需要 is_ground 数据
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
