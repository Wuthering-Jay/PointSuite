"""
快速H5格式数据读取示例 - 全面指南

演示如何读取快速H5格式中的所有数据：
1. Header信息（元数据）
2. Segment数据（点云、分类、强度等）
3. 单个segment读取
4. 批量segment读取
5. 使用Dataset类训练
6. 多文件读取
"""

import h5py
import numpy as np
from pathlib import Path
import time


def example_1_read_header_info(h5_path: str):
    """
    示例1：读取H5文件的header信息
    
    Header包含：
    - 点云元数据（scale, offset, 坐标范围等）
    - LAS格式信息
    - 可用字段列表
    """
    print("="*80)
    print("示例1：读取Header信息")
    print("="*80)
    
    with h5py.File(h5_path, 'r') as f:
        header = f['header']
        
        # 基本元数据
        print("\n【基本元数据】")
        print(f"  总点数: {header.attrs['num_points']:,}")
        print(f"  Point Format: {header.attrs.get('point_format', 'N/A')}")
        print(f"  LAS版本: {header.attrs.get('version_major', 1)}.{header.attrs.get('version_minor', 2)}")
        
        # Scale和Offset（用于坐标转换）
        print("\n【坐标参数】")
        print(f"  X scale/offset: {header.attrs['x_scale']} / {header.attrs['x_offset']}")
        print(f"  Y scale/offset: {header.attrs['y_scale']} / {header.attrs['y_offset']}")
        print(f"  Z scale/offset: {header.attrs['z_scale']} / {header.attrs['z_offset']}")
        
        # CRS信息（坐标系统）
        if 'crs' in header.attrs:
            print(f"\n【坐标系统】")
            print(f"  CRS: {header.attrs['crs']}")
        
        # 可用字段
        if 'available_fields' in header.attrs:
            fields_str = header.attrs['available_fields']
            if isinstance(fields_str, bytes):
                fields_str = fields_str.decode('utf-8')
            fields = fields_str.split(',')
            print(f"\n【可用字段】({len(fields)}个)")
            for i, field in enumerate(fields, 1):
                print(f"  {i:2d}. {field}")
        
        # Segment信息
        print("\n【Segment信息】")
        num_segments = f['segments'].attrs['num_segments']
        print(f"  总segments数: {num_segments}")


def example_2_read_single_segment(h5_path: str, segment_idx: int = 0):
    """
    示例2：读取单个segment的所有数据
    
    每个segment包含：
    - 坐标 (x, y, z)
    - 分类 (classification)
    - 其他字段（intensity, gps_time等）
    """
    print("\n" + "="*80)
    print(f"示例2：读取单个Segment（segment_{segment_idx:04d}）")
    print("="*80)
    
    start = time.time()
    
    with h5py.File(h5_path, 'r') as f:
        seg = f['segments'][f'segment_{segment_idx:04d}']
        
        # 读取坐标（必需字段）
        print("\n【坐标数据】")
        x = seg['x'][:]
        y = seg['y'][:]
        z = seg['z'][:]
        print(f"  点数: {len(x):,}")
        print(f"  X范围: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  Y范围: [{y.min():.2f}, {y.max():.2f}]")
        print(f"  Z范围: [{z.min():.2f}, {z.max():.2f}]")
        
        # 组合为Nx3数组（常用于训练）
        xyz = np.stack([x, y, z], axis=1)
        print(f"  XYZ shape: {xyz.shape}")
        
        # 读取分类（必需字段）
        print("\n【分类数据】")
        classification = seg['classification'][:]
        unique_labels, counts = np.unique(classification, return_counts=True)
        print(f"  唯一类别: {list(unique_labels)}")
        print(f"  类别分布:")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(classification) * 100
            print(f"    类别 {label}: {count:6,} 点 ({percentage:5.2f}%)")
        
        # 读取其他可用字段
        print("\n【其他字段】")
        optional_fields = {
            'intensity': '强度',
            'return_number': '回波编号',
            'number_of_returns': '回波总数',
            'gps_time': 'GPS时间',
            'scan_angle_rank': '扫描角度',
            'point_source_id': '点源ID',
            'user_data': '用户数据',
            'red': '红色通道',
            'green': '绿色通道',
            'blue': '蓝色通道'
        }
        
        for field, description in optional_fields.items():
            if field in seg:
                data = seg[field][:]
                print(f"  {field} ({description}):")
                print(f"    - 范围: [{data.min()}, {data.max()}]")
                print(f"    - 类型: {data.dtype}")
        
        # Metadata
        if 'num_points' in seg.attrs:
            print(f"\n【元数据】")
            print(f"  num_points: {seg.attrs['num_points']}")
    
    elapsed = time.time() - start
    print(f"\n⏱️  读取耗时: {elapsed*1000:.2f} ms")


def example_3_read_multiple_segments(h5_path: str, num_segments: int = 10):
    """
    示例3：批量读取多个segments
    
    展示：
    - 循环读取
    - 性能测试
    - 数据统计
    """
    print("\n" + "="*80)
    print(f"示例3：批量读取{num_segments}个Segments")
    print("="*80)
    
    start = time.time()
    
    all_xyz = []
    all_labels = []
    total_points = 0
    
    with h5py.File(h5_path, 'r') as f:
        max_segments = f['segments'].attrs['num_segments']
        num_to_read = min(num_segments, max_segments)
        
        print(f"\n读取前{num_to_read}个segments...")
        
        for i in range(num_to_read):
            seg = f['segments'][f'segment_{i:04d}']
            
            # 读取xyz
            xyz = np.stack([
                seg['x'][:],
                seg['y'][:],
                seg['z'][:]
            ], axis=1)
            
            # 读取标签
            labels = seg['classification'][:]
            
            all_xyz.append(xyz)
            all_labels.append(labels)
            total_points += len(xyz)
        
    elapsed = time.time() - start
    
    print(f"\n【统计结果】")
    print(f"  读取segments: {len(all_xyz)}")
    print(f"  总点数: {total_points:,}")
    print(f"  平均点数/segment: {total_points/len(all_xyz):.0f}")
    print(f"  总耗时: {elapsed:.3f}秒")
    print(f"  速度: {len(all_xyz)/elapsed:.2f} segments/秒")
    print(f"  平均: {elapsed*1000/len(all_xyz):.2f} ms/segment")
    
    return all_xyz, all_labels


def example_4_efficient_reading_patterns(h5_path: str):
    """
    示例4：高效读取模式
    
    对比：
    - 按需读取（逐个打开文件）
    - 一次性读取（文件保持打开）
    - 预加载到内存
    """
    print("\n" + "="*80)
    print("示例4：不同读取模式性能对比")
    print("="*80)
    
    with h5py.File(h5_path, 'r') as f:
        num_segments = min(50, f['segments'].attrs['num_segments'])
    
    # 模式1：每次都打开关闭文件（慢）
    print("\n【模式1：反复打开文件】")
    start = time.time()
    for i in range(num_segments):
        with h5py.File(h5_path, 'r') as f:
            seg = f['segments'][f'segment_{i:04d}']
            xyz = np.stack([seg['x'][:], seg['y'][:], seg['z'][:]], axis=1)
    elapsed_1 = time.time() - start
    print(f"  耗时: {elapsed_1:.3f}秒 ({num_segments/elapsed_1:.2f} seg/s)")
    
    # 模式2：文件保持打开（快）
    print("\n【模式2：文件保持打开】")
    start = time.time()
    with h5py.File(h5_path, 'r') as f:
        for i in range(num_segments):
            seg = f['segments'][f'segment_{i:04d}']
            xyz = np.stack([seg['x'][:], seg['y'][:], seg['z'][:]], axis=1)
    elapsed_2 = time.time() - start
    print(f"  耗时: {elapsed_2:.3f}秒 ({num_segments/elapsed_2:.2f} seg/s)")
    print(f"  提升: {elapsed_1/elapsed_2:.2f}x")
    
    # 模式3：预加载到内存（最快）
    print("\n【模式3：预加载到内存】")
    start = time.time()
    
    # 预加载阶段
    cache = []
    with h5py.File(h5_path, 'r') as f:
        for i in range(num_segments):
            seg = f['segments'][f'segment_{i:04d}']
            xyz = np.stack([seg['x'][:], seg['y'][:], seg['z'][:]], axis=1)
            labels = seg['classification'][:]
            cache.append((xyz, labels))
    
    preload_time = time.time() - start
    
    # 访问阶段
    start = time.time()
    for xyz, labels in cache:
        pass  # 直接从内存读取
    access_time = time.time() - start
    
    # 避免除以零
    if access_time < 0.001:
        access_time = 0.001
    
    print(f"  预加载: {preload_time:.3f}秒 ({num_segments/preload_time:.2f} seg/s)")
    print(f"  访问: {access_time:.3f}秒 ({num_segments/access_time:.2f} seg/s)")
    print(f"  总提升: {elapsed_1/(preload_time+access_time):.2f}x")
    
    print("\n💡 建议：训练时使用模式3（预加载），推理时使用模式2")


def example_5_use_with_dataset(h5_path: str):
    """
    示例5：使用Dataset类（推荐用于训练）
    
    展示：
    - FastH5Dataset基本使用
    - DataLoader集成
    - 数据增强
    """
    print("\n" + "="*80)
    print("示例5：使用Dataset类进行训练")
    print("="*80)
    
    # 需要导入
    try:
        from h5_dataset_fast import FastH5Dataset, collate_fn
        from torch.utils.data import DataLoader
        
        print("\n【创建Dataset】")
        
        # 方式1：按需加载
        print("\n1. 按需加载模式:")
        dataset = FastH5Dataset(h5_path, preload=False)
        print(f"   - Segments数: {len(dataset)}")
        
        # 读取单个样本
        xyz, labels = dataset[0]
        print(f"   - 样本0: xyz={xyz.shape}, labels={labels.shape}")
        
        # 方式2：预加载（推荐）
        print("\n2. 预加载模式:")
        dataset_preload = FastH5Dataset(h5_path, preload=True)
        
        # 创建DataLoader
        print("\n【创建DataLoader】")
        dataloader = DataLoader(
            dataset_preload,
            batch_size=8,
            shuffle=True,
            num_workers=0,  # 预加载用0
            collate_fn=collate_fn
        )
        
        print(f"   - Batch size: 8")
        print(f"   - Total batches: {len(dataloader)}")
        
        # 迭代几个batch
        print("\n【迭代数据】")
        start = time.time()
        for i, (batch_xyz, batch_labels) in enumerate(dataloader):
            if i >= 5:
                break
            print(f"   Batch {i}: {len(batch_xyz)} segments")
            for j, (xyz, labels) in enumerate(zip(batch_xyz, batch_labels)):
                print(f"     - Segment {j}: {xyz.shape}, labels={labels.shape}")
        
        elapsed = time.time() - start
        print(f"\n   ⏱️  5个batch耗时: {elapsed:.3f}秒")
        
    except ImportError as e:
        print(f"\n❌ 需要安装PyTorch和h5_dataset_fast.py")
        print(f"   错误: {e}")


def example_6_multi_file_reading(h5_dir: str, max_files: int = 3):
    """
    示例6：多文件读取
    
    展示：
    - 多个H5文件的管理
    - 全局索引映射
    - 跨文件读取
    """
    print("\n" + "="*80)
    print("示例6：多文件读取")
    print("="*80)
    
    # 查找H5文件
    h5_dir = Path(h5_dir)
    h5_files = sorted(h5_dir.glob("*.h5"))[:max_files]
    
    if not h5_files:
        print(f"❌ 未找到H5文件: {h5_dir}")
        return
    
    print(f"\n找到 {len(h5_files)} 个H5文件:")
    for i, f in enumerate(h5_files):
        print(f"  {i}. {f.name}")
    
    # 方式1：手动管理多文件
    print("\n【方式1：手动管理】")
    file_segment_map = []
    total_segments = 0
    
    for file_idx, h5_file in enumerate(h5_files):
        with h5py.File(h5_file, 'r') as f:
            num_segs = f['segments'].attrs['num_segments']
            for seg_idx in range(num_segs):
                file_segment_map.append((file_idx, seg_idx))
            total_segments += num_segs
            print(f"  文件 {file_idx}: {num_segs} segments")
    
    print(f"\n  总segments: {total_segments}")
    
    # 随机读取示例
    print("\n  随机读取3个segments:")
    import random
    for global_idx in random.sample(range(total_segments), 3):
        file_idx, seg_idx = file_segment_map[global_idx]
        with h5py.File(h5_files[file_idx], 'r') as f:
            seg = f['segments'][f'segment_{seg_idx:04d}']
            num_points = len(seg['x'])
        print(f"    全局索引{global_idx} -> 文件{file_idx}, segment{seg_idx}, {num_points}点")
    
    # 方式2：使用FastMultiH5Dataset（推荐）
    print("\n【方式2：使用FastMultiH5Dataset（推荐）】")
    try:
        from h5_dataset_fast import FastMultiH5Dataset
        
        dataset = FastMultiH5Dataset(
            [str(f) for f in h5_files],
            preload_strategy="none"
        )
        
        print(f"  总segments: {len(dataset)}")
        print(f"  随机读取示例:")
        
        xyz, labels = dataset[0]
        print(f"    样本0: {xyz.shape}")
        
        xyz, labels = dataset[total_segments // 2]
        print(f"    样本{total_segments // 2}: {xyz.shape}")
        
    except ImportError:
        print("  ⚠️  需要h5_dataset_fast.py")


def example_7_advanced_operations(h5_path: str):
    """
    示例7：高级操作
    
    展示：
    - 统计分析
    - 空间查询
    - 数据筛选
    """
    print("\n" + "="*80)
    print("示例7：高级操作")
    print("="*80)
    
    with h5py.File(h5_path, 'r') as f:
        num_segments = f['segments'].attrs['num_segments']
        
        # 统计所有segments的点数分布
        print("\n【点数统计】")
        point_counts = []
        for i in range(num_segments):
            seg = f['segments'][f'segment_{i:04d}']
            point_counts.append(len(seg['x']))
        
        point_counts = np.array(point_counts)
        print(f"  最小: {point_counts.min():,} 点")
        print(f"  最大: {point_counts.max():,} 点")
        print(f"  平均: {point_counts.mean():.0f} 点")
        print(f"  中位数: {np.median(point_counts):.0f} 点")
        
        # 统计类别分布（所有segments）
        print("\n【全局类别分布】")
        all_labels = []
        for i in range(min(20, num_segments)):  # 采样前20个
            seg = f['segments'][f'segment_{i:04d}']
            all_labels.append(seg['classification'][:])
        
        all_labels = np.concatenate(all_labels)
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        
        print(f"  采样点数: {len(all_labels):,}")
        print(f"  类别分布:")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(all_labels) * 100
            print(f"    类别 {label}: {count:8,} ({percentage:5.2f}%)")
        
        # 查找特定条件的segments
        print("\n【条件筛选】")
        large_segments = [i for i, count in enumerate(point_counts) if count > 50000]
        print(f"  大segments (>50k点): {len(large_segments)}个")
        if large_segments:
            print(f"    索引: {large_segments[:5]}{'...' if len(large_segments) > 5 else ''}")


def main():
    """主函数：运行所有示例"""
    
    # 配置H5文件路径
    h5_path = r"E:\data\云南遥感中心\第一批\h5_fast\train\processed_02.h5"
    h5_dir = r"E:\data\云南遥感中心\第一批\h5_fast\train"
    
    # 检查文件存在
    if not Path(h5_path).exists():
        print(f"❌ 文件不存在: {h5_path}")
        print("\n请修改main()函数中的h5_path变量")
        return
    
    print("\n" + "🚀 "*40)
    print("快速H5格式数据读取 - 全面示例".center(80))
    print("🚀 "*40)
    
    # 运行所有示例
    example_1_read_header_info(h5_path)
    example_2_read_single_segment(h5_path, segment_idx=0)
    example_3_read_multiple_segments(h5_path, num_segments=10)
    example_4_efficient_reading_patterns(h5_path)
    example_5_use_with_dataset(h5_path)
    example_6_multi_file_reading(h5_dir, max_files=3)
    example_7_advanced_operations(h5_path)
    
    print("\n" + "="*80)
    print("✅ 所有示例运行完成！")
    print("="*80)
    print("\n💡 提示:")
    print("  - 训练时使用 FastH5Dataset(preload=True) + DataLoader")
    print("  - 多文件用 FastMultiH5Dataset(preload_strategy='all')")
    print("  - 记得设置 num_workers=0（预加载模式）")
    print("="*80)


if __name__ == "__main__":
    main()
