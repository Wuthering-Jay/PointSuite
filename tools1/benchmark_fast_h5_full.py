"""
完整基准测试 - 读取所有H5文件的全部segments

测试场景：
1. 按需加载 - 遍历所有segments
2. 预加载 - 遍历所有segments
3. DataLoader with different num_workers
"""

import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 导入快速数据集类
import sys
sys.path.append(str(Path(__file__).parent))
from h5_dataset_fast import FastH5Dataset, FastMultiH5Dataset, collate_fn


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}分{secs:.2f}秒"


def benchmark_full_dataset():
    """完整数据集基准测试"""
    
    print("="*80)
    print("完整数据集基准测试 - 19个H5文件")
    print("="*80)
    
    # 查找所有H5文件
    h5_dir = Path(r"E:\data\云南遥感中心\第一批\h5_fast\train")
    h5_files = sorted(h5_dir.glob("*.h5"))
    
    if not h5_files:
        print(f"❌ 未找到H5文件: {h5_dir}")
        return
    
    print(f"\n找到 {len(h5_files)} 个H5文件")
    
    # 创建数据集查看总segment数
    dataset_temp = FastMultiH5Dataset(
        [str(f) for f in h5_files],
        preload_strategy="none"
    )
    total_segments = len(dataset_temp)
    print(f"总segments数: {total_segments}")
    print("="*80)
    
    # ==================== 测试1: 按需加载，单进程 ====================
    print("\n【测试1】按需加载 + num_workers=0")
    print("-"*80)
    
    dataset = FastMultiH5Dataset(
        [str(f) for f in h5_files],
        preload_strategy="none"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    start = time.time()
    total_points = 0
    
    for batch_xyz, batch_labels in tqdm(dataloader, desc="读取进度", unit="batch"):
        for xyz in batch_xyz:
            total_points += len(xyz)
    
    elapsed_1 = time.time() - start
    speed_1 = total_segments / elapsed_1
    
    print(f"\n结果:")
    print(f"  总segments: {total_segments}")
    print(f"  总点数: {total_points:,}")
    print(f"  耗时: {format_time(elapsed_1)}")
    print(f"  速度: {speed_1:.2f} segments/秒")
    print(f"  平均每segment: {elapsed_1*1000/total_segments:.2f} ms")
    
    # ==================== 测试2: 按需加载，多进程 ====================
    print("\n" + "="*80)
    print("【测试2】按需加载 + num_workers=4")
    print("-"*80)
    
    dataset = FastMultiH5Dataset(
        [str(f) for f in h5_files],
        preload_strategy="none"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    
    start = time.time()
    total_points = 0
    
    for batch_xyz, batch_labels in tqdm(dataloader, desc="读取进度", unit="batch"):
        for xyz in batch_xyz:
            total_points += len(xyz)
    
    elapsed_2 = time.time() - start
    speed_2 = total_segments / elapsed_2
    
    print(f"\n结果:")
    print(f"  总segments: {total_segments}")
    print(f"  总点数: {total_points:,}")
    print(f"  耗时: {format_time(elapsed_2)}")
    print(f"  速度: {speed_2:.2f} segments/秒")
    print(f"  平均每segment: {elapsed_2*1000/total_segments:.2f} ms")
    print(f"  对比测试1: {elapsed_1/elapsed_2:.2f}x")
    
    # ==================== 测试3: 全预加载，单进程 ====================
    print("\n" + "="*80)
    print("【测试3】全预加载 + num_workers=0")
    print("-"*80)
    
    dataset = FastMultiH5Dataset(
        [str(f) for f in h5_files],
        preload_strategy="all"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    start = time.time()
    total_points = 0
    
    for batch_xyz, batch_labels in tqdm(dataloader, desc="读取进度", unit="batch"):
        for xyz in batch_xyz:
            total_points += len(xyz)
    
    elapsed_3 = time.time() - start
    speed_3 = total_segments / elapsed_3
    
    print(f"\n结果:")
    print(f"  总segments: {total_segments}")
    print(f"  总点数: {total_points:,}")
    print(f"  耗时: {format_time(elapsed_3)}")
    print(f"  速度: {speed_3:.2f} segments/秒")
    print(f"  平均每segment: {elapsed_3*1000/total_segments:.2f} ms")
    print(f"  对比测试1: {elapsed_1/elapsed_3:.2f}x")
    
    # ==================== 测试4: 随机读取模拟真实训练 ====================
    print("\n" + "="*80)
    print("【测试4】按需加载 + shuffle=True（模拟真实训练）")
    print("-"*80)
    
    dataset = FastMultiH5Dataset(
        [str(f) for f in h5_files],
        preload_strategy="none"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,  # 随机打乱
        num_workers=0,
        collate_fn=collate_fn
    )
    
    start = time.time()
    total_points = 0
    
    for batch_xyz, batch_labels in tqdm(dataloader, desc="读取进度", unit="batch"):
        for xyz in batch_xyz:
            total_points += len(xyz)
    
    elapsed_4 = time.time() - start
    speed_4 = total_segments / elapsed_4
    
    print(f"\n结果:")
    print(f"  总segments: {total_segments}")
    print(f"  总点数: {total_points:,}")
    print(f"  耗时: {format_time(elapsed_4)}")
    print(f"  速度: {speed_4:.2f} segments/秒")
    print(f"  平均每segment: {elapsed_4*1000/total_segments:.2f} ms")
    print(f"  对比测试1（顺序）: {elapsed_1/elapsed_4:.2f}x")
    
    # ==================== 最终总结 ====================
    print("\n" + "="*80)
    print("最终总结")
    print("="*80)
    
    results = [
        ("按需+单进程", elapsed_1, speed_1),
        ("按需+4进程", elapsed_2, speed_2),
        ("预加载+单进程", elapsed_3, speed_3),
        ("随机+单进程", elapsed_4, speed_4)
    ]
    
    print(f"\n{'模式':<20} {'耗时':<15} {'速度 (seg/s)':<20} {'相对性能':<10}")
    print("-"*80)
    
    baseline = elapsed_1
    for name, elapsed, speed in results:
        speedup = baseline / elapsed
        print(f"{name:<20} {format_time(elapsed):<15} {speed:>10.2f}{'':<10} {speedup:>6.2f}x")
    
    # 找出最快的
    best_idx = min(range(len(results)), key=lambda i: results[i][1])
    print(f"\n🏆 最优配置: {results[best_idx][0]}")
    print(f"   - 速度: {results[best_idx][2]:.2f} segments/秒")
    print(f"   - 耗时: {format_time(results[best_idx][1])}")
    
    # 计算与旧版的对比（假设旧版1.5 seg/s）
    print(f"\n📊 与旧版H5格式对比（旧版约1.5 seg/s）:")
    old_speed = 1.5
    for name, elapsed, speed in results:
        improvement = speed / old_speed
        print(f"   {name}: {improvement:.0f}x 提升")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    benchmark_full_dataset()
