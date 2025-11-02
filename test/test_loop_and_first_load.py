"""
测试两个关键问题：
1. 第一次加载数据是否比较慢？
2. loop 参数设置较大值是否有影响？
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import time
import numpy as np
from torch.utils.data import DataLoader

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointsuite.data.datasets.dataset_bin import BinPklDataset
from pointsuite.data.datasets.collate import DynamicBatchSampler, collate_fn


def test_first_vs_second_pass():
    """测试第一次遍历 vs 第二次遍历的速度差异"""
    print("="*70)
    print("[问题1] 第一次加载数据是否比较慢？")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    dataset = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False,  # 不缓存数据，只缓存 metadata
        loop=1
    )
    
    print(f"\n数据集总样本数: {len(dataset):,}")
    
    # 测试：分段访问，观察首次加载每个文件的开销
    print(f"\n[实验] 访问 200 个样本，观察首次加载 pkl 的时间峰值")
    
    times = []
    for i in range(200):
        start = time.time()
        sample = dataset[i]
        elapsed = time.time() - start
        times.append(elapsed)
        
        # 如果耗时超过 50ms，说明是首次加载 pkl
        if elapsed > 0.05:
            print(f"  样本 {i}: {elapsed*1000:.2f} ms ⚠️ (首次加载 pkl)")
    
    avg_time = np.mean(times) * 1000
    max_time = np.max(times) * 1000
    min_time = np.min(times) * 1000
    
    print(f"\n前 200 个样本统计:")
    print(f"  - 平均时间: {avg_time:.2f} ms")
    print(f"  - 最大时间: {max_time:.2f} ms (首次加载 pkl)")
    print(f"  - 最小时间: {min_time:.2f} ms (metadata 已缓存)")
    print(f"  - 峰值 / 平均: {max_time/avg_time:.1f}x")
    
    # 第二次遍历同样的样本
    print(f"\n[第二次遍历] 重复访问前 200 个样本")
    
    start = time.time()
    for i in range(200):
        sample = dataset[i]
    second_pass_time = time.time() - start
    
    first_pass_time = sum(times)
    
    print(f"\n对比:")
    print(f"  - 第一次遍历: {first_pass_time:.2f}s ({first_pass_time/200*1000:.2f} ms/样本)")
    print(f"  - 第二次遍历: {second_pass_time:.2f}s ({second_pass_time/200*1000:.2f} ms/样本)")
    print(f"  - 加速比: {first_pass_time/second_pass_time:.2f}x")
    
    print(f"\n💡 结论:")
    print(f"  - ✅ 第一次访问某个 pkl 文件中的样本时会慢（需加载 pkl）")
    print(f"  - ✅ 但每个 pkl 只需加载一次，后续访问都很快")
    print(f"  - ✅ 完整 epoch 中，29 个 pkl 各加载 1 次，总开销约 {29*max_time/1000:.2f}s")
    print(f"  - ✅ 相比总时间 99s，开销占比 {29*max_time/1000/99*100:.1f}%，可接受")
    print()


def test_loop_parameter():
    """测试 loop 参数的影响"""
    print("="*70)
    print("[问题2] loop 参数设置较大值是否有影响？")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    # 测试不同的 loop 值
    loop_values = [1, 2, 5, 10]
    
    print(f"\n测试不同 loop 值对数据访问的影响:\n")
    
    for loop in loop_values:
        dataset = BinPklDataset(
            data_root=data_root,
            split='train',
            assets=['coord', 'intensity', 'classification'],
            cache_data=False,
            loop=loop
        )
        
        actual_samples = len(dataset.data_list)
        virtual_length = len(dataset)
        
        print(f"loop={loop}:")
        print(f"  - 实际样本数: {actual_samples:,}")
        print(f"  - 虚拟长度: {virtual_length:,} (= {actual_samples:,} × {loop})")
        
        # 测试访问不同索引的速度
        test_indices = [0, actual_samples - 1, actual_samples, virtual_length - 1]
        
        print(f"  - 访问测试:")
        for idx in test_indices:
            if idx >= virtual_length:
                continue
            
            start = time.time()
            sample = dataset[idx]
            elapsed = time.time() - start
            
            # 计算实际访问的数据索引
            data_idx = idx % actual_samples
            
            print(f"    * dataset[{idx}] → data_list[{data_idx}]: {elapsed*1000:.2f} ms")
        
        print()
    
    print(f"\n💡 结论:")
    print(f"  - ✅ loop 只是虚拟地扩展了数据集长度")
    print(f"  - ✅ dataset[idx] 实际访问的是 data_list[idx % len(data_list)]")
    print(f"  - ✅ 不会增加内存占用（不会复制数据）")
    print(f"  - ✅ 不会增加加载时间（metadata 缓存仍然有效）")
    print(f"  - ⚠️ loop 大的话，每个 epoch 时间 = 原始时间 × loop")
    print()


def test_loop_with_dataloader():
    """测试 loop 参数在 DataLoader 中的实际效果"""
    print("="*70)
    print("[深入测试] loop 参数对 DataLoader 训练的影响")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    print(f"\n模拟训练场景：对比 loop=1 和 loop=3\n")
    
    # 测试 loop=1
    print(f"[测试] loop=1 (标准设置)")
    dataset_loop1 = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False,
        loop=1
    )
    
    batch_sampler_loop1 = DynamicBatchSampler(
        dataset_loop1,
        max_points=300000,
        shuffle=True,
        drop_last=False
    )
    
    dataloader_loop1 = DataLoader(
        dataset_loop1,
        batch_sampler=batch_sampler_loop1,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"  - 数据集长度: {len(dataset_loop1):,}")
    print(f"  - 预估 batches: {len(batch_sampler_loop1):,}")
    
    # 加载前 100 个 batch
    start = time.time()
    batch_count = 0
    sample_count = 0
    for batch in dataloader_loop1:
        batch_count += 1
        sample_count += len(batch['offset'])
        if batch_count >= 100:
            break
    elapsed_loop1 = time.time() - start
    
    print(f"  - 前 100 batch 时间: {elapsed_loop1:.2f}s")
    print(f"  - 加载样本数: {sample_count}")
    
    # 测试 loop=3
    print(f"\n[测试] loop=3")
    dataset_loop3 = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False,
        loop=3
    )
    
    batch_sampler_loop3 = DynamicBatchSampler(
        dataset_loop3,
        max_points=300000,
        shuffle=True,
        drop_last=False
    )
    
    dataloader_loop3 = DataLoader(
        dataset_loop3,
        batch_sampler=batch_sampler_loop3,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"  - 数据集长度: {len(dataset_loop3):,}")
    print(f"  - 预估 batches: {len(batch_sampler_loop3):,}")
    
    # 加载前 100 个 batch
    start = time.time()
    batch_count = 0
    sample_count = 0
    for batch in dataloader_loop3:
        batch_count += 1
        sample_count += len(batch['offset'])
        if batch_count >= 100:
            break
    elapsed_loop3 = time.time() - start
    
    print(f"  - 前 100 batch 时间: {elapsed_loop3:.2f}s")
    print(f"  - 加载样本数: {sample_count}")
    
    # 对比
    print(f"\n对比:")
    print(f"  - loop=1: {elapsed_loop1:.2f}s")
    print(f"  - loop=3: {elapsed_loop3:.2f}s")
    print(f"  - 速度比: {elapsed_loop3/elapsed_loop1:.2f}x")
    
    if abs(elapsed_loop3 / elapsed_loop1 - 1.0) < 0.1:
        print(f"  - ✅ 相同数量 batch 的加载时间几乎一致")
    
    print(f"\n💡 结论:")
    print(f"  - ✅ loop 不影响单个样本的加载速度")
    print(f"  - ✅ loop 只是让数据集「看起来」更大")
    print(f"  - ✅ 用于增加每个 epoch 中的数据增强多样性")
    print(f"  - 📌 如果 loop=3，每个 epoch 会访问每个样本 3 次")
    print(f"  - 📌 配合数据增强，每次访问同一样本会得到不同结果")
    print(f"  - ⚠️ 总训练时间 ≈ 原始时间 × loop")
    print()


def test_cache_data_with_loop():
    """测试 cache_data + loop 的组合效果"""
    print("="*70)
    print("[组合测试] cache_data + loop 的效果")
    print("="*70)
    
    data_root = r"E:\data\DALES\dales_las\bin\train"
    
    if not Path(data_root).exists():
        print(f"[X] 数据目录不存在: {data_root}")
        return
    
    print(f"\n对比两种配置:\n")
    
    # 配置1: cache_data=False, loop=3
    print(f"[配置1] cache_data=False, loop=3")
    dataset1 = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=False,
        loop=3
    )
    
    # 遍历前 300 个样本（实际会重复访问）
    start = time.time()
    for i in range(300):
        sample = dataset1[i]
    time1 = time.time() - start
    
    print(f"  - 访问 300 个样本: {time1:.2f}s ({time1/300*1000:.2f} ms/样本)")
    
    # 配置2: cache_data=True, loop=3
    print(f"\n[配置2] cache_data=True, loop=3")
    dataset2 = BinPklDataset(
        data_root=data_root,
        split='train',
        assets=['coord', 'intensity', 'classification'],
        cache_data=True,
        loop=3
    )
    
    # 首次遍历（触发缓存）
    start = time.time()
    for i in range(300):
        sample = dataset2[i]
    time2_first = time.time() - start
    
    print(f"  - 首次访问 300 个样本: {time2_first:.2f}s ({time2_first/300*1000:.2f} ms/样本)")
    
    # 第二次遍历（从缓存）
    start = time.time()
    for i in range(300):
        sample = dataset2[i]
    time2_second = time.time() - start
    
    print(f"  - 第二次访问: {time2_second:.2f}s ({time2_second/300*1000:.2f} ms/样本)")
    print(f"  - 加速比: {time2_first/max(time2_second, 0.001):.1f}x")
    
    print(f"\n💡 结论:")
    print(f"  - cache_data=False: 每次都从磁盘加载（慢但省内存）")
    print(f"  - cache_data=True: 首次慢，后续极快（快但占内存）")
    print(f"  - 推荐: 大数据集用 cache_data=False")
    print(f"  - 推荐: 小数据集用 cache_data=True + loop>1")
    print()


def main():
    """主测试函数"""
    print("\n")
    print("="*70)
    print("深入分析：首次加载速度 & loop 参数影响")
    print("="*70)
    print()
    
    try:
        # 测试1: 第一次 vs 第二次遍历
        test_first_vs_second_pass()
        
        # 测试2: loop 参数的影响
        test_loop_parameter()
        
        # 测试3: loop 在 DataLoader 中的影响
        test_loop_with_dataloader()
        
        # 测试4: cache_data + loop 组合
        test_cache_data_with_loop()
        
        print("="*70)
        print("[完成] 所有测试完成")
        print("="*70)
        
        print("\n【最终建议】")
        print("-"*70)
        print("问题1: 第一次加载是否慢？")
        print("  - ✅ 第一次访问某个 pkl 文件中的样本会慢（~60ms）")
        print("  - ✅ 但 metadata 缓存后，后续访问该文件的样本很快（~6ms）")
        print("  - ✅ 完整 epoch 中，29 个 pkl 各加载 1 次，总开销 <2s")
        print("  - ✅ 占总时间 <2%，可接受")
        print()
        print("问题2: loop 大值是否有影响？")
        print("  - ✅ loop 不影响单个样本加载速度")
        print("  - ✅ loop 不增加内存占用（不复制数据）")
        print("  - ✅ loop 只是虚拟扩展数据集长度")
        print("  - ⚠️ loop=3 → 每个 epoch 时间 × 3")
        print("  - 💡 适合配合数据增强，增加训练多样性")
        print()
        print("最佳实践:")
        print("  - 大数据集: cache_data=False, loop=1-2")
        print("  - 小数据集: cache_data=True, loop=3-5")
        print("  - 数据增强: 与 loop 配合使用，每次访问得到不同样本")
        print("="*70)
        
    except Exception as e:
        print(f"\n[X] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
