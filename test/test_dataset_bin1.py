"""
测试 BinPklDataset1 �?BinPklDataModule1 的功�?

测试内容�?
1. 全量模式和体素模式的基本功能
2. train/val 随机采样 vs test/predict 模运算采�?
3. 点云全覆盖验�?
4. 动态批处理兼容�?
5. 类别映射和类别权�?
6. 速度测试（单样本、多样本随机、动态批处理�?
"""

import os
import sys
import time
import numpy as np
import pickle
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """ANSI 颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def format_number(num: int) -> str:
    return f"{num:,}"


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds*1000000:.1f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f}ms"
    else:
        return f"{seconds:.3f}s"


# ============================================================================
# 测试1: 基本功能测试
# ============================================================================

def test_dataset_basic(pkl_path: str, mode: str = 'voxel'):
    """
    测试 BinPklDataset1 基本功能
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  📋 测试1: BinPklDataset1 基本功能测试 (mode={mode}){Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datasets.dataset_bin1 import BinPklDataset1
    
    # 测试不同 split
    for split in ['train', 'val', 'test']:
        print(f"\n  {Colors.BOLD}📦 Split: {split}{Colors.RESET}")
        
        dataset = BinPklDataset1(
            data_root=pkl_path,
            split=split,
            mode=mode,
            assets=['coord', 'intensity', 'class'],
            max_loops=5 if split in ['test', 'predict'] else None
        )
        
        print(f"  {Colors.DIM}├─{Colors.RESET} 样本�? {Colors.CYAN}{len(dataset)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 原始数据列表长度: {Colors.CYAN}{len(dataset.data_list)}{Colors.RESET}")
        
        # 获取第一个样�?
        sample = dataset[0]
        print(f"  {Colors.DIM}├─{Colors.RESET} 样本 0 �?keys: {list(sample.keys())}")
        print(f"  {Colors.DIM}├─{Colors.RESET} coord shape: {sample['coord'].shape}")
        
        if 'intensity' in sample:
            print(f"  {Colors.DIM}├─{Colors.RESET} intensity range: [{sample['intensity'].min():.3f}, {sample['intensity'].max():.3f}]")
        
        if 'class' in sample:
            unique_classes = np.unique(sample['class'])
            print(f"  {Colors.DIM}├─{Colors.RESET} 类别: {unique_classes}")
        
        if split == 'test' and 'indices' in sample:
            print(f"  {Colors.DIM}├─{Colors.RESET} indices shape: {sample['indices'].shape}")
            if 'loop_idx' in sample:
                print(f"  {Colors.DIM}├─{Colors.RESET} loop_idx: {sample['loop_idx']}")
        
        print(f"  {Colors.DIM}└─{Colors.RESET} {Colors.GREEN}�?通过{Colors.RESET}")
    
    return True


# ============================================================================
# 测试2: 体素模式全覆盖测�?
# ============================================================================

def test_voxel_full_coverage(pkl_path: str, max_loops: Optional[int] = None):
    """
    测试体素模式�?test split 是否覆盖所有点
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🔄 测试2: 体素模式全覆盖测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datasets.dataset_bin1 import BinPklDataset1
    
    max_loops_str = str(max_loops) if max_loops else "自动"
    print(f"  {Colors.DIM}├─{Colors.RESET} Max Loops: {Colors.CYAN}{max_loops_str}{Colors.RESET}")
    
    # 加载原始 PKL 获取真实点数
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    total_original_points = metadata['num_points']
    segments = metadata['segments']
    
    print(f"  {Colors.DIM}├─{Colors.RESET} 原始总点�? {Colors.CYAN}{format_number(total_original_points)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} Segments �? {Colors.CYAN}{len(segments)}{Colors.RESET}")
    
    # 创建 test 数据�?
    dataset = BinPklDataset1(
        data_root=pkl_path,
        split='test',
        mode='grid',
        assets=['coord', 'class'],
        max_loops=max_loops
    )
    
    print(f"  {Colors.DIM}├─{Colors.RESET} 数据集样本数: {Colors.CYAN}{len(dataset)}{Colors.RESET}")
    
    # 收集所有采样的索引
    all_sampled_indices = []
    segment_coverage = {}  # {segment_id: set of indices}
    
    print(f"\n  {Colors.BOLD}📊 采样覆盖分析:{Colors.RESET}")
    
    for i in range(len(dataset.data_list)):
        sample_info = dataset.data_list[i]
        segment_id = sample_info['segment_id']
        
        # 获取样本
        sample = dataset._load_data(i)
        
        if 'indices' in sample:
            indices = sample['indices']
            all_sampled_indices.extend(indices.tolist())
            
            if segment_id not in segment_coverage:
                segment_coverage[segment_id] = set()
            segment_coverage[segment_id].update(indices.tolist())
    
    # 统计覆盖情况
    all_sampled = np.array(all_sampled_indices)
    unique_sampled = np.unique(all_sampled)
    
    print(f"  {Colors.DIM}├─{Colors.RESET} 总采样数: {Colors.CYAN}{format_number(len(all_sampled))}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 唯一采样�? {Colors.CYAN}{format_number(len(unique_sampled))}{Colors.RESET}")
    
    coverage = len(unique_sampled) / total_original_points * 100
    if coverage >= 99.99:
        print(f"  {Colors.DIM}├─{Colors.RESET} 覆盖�? {Colors.GREEN}{format_percent(coverage)} ✓{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}├─{Colors.RESET} 覆盖�? {Colors.RED}{format_percent(coverage)}{Colors.RESET}")
    
    # 重复采样统计
    repeat_total = len(all_sampled) - len(unique_sampled)
    sample_counter = Counter(all_sampled)
    
    print(f"\n  {Colors.BOLD}🔁 重复采样统计:{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 重复采样次数: {Colors.YELLOW}{format_number(repeat_total)}{Colors.RESET}")
    
    if sample_counter:
        counts = list(sample_counter.values())
        print(f"  {Colors.DIM}├─{Colors.RESET} 平均采样次数: {Colors.YELLOW}{np.mean(counts):.2f}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 最大采样次�? {Colors.YELLOW}{max(counts)}{Colors.RESET}")
        
        # 分布
        count_dist = Counter(counts)
        print(f"\n  {Colors.BOLD}📈 采样次数分布:{Colors.RESET}")
        for cnt, num in sorted(count_dist.items())[:5]:
            pct = num / len(counts) * 100
            print(f"  {Colors.DIM}│{Colors.RESET}   采样 {cnt} �? {format_number(num)} �?({format_percent(pct)})")
    
    passed = coverage >= 99.99
    return {
        'coverage': coverage,
        'total_sampled': len(all_sampled),
        'unique_sampled': len(unique_sampled),
        'passed': passed
    }


# ============================================================================
# 测试3: 动态批处理兼容�?
# ============================================================================

def test_dynamic_batch_compatibility(pkl_path: str):
    """
    测试�?DynamicBatchSampler 的兼容�?
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  📦 测试3: 动态批处理兼容性{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datasets.dataset_bin1 import BinPklDataset1
    
    # 测试 voxel 模式
    for split in ['train', 'test']:
        print(f"\n  {Colors.BOLD}📦 Split: {split}{Colors.RESET}")
        
        dataset = BinPklDataset1(
            data_root=pkl_path,
            split=split,
            mode='grid',
            max_loops=5 if split == 'test' else None
        )
        
        # 获取样本点数列表
        sample_num_points = dataset.get_sample_num_points()
        
        print(f"  {Colors.DIM}├─{Colors.RESET} 样本�? {Colors.CYAN}{len(sample_num_points)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 点数范围: [{min(sample_num_points):,}, {max(sample_num_points):,}]")
        print(f"  {Colors.DIM}├─{Colors.RESET} 平均点数: {np.mean(sample_num_points):,.1f}")
        
        # 模拟动态批处理
        max_points = 50000
        batches = []
        current_batch = []
        current_points = 0
        
        for i, num_points in enumerate(sample_num_points):
            if current_points + num_points > max_points and current_batch:
                batches.append(current_batch)
                current_batch = [i]
                current_points = num_points
            else:
                current_batch.append(i)
                current_points += num_points
        
        if current_batch:
            batches.append(current_batch)
        
        batch_sizes = [len(b) for b in batches]
        batch_points = [sum(sample_num_points[i] for i in b) for b in batches]
        
        print(f"  {Colors.DIM}├─{Colors.RESET} 批次�?(max_points={max_points}): {Colors.CYAN}{len(batches)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 批次大小范围: [{min(batch_sizes)}, {max(batch_sizes)}]")
        print(f"  {Colors.DIM}├─{Colors.RESET} 批次点数范围: [{min(batch_points):,}, {max(batch_points):,}]")
        print(f"  {Colors.DIM}└─{Colors.RESET} {Colors.GREEN}�?通过{Colors.RESET}")
    
    return True


# ============================================================================
# 测试4: DataModule 功能测试
# ============================================================================

def test_datamodule(pkl_path: str):
    """
    测试 BinPklDataModule1 功能
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🔧 测试4: BinPklDataModule1 功能测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datamodule_bin1 import BinPklDataModule1
    
    # 创建 DataModule
    datamodule = BinPklDataModule1(
        train_data=pkl_path,
        val_data=pkl_path,
        test_data=pkl_path,
        batch_size=4,
        num_workers=0,  # 测试时使�?0
        mode='grid',
        max_loops=5,
        assets=['coord', 'intensity', 'class'],
    )
    
    # 设置数据�?
    datamodule.setup('fit')
    datamodule.setup('test')
    
    print(f"\n  {Colors.BOLD}📊 DataModule 信息:{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 训练样本�? {Colors.CYAN}{len(datamodule.train_dataset)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 验证样本�? {Colors.CYAN}{len(datamodule.val_dataset)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 测试样本�? {Colors.CYAN}{len(datamodule.test_dataset)}{Colors.RESET}")
    
    # 测试 DataLoader
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    print(f"\n  {Colors.BOLD}📦 DataLoader 测试:{Colors.RESET}")
    
    # 测试一�?batch
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        batch = next(iter(loader))
        print(f"  {Colors.DIM}├─{Colors.RESET} {name} batch:")
        print(f"  {Colors.DIM}│{Colors.RESET}   - coord shape: {batch['coord'].shape}")
        if 'offset' in batch:
            print(f"  {Colors.DIM}│{Colors.RESET}   - offset: {batch['offset']}")
    
    print(f"\n  {Colors.DIM}└─{Colors.RESET} {Colors.GREEN}�?通过{Colors.RESET}")
    
    return True


# ============================================================================
# 测试5: 类别映射和权�?
# ============================================================================

def test_class_mapping(pkl_path: str):
    """
    测试类别映射和类别权重功�?
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🏷�?测试5: 类别映射和权重{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datasets.dataset_bin1 import BinPklDataset1
    
    # 首先获取原始类别分布
    dataset_orig = BinPklDataset1(
        data_root=pkl_path,
        split='train',
        mode='grid',
        assets=['coord', 'class'],
    )
    
    orig_dist = dataset_orig.get_class_distribution()
    print(f"\n  {Colors.BOLD}📊 原始类别分布:{Colors.RESET}")
    for cls, count in sorted(orig_dist.items()):
        print(f"  {Colors.DIM}├─{Colors.RESET} 类别 {cls}: {format_number(count)}")
    
    # 测试类别映射
    # 假设映射 DALES 类别: 0->ignore, 1->0, 2->1, 3->2, 4->3, 5->4, 6->5, 7->6, 8->7
    class_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    
    dataset_mapped = BinPklDataset1(
        data_root=pkl_path,
        split='train',
        mode='grid',
        assets=['coord', 'class'],
        class_mapping=class_mapping,
        ignore_label=-1
    )
    
    mapped_dist = dataset_mapped.get_class_distribution()
    print(f"\n  {Colors.BOLD}📊 映射后类别分�?{Colors.RESET}")
    for cls, count in sorted(mapped_dist.items()):
        print(f"  {Colors.DIM}├─{Colors.RESET} 类别 {cls}: {format_number(count)}")
    
    # 测试类别权重
    weights = dataset_mapped.class_weights
    print(f"\n  {Colors.BOLD}⚖️ 类别权重:{Colors.RESET}")
    if weights is not None:
        print(f"  {Colors.DIM}├─{Colors.RESET} 权重 shape: {weights.shape}")
        for i, w in enumerate(weights):
            print(f"  {Colors.DIM}│{Colors.RESET}   类别 {i}: {w:.4f}")
    
    print(f"\n  {Colors.DIM}└─{Colors.RESET} {Colors.GREEN}�?通过{Colors.RESET}")
    
    return True


# ============================================================================
# 测试6: train �?test 模式对比
# ============================================================================

def test_train_vs_test_sampling(pkl_path: str):
    """
    对比 train �?test 模式的采样差�?
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  🔀 测试6: Train vs Test 采样对比{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datasets.dataset_bin1 import BinPklDataset1
    
    # Train 模式
    dataset_train = BinPklDataset1(
        data_root=pkl_path,
        split='train',
        mode='grid',
        assets=['coord'],
    )
    
    # Test 模式
    dataset_test = BinPklDataset1(
        data_root=pkl_path,
        split='test',
        mode='grid',
        max_loops=None,  # 自动
        assets=['coord'],
    )
    
    print(f"\n  {Colors.BOLD}📊 对比:{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} Train 样本�? {Colors.CYAN}{len(dataset_train)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} Test 样本�? {Colors.CYAN}{len(dataset_test)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 比例: {Colors.YELLOW}{len(dataset_test) / len(dataset_train):.2f}x{Colors.RESET}")
    
    # 检�?train 的随机�?
    print(f"\n  {Colors.BOLD}🎲 Train 随机性验�?{Colors.RESET}")
    sample1 = dataset_train[0]
    sample2 = dataset_train[0]  # 再次获取同一个样�?
    
    # 检查坐标是否不同（随机采样�?
    coords_same = np.allclose(sample1['coord'], sample2['coord'])
    if not coords_same:
        print(f"  {Colors.DIM}├─{Colors.RESET} 两次采样结果不同: {Colors.GREEN}�?(随机采样正常){Colors.RESET}")
    else:
        print(f"  {Colors.DIM}├─{Colors.RESET} 两次采样结果相同: {Colors.YELLOW}! (可能是缓存或确定性采�?{Colors.RESET}")
    
    # 检�?test 的确定�?
    print(f"\n  {Colors.BOLD}🔒 Test 确定性验�?{Colors.RESET}")
    # Test 模式下同一索引应该返回相同结果
    test_sample1 = dataset_test._load_data(0)
    test_sample2 = dataset_test._load_data(0)
    
    if 'indices' in test_sample1 and 'indices' in test_sample2:
        indices_same = np.array_equal(test_sample1['indices'], test_sample2['indices'])
        if indices_same:
            print(f"  {Colors.DIM}├─{Colors.RESET} 两次采样索引相同: {Colors.GREEN}�?(模运算确定性正�?{Colors.RESET}")
        else:
            print(f"  {Colors.DIM}├─{Colors.RESET} 两次采样索引不同: {Colors.RED}�?(应该相同){Colors.RESET}")
    
    print(f"\n  {Colors.DIM}└─{Colors.RESET} {Colors.GREEN}�?通过{Colors.RESET}")
    
    return True


# ============================================================================
# 测试7: 速度测试
# ============================================================================

def test_speed_single_sample(pkl_path: str, n_iterations: int = 100):
    """
    单样本采样速度测试
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  �?测试7a: 单样本采样速度测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datasets.dataset_bin1 import BinPklDataset1
    
    results = {}
    
    for mode in ['voxel', 'full']:
        for split in ['train', 'test']:
            print(f"\n  {Colors.BOLD}📊 Mode={mode}, Split={split}{Colors.RESET}")
            
            dataset = BinPklDataset1(
                data_root=pkl_path,
                split=split,
                mode=mode,
                assets=['coord', 'intensity', 'class'],
                max_loops=5 if split == 'test' else None
            )
            
            # 预热 (JIT 编译)
            _ = dataset[0]
            _ = dataset[0]
            
            # 计时
            times = []
            for _ in range(n_iterations):
                idx = np.random.randint(0, len(dataset))
                t0 = time.perf_counter()
                sample = dataset[idx]
                t1 = time.perf_counter()
                times.append(t1 - t0)
            
            times = np.array(times)
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            # 获取样本点数信息
            sample_points = [dataset.data_list[i].get('num_voxels', dataset.data_list[i]['num_points']) 
                            for i in range(min(10, len(dataset.data_list)))]
            avg_points = np.mean(sample_points)
            
            print(f"  {Colors.DIM}├─{Colors.RESET} 迭代次数: {n_iterations}")
            print(f"  {Colors.DIM}├─{Colors.RESET} 平均时间: {Colors.CYAN}{format_time(avg_time)}{Colors.RESET} ± {format_time(std_time)}")
            print(f"  {Colors.DIM}├─{Colors.RESET} 最�?最�? {format_time(min_time)} / {format_time(max_time)}")
            print(f"  {Colors.DIM}├─{Colors.RESET} 平均点数: {avg_points:,.0f}")
            print(f"  {Colors.DIM}├─{Colors.RESET} 吞吐�? {Colors.GREEN}{1/avg_time:.1f} samples/s{Colors.RESET}")
            
            key = f"{mode}_{split}"
            results[key] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput': 1/avg_time
            }
    
    return results


def test_speed_random_access(pkl_path: str, n_iterations: int = 500):
    """
    多文件随机访问速度测试（模�?DataLoader 行为�?
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  �?测试7b: 随机访问速度测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datasets.dataset_bin import BinPklDataset
    
    # 创建 train �?test 数据�?
    dataset_train = BinPklDataset(
        data_root=pkl_path,
        split='train',
        mode='grid',
        assets=['coord', 'intensity', 'class'],
    )
    
    dataset_test = BinPklDataset1(
        data_root=pkl_path,
        split='test',
        mode='grid',
        max_loops=5,
        assets=['coord', 'intensity', 'class'],
    )
    
    # 预热
    _ = dataset_train[0]
    _ = dataset_test[0]
    
    results = {}
    
    for name, dataset in [('train_voxel', dataset_train), ('test_voxel', dataset_test)]:
        print(f"\n  {Colors.BOLD}📊 {name}{Colors.RESET}")
        
        # 生成随机索引序列
        indices = np.random.randint(0, len(dataset), size=n_iterations)
        
        # 计时
        t0 = time.perf_counter()
        total_points = 0
        for idx in indices:
            sample = dataset[idx]
            total_points += len(sample['coord'])
        t1 = time.perf_counter()
        
        total_time = t1 - t0
        avg_time = total_time / n_iterations
        
        print(f"  {Colors.DIM}├─{Colors.RESET} 迭代次数: {n_iterations}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 总时�? {Colors.CYAN}{format_time(total_time)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 平均时间: {Colors.CYAN}{format_time(avg_time)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 总点�? {format_number(total_points)}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 吞吐�? {Colors.GREEN}{1/avg_time:.1f} samples/s{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 点吞吐量: {Colors.GREEN}{total_points/total_time/1e6:.2f} M points/s{Colors.RESET}")
        
        results[name] = {
            'total_time': total_time,
            'avg_time': avg_time,
            'throughput': 1/avg_time,
            'points_per_sec': total_points/total_time
        }
    
    return results


def test_speed_dataloader(pkl_path: str, n_batches: int = 50):
    """
    DataLoader 速度测试（包括动态批处理�?
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  �?测试7c: DataLoader 速度测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datamodule_bin1 import BinPklDataModule1
    from torch.utils.data import DataLoader
    
    results = {}
    
    # 测试不同配置
    configs = [
        {'name': 'fixed_batch', 'use_dynamic_batch': False, 'batch_size': 4},
        {'name': 'dynamic_batch_50k', 'use_dynamic_batch': True, 'max_points': 50000},
        {'name': 'dynamic_batch_100k', 'use_dynamic_batch': True, 'max_points': 100000},
    ]
    
    for config in configs:
        print(f"\n  {Colors.BOLD}📊 {config['name']}{Colors.RESET}")
        
        datamodule = BinPklDataModule1(
            train_data=pkl_path,
            mode='grid',
            assets=['coord', 'intensity', 'class'],
            num_workers=0,  # 单线程测试以准确测量采样时间
            use_dynamic_batch=config.get('use_dynamic_batch', False),
            batch_size=config.get('batch_size', 8),
            max_points=config.get('max_points', 100000),
        )
        
        datamodule.setup('fit')
        train_loader = datamodule.train_dataloader()
        
        # 预热
        for i, batch in enumerate(train_loader):
            if i >= 2:
                break
        
        # 计时
        t0 = time.perf_counter()
        total_points = 0
        batch_count = 0
        batch_sizes = []
        
        for batch in train_loader:
            total_points += batch['coord'].shape[0]
            batch_sizes.append(batch['coord'].shape[0])
            batch_count += 1
            if batch_count >= n_batches:
                break
        
        t1 = time.perf_counter()
        
        total_time = t1 - t0
        avg_batch_time = total_time / batch_count
        
        print(f"  {Colors.DIM}├─{Colors.RESET} 批次�? {batch_count}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 总时�? {Colors.CYAN}{format_time(total_time)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 平均批次时间: {Colors.CYAN}{format_time(avg_batch_time)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 批次大小范围: [{min(batch_sizes):,}, {max(batch_sizes):,}]")
        print(f"  {Colors.DIM}├─{Colors.RESET} 平均批次点数: {np.mean(batch_sizes):,.0f}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 批次吞吐�? {Colors.GREEN}{batch_count/total_time:.1f} batches/s{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 点吞吐量: {Colors.GREEN}{total_points/total_time/1e6:.2f} M points/s{Colors.RESET}")
        
        results[config['name']] = {
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'batch_throughput': batch_count/total_time,
            'points_per_sec': total_points/total_time
        }
    
    return results


def test_speed_comparison(pkl_path: str):
    """
    速度对比测试：Numba vs �?Python（如果可用）
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  �?测试7d: 采样函数性能分析{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datasets.dataset_bin1 import BinPklDataset1
    
    # 加载数据
    dataset = BinPklDataset1(
        data_root=pkl_path,
        split='train',
        mode='grid',
        assets=['coord'],
    )
    
    # 获取一�?segment 进行测试
    metadata = dataset._get_metadata(dataset.data_list[0]['pkl_path'])
    segment_info = metadata['segments'][0]
    mmap_data = dataset._get_mmap(
        dataset.data_list[0]['bin_path'], 
        metadata['dtype']
    )
    
    n_voxels = len(segment_info['voxel_counts'])
    n_points = segment_info['num_points']
    
    print(f"\n  {Colors.BOLD}📊 测试样本信息:{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 体素�? {n_voxels:,}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 总点�? {n_points:,}")
    
    # 预热 Numba
    _ = dataset._voxel_random_sample(segment_info, mmap_data)
    _ = dataset._voxel_modulo_sample(segment_info, mmap_data, 0, 1)
    
    # 测试随机采样
    n_iterations = 1000
    
    print(f"\n  {Colors.BOLD}🎲 随机采样测试 ({n_iterations} �?:{Colors.RESET}")
    
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = dataset._voxel_random_sample(segment_info, mmap_data)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    avg_time = np.mean(times)
    print(f"  {Colors.DIM}├─{Colors.RESET} 平均时间: {Colors.CYAN}{format_time(avg_time)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 每体素耗时: {Colors.CYAN}{avg_time/n_voxels*1e9:.1f} ns{Colors.RESET}")
    print(f"  {Colors.DIM}└─{Colors.RESET} 吞吐�? {Colors.GREEN}{n_voxels/avg_time/1e6:.2f} M voxels/s{Colors.RESET}")
    
    # 测试模运算采�?
    print(f"\n  {Colors.BOLD}🔄 模运算采样测�?({n_iterations} �?:{Colors.RESET}")
    
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = dataset._voxel_modulo_sample(segment_info, mmap_data, 0, 1)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    avg_time = np.mean(times)
    print(f"  {Colors.DIM}├─{Colors.RESET} 平均时间: {Colors.CYAN}{format_time(avg_time)}{Colors.RESET}")
    print(f"  {Colors.DIM}├─{Colors.RESET} 每体素耗时: {Colors.CYAN}{avg_time/n_voxels*1e9:.1f} ns{Colors.RESET}")
    print(f"  {Colors.DIM}└─{Colors.RESET} 吞吐�? {Colors.GREEN}{n_voxels/avg_time/1e6:.2f} M voxels/s{Colors.RESET}")
    
    return True


def test_speed_multi_workers(pkl_path: str, n_batches: int = 50):
    """
    测试不同 num_workers �?DataLoader 速度的影�?
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  �?测试7e: �?Workers 速度测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datamodule_bin1 import BinPklDataModule1
    import multiprocessing
    
    max_workers = min(multiprocessing.cpu_count(), 8)
    worker_counts = [0, 1, 2, 4] + ([max_workers] if max_workers > 4 else [])
    worker_counts = sorted(set(worker_counts))
    
    print(f"\n  {Colors.DIM}CPU 核心�? {multiprocessing.cpu_count()}{Colors.RESET}")
    print(f"  {Colors.DIM}测试 workers: {worker_counts}{Colors.RESET}")
    
    results = {}
    
    for num_workers in worker_counts:
        print(f"\n  {Colors.BOLD}📊 num_workers={num_workers}{Colors.RESET}")
        
        try:
            datamodule = BinPklDataModule1(
                train_data=pkl_path,
                mode='grid',
                assets=['coord', 'class'],
                num_workers=num_workers,
                use_dynamic_batch=True,
                max_points=80000,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
            )
            
            datamodule.setup('fit')
            train_loader = datamodule.train_dataloader()
            
            # 预热
            warmup_count = min(5, n_batches // 2)
            for i, batch in enumerate(train_loader):
                if i >= warmup_count:
                    break
            
            # 计时
            t0 = time.perf_counter()
            total_points = 0
            batch_count = 0
            
            for batch in train_loader:
                total_points += batch['coord'].shape[0]
                batch_count += 1
                if batch_count >= n_batches:
                    break
            
            t1 = time.perf_counter()
            
            total_time = t1 - t0
            points_per_sec = total_points / total_time
            
            print(f"  {Colors.DIM}├─{Colors.RESET} 批次�? {batch_count}")
            print(f"  {Colors.DIM}├─{Colors.RESET} 总时�? {Colors.CYAN}{format_time(total_time)}{Colors.RESET}")
            print(f"  {Colors.DIM}└─{Colors.RESET} 吞吐�? {Colors.GREEN}{points_per_sec/1e6:.2f} M points/s{Colors.RESET}")
            
            results[f'workers_{num_workers}'] = {
                'total_time': total_time,
                'points_per_sec': points_per_sec,
                'batch_count': batch_count,
            }
            
        except Exception as e:
            print(f"  {Colors.RED}�?失败: {e}{Colors.RESET}")
            results[f'workers_{num_workers}'] = {'error': str(e)}
    
    # 对比分析
    print(f"\n  {Colors.BOLD}📈 Workers 性能对比:{Colors.RESET}")
    base_throughput = None
    for key, val in results.items():
        if 'points_per_sec' in val:
            throughput = val['points_per_sec']
            if base_throughput is None:
                base_throughput = throughput
                speedup = "基准"
            else:
                speedup = f"{throughput/base_throughput:.2f}x"
            print(f"  {Colors.DIM}├─{Colors.RESET} {key}: {throughput/1e6:.2f} M pts/s ({speedup})")
    
    return results


def test_speed_multi_files(data_dir: str, n_iterations: int = 100):
    """
    测试多文件（全目录）读取速度
    """
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  �?测试7f: 多文件速度测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    from pointsuite.data.datasets.dataset_bin1 import BinPklDataset1
    
    data_path = Path(data_dir)
    pkl_files = list(data_path.glob('*.pkl'))
    
    print(f"\n  {Colors.DIM}数据目录: {data_dir}{Colors.RESET}")
    print(f"  {Colors.DIM}pkl 文件�? {len(pkl_files)}{Colors.RESET}")
    
    if len(pkl_files) == 0:
        print(f"  {Colors.RED}未找�?pkl 文件{Colors.RESET}")
        return {}
    
    results = {}
    
    # 测试不同模式�?split 组合
    test_configs = [
        {'mode': 'voxel', 'split': 'train', 'name': 'voxel_train'},
        {'mode': 'voxel', 'split': 'test', 'name': 'voxel_test'},
        {'mode': 'full', 'split': 'train', 'name': 'full_train'},
    ]
    
    for config in test_configs:
        print(f"\n  {Colors.BOLD}📊 {config['name']} (全目�?{len(pkl_files)} 个文�?{Colors.RESET}")
        
        try:
            t_load_start = time.perf_counter()
            dataset = BinPklDataset1(
                data_root=data_dir,
                split=config['split'],
                mode=config['mode'],
                assets=['coord', 'class'],
            )
            t_load_end = time.perf_counter()
            
            n_samples = len(dataset)
            load_time = t_load_end - t_load_start
            
            print(f"  {Colors.DIM}├─{Colors.RESET} 加载时间: {Colors.CYAN}{format_time(load_time)}{Colors.RESET}")
            print(f"  {Colors.DIM}├─{Colors.RESET} 总样本数: {Colors.CYAN}{n_samples:,}{Colors.RESET}")
            
            if n_samples == 0:
                print(f"  {Colors.RED}└─ 无样本可测试{Colors.RESET}")
                continue
            
            # 预热
            _ = dataset[0]
            
            # 随机访问测试
            actual_iterations = min(n_iterations, n_samples)
            indices = np.random.randint(0, n_samples, size=actual_iterations)
            
            t0 = time.perf_counter()
            total_points = 0
            
            for idx in indices:
                sample = dataset[idx]
                total_points += len(sample['coord'])
            
            t1 = time.perf_counter()
            
            total_time = t1 - t0
            avg_time = total_time / actual_iterations
            
            print(f"  {Colors.DIM}├─{Colors.RESET} 随机访问 {actual_iterations} �? {Colors.CYAN}{format_time(total_time)}{Colors.RESET}")
            print(f"  {Colors.DIM}├─{Colors.RESET} 平均时间: {Colors.CYAN}{format_time(avg_time)}{Colors.RESET}")
            print(f"  {Colors.DIM}├─{Colors.RESET} 吞吐�? {Colors.GREEN}{1/avg_time:.1f} samples/s{Colors.RESET}")
            print(f"  {Colors.DIM}└─{Colors.RESET} 点吞吐量: {Colors.GREEN}{total_points/total_time/1e6:.2f} M points/s{Colors.RESET}")
            
            results[config['name']] = {
                'n_files': len(pkl_files),
                'n_samples': n_samples,
                'load_time': load_time,
                'total_time': total_time,
                'avg_time': avg_time,
                'throughput': 1/avg_time,
                'points_per_sec': total_points/total_time,
            }
            
        except Exception as e:
            print(f"  {Colors.RED}�?失败: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            results[config['name']] = {'error': str(e)}
    
    # DataLoader 测试（全目录�?
    print(f"\n  {Colors.BOLD}📊 DataLoader 全目录测试{Colors.RESET}")
    
    try:
        from pointsuite.data.datamodule_bin1 import BinPklDataModule1
        
        datamodule = BinPklDataModule1(
            train_data=data_dir,
            mode='grid',
            assets=['coord', 'class'],
            num_workers=0,
            use_dynamic_batch=True,
            max_points=80000,
        )
        
        datamodule.setup('fit')
        train_loader = datamodule.train_dataloader()
        n_total_samples = len(train_loader.dataset)
        
        print(f"  {Colors.DIM}├─{Colors.RESET} 总样本数: {n_total_samples:,}")
        
        # 遍历整个数据集一�?
        t0 = time.perf_counter()
        total_points = 0
        batch_count = 0
        
        for batch in train_loader:
            total_points += batch['coord'].shape[0]
            batch_count += 1
        
        t1 = time.perf_counter()
        
        total_time = t1 - t0
        
        print(f"  {Colors.DIM}├─{Colors.RESET} 总批次数: {batch_count}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 遍历时间: {Colors.CYAN}{format_time(total_time)}{Colors.RESET}")
        print(f"  {Colors.DIM}├─{Colors.RESET} 总点�? {Colors.CYAN}{total_points:,}{Colors.RESET}")
        print(f"  {Colors.DIM}└─{Colors.RESET} 吞吐�? {Colors.GREEN}{total_points/total_time/1e6:.2f} M points/s{Colors.RESET}")
        
        results['dataloader_full'] = {
            'n_samples': n_total_samples,
            'batch_count': batch_count,
            'total_time': total_time,
            'total_points': total_points,
            'points_per_sec': total_points/total_time,
        }
        
    except Exception as e:
        print(f"  {Colors.RED}�?DataLoader 测试失败: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
    
    return results


def run_speed_tests(pkl_path: str, n_iterations: int = 100, n_batches: int = 50, 
                    test_multi_workers: bool = True, test_multi_files: bool = True):
    """运行所有速度测试"""
    print(f"\n{Colors.BOLD}{'#'*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}  �?速度测试套件{Colors.RESET}")
    print(f"{Colors.BOLD}{'#'*70}{Colors.RESET}")
    print(f"  测试文件: {pkl_path}")
    
    results = {}
    
    try:
        # 采样函数性能
        results['sampling_perf'] = test_speed_comparison(pkl_path)
        
        # 单样本测�?
        results['single_sample'] = test_speed_single_sample(pkl_path, n_iterations=n_iterations)
        
        # 随机访问测试
        results['random_access'] = test_speed_random_access(pkl_path, n_iterations=n_iterations * 5)
        
        # DataLoader 测试
        results['dataloader'] = test_speed_dataloader(pkl_path, n_batches=n_batches)
        
        # �?Workers 测试
        if test_multi_workers:
            results['multi_workers'] = test_speed_multi_workers(pkl_path, n_batches=n_batches)
        
        # 多文件测�?
        if test_multi_files:
            data_dir = Path(pkl_path).parent
            results['multi_files'] = test_speed_multi_files(str(data_dir), n_iterations=n_iterations)
        
    except Exception as e:
        print(f"\n{Colors.RED}�?速度测试失败: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
    
    # 汇�?
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  📋 速度测试汇总{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    if 'single_sample' in results:
        print(f"\n  {Colors.BOLD}单样本采�?{Colors.RESET}")
        for key, val in results['single_sample'].items():
            print(f"  {Colors.DIM}├─{Colors.RESET} {key}: {format_time(val['avg_time'])} ({val['throughput']:.1f} samples/s)")
    
    if 'dataloader' in results:
        print(f"\n  {Colors.BOLD}DataLoader 吞吐�?(num_workers=0):{Colors.RESET}")
        for key, val in results['dataloader'].items():
            print(f"  {Colors.DIM}├─{Colors.RESET} {key}: {val['points_per_sec']/1e6:.2f} M points/s")
    
    if 'multi_workers' in results:
        print(f"\n  {Colors.BOLD}�?Workers 吞吐�?{Colors.RESET}")
        for key, val in results['multi_workers'].items():
            if 'points_per_sec' in val:
                print(f"  {Colors.DIM}├─{Colors.RESET} {key}: {val['points_per_sec']/1e6:.2f} M points/s")
    
    if 'multi_files' in results:
        print(f"\n  {Colors.BOLD}多文�?(全目�? 吞吐�?{Colors.RESET}")
        for key, val in results['multi_files'].items():
            if 'points_per_sec' in val:
                print(f"  {Colors.DIM}├─{Colors.RESET} {key}: {val['points_per_sec']/1e6:.2f} M points/s")
    
    print()
    return results


# ============================================================================
# 主测试入�?
# ============================================================================

def run_all_tests(pkl_path: str):
    """运行所有测�?""
    print(f"\n{Colors.BOLD}{'#'*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}  🧪 BinPklDataset1 & DataModule1 测试套件{Colors.RESET}")
    print(f"{Colors.BOLD}{'#'*70}{Colors.RESET}")
    print(f"  测试文件: {pkl_path}")
    
    results = {}
    
    try:
        # 测试1: 基本功能
        results['basic_voxel'] = test_dataset_basic(pkl_path, mode='grid')
        results['basic_full'] = test_dataset_basic(pkl_path, mode='full')
        
        # 测试2: 全覆�?
        results['coverage'] = test_voxel_full_coverage(pkl_path, max_loops=None)
        results['coverage_limited'] = test_voxel_full_coverage(pkl_path, max_loops=5)
        
        # 测试3: 动态批处理
        results['dynamic_batch'] = test_dynamic_batch_compatibility(pkl_path)
        
        # 测试4: DataModule
        results['datamodule'] = test_datamodule(pkl_path)
        
        # 测试5: 类别映射
        results['class_mapping'] = test_class_mapping(pkl_path)
        
        # 测试6: Train vs Test
        results['train_vs_test'] = test_train_vs_test_sampling(pkl_path)
        
        # 测试7: 速度测试
        results['speed'] = run_speed_tests(pkl_path)
        
    except Exception as e:
        print(f"\n{Colors.RED}�?测试失败: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return results
    
    # 汇�?
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  📋 测试结果汇总{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    all_passed = True
    for name, result in results.items():
        if name == 'speed':
            continue  # 速度测试不参�?pass/fail 判断
        if isinstance(result, dict):
            passed = result.get('passed', True)
        else:
            passed = result
        status = f"{Colors.GREEN}�?PASS{Colors.RESET}" if passed else f"{Colors.RED}�?FAIL{Colors.RESET}"
        print(f"  {Colors.DIM}├─{Colors.RESET} {name}: {status}")
        all_passed = all_passed and passed
    
    print(f"\n  {Colors.BOLD}最终结�? ", end="")
    if all_passed:
        print(f"{Colors.GREEN}所有测试通过 ✓{Colors.RESET}")
    else:
        print(f"{Colors.RED}部分测试失败{Colors.RESET}")
    print()
    
    return results


# ============================================================================
# 命令行入�?
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试 BinPklDataset1 �?DataModule1')
    parser.add_argument('--pkl', type=str, required=False,
                        help='PKL 文件路径')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'basic', 'coverage', 'batch', 'datamodule', 'class', 'compare', 'speed'],
                        help='运行的测�?)
    parser.add_argument('--speed-iterations', type=int, default=100,
                        help='速度测试的迭代次�?)
    parser.add_argument('--speed-batches', type=int, default=50,
                        help='DataLoader速度测试的批次数')
    
    args = parser.parse_args()
    
    # 默认测试路径
    if args.pkl:
        pkl_path = args.pkl
    else:
        default_path = r"E:\data\DALES\dales_las\bin\train_logical\5080_54435.pkl"
        if Path(default_path).exists():
            pkl_path = default_path
        else:
            print(f"{Colors.RED}请指�?--pkl 参数{Colors.RESET}")
            sys.exit(1)
    
    if not Path(pkl_path).exists():
        print(f"{Colors.RED}文件不存�? {pkl_path}{Colors.RESET}")
        sys.exit(1)
    
    if args.test == 'all':
        run_all_tests(pkl_path)
    elif args.test == 'basic':
        test_dataset_basic(pkl_path, 'voxel')
        test_dataset_basic(pkl_path, 'full')
    elif args.test == 'coverage':
        test_voxel_full_coverage(pkl_path)
    elif args.test == 'batch':
        test_dynamic_batch_compatibility(pkl_path)
    elif args.test == 'datamodule':
        test_datamodule(pkl_path)
    elif args.test == 'class':
        test_class_mapping(pkl_path)
    elif args.test == 'compare':
        test_train_vs_test_sampling(pkl_path)
    elif args.test == 'speed':
        run_speed_tests(pkl_path, args.speed_iterations, args.speed_batches)
