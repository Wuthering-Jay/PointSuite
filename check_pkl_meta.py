import pickle
from pathlib import Path
from collections import Counter

test_dir = Path(r'E:\data\DALES\dales_las\bin_logical\test')

import sys
sys.path.insert(0, str(Path(r'e:\code\PointSuite')))
from pointsuite.data.datasets.dataset_bin1 import BinPklDataset1
from pointsuite.data.datasets.collate import collate_fn, DynamicBatchSampler
from torch.utils.data import DataLoader

# 创建 test 数据集
dataset = BinPklDataset1(
    data_root=test_dir,
    split='test',
    mode='voxel',
    max_loops=None,
)

print(f"\nDataset length: {len(dataset)}")

# 统计每个文件应有的样本数
expected_counts = Counter(Path(s['pkl_path']).stem for s in dataset.data_list)
print(f"\nExpected samples per file (from data_list):")
for name, count in sorted(expected_counts.items()):
    print(f"  {name}: {count}")

# 使用 DynamicBatchSampler
sampler = DynamicBatchSampler(
    dataset=dataset,
    max_points=800000,
    shuffle=False,
    drop_last=False,
)

print(f"\nDynamicBatchSampler info:")
print(f"  Estimated batches: {len(sampler)}")

# 统计实际遍历的样本
actual_indices = []
for batch_indices in sampler:
    actual_indices.extend(batch_indices)

print(f"  Actual samples yielded: {len(actual_indices)}")
print(f"  Missing samples: {len(dataset) - len(actual_indices)}")

# 统计实际遍历的文件
actual_counts = Counter()
for idx in actual_indices:
    file_stem = Path(dataset.data_list[idx]['pkl_path']).stem
    actual_counts[file_stem] += 1

print(f"\nActual samples per file (from sampler):")
for name, count in sorted(actual_counts.items()):
    print(f"  {name}: {count} (expected: {expected_counts[name]})")

# 检查是否有丢失
missing = set(expected_counts.keys()) - set(actual_counts.keys())
if missing:
    print(f"\n🚨 Missing files: {missing}")
else:
    print(f"\n✅ All files are covered")






