"""检查遗漏的 pkl 文件问题 - 深入调试"""
import pickle
from pathlib import Path
import sys
import os
sys.path.insert(0, r'E:\code\PointSuite')

from pointsuite.data.datasets import BinPklDataset
from pointsuite.data.datasets.collate import DynamicBatchSampler, collate_fn
from torch.utils.data import DataLoader
from pointsuite.data.transforms import CenterShift, Collect, ToTensor
from collections import Counter, defaultdict
from tqdm import tqdm

# 检查所有测试文件
test_dir = Path(r'E:\data\DALES\dales_las\bin_logical\test')
pkl_files = sorted(test_dir.glob('*.pkl'))

print("=" * 60)
print("检查 batch 中 bin_file 的数据结构")
print("=" * 60)

# 预测时的 transforms
predict_transforms = [
    CenterShift(),
    Collect(keys=['coord', 'indices', 'bin_file', 'bin_path', 'pkl_path'],
            feat_keys={'feat': ['coord', 'echo']}),
    ToTensor(),
]

# 创建数据集实例
dataset = BinPklDataset(
    data_root=test_dir,
    split='predict',
    mode='voxel',
    max_loops=None,
    assets=['coord', 'class', 'echo'],
    transform=predict_transforms,
)

# 创建 DynamicBatchSampler
batch_sampler = DynamicBatchSampler(
    dataset=dataset,
    max_points=125000,
    shuffle=False,
    drop_last=False,
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=collate_fn,
    num_workers=0,
)

# 只检查前几个 batch
for batch_idx, batch in enumerate(dataloader):
    print(f"\nBatch {batch_idx}:")
    print(f"  Keys: {list(batch.keys())}")
    
    if 'bin_file' in batch:
        bf = batch['bin_file']
        print(f"  bin_file type: {type(bf)}")
        if isinstance(bf, list):
            print(f"  bin_file count: {len(bf)}")
            print(f"  bin_file sample: {bf[:3] if len(bf) > 3 else bf}")
        else:
            print(f"  bin_file value: {bf}")
    else:
        print("  ⚠️ bin_file NOT in batch!")
        
    if 'offset' in batch:
        print(f"  offset shape: {batch['offset'].shape}")
        print(f"  offset values: {batch['offset'].tolist()[:5]}...")
        
    if batch_idx >= 2:
        break

# 计算 5175_54395 的样本起始索引
print("\n" + "=" * 60)
print("检查 5175_54395 的样本索引范围")
print("=" * 60)

file_indices = defaultdict(list)
for idx, item in enumerate(dataset.data_list):
    file_indices[item['file_name']].append(idx)

for file_name in sorted(file_indices.keys()):
    indices = file_indices[file_name]
    print(f"{file_name}: indices [{indices[0]}..{indices[-1]}], 共 {len(indices)} 个样本")
