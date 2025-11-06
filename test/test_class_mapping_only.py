"""测试类别映射功能"""
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointsuite.data.datasets.dataset_bin import BinPklDataset

data_root = r"E:\data\Dales\dales_las\bin\train"

# 定义类别映射
class_mapping = {
    0: -1,  # 噪声 -> -1 (ignore_label)
    1: 0,   # 地面 -> 0
    2: 1,   # 植被 -> 1
    3: 2,   # 汽车 -> 2
    4: 3,   # 卡车 -> 3
    5: 4,   # 电线杆 -> 4
    6: 5,   # 围栏 -> 5
    7: 6,   # 建筑 -> 6
    8: 7,   # 其他 -> 7
}

print("创建带类别映射的数据集...")
dataset = BinPklDataset(
    data_root=data_root,
    split='train',
    assets=['coord', 'class'],
    transform=None,
    ignore_label=-1,
    class_mapping=class_mapping
)

print(f"数据集创建成功，加载第一个样本...")
sample = dataset[0]

print(f"\n样本加载完成")
print(f"键: {list(sample.keys())}")
print(f"class shape: {sample['class'].shape}")
print(f"class dtype: {sample['class'].dtype}")

unique_classes, counts = np.unique(sample['class'], return_counts=True)
print(f"\n映射后的类别分布:")
for cls, cnt in zip(unique_classes, counts):
    label_name = "ignore" if cls == -1 else f"类别 {cls}"
    print(f"  {label_name}: {cnt:,} 点 ({cnt/len(sample['class'])*100:.2f}%)")
