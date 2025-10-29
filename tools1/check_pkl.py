import pickle
from pathlib import Path

# 检查一个pkl文件中保存的字段
pkl_file = Path(r"E:\data\Dales\dales_las\tile_bin\test\5080_54400.pkl")

with open(pkl_file, 'rb') as f:
    metadata = pickle.load(f)

print("=" * 60)
print(f"文件: {pkl_file.name}")
print("=" * 60)
print(f"\n总点数: {metadata['num_points']}")
print(f"分块数: {metadata['num_segments']}")
print(f"\n保存的字段 ({len(metadata['fields'])} 个):")
for i, field in enumerate(metadata['fields'], 1):
    print(f"  {i}. {field}")

print(f"\n数据类型:")
for field_name, dtype in metadata['dtype']:
    print(f"  {field_name}: {dtype}")

print(f"\n类别分布:")
for label, count in metadata['label_counts'].items():
    print(f"  类别 {label}: {count:,} 点")
