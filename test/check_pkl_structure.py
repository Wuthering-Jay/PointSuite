"""
快速检查 pkl 文件的结构
"""
import pickle
from pathlib import Path

pkl_file = Path(r"E:\data\DALES\dales_las\bin\train").glob('*.pkl')
pkl_file = next(pkl_file)

print(f"检查文件: {pkl_file}")

with open(pkl_file, 'rb') as f:
    metadata = pickle.load(f)

print("\nMetadata keys:")
for key in metadata.keys():
    print(f"  - {key}")

print("\n第一个 segment 的内容:")
if 'segments' in metadata and len(metadata['segments']) > 0:
    seg = metadata['segments'][0]
    for key, value in seg.items():
        print(f"  - {key}: {value}")
