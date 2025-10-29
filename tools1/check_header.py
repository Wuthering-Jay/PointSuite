import pickle
from pathlib import Path

# 检查完整的头文件信息
pkl_file = Path(r"E:\data\Dales\dales_las\tile_bin\test\5080_54400.pkl")

with open(pkl_file, 'rb') as f:
    metadata = pickle.load(f)

header_info = metadata['header_info']

print("=" * 70)
print(f"完整头文件信息: {pkl_file.name}")
print("=" * 70)

print("\n【基本信息】")
print(f"  LAS版本: {header_info.get('version')}")
print(f"  点格式: {header_info.get('point_format')}")
print(f"  点数量: {header_info.get('point_count'):,}")

print("\n【缩放和偏移】")
print(f"  X scale/offset: {header_info.get('x_scale')} / {header_info.get('x_offset')}")
print(f"  Y scale/offset: {header_info.get('y_scale')} / {header_info.get('y_offset')}")
print(f"  Z scale/offset: {header_info.get('z_scale')} / {header_info.get('z_offset')}")

print("\n【边界范围】")
print(f"  X: [{header_info.get('x_min')}, {header_info.get('x_max')}]")
print(f"  Y: [{header_info.get('y_min')}, {header_info.get('y_max')}]")
print(f"  Z: [{header_info.get('z_min')}, {header_info.get('z_max')}]")

if 'system_identifier' in header_info:
    print(f"\n【系统信息】")
    print(f"  系统标识: {header_info.get('system_identifier')}")

if 'generating_software' in header_info:
    print(f"  生成软件: {header_info.get('generating_software')}")

if 'creation_date' in header_info:
    print(f"  创建日期: {header_info.get('creation_date')}")

if 'global_encoding' in header_info:
    print(f"  全局编码: {header_info.get('global_encoding')}")

if 'vlrs' in header_info and header_info['vlrs']:
    print(f"\n【坐标系信息 (VLRs)】")
    print(f"  VLR数量: {len(header_info['vlrs'])}")
    for i, vlr in enumerate(header_info['vlrs'], 1):
        print(f"  VLR #{i}:")
        print(f"    User ID: {vlr.get('user_id')}")
        print(f"    Record ID: {vlr.get('record_id')}")
        print(f"    Description: {vlr.get('description')}")
        if 'record_data' in vlr:
            print(f"    Data size: {len(vlr['record_data'])} bytes")

if 'crs' in header_info and header_info['crs']:
    print(f"\n【坐标参考系统 (CRS)】")
    print(f"  {header_info.get('crs')}")

print(f"\n【数据字段】")
print(f"  字段数量: {len(metadata['fields'])}")
print(f"  字段列表: {', '.join(metadata['fields'])}")

print(f"\n【分块统计】")
print(f"  总分块数: {metadata['num_segments']}")
print(f"  窗口大小: {metadata['window_size']}")
print(f"  最小点数: {metadata['min_points']}")
print(f"  最大点数: {metadata['max_points']}")
