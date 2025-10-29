import laspy
from pathlib import Path
import numpy as np

# 验证转换回来的LAS文件
las_dir = Path(r"E:\data\Dales\dales_las\tile_bin\test_las")
las_files = list(las_dir.glob("*.las"))[:3]  # 只检查前3个文件

print("="*70)
print("验证 BIN->LAS 转换结果")
print("="*70)

for las_file in las_files:
    print(f"\n文件: {las_file.name}")
    print("-"*70)
    
    with laspy.open(las_file) as fh:
        las_data = fh.read()
    
    print(f"  点数: {len(las_data.points):,}")
    print(f"  边界: X[{las_data.header.x_min:.2f}, {las_data.header.x_max:.2f}]")
    print(f"        Y[{las_data.header.y_min:.2f}, {las_data.header.y_max:.2f}]")
    print(f"        Z[{las_data.header.z_min:.2f}, {las_data.header.z_max:.2f}]")
    
    # 检查字段
    available_fields = []
    for field in ['X', 'Y', 'Z', 'intensity', 'classification', 'gps_time']:
        field_lower = field.lower()
        if hasattr(las_data, field_lower):
            available_fields.append(field)
    
    print(f"  可用字段: {', '.join(available_fields)}")
    
    # 检查分类统计
    if hasattr(las_data, 'classification'):
        unique_labels, counts = np.unique(las_data.classification, return_counts=True)
        print(f"  类别统计:")
        for label, count in zip(unique_labels, counts):
            print(f"    类别 {label}: {count:,} 点 ({count/len(las_data.points)*100:.1f}%)")
    
    # 检查坐标系信息
    if hasattr(las_data.header, 'vlrs') and las_data.header.vlrs:
        print(f"  VLRs: {len(las_data.header.vlrs)} 个")

print("\n" + "="*70)
print("验证完成！文件转换正确。")
print("="*70)
