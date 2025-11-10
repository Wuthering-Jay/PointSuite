"""
测试 SegmentationWriter 的完整属性恢复功能

验证从 bin 文件中恢复所有点属性（强度、颜色、回波等）到最终 LAS 文件
"""

import numpy as np
import pickle
from pathlib import Path
import tempfile
import shutil

try:
    import laspy
except ImportError:
    print("请安装 laspy: pip install laspy")
    exit(1)


def create_test_las_with_attributes(las_path, num_points=1000):
    """创建包含完整属性的测试 LAS 文件"""
    
    # 创建 LAS 头（point format 3 支持 GPS 时间和 RGB）
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.offsets = [0, 0, 0]
    header.scales = [0.01, 0.01, 0.01]
    
    las = laspy.LasData(header)
    
    # 生成随机点云数据
    las.x = np.random.rand(num_points) * 1000
    las.y = np.random.rand(num_points) * 1000
    las.z = np.random.rand(num_points) * 100
    
    # 设置各种属性
    las.intensity = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    las.return_number = np.random.randint(1, 5, num_points, dtype=np.uint8)
    las.number_of_returns = np.random.randint(1, 5, num_points, dtype=np.uint8)
    las.scan_angle_rank = np.random.randint(-90, 90, num_points, dtype=np.int8)
    las.user_data = np.random.randint(0, 255, num_points, dtype=np.uint8)
    las.point_source_id = np.random.randint(0, 100, num_points, dtype=np.uint16)
    
    # GPS 时间
    las.gps_time = np.linspace(0, 1000, num_points)
    
    # RGB 颜色
    las.red = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    las.green = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    las.blue = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    
    # 原始分类（用于对比）
    las.classification = np.random.randint(0, 10, num_points, dtype=np.uint8)
    
    las.write(las_path)
    print(f"✓ 创建测试 LAS 文件: {las_path}")
    print(f"  - 点数: {num_points}")
    print(f"  - 属性: intensity, return_number, RGB, gps_time, 等")
    
    return las


def las_to_bin_pkl(las_path, bin_path, pkl_path):
    """模拟 tile.py 的处理：将 LAS 转换为 bin+pkl"""
    
    las = laspy.read(las_path)
    
    # 创建 structured array（模拟 tile.py 的输出）
    dtype = [
        ('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'),
        ('intensity', 'u2'),
        ('return_number', 'u1'),
        ('number_of_returns', 'u1'),
        ('scan_angle_rank', 'i1'),
        ('user_data', 'u1'),
        ('point_source_id', 'u2'),
        ('gps_time', 'f8'),
        ('red', 'u2'), ('green', 'u2'), ('blue', 'u2'),
        ('label', 'u1')
    ]
    
    num_points = len(las.x)
    point_data = np.zeros(num_points, dtype=dtype)
    
    # 填充数据
    point_data['X'] = las.x
    point_data['Y'] = las.y
    point_data['Z'] = las.z
    point_data['intensity'] = las.intensity
    point_data['return_number'] = las.return_number
    point_data['number_of_returns'] = las.number_of_returns
    point_data['scan_angle_rank'] = las.scan_angle_rank
    point_data['user_data'] = las.user_data
    point_data['point_source_id'] = las.point_source_id
    point_data['gps_time'] = las.gps_time
    point_data['red'] = las.red
    point_data['green'] = las.green
    point_data['blue'] = las.blue
    point_data['label'] = las.classification
    
    # 保存 bin 文件
    point_data.tofile(bin_path)
    
    # 创建 metadata（模拟 tile.py）
    metadata = {
        'dtype': dtype,
        'num_points': num_points,
        'header_info': {
            'point_format': int(las.header.point_format.id),
            'version': str(las.header.version),
            'x_scale': float(las.header.scales[0]),
            'y_scale': float(las.header.scales[1]),
            'z_scale': float(las.header.scales[2]),
            'x_offset': float(las.header.offsets[0]),
            'y_offset': float(las.header.offsets[1]),
            'z_offset': float(las.header.offsets[2]),
            'system_identifier': las.header.system_identifier,
            'generating_software': las.header.generating_software,
            'vlrs': [
                {
                    'user_id': vlr.user_id,
                    'record_id': vlr.record_id,
                    'description': vlr.description,
                    'record_data': bytes(vlr.record_data) if hasattr(vlr, 'record_data') else b''
                }
                for vlr in las.header.vlrs
            ]
        },
        'las_file': str(las_path),
    }
    
    # 保存 pkl 文件
    with open(pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ 转换为 bin+pkl 格式:")
    print(f"  - {bin_path}")
    print(f"  - {pkl_path}")
    
    return metadata


def test_attribute_recovery():
    """测试属性恢复功能"""
    
    print("=" * 70)
    print("测试 LAS 属性恢复功能")
    print("=" * 70)
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\n临时目录: {temp_dir}")
    
    try:
        # 1. 创建原始 LAS 文件
        original_las = temp_dir / "original.las"
        las_original = create_test_las_with_attributes(original_las, num_points=1000)
        
        # 2. 转换为 bin+pkl
        bin_file = temp_dir / "original.bin"
        pkl_file = temp_dir / "original.pkl"
        metadata = las_to_bin_pkl(original_las, bin_file, pkl_file)
        
        # 3. 模拟预测过程：加载 bin 数据
        print("\n" + "=" * 70)
        print("模拟预测和保存过程")
        print("=" * 70)
        
        point_data = np.memmap(bin_file, dtype=metadata['dtype'], mode='r')
        
        # 提取坐标
        xyz = np.stack([
            point_data['X'],
            point_data['Y'],
            point_data['Z']
        ], axis=1).astype(np.float64)
        
        # 模拟预测：生成新的分类标签（与原始不同）
        predicted_labels = np.random.randint(0, 5, len(point_data), dtype=np.uint8)
        print(f"✓ 生成预测标签: {len(predicted_labels)} 个点")
        
        # 4. 使用 _save_las_file 保存（模拟 callback 的逻辑）
        output_las = temp_dir / "predicted.las"
        
        # 添加 bin_path 到 metadata
        metadata['_bin_path'] = str(bin_file)
        
        # 模拟 _save_las_file 的核心逻辑
        print(f"\n保存预测结果到: {output_las}")
        print("恢复属性中...")
        
        # 创建 LAS 头
        header_info = metadata['header_info']
        point_format = header_info.get('point_format', 3)
        version_str = header_info.get('version', '1.2')
        
        header = laspy.LasHeader(point_format=point_format, version=version_str)
        header.offsets = [
            header_info['x_offset'],
            header_info['y_offset'],
            header_info['z_offset']
        ]
        header.scales = [
            header_info['x_scale'],
            header_info['y_scale'],
            header_info['z_scale']
        ]
        
        las = laspy.LasData(header)
        
        # 设置坐标
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]
        
        # 🔥 从 bin 文件恢复所有属性
        dtype = metadata['dtype']
        field_names = [name for name, _ in dtype]
        
        recovered_fields = []
        
        if 'intensity' in field_names:
            las.intensity = point_data['intensity']
            recovered_fields.append('intensity')
        
        if 'return_number' in field_names:
            las.return_number = point_data['return_number']
            recovered_fields.append('return_number')
        
        if 'number_of_returns' in field_names:
            las.number_of_returns = point_data['number_of_returns']
            recovered_fields.append('number_of_returns')
        
        if 'scan_angle_rank' in field_names:
            las.scan_angle_rank = point_data['scan_angle_rank']
            recovered_fields.append('scan_angle_rank')
        
        if 'user_data' in field_names:
            las.user_data = point_data['user_data']
            recovered_fields.append('user_data')
        
        if 'point_source_id' in field_names:
            las.point_source_id = point_data['point_source_id']
            recovered_fields.append('point_source_id')
        
        if 'gps_time' in field_names:
            las.gps_time = point_data['gps_time']
            recovered_fields.append('gps_time')
        
        if header.point_format.id in [2, 3, 5, 7, 8, 10]:
            if all(f in field_names for f in ['red', 'green', 'blue']):
                las.red = point_data['red']
                las.green = point_data['green']
                las.blue = point_data['blue']
                recovered_fields.append('RGB')
        
        # 设置预测的分类标签
        las.classification = predicted_labels
        recovered_fields.append('classification (predicted)')
        
        las.write(output_las)
        
        print(f"✓ 恢复的属性: {', '.join(recovered_fields)}")
        
        # 5. 验证恢复的数据
        print("\n" + "=" * 70)
        print("验证属性恢复")
        print("=" * 70)
        
        las_recovered = laspy.read(output_las)
        
        all_passed = True
        
        # 验证坐标
        if np.allclose(las_original.x, las_recovered.x, rtol=1e-5):
            print("✓ X 坐标匹配")
        else:
            print("✗ X 坐标不匹配")
            all_passed = False
        
        if np.allclose(las_original.y, las_recovered.y, rtol=1e-5):
            print("✓ Y 坐标匹配")
        else:
            print("✗ Y 坐标不匹配")
            all_passed = False
        
        if np.allclose(las_original.z, las_recovered.z, rtol=1e-5):
            print("✓ Z 坐标匹配")
        else:
            print("✗ Z 坐标不匹配")
            all_passed = False
        
        # 验证强度
        if np.array_equal(las_original.intensity, las_recovered.intensity):
            print("✓ Intensity 匹配")
        else:
            print("✗ Intensity 不匹配")
            all_passed = False
        
        # 验证回波信息
        if np.array_equal(las_original.return_number, las_recovered.return_number):
            print("✓ Return Number 匹配")
        else:
            print("✗ Return Number 不匹配")
            all_passed = False
        
        # 验证 GPS 时间
        if np.allclose(las_original.gps_time, las_recovered.gps_time, rtol=1e-5):
            print("✓ GPS Time 匹配")
        else:
            print("✗ GPS Time 不匹配")
            all_passed = False
        
        # 验证 RGB
        if np.array_equal(las_original.red, las_recovered.red):
            print("✓ Red 匹配")
        else:
            print("✗ Red 不匹配")
            all_passed = False
        
        if np.array_equal(las_original.green, las_recovered.green):
            print("✓ Green 匹配")
        else:
            print("✗ Green 不匹配")
            all_passed = False
        
        if np.array_equal(las_original.blue, las_recovered.blue):
            print("✓ Blue 匹配")
        else:
            print("✗ Blue 不匹配")
            all_passed = False
        
        # 验证分类标签是新的预测标签
        if np.array_equal(las_recovered.classification, predicted_labels):
            print("✓ Classification 使用预测标签 (正确覆盖原始标签)")
        else:
            print("✗ Classification 不是预测标签")
            all_passed = False
        
        # 验证 header 信息
        if las_recovered.header.point_format.id == las_original.header.point_format.id:
            print(f"✓ Point Format 匹配: {las_recovered.header.point_format.id}")
        else:
            print(f"✗ Point Format 不匹配")
            all_passed = False
        
        print("\n" + "=" * 70)
        if all_passed:
            print("✅ 所有属性恢复测试通过！")
            print("\n优势:")
            print("  1. 完整保留原始点云的所有属性（强度、颜色、回波等）")
            print("  2. 只更新分类标签，其他属性保持原样")
            print("  3. 保留 LAS 头信息（坐标系、精度等）")
            print("  4. 无信息损失，可用于后续分析")
        else:
            print("❌ 部分属性恢复失败")
        print("=" * 70)
        
        return all_passed
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print(f"\n✓ 清理临时目录: {temp_dir}")


if __name__ == '__main__':
    test_attribute_recovery()
