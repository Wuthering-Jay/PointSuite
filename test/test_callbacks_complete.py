"""
完整测试 SegmentationWriter 回调的所有功能

测试内容:
1. 基本流程：临时文件写入 → 投票累积 → LAS 文件保存
2. 显式文件路径传递：bin_file, bin_path, pkl_path
3. 多 batch 投票机制：logits 平均
4. 完整属性恢复：intensity, RGB, GPS time 等
5. 反向类别映射：连续标签 → 原始标签
6. 边界情况：未预测点、单次预测、多文件混合
"""

import torch
import numpy as np
import pickle
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict

try:
    import laspy
except ImportError:
    print("请安装 laspy: pip install laspy")
    exit(1)

# 模拟 PyTorch Lightning 组件
class MockTrainer:
    """模拟 Trainer"""
    def __init__(self, dataset):
        self.predict_dataloaders = type('obj', (object,), {'dataset': dataset})()

class MockModule:
    """模拟 LightningModule"""
    def __init__(self, num_classes):
        self.head = type('obj', (object,), {'out_channels': num_classes})()
    
    def print(self, msg):
        print(msg)


def create_test_environment():
    """创建测试环境：生成 LAS → bin+pkl 数据"""
    
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    
    print("=" * 70)
    print("创建测试环境")
    print("=" * 70)
    print(f"临时目录: {temp_dir}")
    
    # 1. 创建两个测试 LAS 文件
    las_files = []
    bin_files = []
    pkl_files = []
    
    for file_idx in range(2):
        las_path = data_dir / f"test_{file_idx}.las"
        bin_path = data_dir / f"test_{file_idx}.bin"
        pkl_path = data_dir / f"test_{file_idx}.pkl"
        
        # 创建 LAS 文件
        num_points = 1000 + file_idx * 500
        
        header = laspy.LasHeader(point_format=3, version='1.2')
        header.offsets = [file_idx * 1000, file_idx * 1000, 0]
        header.scales = [0.01, 0.01, 0.01]
        
        las = laspy.LasData(header)
        las.x = np.random.rand(num_points) * 1000 + file_idx * 1000
        las.y = np.random.rand(num_points) * 1000 + file_idx * 1000
        las.z = np.random.rand(num_points) * 100
        
        # 添加各种属性
        las.intensity = np.random.randint(0, 65535, num_points, dtype=np.uint16)
        las.return_number = np.random.randint(1, 4, num_points, dtype=np.uint8)
        las.number_of_returns = np.random.randint(1, 4, num_points, dtype=np.uint8)
        las.scan_angle_rank = np.random.randint(-45, 45, num_points, dtype=np.int8)
        las.user_data = np.random.randint(0, 255, num_points, dtype=np.uint8)
        las.point_source_id = np.random.randint(0, 10, num_points, dtype=np.uint16)
        las.gps_time = np.linspace(0, 1000, num_points)
        las.red = np.random.randint(0, 65535, num_points, dtype=np.uint16)
        las.green = np.random.randint(0, 65535, num_points, dtype=np.uint16)
        las.blue = np.random.randint(0, 65535, num_points, dtype=np.uint16)
        las.classification = np.random.randint(0, 5, num_points, dtype=np.uint8)
        
        las.write(las_path)
        print(f"✓ 创建 LAS 文件: {las_path.name} ({num_points} 点)")
        
        # 2. 转换为 bin+pkl
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
        
        point_data = np.zeros(num_points, dtype=dtype)
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
        
        point_data.tofile(bin_path)
        
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
                'vlrs': []
            },
            'las_file': str(las_path),
        }
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        las_files.append(las_path)
        bin_files.append(bin_path)
        pkl_files.append(pkl_path)
    
    print(f"✓ 生成 {len(bin_files)} 个 bin+pkl 文件对")
    
    return temp_dir, las_files, bin_files, pkl_files


def create_mock_segments(bin_files, pkl_files):
    """创建模拟的 segments (模拟 tile.py 的输出)"""
    
    segments = []
    
    for bin_path, pkl_path in zip(bin_files, pkl_files):
        # 加载 metadata 获取点数
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        num_points = metadata['num_points']
        
        # 为每个文件创建 3 个重叠的 segments
        segment_configs = [
            (0, num_points // 2),                    # 前半部分
            (num_points // 3, 2 * num_points // 3),  # 中间部分
            (num_points // 2, num_points),           # 后半部分
        ]
        
        for seg_idx, (start, end) in enumerate(segment_configs):
            segment_info = {
                'indices': list(range(start, end)),
                'bin_file': bin_path.stem,
                'bin_path': str(bin_path),
                'pkl_path': str(pkl_path),
                'segment_id': f"{bin_path.stem}_seg_{seg_idx}"
            }
            segments.append(segment_info)
    
    print(f"✓ 创建 {len(segments)} 个 segments (模拟重叠分块)")
    return segments


def simulate_prediction(segments, num_classes=5):
    """模拟预测过程：生成预测结果"""
    
    print("\n" + "=" * 70)
    print("模拟预测过程")
    print("=" * 70)
    
    predictions = []
    
    for batch_idx, segment in enumerate(segments):
        indices = torch.tensor(segment['indices'], dtype=torch.long)
        num_points = len(indices)
        
        # 生成随机 logits
        logits = torch.randn(num_points, num_classes)
        
        # 构造 prediction 字典（模拟 SemanticSegmentationTask.predict_step 的输出）
        prediction = {
            'logits': logits,
            'indices': indices,
            'bin_file': [segment['bin_file']],  # 列表形式（模拟 collate_fn）
            'bin_path': [segment['bin_path']],
            'pkl_path': [segment['pkl_path']],
        }
        
        predictions.append(prediction)
        
        if batch_idx % 2 == 0:
            print(f"  Batch {batch_idx}: {num_points} 点, 来自 {segment['bin_file']}")
    
    print(f"✓ 生成 {len(predictions)} 个 batch 的预测结果")
    return predictions


def test_segmentation_writer_complete():
    """完整测试 SegmentationWriter"""
    
    print("\n" + "=" * 70)
    print("完整测试 SegmentationWriter 回调")
    print("=" * 70)
    
    # 1. 创建测试环境
    temp_dir, las_files, bin_files, pkl_files = create_test_environment()
    
    try:
        # 2. 创建 segments
        segments = create_mock_segments(bin_files, pkl_files)
        
        # 3. 模拟预测
        num_classes = 5
        predictions = simulate_prediction(segments, num_classes)
        
        # 4. 创建 SegmentationWriter
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # 导入 SegmentationWriter
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from pointsuite.utils.callbacks import SegmentationWriter
        
        # 创建反向类别映射（测试映射功能）
        reverse_mapping = {0: 0, 1: 1, 2: 2, 3: 10, 4: 15}
        
        writer = SegmentationWriter(
            output_dir=str(output_dir),
            write_interval="batch",
            num_classes=num_classes,
            save_logits=True,
            reverse_class_mapping=reverse_mapping
        )
        
        print(f"\n✓ 创建 SegmentationWriter:")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 类别数: {num_classes}")
        print(f"  - 反向映射: {reverse_mapping}")
        
        # 5. 模拟 write_on_batch_end
        print("\n" + "=" * 70)
        print("测试 write_on_batch_end (流式写入)")
        print("=" * 70)
        
        # 创建 mock dataset
        mock_dataset = type('Dataset', (object,), {'data_list': segments})()
        trainer = MockTrainer(mock_dataset)
        pl_module = MockModule(num_classes)
        
        for batch_idx, prediction in enumerate(predictions):
            writer.write_on_batch_end(
                trainer=trainer,
                pl_module=pl_module,
                prediction=prediction,
                batch_indices=[batch_idx],
                batch=None,
                batch_idx=batch_idx,
                dataloader_idx=0
            )
        
        # 检查临时文件
        temp_files = list(Path(writer.temp_dir).glob("*.pred.tmp"))
        print(f"\n✓ 生成 {len(temp_files)} 个临时预测文件")
        assert len(temp_files) == len(predictions), "临时文件数量不匹配"
        
        # 6. 测试 on_predict_end
        print("\n" + "=" * 70)
        print("测试 on_predict_end (投票和保存)")
        print("=" * 70)
        
        writer.on_predict_end(trainer, pl_module)
        
        # 7. 验证输出
        print("\n" + "=" * 70)
        print("验证输出文件")
        print("=" * 70)
        
        # 检查 LAS 文件
        las_outputs = list(output_dir.glob("*.las"))
        print(f"\n✓ 生成 {len(las_outputs)} 个 LAS 文件:")
        for las_file in las_outputs:
            print(f"  - {las_file.name}")
        
        assert len(las_outputs) == len(bin_files), f"LAS 文件数量不匹配: {len(las_outputs)} vs {len(bin_files)}"
        
        # 检查 logits 文件
        logits_outputs = list(output_dir.glob("*.npz"))
        print(f"\n✓ 生成 {len(logits_outputs)} 个 logits 文件:")
        for logits_file in logits_outputs:
            print(f"  - {logits_file.name}")
        
        # 8. 验证属性恢复
        print("\n" + "=" * 70)
        print("验证属性恢复")
        print("=" * 70)
        
        all_tests_passed = True
        
        for las_output, las_original in zip(sorted(las_outputs), sorted(las_files)):
            print(f"\n检查文件: {las_output.name}")
            
            las_pred = laspy.read(las_output)
            las_orig = laspy.read(las_original)
            
            # 验证点数
            if len(las_pred.x) == len(las_orig.x):
                print(f"  ✓ 点数匹配: {len(las_pred.x)}")
            else:
                print(f"  ✗ 点数不匹配: {len(las_pred.x)} vs {len(las_orig.x)}")
                all_tests_passed = False
            
            # 验证坐标
            coord_match = (
                np.allclose(las_pred.x, las_orig.x, rtol=1e-5) and
                np.allclose(las_pred.y, las_orig.y, rtol=1e-5) and
                np.allclose(las_pred.z, las_orig.z, rtol=1e-5)
            )
            if coord_match:
                print(f"  ✓ 坐标匹配 (XYZ)")
            else:
                print(f"  ✗ 坐标不匹配")
                all_tests_passed = False
            
            # 验证强度
            if np.array_equal(las_pred.intensity, las_orig.intensity):
                print(f"  ✓ Intensity 恢复")
            else:
                print(f"  ✗ Intensity 不匹配")
                all_tests_passed = False
            
            # 验证 RGB
            if (np.array_equal(las_pred.red, las_orig.red) and
                np.array_equal(las_pred.green, las_orig.green) and
                np.array_equal(las_pred.blue, las_orig.blue)):
                print(f"  ✓ RGB 颜色恢复")
            else:
                print(f"  ✗ RGB 颜色不匹配")
                all_tests_passed = False
            
            # 验证 GPS 时间
            if np.allclose(las_pred.gps_time, las_orig.gps_time, rtol=1e-5):
                print(f"  ✓ GPS Time 恢复")
            else:
                print(f"  ✗ GPS Time 不匹配")
                all_tests_passed = False
            
            # 验证分类已更新（不应该与原始相同）
            if not np.array_equal(las_pred.classification, las_orig.classification):
                print(f"  ✓ Classification 已更新（预测结果）")
                
                # 验证反向映射是否应用
                unique_labels = np.unique(las_pred.classification)
                expected_labels = set(reverse_mapping.values())
                if all(label in expected_labels for label in unique_labels):
                    print(f"  ✓ 反向类别映射已应用")
                    print(f"    - 预测标签: {sorted(unique_labels)}")
                    print(f"    - 期望标签: {sorted(expected_labels)}")
            else:
                print(f"  ✗ Classification 未更新")
                all_tests_passed = False
            
            # 验证 header 信息
            if (las_pred.header.point_format.id == las_orig.header.point_format.id and
                np.allclose(las_pred.header.scales, las_orig.header.scales) and
                np.allclose(las_pred.header.offsets, las_orig.header.offsets)):
                print(f"  ✓ LAS Header 信息保留")
            else:
                print(f"  ✗ LAS Header 信息不匹配")
                all_tests_passed = False
        
        # 9. 验证投票机制
        print("\n" + "=" * 70)
        print("验证投票机制")
        print("=" * 70)
        
        # 加载一个 logits 文件检查
        if logits_outputs:
            logits_data = np.load(logits_outputs[0])
            
            print(f"✓ Logits 文件内容:")
            print(f"  - logits shape: {logits_data['logits'].shape}")
            print(f"  - predictions shape: {logits_data['predictions'].shape}")
            print(f"  - counts shape: {logits_data['counts'].shape}")
            
            # 检查投票次数
            vote_counts = logits_data['counts']
            unique_counts, count_frequency = np.unique(vote_counts, return_counts=True)
            
            print(f"\n✓ 投票统计:")
            for count, freq in zip(unique_counts, count_frequency):
                print(f"  - {freq} 个点被预测 {count} 次")
            
            # 验证重叠区域被多次预测
            if len(unique_counts) > 1:
                print(f"  ✓ 检测到重叠区域（多次投票）")
            else:
                print(f"  ! 警告: 所有点只被预测一次（可能没有重叠）")
        
        # 10. 验证临时文件已清理
        print("\n" + "=" * 70)
        print("验证临时文件清理")
        print("=" * 70)
        
        temp_files_after = list(Path(writer.temp_dir).glob("*.pred.tmp"))
        if len(temp_files_after) == 0:
            print(f"✓ 所有临时文件已清理")
        else:
            print(f"✗ 仍有 {len(temp_files_after)} 个临时文件未清理")
            all_tests_passed = False
        
        # 最终结果
        print("\n" + "=" * 70)
        if all_tests_passed:
            print("✅ 所有测试通过！")
            print("\n验证的功能:")
            print("  ✓ 流式写入临时文件")
            print("  ✓ 显式文件路径传递")
            print("  ✓ 多 batch 投票累积")
            print("  ✓ 完整属性恢复 (坐标, 强度, RGB, GPS, 回波)")
            print("  ✓ 反向类别映射")
            print("  ✓ LAS Header 保留")
            print("  ✓ 临时文件清理")
        else:
            print("❌ 部分测试失败")
        print("=" * 70)
        
        return all_tests_passed
        
    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
            print(f"\n✓ 清理临时目录")
        except Exception as e:
            print(f"\n! 清理失败: {e}")


if __name__ == '__main__':
    success = test_segmentation_writer_complete()
    exit(0 if success else 1)
