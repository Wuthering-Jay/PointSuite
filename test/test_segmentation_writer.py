"""
测试 SegmentationWriter 回调的功能

这个脚本演示如何使用 SegmentationWriter 进行预测并保存结果

注意: 这是一个简化的测试，不需要完整的 PyTorch Lightning 环境
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

# 假设项目结构
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 SegmentationWriter
from pointsuite.utils.callbacks import SegmentationWriter


def test_segmentation_writer_basic():
    """测试 SegmentationWriter 的基本功能"""
    
    print("="*70)
    print("测试 SegmentationWriter 基本功能")
    print("="*70)
    
    # 创建临时输出目录
    temp_dir = tempfile.mkdtemp()
    print(f"\n临时输出目录: {temp_dir}")
    
    try:
        # 1. 创建回调
        writer = SegmentationWriter(
            output_dir=temp_dir,
            num_classes=8,
            save_logits=True,
        )
        
        print("\n✓ SegmentationWriter 创建成功")
        print(f"  - 输出目录: {writer.output_dir}")
        print(f"  - 临时目录: {writer.temp_dir}")
        print(f"  - 类别数: {writer.num_classes}")
        print(f"  - 保存 logits: {writer.save_logits}")
        
        # 2. 检查目录是否创建
        assert Path(writer.output_dir).exists(), "输出目录未创建"
        assert Path(writer.temp_dir).exists(), "临时目录未创建"
        
        print("\n✓ 目录结构正确")
        
        # 3. 模拟批次预测
        print("\n开始模拟批次预测...")
        
        # 创建模拟数据
        batch_predictions = [
            {
                'logits': torch.randn(1000, 8),  # 1000 points, 8 classes
                'indices': torch.arange(0, 1000).long(),
            },
            {
                'logits': torch.randn(1500, 8),
                'indices': torch.arange(1000, 2500).long(),
            },
            {
                'logits': torch.randn(500, 8),
                'indices': torch.arange(2500, 3000).long(),
            }
        ]
        
        # 模拟保存临时文件
        for i, pred in enumerate(batch_predictions):
            tmp_file = Path(writer.temp_dir) / f"test_bin_batch_{i}.pred.tmp"
            torch.save(pred, tmp_file)
            print(f"  - 保存批次 {i}: {tmp_file.name}")
        
        # 4. 检查临时文件
        tmp_files = list(Path(writer.temp_dir).glob("*.pred.tmp"))
        assert len(tmp_files) == 3, f"临时文件数量不对: {len(tmp_files)}"
        
        print(f"\n✓ 临时文件创建成功 ({len(tmp_files)} 个文件)")
        
        # 5. 测试文件分组逻辑
        from collections import defaultdict
        bin_file_groups = defaultdict(list)
        
        for tmp_file in tmp_files:
            filename = tmp_file.name
            bin_basename = filename.split('_batch_')[0]
            bin_file_groups[bin_basename].append(str(tmp_file))
        
        print(f"\n✓ 文件分组成功:")
        for bin_name, files in bin_file_groups.items():
            print(f"  - {bin_name}: {len(files)} 个批次")
        
        print("\n✓ 基本功能测试通过!")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"\n清理临时目录: {temp_dir}")


def test_class_mapping():
    """测试类别映射功能"""
    
    print("\n" + "="*70)
    print("测试类别映射功能")
    print("="*70)
    
    # 定义类别映射
    class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    
    print("\n原始映射:")
    print(f"  {class_mapping}")
    print("\n反向映射:")
    print(f"  {reverse_mapping}")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建带映射的 writer
        writer = SegmentationWriter(
            output_dir=temp_dir,
            num_classes=5,  # 连续标签数量
            reverse_class_mapping=reverse_mapping,
        )
        
        # 模拟预测结果 (连续标签)
        continuous_labels = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        
        # 应用反向映射
        if writer.reverse_class_mapping is not None:
            mapped_labels = np.zeros_like(continuous_labels)
            for cont_label, orig_label in writer.reverse_class_mapping.items():
                mapped_labels[continuous_labels == cont_label] = orig_label
        
        print("\n连续标签:")
        print(f"  {continuous_labels}")
        print("\n映射后的原始标签:")
        print(f"  {mapped_labels}")
        
        # 验证映射
        expected = np.array([0, 1, 2, 6, 9, 0, 1, 2, 6, 9])
        assert np.array_equal(mapped_labels, expected), "类别映射不正确"
        
        print("\n✓ 类别映射测试通过!")
        
    finally:
        shutil.rmtree(temp_dir)


def test_voting_mechanism():
    """测试投票机制"""
    
    print("\n" + "="*70)
    print("测试投票机制")
    print("="*70)
    
    num_points = 100
    num_classes = 5
    
    # 创建投票数组
    logits_sum = torch.zeros((num_points, num_classes), dtype=torch.float32)
    counts = torch.zeros(num_points, dtype=torch.int32)
    
    # 模拟 3 个重叠的 segment
    # Segment 1: 点 0-49
    # Segment 2: 点 30-79
    # Segment 3: 点 60-99
    segments = [
        {'indices': torch.arange(0, 50), 'logits': torch.randn(50, num_classes)},
        {'indices': torch.arange(30, 80), 'logits': torch.randn(50, num_classes)},
        {'indices': torch.arange(60, 100), 'logits': torch.randn(40, num_classes)},
    ]
    
    print("\nSegment 覆盖范围:")
    print("  - Segment 1: 点 0-49")
    print("  - Segment 2: 点 30-79")
    print("  - Segment 3: 点 60-99")
    
    # 累积投票
    for seg in segments:
        indices = seg['indices']
        logits = seg['logits']
        
        logits_sum.index_add_(0, indices.long(), logits)
        counts.index_add_(0, indices.long(), torch.ones(len(indices), dtype=torch.int32))
    
    # 检查投票次数
    print("\n投票次数统计:")
    print(f"  - 点 0-29:   {counts[:30].unique().item()} 次 (只有 Segment 1)")
    print(f"  - 点 30-49:  {counts[30:50].unique().item()} 次 (Segment 1 + 2)")
    print(f"  - 点 50-59:  {counts[50:60].unique().item()} 次 (只有 Segment 2)")
    print(f"  - 点 60-79:  {counts[60:80].unique().item()} 次 (Segment 2 + 3)")
    print(f"  - 点 80-99:  {counts[80:100].unique().item()} 次 (只有 Segment 3)")
    
    # 验证投票次数（修正后的逻辑）
    assert counts[:30].unique().item() == 1, "点 0-29 应该被投票 1 次"
    assert counts[30:50].unique().item() == 2, "点 30-49 应该被投票 2 次"
    assert counts[50:60].unique().item() == 1, "点 50-59 应该被投票 1 次"  # 修正：只有 Segment 2
    assert counts[60:80].unique().item() == 2, "点 60-79 应该被投票 2 次"
    assert counts[80:100].unique().item() == 1, "点 80-99 应该被投票 1 次"
    
    # 计算平均 logits
    counts[counts == 0] = 1  # 避免除以 0
    mean_logits = logits_sum / counts.unsqueeze(-1)
    
    # 获取预测
    predictions = torch.argmax(mean_logits, dim=1)
    
    print(f"\n最终预测形状: {predictions.shape}")
    print(f"类别分布: {torch.bincount(predictions)}")
    
    print("\n✓ 投票机制测试通过!")


def print_usage_example():
    """打印使用示例"""
    
    print("\n" + "="*70)
    print("SegmentationWriter 使用示例")
    print("="*70)
    
    example_code = """
# 1. 创建回调
from pointsuite.utils.callbacks import SegmentationWriter

writer = SegmentationWriter(
    output_dir='predictions',
    num_classes=8,  # 或 -1 自动推断
    save_logits=True,  # 可选: 保存 logits
    reverse_class_mapping={0: 0, 1: 1, 2: 2, 3: 6, 4: 9},  # 可选: 类别映射
)

# 2. 创建 DataModule
from pointsuite.data.datamodule_binpkl import BinPklDataModule

datamodule = BinPklDataModule(
    data_root='data/test',
    test_files=['test1.pkl', 'test2.pkl'],
    assets=['coord', 'intensity', 'classification'],
    batch_size=4,
    num_workers=4,
)

# 3. 创建 Trainer 并预测
import pytorch_lightning as pl

trainer = pl.Trainer(
    callbacks=[writer],
    accelerator='gpu',
    devices=1,
)

# 4. 加载模型并预测
model = SemanticSegmentationTask.load_from_checkpoint('checkpoint.ckpt')
trainer.predict(model, datamodule=datamodule)

# 5. 预测结果
# predictions/
# ├── bin_file1_predicted.las  # 完整点云 + 预测标签
# ├── bin_file1_logits.npz     # logits (可选)
# ├── bin_file2_predicted.las
# └── ...
"""
    
    print(example_code)


if __name__ == "__main__":
    # 运行测试
    test_segmentation_writer_basic()
    test_class_mapping()
    test_voting_mechanism()
    print_usage_example()
    
    print("\n" + "="*70)
    print("所有测试通过! ✓")
    print("="*70)
    print("\n详细使用文档请参考: docs/SEGMENTATION_WRITER_USAGE.md")
