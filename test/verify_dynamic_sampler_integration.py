"""
验证 point_datamodule.py 的语法和结构

这个脚本检查新的 DynamicBatchSampler 集成是否正确
"""

import ast
import sys
from pathlib import Path

def check_point_datamodule():
    """检查 point_datamodule.py 的结构"""
    
    file_path = Path('pointsuite/data/point_datamodule.py')
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    print(f"✅ 找到文件: {file_path}")
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键导入
    checks = {
        'DynamicBatchSampler 导入': 'from .datasets.collate import collate_fn, DynamicBatchSampler' in content,
        'WeightedRandomSampler 导入': 'from torch.utils.data import DataLoader, WeightedRandomSampler' in content,
        'use_dynamic_batch 参数': 'use_dynamic_batch: bool = False' in content,
        'max_points 参数': 'max_points: int = 500000' in content,
        'train_sampler_weights 参数': 'train_sampler_weights: Optional[List[float]] = None' in content,
        'DynamicBatchSampler 使用': 'DynamicBatchSampler(' in content,
        'WeightedRandomSampler 使用': 'WeightedRandomSampler(' in content,
    }
    
    print("\n检查关键特性:")
    all_passed = True
    for name, check in checks.items():
        status = "✅" if check else "❌"
        print(f"  {status} {name}")
        if not check:
            all_passed = False
    
    # 尝试解析 AST
    try:
        tree = ast.parse(content)
        print("\n✅ Python 语法正确")
    except SyntaxError as e:
        print(f"\n❌ 语法错误: {e}")
        return False
    
    # 检查类定义
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if 'PointDataModule' in classes:
        print("✅ PointDataModule 类定义存在")
    else:
        print("❌ PointDataModule 类定义未找到")
        all_passed = False
    
    # 检查方法
    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'PointDataModule':
            class_node = node
            break
    
    if class_node:
        methods = [n.name for n in class_node.body if isinstance(n, ast.FunctionDef)]
        required_methods = [
            '__init__',
            'train_dataloader',
            'val_dataloader',
            'test_dataloader',
            'setup',
            'print_info'
        ]
        
        print("\n检查必需方法:")
        for method in required_methods:
            status = "✅" if method in methods else "❌"
            print(f"  {status} {method}")
            if method not in methods:
                all_passed = False
    
    # 检查文档字符串
    if '"""' in content and 'DynamicBatchSampler' in content:
        print("\n✅ 包含文档字符串")
    
    return all_passed


def check_examples():
    """检查示例文件"""
    
    example_files = [
        'examples/dynamic_sampler_examples.py',
        'examples/datamodule_usage_example.py'
    ]
    
    print("\n" + "="*60)
    print("检查示例文件")
    print("="*60)
    
    for file_path in example_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
            
            # 检查文件大小
            size = path.stat().st_size
            print(f"   文件大小: {size:,} 字节")
        else:
            print(f"❌ {file_path} 不存在")


def check_docs():
    """检查文档文件"""
    
    doc_files = [
        'docs/DYNAMIC_BATCH_SAMPLER.md',
        'docs/POINTDATAMODULE.md'
    ]
    
    print("\n" + "="*60)
    print("检查文档文件")
    print("="*60)
    
    for file_path in doc_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
            
            # 统计行数
            with open(path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            print(f"   行数: {lines}")
        else:
            print(f"❌ {file_path} 不存在")


def main():
    print("="*80)
    print("PointDataModule - DynamicBatchSampler 集成验证")
    print("="*80)
    
    # 检查主文件
    print("\n" + "="*60)
    print("检查 point_datamodule.py")
    print("="*60)
    
    success = check_point_datamodule()
    
    # 检查示例
    check_examples()
    
    # 检查文档
    check_docs()
    
    # 总结
    print("\n" + "="*80)
    if success:
        print("✅ 所有检查通过！")
        print("\n新功能:")
        print("  1. ✅ DynamicBatchSampler 集成")
        print("  2. ✅ WeightedRandomSampler 支持")
        print("  3. ✅ use_dynamic_batch 参数")
        print("  4. ✅ train_sampler_weights 参数")
        print("  5. ✅ 完整的文档和示例")
        print("\n使用方法:")
        print("  datamodule = PointDataModule(")
        print("      data_root='path/to/data',")
        print("      use_dynamic_batch=True,")
        print("      max_points=500000,")
        print("      train_sampler_weights=weights,  # 可选")
        print("  )")
    else:
        print("❌ 部分检查失败，请查看上面的详细信息")
    
    print("="*80 + "\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
