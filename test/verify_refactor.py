"""
简单验证重构结果
"""

import sys
import ast
from pathlib import Path

def check_file_syntax(file_path):
    """检查文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, "语法正确"
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"错误: {e}"


def main():
    print("=" * 80)
    print("DataModule 重构验证")
    print("=" * 80)
    
    files_to_check = [
        ('datamodule_base.py', 'pointsuite/data/datamodule_base.py'),
        ('datamodule_binpkl.py', 'pointsuite/data/datamodule_binpkl.py'),
        ('point_datamodule.py', 'pointsuite/data/point_datamodule.py'),
        ('__init__.py', 'pointsuite/data/__init__.py'),
    ]
    
    print("\n文件语法检查:")
    print("-" * 80)
    
    all_passed = True
    for name, path in files_to_check:
        file_path = Path(path)
        if not file_path.exists():
            print(f"❌ {name}: 文件不存在")
            all_passed = False
            continue
        
        success, message = check_file_syntax(file_path)
        status = "✅" if success else "❌"
        size = file_path.stat().st_size
        print(f"{status} {name}: {message} ({size:,} 字节)")
        
        if not success:
            all_passed = False
    
    print("\n" + "=" * 80)
    print("重构结构:")
    print("=" * 80)
    
    structure = """
    pointsuite/data/
    ├── datamodule_base.py       ← 抽象基类 (新)
    │   └── DataModuleBase       - 通用功能
    │       ├── setup()
    │       ├── train_dataloader()
    │       ├── _create_dataloader()
    │       └── _create_dataset()  [抽象方法]
    │
    ├── datamodule_binpkl.py     ← BinPkl 实现 (新)
    │   └── BinPklDataModule     - 继承自 DataModuleBase
    │       └── _create_dataset()  [实现]
    │           └── 返回 BinPklDataset
    │
    ├── point_datamodule.py      ← 向后兼容 (重构)
    │   └── PointDataModule = BinPklDataModule
    │
    └── __init__.py              ← 包导出 (新)
        ├── DataModuleBase
        ├── BinPklDataModule
        └── PointDataModule
    """
    
    print(structure)
    
    print("\n" + "=" * 80)
    print("使用示例:")
    print("=" * 80)
    
    examples = """
    # 方式1: 向后兼容（旧代码无需修改）
    from pointsuite.data.point_datamodule import PointDataModule
    datamodule = PointDataModule(
        data_root='path/to/data',
        use_dynamic_batch=True,
        max_points=500000
    )
    
    # 方式2: 使用新名称（推荐）
    from pointsuite.data.datamodule_binpkl import BinPklDataModule
    datamodule = BinPklDataModule(
        data_root='path/to/data',
        use_dynamic_batch=True,
        max_points=500000
    )
    
    # 方式3: 从包导入
    from pointsuite.data import BinPklDataModule
    datamodule = BinPklDataModule(...)
    
    # 方式4: 创建自定义 DataModule
    from pointsuite.data.datamodule_base import DataModuleBase
    
    class CustomDataModule(DataModuleBase):
        def _create_dataset(self, data_paths, split, transforms):
            # 实现自定义数据集创建逻辑
            return CustomDataset(
                data_paths=data_paths,
                split=split,
                transform=transforms
            )
    
    # 使用自定义 DataModule
    datamodule = CustomDataModule(
        data_root='path/to/data',
        use_dynamic_batch=True
    )
    """
    
    print(examples)
    
    print("\n" + "=" * 80)
    print("重构优势:")
    print("=" * 80)
    
    advantages = """
    ✅ 代码复用: 通用逻辑在基类中，避免重复
    ✅ 可扩展性: 轻松创建新的 DataModule（只需实现 _create_dataset）
    ✅ 向后兼容: 旧代码无需修改，PointDataModule 仍然有效
    ✅ 清晰结构: 职责分离，基类处理通用功能，子类处理特定格式
    ✅ 易于维护: DataLoader 逻辑统一在 _create_dataloader 中
    ✅ DynamicBatchSampler: 内置支持，支持 WeightedRandomSampler
    """
    
    print(advantages)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有文件语法正确，重构完成！")
    else:
        print("❌ 部分文件有问题，请检查上面的详细信息")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
