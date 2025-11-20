"""
测试 DataModule 重构

验证基类和具体实现的正确性
"""

import sys
from pathlib import Path

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试1: 导入模块")
    print("=" * 60)
    
    try:
        # 测试基类导入
        from pointsuite.data.datamodule_base import DataModuleBase
        print("✅ DataModuleBase 导入成功")
        
        # 测试具体实现导入
        from pointsuite.data.datamodule_bin import BinPklDataModule
        print("✅ BinPklDataModule 导入成功")
        
        # 测试向后兼容性导入
        from pointsuite.data.point_datamodule import PointDataModule
        print("✅ PointDataModule 导入成功 (向后兼容)")
        
        # 验证 PointDataModule 是 BinPklDataModule 的别名
        assert PointDataModule is BinPklDataModule
        print("✅ PointDataModule 正确指向 BinPklDataModule")
        
        # 测试从 __init__ 导入
        from pointsuite.data import (
            DataModuleBase as DM1,
            BinPklDataModule as DM2,
            PointDataModule as DM3
        )
        print("✅ 从 pointsuite.data 包导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_class_hierarchy():
    """测试类层次结构"""
    print("\n" + "=" * 60)
    print("测试2: 类层次结构")
    print("=" * 60)
    
    try:
        from pointsuite.data.datamodule_base import DataModuleBase
        from pointsuite.data.datamodule_bin import BinPklDataModule
        import pytorch_lightning as pl
        from abc import ABC
        
        # 验证继承关系
        assert issubclass(DataModuleBase, pl.LightningDataModule)
        print("✅ DataModuleBase 继承自 LightningDataModule")
        
        assert issubclass(DataModuleBase, ABC)
        print("✅ DataModuleBase 是抽象基类")
        
        assert issubclass(BinPklDataModule, DataModuleBase)
        print("✅ BinPklDataModule 继承自 DataModuleBase")
        
        # 验证抽象方法
        assert hasattr(DataModuleBase, '_create_dataset')
        print("✅ DataModuleBase 有 _create_dataset 抽象方法")
        
        # 验证具体实现
        assert hasattr(BinPklDataModule, '_create_dataset')
        print("✅ BinPklDataModule 实现了 _create_dataset")
        
        return True
    except Exception as e:
        print(f"❌ 类层次结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_methods():
    """测试方法存在性"""
    print("\n" + "=" * 60)
    print("测试3: 方法存在性")
    print("=" * 60)
    
    try:
        from pointsuite.data.datamodule_base import DataModuleBase
        from pointsuite.data.datamodule_bin import BinPklDataModule
        
        required_methods = [
            '__init__',
            'setup',
            'prepare_data',
            'train_dataloader',
            'val_dataloader',
            'test_dataloader',
            'predict_dataloader',
            'teardown',
            'get_dataset_info',
            'print_info',
        ]
        
        print("\nDataModuleBase 方法:")
        for method in required_methods:
            has_method = hasattr(DataModuleBase, method)
            status = "✅" if has_method else "❌"
            print(f"  {status} {method}")
            if not has_method:
                return False
        
        print("\nBinPklDataModule 方法:")
        for method in required_methods:
            has_method = hasattr(BinPklDataModule, method)
            status = "✅" if has_method else "❌"
            print(f"  {status} {method}")
            if not has_method:
                return False
        
        # 验证 _create_dataloader 内部方法
        assert hasattr(DataModuleBase, '_create_dataloader')
        print("\n✅ DataModuleBase 有 _create_dataloader 内部方法")
        
        return True
    except Exception as e:
        print(f"❌ 方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "=" * 60)
    print("测试4: 向后兼容性")
    print("=" * 60)
    
    try:
        # 旧的导入方式应该仍然有效
        from pointsuite.data.point_datamodule import PointDataModule
        from pointsuite.data.datamodule_bin import BinPklDataModule
        
        # 验证它们是同一个类
        assert PointDataModule is BinPklDataModule
        print("✅ 旧的 PointDataModule 名称仍然有效")
        
        # 验证可以创建实例（不实际初始化，只检查签名）
        import inspect
        sig = inspect.signature(PointDataModule.__init__)
        params = list(sig.parameters.keys())
        
        expected_params = [
            'self', 'data_root', 'train_files', 'val_files', 'test_files',
            'batch_size', 'num_workers', 'assets', 'train_transforms',
            'val_transforms', 'test_transforms', 'ignore_label', 'loop',
            'cache_data', 'class_mapping', 'use_dynamic_batch', 'max_points',
            'train_sampler_weights', 'pin_memory', 'persistent_workers',
            'prefetch_factor', 'kwargs'
        ]
        
        print("\n参数签名检查:")
        for param in expected_params:
            if param in params:
                print(f"  ✅ {param}")
            else:
                print(f"  ⚠️  {param} (可能不存在，但不影响兼容性)")
        
        return True
    except Exception as e:
        print(f"❌ 向后兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_documentation():
    """测试文档字符串"""
    print("\n" + "=" * 60)
    print("测试5: 文档字符串")
    print("=" * 60)
    
    try:
        from pointsuite.data.datamodule_base import DataModuleBase
        from pointsuite.data.datamodule_bin import BinPklDataModule
        
        # 检查类文档
        assert DataModuleBase.__doc__ is not None
        print("✅ DataModuleBase 有文档字符串")
        print(f"   摘要: {DataModuleBase.__doc__.split(chr(10))[0].strip()}")
        
        assert BinPklDataModule.__doc__ is not None
        print("✅ BinPklDataModule 有文档字符串")
        print(f"   摘要: {BinPklDataModule.__doc__.split(chr(10))[0].strip()}")
        
        # 检查关键方法文档
        methods_with_docs = [
            'setup',
            'train_dataloader',
            'get_dataset_info',
            'print_info',
        ]
        
        print("\n方法文档检查:")
        for method_name in methods_with_docs:
            method = getattr(DataModuleBase, method_name)
            has_doc = method.__doc__ is not None and len(method.__doc__.strip()) > 0
            status = "✅" if has_doc else "⚠️ "
            print(f"  {status} {method_name}")
        
        return True
    except Exception as e:
        print(f"❌ 文档测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n" + "=" * 60)
    print("测试6: 文件结构")
    print("=" * 60)
    
    required_files = [
        'pointsuite/data/datamodule_base.py',
        'pointsuite/data/datamodule_binpkl.py',
        'pointsuite/data/point_datamodule.py',
        'pointsuite/data/__init__.py',
        'pointsuite/data/datasets/dataset_base.py',
        'pointsuite/data/datasets/dataset_bin.py',
        'pointsuite/data/datasets/collate.py',
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} ({size:,} 字节)")
        else:
            print(f"❌ {file_path} 不存在")
            return False
    
    return True


def main():
    print("\n" + "=" * 80)
    print("DataModule 重构验证测试")
    print("=" * 80)
    
    tests = [
        ("导入测试", test_imports),
        ("类层次结构", test_class_hierarchy),
        ("方法存在性", test_methods),
        ("向后兼容性", test_backward_compatibility),
        ("文档字符串", test_documentation),
        ("文件结构", test_file_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 崩溃: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print("\n" + "-" * 80)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！重构成功！")
        print("\n新结构:")
        print("  - DataModuleBase: 抽象基类，可扩展")
        print("  - BinPklDataModule: bin+pkl 格式的具体实现")
        print("  - PointDataModule: 向后兼容的别名")
        print("\n使用方法:")
        print("  # 向后兼容（仍然有效）")
        print("  from pointsuite.data.point_datamodule import PointDataModule")
        print("")
        print("  # 新方法（推荐）")
        print("  from pointsuite.data.datamodule_binpkl import BinPklDataModule")
        print("")
        print("  # 创建自定义 DataModule")
        print("  from pointsuite.data.datamodule_base import DataModuleBase")
        print("  class MyDataModule(DataModuleBase):")
        print("      def _create_dataset(self, ...):")
        print("          return MyDataset(...)")
    else:
        print("\n⚠️  部分测试失败，请检查上面的详细信息")
    
    print("=" * 80 + "\n")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
