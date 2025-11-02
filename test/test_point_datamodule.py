"""
Test script for PointDataModule

Simple tests to verify the DataModule works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pointsuite.data.point_datamodule import PointDataModule


def test_initialization():
    """Test DataModule initialization"""
    print("\n" + "="*60)
    print("Test 1: DataModule Initialization")
    print("="*60)
    
    try:
        # This will fail if data_root doesn't exist, but tests initialization logic
        datamodule = PointDataModule(
            data_root='.',  # Use current dir for testing
            batch_size=4,
            num_workers=0,  # No workers for testing
        )
        print("✅ DataModule initialized successfully")
        print(f"   - Batch size: {datamodule.batch_size}")
        print(f"   - Num workers: {datamodule.num_workers}")
        print(f"   - Assets: {datamodule.assets}")
        print(f"   - Collate function: {type(datamodule.collate_fn).__name__}")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_hyperparameters():
    """Test hyperparameter saving"""
    print("\n" + "="*60)
    print("Test 2: Hyperparameter Saving")
    print("="*60)
    
    try:
        datamodule = PointDataModule(
            data_root='.',
            batch_size=8,
            num_workers=4,
            loop=2,
            cache_data=True,
        )
        
        # Check if hyperparameters are saved
        assert hasattr(datamodule, 'hparams'), "hparams not saved"
        assert datamodule.hparams.batch_size == 8, "batch_size not saved correctly"
        assert datamodule.hparams.num_workers == 4, "num_workers not saved correctly"
        assert datamodule.hparams.loop == 2, "loop not saved correctly"
        
        print("✅ Hyperparameters saved correctly")
        print(f"   - Saved hparams: {list(datamodule.hparams.keys())}")
        return True
    except Exception as e:
        print(f"❌ Hyperparameter test failed: {e}")
        return False


def test_collate_functions():
    """Test different collate function types"""
    print("\n" + "="*60)
    print("Test 3: Collate Functions")
    print("="*60)
    
    try:
        # Test default collate
        dm1 = PointDataModule(
            data_root='.',
            collate_fn_type='default',
        )
        assert dm1.collate_fn is not None
        print("✅ Default collate function initialized")
        
        # Test limited collate
        dm2 = PointDataModule(
            data_root='.',
            collate_fn_type='limited',
            max_points=100000,
        )
        assert dm2.collate_fn is not None
        print("✅ Limited collate function initialized")
        print(f"   - Max points: {dm2.collate_fn.max_points if hasattr(dm2.collate_fn, 'max_points') else 'N/A'}")
        
        return True
    except Exception as e:
        print(f"❌ Collate function test failed: {e}")
        return False


def test_class_mapping():
    """Test class mapping parameter"""
    print("\n" + "="*60)
    print("Test 4: Class Mapping")
    print("="*60)
    
    try:
        class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
        
        datamodule = PointDataModule(
            data_root='.',
            class_mapping=class_mapping,
        )
        
        assert datamodule.class_mapping == class_mapping
        print("✅ Class mapping set correctly")
        print(f"   - Mapping: {class_mapping}")
        return True
    except Exception as e:
        print(f"❌ Class mapping test failed: {e}")
        return False


def test_assets_configuration():
    """Test different asset configurations"""
    print("\n" + "="*60)
    print("Test 5: Assets Configuration")
    print("="*60)
    
    try:
        # Test default assets
        dm1 = PointDataModule(data_root='.')
        assert dm1.assets == ['coord', 'intensity', 'classification']
        print("✅ Default assets: ", dm1.assets)
        
        # Test custom assets
        custom_assets = ['coord', 'intensity', 'color', 'classification']
        dm2 = PointDataModule(data_root='.', assets=custom_assets)
        assert dm2.assets == custom_assets
        print("✅ Custom assets: ", dm2.assets)
        
        return True
    except Exception as e:
        print(f"❌ Assets configuration test failed: {e}")
        return False


def test_file_specification():
    """Test file specification options"""
    print("\n" + "="*60)
    print("Test 6: File Specification")
    print("="*60)
    
    try:
        # Test with file lists
        datamodule = PointDataModule(
            data_root='.',
            train_files=['train1.pkl', 'train2.pkl'],
            val_files=['val.pkl'],
            test_files=['test.pkl'],
        )
        
        assert datamodule.train_files == ['train1.pkl', 'train2.pkl']
        assert datamodule.val_files == ['val.pkl']
        assert datamodule.test_files == ['test.pkl']
        
        print("✅ File specification works correctly")
        print(f"   - Train files: {datamodule.train_files}")
        print(f"   - Val files: {datamodule.val_files}")
        print(f"   - Test files: {datamodule.test_files}")
        
        return True
    except Exception as e:
        print(f"❌ File specification test failed: {e}")
        return False


def test_methods_exist():
    """Test that all required methods exist"""
    print("\n" + "="*60)
    print("Test 7: Method Existence")
    print("="*60)
    
    try:
        datamodule = PointDataModule(data_root='.')
        
        required_methods = [
            'prepare_data',
            'setup',
            'train_dataloader',
            'val_dataloader',
            'test_dataloader',
            'predict_dataloader',
            'teardown',
            'get_dataset_info',
            'print_info',
        ]
        
        for method_name in required_methods:
            assert hasattr(datamodule, method_name), f"Missing method: {method_name}"
            assert callable(getattr(datamodule, method_name)), f"Not callable: {method_name}"
        
        print("✅ All required methods exist and are callable")
        print(f"   - Methods: {', '.join(required_methods)}")
        
        return True
    except Exception as e:
        print(f"❌ Method existence test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("PointDataModule Test Suite")
    print("="*80)
    
    tests = [
        test_initialization,
        test_hyperparameters,
        test_collate_functions,
        test_class_mapping,
        test_assets_configuration,
        test_file_specification,
        test_methods_exist,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    print("="*80 + "\n")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
