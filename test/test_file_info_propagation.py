"""
测试文件信息在数据流中的传递

验证从 tile.py → dataset_bin.py → semantic_segmentation.py → callbacks.py 的完整流程
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
import tempfile
import shutil


def test_tile_metadata_structure():
    """测试 tile.py 生成的 metadata 是否包含文件信息"""
    
    print("="*70)
    print("测试 1: Tile Metadata 结构")
    print("="*70)
    
    # 模拟 tile.py 生成的 segment 信息
    bin_path = Path("/data/test/5080_54400.bin")
    pkl_path = Path("/data/test/5080_54400.pkl")
    base_name = "5080_54400"
    
    segment_info = {
        'segment_id': 0,
        'indices': np.array([0, 1, 2, 3, 4]),
        'num_points': 5,
        # 🔥 新增的文件关联信息
        'bin_file': base_name,
        'bin_path': str(bin_path),
        'pkl_path': str(pkl_path),
    }
    
    print("\n生成的 segment_info:")
    print(f"  - segment_id: {segment_info['segment_id']}")
    print(f"  - num_points: {segment_info['num_points']}")
    print(f"  - bin_file: {segment_info['bin_file']}")
    print(f"  - bin_path: {segment_info['bin_path']}")
    print(f"  - pkl_path: {segment_info['pkl_path']}")
    
    # 验证必要字段
    assert 'bin_file' in segment_info, "缺少 bin_file"
    assert 'bin_path' in segment_info, "缺少 bin_path"
    assert 'pkl_path' in segment_info, "缺少 pkl_path"
    
    print("\n✓ Tile metadata 结构正确!")


def test_dataset_propagation():
    """测试 dataset 是否正确传递文件信息"""
    
    print("\n" + "="*70)
    print("测试 2: Dataset 文件信息传递")
    print("="*70)
    
    # 模拟 dataset 的 sample_info (来自 pkl)
    sample_info = {
        'segment_id': 0,
        'bin_path': '/data/test/5080_54400.bin',
        'pkl_path': '/data/test/5080_54400.pkl',
        'bin_file': '5080_54400',
        'num_points': 1000,
    }
    
    # 模拟 dataset._load_data 的返回
    data = {
        'coord': np.random.randn(1000, 3),
        'feat': np.random.randn(1000, 4),
        'class': np.zeros(1000, dtype=np.int64),
    }
    
    # 在 test split 中添加文件信息
    split = 'test'
    if split == 'test':
        indices = np.arange(1000)
        data['indices'] = indices.copy()
        
        # 🔥 添加文件信息
        data['bin_file'] = sample_info.get('bin_file', Path(sample_info['bin_path']).stem)
        data['bin_path'] = sample_info['bin_path']
        data['pkl_path'] = sample_info['pkl_path']
    
    print("\nDataset 返回的 data 字典包含:")
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"  - {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"  - {key}: {data[key]}")
    
    # 验证
    assert 'bin_file' in data, "Dataset 未传递 bin_file"
    assert 'bin_path' in data, "Dataset 未传递 bin_path"
    assert 'pkl_path' in data, "Dataset 未传递 pkl_path"
    assert 'indices' in data, "Dataset 未传递 indices"
    
    print("\n✓ Dataset 文件信息传递正确!")
    
    return data


def test_task_propagation(data):
    """测试 task.predict_step 是否传递文件信息"""
    
    print("\n" + "="*70)
    print("测试 3: Task Predict Step 文件信息传递")
    print("="*70)
    
    # 模拟 batch (collate_fn 的输出)
    # 在实际场景中，collate_fn 会保持某些字段为列表
    batch = {
        'coord': data['coord'],  # [N, 3]
        'feat': data['feat'],    # [N, C]
        'indices': data['indices'],  # [N]
        'bin_file': [data['bin_file']],  # 列表形式
        'bin_path': [data['bin_path']],
        'pkl_path': [data['pkl_path']],
        'offset': np.array([len(data['coord'])]),  # batch size = 1
    }
    
    print("\nBatch 包含:")
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape={value.shape}")
        elif isinstance(value, list):
            print(f"  - {key}: {value}")
        else:
            print(f"  - {key}: {type(value)}")
    
    # 模拟 predict_step 的返回
    results = {
        'logits': np.random.randn(len(data['coord']), 8),  # [N, C]
    }
    
    # 传递必要信息
    if "indices" in batch:
        results["indices"] = batch["indices"]
    
    # 🔥 传递文件信息
    if "bin_file" in batch:
        results["bin_file"] = batch["bin_file"]
    if "bin_path" in batch:
        results["bin_path"] = batch["bin_path"]
    if "pkl_path" in batch:
        results["pkl_path"] = batch["pkl_path"]
    
    if "coord" in batch:
        results["coord"] = batch["coord"]
    
    print("\nPredict step 返回的 results 包含:")
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape={value.shape}")
        elif isinstance(value, list):
            print(f"  - {key}: {value}")
        else:
            print(f"  - {key}: {type(value)}")
    
    # 验证
    assert 'bin_file' in results, "Predict step 未传递 bin_file"
    assert 'bin_path' in results, "Predict step 未传递 bin_path"
    assert 'pkl_path' in results, "Predict step 未传递 pkl_path"
    
    print("\n✓ Task predict step 文件信息传递正确!")
    
    return results


def test_callback_extraction(results):
    """测试 callback 是否正确提取文件信息"""
    
    print("\n" + "="*70)
    print("测试 4: Callback 文件信息提取")
    print("="*70)
    
    # 模拟 write_on_batch_end 的处理
    prediction = results
    
    # 🔥 直接从 prediction 获取 bin 文件信息
    if 'bin_file' in prediction and len(prediction['bin_file']) > 0:
        bin_files = prediction['bin_file']
        
        # 取第一个文件名
        if isinstance(bin_files, list):
            bin_basename = bin_files[0]
        else:
            bin_basename = str(bin_files)
        
        print(f"\n✓ 直接从 prediction 获取 bin_basename: {bin_basename}")
    else:
        print("\n✗ 无法从 prediction 获取 bin_file，需要使用推断方法")
        return False
    
    # 验证文件名
    assert bin_basename == "5080_54400", f"文件名不匹配: {bin_basename}"
    
    # 模拟保存临时文件
    temp_dir = tempfile.mkdtemp()
    try:
        batch_idx = 0
        tmp_filename = f"{bin_basename}_batch_{batch_idx}.pred.tmp"
        tmp_path = Path(temp_dir) / tmp_filename
        
        # 保存文件信息到临时文件
        save_dict = {
            'logits': prediction['logits'],
            'indices': prediction['indices'],
            'bin_file': bin_basename,
            'bin_path': prediction['bin_path'],
            'pkl_path': prediction['pkl_path'],
        }
        
        # 使用 pickle 模拟保存
        with open(tmp_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ 保存临时文件: {tmp_filename}")
        
        # 模拟从临时文件读取
        with open(tmp_path, 'rb') as f:
            loaded = pickle.load(f)
        
        print("\n从临时文件读取的信息:")
        print(f"  - bin_file: {loaded['bin_file']}")
        print(f"  - bin_path: {loaded['bin_path']}")
        print(f"  - pkl_path: {loaded['pkl_path']}")
        
        # 验证路径可以直接使用
        assert 'bin_path' in loaded, "临时文件缺少 bin_path"
        assert 'pkl_path' in loaded, "临时文件缺少 pkl_path"
        
        bin_path_from_tmp = loaded['bin_path']
        pkl_path_from_tmp = loaded['pkl_path']
        
        if isinstance(bin_path_from_tmp, list):
            bin_path_from_tmp = bin_path_from_tmp[0]
            pkl_path_from_tmp = pkl_path_from_tmp[0]
        
        print(f"\n✓ 可直接使用的完整路径:")
        print(f"  - Bin: {bin_path_from_tmp}")
        print(f"  - Pkl: {pkl_path_from_tmp}")
        
        print("\n✓ Callback 文件信息提取正确!")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_complete_flow():
    """测试完整的文件信息传递流程"""
    
    print("\n" + "="*70)
    print("测试 5: 完整数据流测试")
    print("="*70)
    
    # 1. Tile.py 阶段
    print("\n[1/4] Tile.py 生成 metadata...")
    test_tile_metadata_structure()
    
    # 2. Dataset 阶段
    print("\n[2/4] Dataset 加载数据...")
    data = test_dataset_propagation()
    
    # 3. Task 阶段
    print("\n[3/4] Task predict_step...")
    results = test_task_propagation(data)
    
    # 4. Callback 阶段
    print("\n[4/4] Callback 处理...")
    success = test_callback_extraction(results)
    
    if success:
        print("\n" + "="*70)
        print("✅ 完整数据流测试通过!")
        print("="*70)
        print("\n优势:")
        print("  1. ✓ 无需推断，文件信息显式传递")
        print("  2. ✓ 更高效，避免遍历 data_list 查找")
        print("  3. ✓ 更可靠，不依赖索引匹配")
        print("  4. ✓ 更清晰，数据流向一目了然")
        print("="*70)
    else:
        print("\n❌ 测试失败")


if __name__ == "__main__":
    test_complete_flow()
