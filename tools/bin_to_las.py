"""
从bin+pkl文件中提取每个segment并保存为独立的LAS文件
用于在专业软件中可视化检查分块效果
"""
import numpy as np
import pickle
import laspy
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm


def create_las_from_segment(segment_data: np.ndarray, 
                            header_info: dict,
                            output_path: Union[str, Path]):
    """
    根据segment数据创建LAS文件
    
    Args:
        segment_data: 结构化数组，包含所有点属性
        header_info: 原始LAS文件的头信息
        output_path: 输出LAS文件路径
    """
    output_path = Path(output_path)
    
    # 创建LAS头
    header = laspy.LasHeader(
        point_format=header_info['point_format'],
        version=header_info['version']
    )
    
    # 设置坐标缩放和偏移
    header.x_scale = header_info['x_scale']
    header.y_scale = header_info['y_scale']
    header.z_scale = header_info['z_scale']
    header.x_offset = header_info['x_offset']
    header.y_offset = header_info['y_offset']
    header.z_offset = header_info['z_offset']
    
    # 设置其他头信息
    if 'system_identifier' in header_info:
        header.system_identifier = header_info['system_identifier']
    if 'generating_software' in header_info:
        header.generating_software = header_info['generating_software']
    
    # 创建LAS数据对象
    las = laspy.LasData(header)
    
    # 设置坐标（必须字段）
    las.x = segment_data['X']
    las.y = segment_data['Y']
    las.z = segment_data['Z']
    
    # 设置其他属性（如果存在）
    field_names = segment_data.dtype.names
    
    if 'intensity' in field_names:
        las.intensity = segment_data['intensity']
    if 'return_number' in field_names:
        las.return_number = segment_data['return_number']
    if 'number_of_returns' in field_names:
        las.number_of_returns = segment_data['number_of_returns']
    if 'classification' in field_names:
        las.classification = segment_data['classification']
    if 'scan_angle_rank' in field_names:
        las.scan_angle_rank = segment_data['scan_angle_rank']
    if 'user_data' in field_names:
        las.user_data = segment_data['user_data']
    if 'point_source_id' in field_names:
        las.point_source_id = segment_data['point_source_id']
    if 'gps_time' in field_names:
        las.gps_time = segment_data['gps_time']
    
    # RGB颜色（如果存在）
    if 'red' in field_names and 'green' in field_names and 'blue' in field_names:
        las.red = segment_data['red']
        las.green = segment_data['green']
        las.blue = segment_data['blue']
    if 'nir' in field_names:
        las.nir = segment_data['nir']
    
    # 额外字段（通过extra_bytes写入）
    extra_fields_to_add = []
    
    # is_ground 字段
    if 'is_ground' in field_names:
        extra_fields_to_add.append(('is_ground', segment_data['is_ground'], np.uint8))
    
    # 如果有额外字段，添加到LAS文件
    if extra_fields_to_add:
        # 为每个额外字段创建ExtraBytesParams
        for field_name, field_data, field_dtype in extra_fields_to_add:
            try:
                # 创建extra bytes定义
                extra_bytes = laspy.ExtraBytesParams(
                    name=field_name,
                    type=field_dtype
                )
                # 添加到header
                las.add_extra_dim(extra_bytes)
                # 设置数据
                setattr(las, field_name, field_data)
            except Exception as e:
                print(f"  ⚠️ 警告: 无法添加字段 {field_name}: {e}")
    
    # 保存LAS文件
    las.write(output_path)


def extract_segments_to_las(bin_path: Union[str, Path],
                            pkl_path: Union[str, Path],
                            output_dir: Union[str, Path],
                            segment_ids: Optional[list] = None,
                            max_segments: Optional[int] = None,
                            add_segment_id_suffix: bool = True):
    """
    从bin+pkl文件中提取segments并保存为LAS文件
    
    Args:
        bin_path: bin文件路径
        pkl_path: pkl文件路径
        output_dir: 输出目录
        segment_ids: 要提取的segment ID列表（None表示全部）
        max_segments: 最多提取多少个segment（None表示不限制）
        add_segment_id_suffix: 是否在文件名中添加segment_id后缀
    """
    bin_path = Path(bin_path)
    pkl_path = Path(pkl_path)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    print("="*70)
    print(f"从 {bin_path.name} 提取Segments到LAS文件")
    print("="*70)
    
    # 加载pkl元数据
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    total_segments = metadata['num_segments']
    header_info = metadata['header_info']
    base_name = bin_path.stem
    
    print(f"\n📊 文件信息:")
    print(f"  - 总点数: {metadata['num_points']:,}")
    print(f"  - 总segment数: {total_segments}")
    print(f"  - Grid Size: {metadata.get('grid_size', 'N/A')}")
    
    # 确定要提取的segment IDs
    if segment_ids is None:
        segment_ids = list(range(total_segments))
    
    if max_segments is not None:
        segment_ids = segment_ids[:max_segments]
    
    print(f"  - 将提取: {len(segment_ids)} 个segments")
    print(f"  - 输出目录: {output_dir}")
    
    # 使用memmap加载bin文件（节省内存）
    dtype = np.dtype(metadata['dtype'])
    mmap_data = np.memmap(bin_path, dtype=dtype, mode='r')
    
    print(f"\n🔄 开始提取segments...")
    
    # 提取每个segment
    success_count = 0
    for seg_id in tqdm(segment_ids, desc="提取segments", unit="seg"):
        try:
            segment_info = metadata['segments'][seg_id]
            indices = segment_info['indices']
            
            # 从memmap中读取数据
            segment_data = mmap_data[indices]
            
            # 生成输出文件名
            if add_segment_id_suffix:
                output_name = f"{base_name}_seg{seg_id:04d}.las"
            else:
                output_name = f"{base_name}_{seg_id}.las"
            
            output_path = output_dir / output_name
            
            # 创建LAS文件
            create_las_from_segment(segment_data, header_info, output_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n⚠️ Segment {seg_id} 提取失败: {e}")
            continue
    
    print(f"\n✅ 提取完成!")
    print(f"  - 成功: {success_count}/{len(segment_ids)} 个segments")
    print(f"  - 保存位置: {output_dir}")
    print("="*70)


def batch_extract_from_directory(bin_dir: Union[str, Path],
                                 output_base_dir: Union[str, Path],
                                 max_segments_per_file: Optional[int] = None):
    """
    批量处理目录下所有bin+pkl文件
    
    Args:
        bin_dir: 包含bin和pkl文件的目录
        output_base_dir: 输出根目录
        max_segments_per_file: 每个bin文件最多提取多少个segment
    """
    bin_dir = Path(bin_dir)
    output_base_dir = Path(output_base_dir)
    
    # 查找所有bin文件
    bin_files = list(bin_dir.glob('*.bin'))
    
    if not bin_files:
        print(f"❌ 目录 {bin_dir} 中没有找到bin文件")
        return
    
    print("="*70)
    print(f"批量提取Segments")
    print("="*70)
    print(f"输入目录: {bin_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"找到 {len(bin_files)} 个bin文件")
    print("="*70)
    
    for bin_file in bin_files:
        pkl_file = bin_file.with_suffix('.pkl')
        
        if not pkl_file.exists():
            print(f"\n⚠️ 跳过 {bin_file.name}: 找不到对应的pkl文件")
            continue
        
        # 为每个bin文件创建独立的输出目录
        output_dir = output_base_dir / bin_file.stem
        
        try:
            extract_segments_to_las(
                bin_path=bin_file,
                pkl_path=pkl_file,
                output_dir=output_dir,
                max_segments=max_segments_per_file
            )
            print()
        except Exception as e:
            print(f"\n❌ 处理 {bin_file.name} 失败: {e}")
            import traceback
            traceback.print_exc()
            print()


def extract_specific_segments(bin_path: Union[str, Path],
                              pkl_path: Union[str, Path],
                              output_dir: Union[str, Path],
                              segment_ids: list):
    """
    提取指定ID的segments
    
    Args:
        bin_path: bin文件路径
        pkl_path: pkl文件路径
        output_dir: 输出目录
        segment_ids: 要提取的segment ID列表
    """
    extract_segments_to_las(
        bin_path=bin_path,
        pkl_path=pkl_path,
        output_dir=output_dir,
        segment_ids=segment_ids
    )


if __name__ == "__main__":
    # ==================== 使用示例 ====================
    
    # 示例1: 提取单个bin文件的所有segments（限制数量避免生成过多文件）
    bin_file = r"E:\data\Dales\dales_las\bin\train\5080_54435.bin"
    pkl_file = r"E:\data\Dales\dales_las\bin\train\5080_54435.pkl"
    output_dir = r"E:\data\Dales\dales_las\bin\train\5080_54435_output"
    
    if Path(bin_file).exists() and Path(pkl_file).exists():
        # 只提取前20个segment作为示例
        extract_segments_to_las(
            bin_path=bin_file,
            pkl_path=pkl_file,
            output_dir=output_dir,
            # max_segments=20  # 限制提取数量
        )
    
    # 示例2: 提取指定的segments
    # specific_ids = [0, 5, 10, 15, 20]  # 指定要提取的segment IDs
    # extract_specific_segments(
    #     bin_path=bin_file,
    #     pkl_path=pkl_file,
    #     output_dir=output_dir,
    #     segment_ids=specific_ids
    # )
    
    # 示例3: 批量处理整个目录
    # bin_dir = r"E:\data\云南遥感中心\第一批\bin\train_with_gridsample"
    # output_base_dir = r"E:\data\云南遥感中心\第一批\las_tiles"
    # batch_extract_from_directory(
    #     bin_dir=bin_dir,
    #     output_base_dir=output_base_dir,
    #     max_segments_per_file=20  # 每个bin文件最多提取20个segment
    # )
