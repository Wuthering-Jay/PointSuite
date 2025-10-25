"""
将H5格式中的分块点云数据转换为LAS文件
"""

import numpy as np
import laspy
import h5py
from pathlib import Path
from typing import Union, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse


def convert_segment_to_las(args: Tuple) -> Tuple[int, bool, str]:
    """
    Worker function: 转换单个segment到LAS文件（快速格式）
    
    Args:
        args: (h5_path, segment_idx, output_dir, header_info)
    
    Returns:
        (segment_idx, success, message)
    """
    h5_path, seg_idx, output_dir, header_info = args
    
    try:
        with h5py.File(h5_path, 'r') as f:
            seg_group = f['segments'][f'segment_{seg_idx:04d}']
            
            # 读取xyz（可能是X,Y,Z或x,y,z）
            x_key = 'X' if 'X' in seg_group else 'x'
            y_key = 'Y' if 'Y' in seg_group else 'y'
            z_key = 'Z' if 'Z' in seg_group else 'z'
            
            x_data = seg_group[x_key][:]
            y_data = seg_group[y_key][:]
            z_data = seg_group[z_key][:]
            num_points = len(x_data)
            
            # 创建LAS header
            header = laspy.LasHeader(
                point_format=header_info['point_format'],
                version=header_info['version']
            )
            header.x_scale = header_info['x_scale']
            header.y_scale = header_info['y_scale']
            header.z_scale = header_info['z_scale']
            header.x_offset = header_info['x_offset']
            header.y_offset = header_info['y_offset']
            header.z_offset = header_info['z_offset']
            
            # 创建LAS数据
            new_las = laspy.LasData(header)
            new_las.points = laspy.ScaleAwarePointRecord.zeros(num_points, header=header)
            
            # 设置坐标
            new_las.x = x_data
            new_las.y = y_data
            new_las.z = z_data
            
            # 读取所有其他可用字段
            available_fields = header_info.get('available_fields', [])
            for field in available_fields:
                # 跳过已处理的坐标字段
                if field in ['x', 'y', 'z', 'X', 'Y', 'Z']:
                    continue
                
                # H5中的字段名（大写）
                h5_field = field
                # LAS中的字段名（小写）
                las_field = field.lower()
                
                if h5_field in seg_group and hasattr(new_las, las_field):
                    try:
                        field_data = seg_group[h5_field][:]
                        setattr(new_las, las_field, field_data)
                    except Exception as e:
                        print(f"警告: 设置字段 {field} 失败: {e}")
            
            # 设置CRS（如果有）
            if header_info.get('crs'):
                try:
                    new_las.crs = header_info['crs']
                except:
                    pass
            
            # 保存文件
            h5_stem = Path(h5_path).stem
            output_path = Path(output_dir) / f"{h5_stem}_segment_{seg_idx:04d}.las"
            new_las.write(output_path)
            
            return (seg_idx, True, f"Success ({num_points} points)")
            
    except Exception as e:
        return (seg_idx, False, str(e))


def h5_fast_to_las(
    h5_path: str,
    output_dir: Optional[str] = None,
    segment_indices: Optional[List[int]] = None,
    n_workers: int = 1
) -> None:
    """
    将快速H5格式转换为LAS文件（支持并行）
    
    Args:
        h5_path: H5文件路径
        output_dir: 输出目录
        segment_indices: 要转换的segment索引列表（None表示全部）
        n_workers: 并行worker数量（1表示串行）
    """
    h5_path = Path(h5_path)
    
    if output_dir is None:
        output_dir = h5_path.parent / f"{h5_path.stem}_segments"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"打开H5文件: {h5_path.name}")
    
    # 读取header信息
    with h5py.File(h5_path, 'r') as f:
        header_group = f['header']
        
        # 快速格式的header结构
        header_info = {
            'point_format': int(header_group.attrs.get('point_format', 3)),
            'version_major': int(header_group.attrs.get('version_major', 1)),
            'version_minor': int(header_group.attrs.get('version_minor', 2)),
            'x_scale': float(header_group.attrs['x_scale']),
            'y_scale': float(header_group.attrs['y_scale']),
            'z_scale': float(header_group.attrs['z_scale']),
            'x_offset': float(header_group.attrs['x_offset']),
            'y_offset': float(header_group.attrs['y_offset']),
            'z_offset': float(header_group.attrs['z_offset']),
            'crs': header_group.attrs.get('crs', None),
        }
        
        # 构建version字符串
        header_info['version'] = f"{header_info['version_major']}.{header_info['version_minor']}"
        
        # 读取可用字段列表
        if 'available_fields' in header_group.attrs:
            available_fields_str = header_group.attrs['available_fields']
            if isinstance(available_fields_str, bytes):
                available_fields_str = available_fields_str.decode('utf-8')
            header_info['available_fields'] = available_fields_str.split(',')
        else:
            header_info['available_fields'] = []
        
        num_segments = f['segments'].attrs['num_segments']
    
    # 确定要转换的segments
    if segment_indices is None:
        segment_indices = list(range(num_segments))
    
    print(f"将 {len(segment_indices)} 个segments转换为LAS文件...")
    
    # 准备参数
    args_list = [(h5_path, seg_idx, output_dir, header_info) 
                for seg_idx in segment_indices]
    
    # 转换
    success_count = 0
    failed_segments = []
    total_points = 0
    
    if n_workers > 1:
        # 并行处理
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(convert_segment_to_las, args): args[1] 
                      for args in args_list}
            
            for future in tqdm(as_completed(futures), total=len(segment_indices),
                             desc="Converting", unit="segment"):
                seg_idx, success, message = future.result()
                if success:
                    success_count += 1
                    # 提取点数
                    if "points)" in message:
                        points = int(message.split("(")[1].split(" ")[0])
                        total_points += points
                else:
                    failed_segments.append((seg_idx, message))
    else:
        # 串行处理
        for args in tqdm(args_list, desc="Converting", unit="segment"):
            seg_idx, success, message = convert_segment_to_las(args)
            if success:
                success_count += 1
                if "points)" in message:
                    points = int(message.split("(")[1].split(" ")[0])
                    total_points += points
            else:
                failed_segments.append((seg_idx, message))
    
    # 报告结果
    print(f"\n✅ 成功转化: {success_count}/{len(segment_indices)} segments")
    print(f"  总点数: {total_points:,}")
    
    if failed_segments:
        print(f"✗ 失败: {len(failed_segments)} segments")
        for seg_idx, error in failed_segments[:5]:
            print(f"  Segment {seg_idx}: {error}")
        if len(failed_segments) > 5:
            print(f"  ... and {len(failed_segments)-5} more")
    
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":

    h5_path = r"E:\data\云南遥感中心\第一批\h5_fast\train\processed_02.h5"
    output_dir = r"E:\data\云南遥感中心\第一批\h5_fast\train\processed_02"
    segment_indices = None
    n_workers = 4

    print("="*70)
    print("H5快速格式转换为LAS文件")
    print("="*70)
    print(f"输入H5文件: {h5_path}")
    print(f"输出目录: {output_dir}")
    if segment_indices is None:
        print("转换所有 segments")
    else:
        print(f"转换指定 segments: {segment_indices}")
    print(f"使用并行 workers: {n_workers}")
    print("="*70)
    
    # 转换
    h5_fast_to_las(
        h5_path=h5_path,
        output_dir=output_dir,
        segment_indices=segment_indices,
        n_workers=n_workers
    )
    
