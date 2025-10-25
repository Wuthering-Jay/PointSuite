"""
H5到LAS转换工具 - 并行加速版本

这是h5_to_las.py的优化版本，支持并行处理以加速转换。

使用方式:
    # 串行处理（默认）
    python h5_to_las_parallel.py <h5_file>
    
    # 并行处理（推荐）
    python h5_to_las_parallel.py <h5_file> --workers 8
    
    # 只转换指定segments
    python h5_to_las_parallel.py <h5_file> --segments 0-100
"""

import json
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
    Worker function: 转换单个segment到LAS文件
    
    Args:
        args: (h5_path, segment_idx, output_dir, header_info)
    
    Returns:
        (segment_idx, success, message)
    """
    h5_path, seg_idx, output_dir, header_info = args
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # 读取segment indices
            indices = f['segments'][f'segment_{seg_idx:04d}']['indices'][:]
            
            # 排序indices（H5要求）
            if not np.all(indices[:-1] <= indices[1:]):
                sort_order = np.argsort(indices)
                sorted_indices = indices[sort_order]
                unsort_order = np.argsort(sort_order)
            else:
                sorted_indices = indices
                unsort_order = None
            
            # 创建LAS header
            header = laspy.LasHeader(
                point_format=header_info['point_format'],
                version=f"{header_info['version_major']}.{header_info['version_minor']}"
            )
            header.x_scale = header_info['x_scale']
            header.y_scale = header_info['y_scale']
            header.z_scale = header_info['z_scale']
            header.x_offset = header_info['x_offset']
            header.y_offset = header_info['y_offset']
            header.z_offset = header_info['z_offset']
            
            # 创建LAS数据
            new_las = laspy.LasData(header)
            new_las.points = laspy.ScaleAwarePointRecord.zeros(len(indices), header=header)
            
            # 读取数据
            x_data = f['data']['x'][sorted_indices]
            y_data = f['data']['y'][sorted_indices]
            z_data = f['data']['z'][sorted_indices]
            class_data = f['data']['classification'][sorted_indices]
            
            # 恢复原始顺序
            if unsort_order is not None:
                new_las.x = x_data[unsort_order]
                new_las.y = y_data[unsort_order]
                new_las.z = z_data[unsort_order]
                new_las.classification = class_data[unsort_order]
            else:
                new_las.x = x_data
                new_las.y = y_data
                new_las.z = z_data
                new_las.classification = class_data
            
            # 读取其他字段
            for field in header_info['available_fields']:
                if field in f['data'] and hasattr(new_las, field):
                    try:
                        field_data = f['data'][field][sorted_indices]
                        if unsort_order is not None:
                            field_data = field_data[unsort_order]
                        setattr(new_las, field, field_data)
                    except:
                        pass
            
            # 设置CRS
            if header_info.get('crs'):
                try:
                    new_las.crs = header_info['crs']
                except:
                    pass
            
            # 保存文件
            h5_stem = Path(h5_path).stem
            output_path = Path(output_dir) / f"{h5_stem}_segment_{seg_idx:04d}.las"
            new_las.write(output_path)
            
            return (seg_idx, True, f"Success")
            
    except Exception as e:
        return (seg_idx, False, str(e))


def h5_to_las_parallel(
    h5_path: str,
    output_dir: Optional[str] = None,
    segment_indices: Optional[List[int]] = None,
    n_workers: int = 1
) -> None:
    """
    将H5文件转换为LAS文件（支持并行）
    
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
    
    print(f"Opening H5 file: {h5_path.name}")
    
    # 读取header信息
    with h5py.File(h5_path, 'r') as f:
        header_group = f['header']
        header_info = {
            'point_format': int(header_group.attrs['point_format']),
            'version_major': int(header_group.attrs['version_major']),
            'version_minor': int(header_group.attrs['version_minor']),
            'x_scale': float(header_group.attrs['x_scale']),
            'y_scale': float(header_group.attrs['y_scale']),
            'z_scale': float(header_group.attrs['z_scale']),
            'x_offset': float(header_group.attrs['x_offset']),
            'y_offset': float(header_group.attrs['y_offset']),
            'z_offset': float(header_group.attrs['z_offset']),
            'crs': header_group.attrs.get('crs', None),
            'available_fields': json.loads(f['data'].attrs['available_fields'])
        }
        
        num_segments = f['segments'].attrs['num_segments']
    
    # 确定要转换的segments
    if segment_indices is None:
        segment_indices = list(range(num_segments))
    
    print(f"Converting {len(segment_indices)} segments to LAS files...")
    
    if n_workers > 1:
        print(f"Using {n_workers} parallel workers")
    
    # 准备参数
    args_list = [(h5_path, seg_idx, output_dir, header_info) 
                for seg_idx in segment_indices]
    
    # 转换
    success_count = 0
    failed_segments = []
    
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
                else:
                    failed_segments.append((seg_idx, message))
    else:
        # 串行处理
        for args in tqdm(args_list, desc="Converting", unit="segment"):
            seg_idx, success, message = convert_segment_to_las(args)
            if success:
                success_count += 1
            else:
                failed_segments.append((seg_idx, message))
    
    # 报告结果
    print(f"\n✓ Successfully converted: {success_count}/{len(segment_indices)} segments")
    
    if failed_segments:
        print(f"✗ Failed: {len(failed_segments)} segments")
        for seg_idx, error in failed_segments[:5]:
            print(f"  Segment {seg_idx}: {error}")
        if len(failed_segments) > 5:
            print(f"  ... and {len(failed_segments)-5} more")
    
    print(f"\nOutput directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert H5 files to LAS segments (parallel version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default settings (serial)
  python h5_to_las_parallel.py file.h5
  
  # Convert with 8 workers (parallel, recommended)
  python h5_to_las_parallel.py file.h5 --workers 8
  
  # Convert only segments 0-99
  python h5_to_las_parallel.py file.h5 --segments 0-99 --workers 4
  
  # Specify output directory
  python h5_to_las_parallel.py file.h5 --output ./segments --workers 8
        """
    )
    
    parser.add_argument('h5_file', help='Input H5 file path')
    parser.add_argument('--output', '-o', help='Output directory (default: <h5file>_segments)')
    parser.add_argument('--workers', '-w', type=int, default=1,
                       help='Number of parallel workers (default: 1, recommended: 4-8)')
    parser.add_argument('--segments', '-s', help='Segment range to convert (e.g., "0-99" or "0,5,10-20")')
    
    args = parser.parse_args()
    
    # 检查文件
    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        print(f"Error: File not found: {h5_path}")
        return 1
    
    # 解析segment范围
    segment_indices = None
    if args.segments:
        segment_indices = []
        for part in args.segments.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                segment_indices.extend(range(start, end + 1))
            else:
                segment_indices.append(int(part))
    
    # 转换
    h5_to_las_parallel(
        h5_path=str(h5_path),
        output_dir=args.output,
        segment_indices=segment_indices,
        n_workers=args.workers
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
