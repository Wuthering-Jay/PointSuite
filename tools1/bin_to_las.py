import numpy as np
import laspy
import pickle
from pathlib import Path
from typing import Union, List
from tqdm import tqdm


class BinToLASConverter:
    def __init__(self,
                 input_dir: Union[str, Path],
                 output_dir: Union[str, Path] = None):
        """
        Initialize converter to transform bin+pkl back to LAS segment files.
        
        Args:
            input_dir: Directory containing bin and pkl files
            output_dir: Directory to save LAS segment files (default: input_dir/las_segments)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "las_segments"
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        self.bin_files = self._find_bin_files()
    
    def _find_bin_files(self) -> List[Path]:
        """Find all bin files in the input directory."""
        bin_files = list(self.input_dir.glob('*.bin'))
        if not bin_files:
            raise ValueError(f"No bin files found in {self.input_dir}")
        return bin_files
    
    def convert_all_files(self):
        """Convert all bin+pkl files to LAS segments."""
        import time
        start_time = time.time()
        
        print("="*70)
        print(f"Starting BIN/PKL to LAS conversion")
        print("="*70)
        print(f"Total files: {len(self.bin_files)}")
        print(f"Output directory: {self.output_dir}")
        print("-"*70)
        
        total_segments = 0
        for bin_file in tqdm(self.bin_files, desc="Progress", unit="file",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            pkl_file = bin_file.with_suffix('.pkl')
            if not pkl_file.exists():
                print(f"\n[WARNING] PKL file not found for {bin_file.name}, skipping...")
                continue
            
            segments_count = self.convert_single_file(bin_file, pkl_file)
            total_segments += segments_count
        
        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print(f"Conversion completed successfully!")
        print(f"Total segments created: {total_segments}")
        print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")
        print(f"Average: {elapsed_time/len(self.bin_files):.2f}s per file")
        print("="*70)
    
    def convert_single_file(self, bin_path: Path, pkl_path: Path):
        """
        Convert a single bin+pkl pair to multiple LAS segment files.
        
        Args:
            bin_path: Path to bin file
            pkl_path: Path to pkl file
        """
        # 加载元数据
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 使用memmap加载bin文件
        dtype = np.dtype(metadata['dtype'])
        mmap_data = np.memmap(bin_path, dtype=dtype, mode='r')
        
        base_name = bin_path.stem
        segments = metadata['segments']
        header_info = metadata['header_info']
        
        for segment_info in segments:
            segment_id = segment_info['segment_id']
            indices = segment_info['indices']
            
            # 从memmap中提取该分块的数据
            segment_data = mmap_data[indices]
            
            # 创建LAS文件
            self._create_las_from_segment(
                segment_data=segment_data,
                segment_id=segment_id,
                base_name=base_name,
                header_info=header_info,
                metadata=metadata
            )
        
        return len(segments)
    
    def _create_las_from_segment(self, 
                                 segment_data: np.ndarray,
                                 segment_id: int,
                                 base_name: str,
                                 header_info: dict,
                                 metadata: dict):
        """
        Create a LAS file from segment data.
        
        Args:
            segment_data: Structured numpy array containing point data
            segment_id: ID of the segment
            base_name: Base name for the output file
            header_info: LAS header information from metadata
            metadata: Full metadata dictionary
        """
        # 确定point_format
        point_format = header_info.get('point_format', 0)
        
        # 解析版本
        version_str = header_info.get('version', '1.2')
        version_parts = version_str.split('.')
        major_version = int(version_parts[0])
        minor_version = int(version_parts[1])
        
        # 创建LAS header
        header = laspy.LasHeader(
            point_format=point_format,
            version=f"{major_version}.{minor_version}"
        )
        
        # 设置scale和offset
        header.x_scale = header_info.get('x_scale', 0.001)
        header.y_scale = header_info.get('y_scale', 0.001)
        header.z_scale = header_info.get('z_scale', 0.001)
        header.x_offset = header_info.get('x_offset', 0.0)
        header.y_offset = header_info.get('y_offset', 0.0)
        header.z_offset = header_info.get('z_offset', 0.0)
        
        # 恢复其他头文件属性
        if 'system_identifier' in header_info:
            header.system_identifier = header_info['system_identifier']
        if 'generating_software' in header_info:
            header.generating_software = header_info['generating_software']
        # global_encoding不需要手动设置，laspy会自动处理
        
        # 恢复VLRs（坐标系信息等）
        if 'vlrs' in header_info and header_info['vlrs']:
            for vlr_dict in header_info['vlrs']:
                try:
                    from laspy.vlrs.known import VLR
                    vlr = VLR(
                        user_id=vlr_dict['user_id'],
                        record_id=vlr_dict['record_id'],
                        description=vlr_dict['description'],
                        record_data=vlr_dict.get('record_data', b'')
                    )
                    header.vlrs.append(vlr)
                except Exception as e:
                    print(f"Warning: Could not restore VLR: {e}")
        
        # 创建LAS data
        las_data = laspy.LasData(header)
        
        # 创建points数组
        las_data.points = laspy.ScaleAwarePointRecord.zeros(
            len(segment_data),
            header=header
        )
        
        # 复制所有字段数据
        fields = metadata['fields']
        available_dims = las_data.point_format.dimension_names
        
        for field in fields:
            if field in segment_data.dtype.names:
                # laspy使用小写字段名访问
                field_lower = field.lower()
                if field in available_dims or field_lower in [d.lower() for d in available_dims]:
                    # 该字段在LAS格式中存在，直接复制
                    try:
                        setattr(las_data, field_lower, segment_data[field])
                    except:
                        # 如果小写失败，尝试原始字段名
                        setattr(las_data, field, segment_data[field])
        
        # 恢复CRS信息（如果有）
        if 'crs' in header_info and header_info['crs']:
            try:
                las_data.crs = header_info['crs']
            except:
                pass
        
        # 保存LAS文件
        output_path = self.output_dir / f"{base_name}_segment_{segment_id:04d}.las"
        las_data.write(output_path)


def convert_bin_to_las(input_dir: Union[str, Path], 
                       output_dir: Union[str, Path] = None):
    """
    Convert bin+pkl files to LAS segment files.
    
    Args:
        input_dir: Directory containing bin and pkl files
        output_dir: Directory to save LAS segment files
    """
    converter = BinToLASConverter(input_dir=input_dir, output_dir=output_dir)
    converter.convert_all_files()


def convert_single_bin_to_las(bin_path: Union[str, Path],
                              pkl_path: Union[str, Path],
                              output_dir: Union[str, Path] = None):
    """
    Convert a single bin+pkl pair to LAS segment files.
    
    Args:
        bin_path: Path to bin file
        pkl_path: Path to pkl file
        output_dir: Directory to save LAS segment files
    """
    bin_path = Path(bin_path)
    pkl_path = Path(pkl_path)
    
    if output_dir is None:
        output_dir = bin_path.parent / "las_segments"
    
    converter = BinToLASConverter(input_dir=bin_path.parent, output_dir=output_dir)
    converter.convert_single_file(bin_path, pkl_path)


if __name__ == "__main__":
    # 示例：转换整个文件夹
    input_dir = r"E:\data\Dales\dales_las\tile_bin\test"
    output_dir = r"E:\data\Dales\dales_las\tile_bin\test_las"
    
    convert_bin_to_las(
        input_dir=input_dir,
        output_dir=output_dir
    )
    
    print("\n" + "="*50)
    print("转换完成！")
    print("="*50)
    
    # 示例：转换单个文件
    # bin_file = r"E:\data\DALES\dales_las\tile_bin\train\5080_54435.bin"
    # pkl_file = r"E:\data\DALES\dales_las\tile_bin\train\5080_54435.pkl"
    # output_dir = r"E:\data\DALES\dales_las\tile_bin_to_las\train"
    # 
    # convert_single_bin_to_las(
    #     bin_path=bin_file,
    #     pkl_path=pkl_file,
    #     output_dir=output_dir
    # )
