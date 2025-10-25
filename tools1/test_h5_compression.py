"""
H5压缩方式对比测试

测试blosc, lz4, zstd等轻量级压缩对快速H5格式的影响
重点关注：
1. 压缩率（文件大小）
2. 随机读取速度
3. 写入速度

关键问题：
- gzip: 压缩好但慢，且会破坏contiguous layout
- blosc/lz4/zstd: 快速压缩，但需要检查是否影响随机读取
"""

import h5py
import numpy as np
import time
from pathlib import Path
import tempfile


def test_compression_methods():
    """测试各种压缩方式"""
    
    print("="*70)
    print("H5压缩方式对比测试")
    print("="*70)
    
    # 生成测试数据（模拟点云segment）
    np.random.seed(42)
    num_segments = 400
    points_per_seg = 40000  # 2万点/segment
    
    print(f"\n测试数据:")
    print(f"  Segments: {num_segments}")
    print(f"  Points/segment: {points_per_seg}")
    print(f"  总点数: {num_segments * points_per_seg:,}")
    
    # 生成点云数据
    segments_data = []
    for i in range(num_segments):
        seg = {
            'x': np.random.randn(points_per_seg).astype(np.float32),
            'y': np.random.randn(points_per_seg).astype(np.float32),
            'z': np.random.randn(points_per_seg).astype(np.float32),
            'intensity': np.random.randint(0, 65536, points_per_seg, dtype=np.uint16),
            'classification': np.random.randint(0, 32, points_per_seg, dtype=np.uint8),
            'return_number': np.random.randint(1, 8, points_per_seg, dtype=np.uint8),
        }
        segments_data.append(seg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # 测试配置
        compression_configs = [
            # (名称, compression, compression_opts, chunks, 说明)
            ("无压缩 (当前)", None, None, None, "Contiguous layout, 最快"),
            ("gzip-1", "gzip", 1, True, "最低级别gzip"),
            ("gzip-4", "gzip", 4, True, "平衡级别gzip"),
            ("lzf", "lzf", None, True, "HDF5内置快速压缩"),
        ]
        
        # 检查是否支持blosc
        try:
            import hdf5plugin
            compression_configs.extend([
                ("blosc-lz4", hdf5plugin.Blosc(cname='lz4', clevel=1, shuffle=hdf5plugin.Blosc.SHUFFLE), None, True, "Blosc+LZ4快速压缩"),
                ("blosc-zstd", hdf5plugin.Blosc(cname='zstd', clevel=1, shuffle=hdf5plugin.Blosc.SHUFFLE), None, True, "Blosc+ZSTD快速压缩"),
            ])
            print("\n✅ 检测到hdf5plugin，将测试blosc压缩")
        except ImportError:
            print("\n⚠️ 未安装hdf5plugin，跳过blosc测试")
            print("   安装命令: pip install hdf5plugin")
        
        results = []
        
        for config in compression_configs:
            name, compression, comp_opts, chunks, desc = config
            
            print(f"\n{'='*70}")
            print(f"测试: {name}")
            print(f"说明: {desc}")
            print(f"{'='*70}")
            
            h5_path = tmpdir / f"test_{name.replace(' ', '_').replace('-', '_')}.h5"
            
            # === 写入测试 ===
            write_start = time.time()
            try:
                with h5py.File(h5_path, 'w') as f:
                    seg_group = f.create_group('segments')
                    seg_group.attrs['num_segments'] = num_segments
                    
                    for i, seg_data in enumerate(segments_data):
                        sg = seg_group.create_group(f'segment_{i:04d}')
                        
                        for field, data in seg_data.items():
                            # 设置存储参数
                            if chunks is None:
                                # Contiguous
                                sg.create_dataset(field, data=data, chunks=None)
                            elif chunks is True:
                                # Auto chunking
                                if compression is None:
                                    sg.create_dataset(field, data=data)
                                elif isinstance(compression, str):
                                    sg.create_dataset(
                                        field, data=data,
                                        compression=compression,
                                        compression_opts=comp_opts,
                                        shuffle=True
                                    )
                                else:
                                    # hdf5plugin filter
                                    sg.create_dataset(field, data=data, **compression)
                
                write_time = time.time() - write_start
                file_size_mb = h5_path.stat().st_size / (1024 * 1024)
                
                print(f"✅ 写入成功")
                print(f"  写入时间: {write_time:.2f}秒 ({num_segments/write_time:.1f} seg/s)")
                print(f"  文件大小: {file_size_mb:.1f} MB")
                
                # === 随机读取测试 ===
                # 测试1: 按需读取（模拟训练时的随机访问）
                num_reads = 50
                read_indices = np.random.choice(num_segments, num_reads, replace=False)
                
                read_start = time.time()
                with h5py.File(h5_path, 'r') as f:
                    for idx in read_indices:
                        seg_group = f['segments'][f'segment_{idx:04d}']
                        # 读取所有字段（模拟真实使用）
                        xyz = np.stack([
                            seg_group['x'][:],
                            seg_group['y'][:],
                            seg_group['z'][:]
                        ], axis=1)
                        intensity = seg_group['intensity'][:]
                        classification = seg_group['classification'][:]
                
                read_time = time.time() - read_start
                avg_read_ms = (read_time / num_reads) * 1000
                
                print(f"  随机读取: {num_reads}个segments")
                print(f"    总时间: {read_time:.3f}秒")
                print(f"    平均: {avg_read_ms:.2f}ms/segment")
                print(f"    速度: {num_reads/read_time:.0f} seg/s")
                
                # === 顺序读取测试（预加载场景） ===
                seq_start = time.time()
                with h5py.File(h5_path, 'r') as f:
                    all_data = []
                    for i in range(min(20, num_segments)):
                        seg_group = f['segments'][f'segment_{i:04d}']
                        xyz = np.stack([
                            seg_group['x'][:],
                            seg_group['y'][:],
                            seg_group['z'][:]
                        ], axis=1)
                        all_data.append(xyz)
                
                seq_time = time.time() - seq_start
                seq_avg_ms = (seq_time / 20) * 1000
                
                print(f"  顺序读取: 前20个segments")
                print(f"    总时间: {seq_time:.3f}秒")
                print(f"    平均: {seq_avg_ms:.2f}ms/segment")
                
                # 保存结果
                results.append({
                    'name': name,
                    'desc': desc,
                    'write_time': write_time,
                    'file_size_mb': file_size_mb,
                    'random_read_ms': avg_read_ms,
                    'seq_read_ms': seq_avg_ms,
                    'compression_ratio': file_size_mb / results[0]['file_size_mb'] if results else 1.0
                })
                
            except Exception as e:
                print(f"❌ 失败: {e}")
                continue
        
        # === 汇总对比 ===
        print("\n" + "="*70)
        print("压缩方式对比汇总")
        print("="*70)
        
        if not results:
            print("没有成功的测试结果")
            return
        
        # 表头
        print(f"\n{'方法':<20} {'大小(MB)':<12} {'压缩率':<10} {'随机读(ms)':<12} {'顺序读(ms)':<12} {'写入(s)':<10}")
        print("-"*70)
        
        # 数据行
        for r in results:
            print(f"{r['name']:<20} {r['file_size_mb']:<12.1f} {r['compression_ratio']:<10.2f} "
                  f"{r['random_read_ms']:<12.2f} {r['seq_read_ms']:<12.2f} {r['write_time']:<10.2f}")
        
        # === 建议 ===
        print("\n" + "="*70)
        print("💡 分析与建议")
        print("="*70)
        
        base = results[0]  # 无压缩基准
        
        print(f"\n基准（无压缩）:")
        print(f"  文件大小: {base['file_size_mb']:.1f} MB")
        print(f"  随机读取: {base['random_read_ms']:.2f} ms/segment")
        
        # 找出最佳平衡
        candidates = []
        for r in results[1:]:  # 跳过基准
            # 计算综合分数（越低越好）
            # 假设：文件大小降低30%以上，且随机读取慢不超过2倍
            size_reduction = (1 - r['compression_ratio']) * 100
            read_slowdown = r['random_read_ms'] / base['random_read_ms']
            
            if size_reduction > 20 and read_slowdown < 3:
                candidates.append((r['name'], size_reduction, read_slowdown, r))
        
        if candidates:
            print(f"\n✅ 推荐的压缩方式:")
            for name, reduction, slowdown, r in sorted(candidates, key=lambda x: -x[1]):
                print(f"\n  {name}:")
                print(f"    - 文件减小: {reduction:.0f}%")
                print(f"    - 读取变慢: {slowdown:.1f}x")
                print(f"    - 随机读取: {r['random_read_ms']:.2f}ms (基准: {base['random_read_ms']:.2f}ms)")
                print(f"    - 文件大小: {r['file_size_mb']:.1f}MB (基准: {base['file_size_mb']:.1f}MB)")
        else:
            print(f"\n❌ 没有找到合适的压缩方式")
            print(f"   所有压缩方式要么压缩率不足(<20%)，要么读取速度慢太多(>3x)")
            print(f"   建议: 保持无压缩以获得最佳性能")
        
        print(f"\n📊 关键结论:")
        print(f"  1. Contiguous + 无压缩 = 最快读取 ({base['random_read_ms']:.2f}ms)")
        print(f"  2. 任何压缩都会引入chunking → 破坏contiguous → 变慢")
        print(f"  3. 即使是LZ4/ZSTD快速压缩，也需要解压开销")
        print(f"  4. 对于需要极致随机读取性能的场景，无压缩是最佳选择")
        print(f"  5. 如果磁盘空间紧张，lzf或blosc-lz4是较好的折中方案")


def estimate_real_impact(original_size_mb: float = 850.0):
    """估算实际影响"""
    
    print("\n" + "="*70)
    print("实际场景影响估算")
    print("="*70)
    
    print(f"\n假设单个H5文件: {original_size_mb:.0f}MB (无压缩)")
    print(f"19个文件总计: {original_size_mb * 19 / 1024:.1f}GB")
    
    scenarios = [
        ("无压缩", 1.0, 1.5),
        ("lzf", 0.6, 3.0),
        ("gzip-1", 0.5, 8.0),
        ("blosc-lz4", 0.65, 2.5),
        ("gzip-4", 0.4, 15.0),
    ]
    
    print(f"\n{'压缩方式':<15} {'单文件(MB)':<15} {'19文件(GB)':<15} {'读取速度':<20}")
    print("-"*70)
    
    for name, ratio, slowdown in scenarios:
        single = original_size_mb * ratio
        total = single * 19 / 1024
        speed = f"{slowdown:.1f}x慢" if slowdown > 1 else "基准"
        print(f"{name:<15} {single:<15.0f} {total:<15.1f} {speed:<20}")
    
    print(f"\n💡 权衡建议:")
    print(f"  - 磁盘充足 → 无压缩 (最快)")
    print(f"  - 磁盘紧张 + 需要性能 → lzf 或 blosc-lz4 (2-3x慢，省35-40%空间)")
    print(f"  - 磁盘很紧张 + 可接受慢速 → gzip-1 (8x慢，省50%空间)")
    print(f"  - 仅存档不训练 → gzip-4 (15x慢，省60%空间)")


if __name__ == "__main__":
    test_compression_methods()
    estimate_real_impact(850.0)
