"""
多H5文件高效读取策略说明

解答关于预加载、内存、多文件和线程数的问题
"""

import h5py
import numpy as np
from pathlib import Path
import psutil
import os


def analyze_memory_requirements():
    """分析内存需求"""
    
    print("="*70)
    print("内存需求分析")
    print("="*70)
    
    # 示例H5文件
    h5_path = r"E:\data\云南遥感中心\第一批\h5\train\processed_02.h5"
    
    with h5py.File(h5_path, 'r') as f:
        total_points = len(f['data']['x'])
        num_segments = f['segments'].attrs['num_segments']
        
        # 计算内存占用
        x_size = f['data']['x'].nbytes / (1024**2)
        y_size = f['data']['y'].nbytes / (1024**2)
        z_size = f['data']['z'].nbytes / (1024**2)
        label_size = f['data']['classification'].nbytes / (1024**2)
        
        total_memory = x_size + y_size + z_size + label_size
        
        print(f"\n单个H5文件内存需求:")
        print(f"  总点数: {total_points:,}")
        print(f"  Segments: {num_segments}")
        print(f"  X数组: {x_size:.1f} MB")
        print(f"  Y数组: {y_size:.1f} MB")
        print(f"  Z数组: {z_size:.1f} MB")
        print(f"  标签数组: {label_size:.1f} MB")
        print(f"  总计: {total_memory:.1f} MB")
        
    # 系统可用内存
    available_memory = psutil.virtual_memory().available / (1024**2)
    total_system_memory = psutil.virtual_memory().total / (1024**2)
    
    print(f"\n系统内存:")
    print(f"  总内存: {total_system_memory:.1f} MB")
    print(f"  可用内存: {available_memory:.1f} MB")
    
    # 多文件场景分析
    print(f"\n多文件场景分析:")
    for num_files in [1, 5, 10, 20]:
        required = total_memory * num_files
        feasible = "✅ 可行" if required < available_memory * 0.7 else "❌ 不推荐"
        print(f"  {num_files}个H5文件: {required:.1f} MB {feasible}")
    
    return total_memory


def explain_loading_strategies():
    """解释不同的加载策略"""
    
    print("\n" + "="*70)
    print("加载策略对比")
    print("="*70)
    
    strategies = """
【策略1：全预加载（当前h5_dataset.py的preload=True）】
原理:
  - 初始化时：读取整个H5文件到内存
  - 训练时：从内存直接索引（超快）

优点:
  ✅ 速度极快（900+ seg/s）
  ✅ 训练时无IO等待

缺点:
  ❌ 内存占用大（每个H5约500MB）
  ❌ 20个H5需要10GB内存
  ❌ 初始化慢（每个H5加载1-2秒）

适用场景:
  - 单个或少量H5文件（1-5个）
  - 内存充足（>16GB）
  - 追求极致训练速度

---

【策略2：按需加载（h5_dataset.py的preload=False）】
原理:
  - 初始化时：只读取indices信息（很小）
  - 训练时：每次从磁盘读取需要的segment

优点:
  ✅ 内存占用极小（几MB）
  ✅ 支持任意数量H5文件
  ✅ 初始化快

缺点:
  ❌ 速度慢（1.5 seg/s with 4 workers）
  ❌ 频繁磁盘IO

适用场景:
  - 大量H5文件（>10个）
  - 内存不足
  - 数据集总大小 > 内存

---

【策略3：LRU缓存（推荐，待实现）】
原理:
  - 缓存最近使用的N个H5文件的数据
  - 超过缓存大小时，淘汰最久未使用的

优点:
  ✅ 平衡内存和速度
  ✅ 自动管理内存
  ✅ 适应训练过程

缺点:
  ⚠️ 需要实现缓存逻辑
  ⚠️ 缓存命中率影响性能

适用场景:
  - 多个H5文件（5-20个）
  - 中等内存（8-16GB）
  - **推荐用于你的20个H5场景**

---

【策略4：分布式数据加载】
原理:
  - 每个GPU/进程只加载部分H5文件
  - 使用DistributedSampler分配数据

优点:
  ✅ 内存分散到多个进程
  ✅ 适合大规模训练

缺点:
  ⚠️ 需要多GPU或多机
  ⚠️ 实现复杂

适用场景:
  - 多GPU训练
  - 数据集超大
    """
    
    print(strategies)


def explain_num_workers():
    """解释为什么workers越多越慢"""
    
    print("\n" + "="*70)
    print("num_workers参数详解")
    print("="*70)
    
    explanation = """
【为什么预加载模式下，workers越多越慢？】

1. 预加载模式的数据流:
   初始化 → 数据在主进程内存中
            ↓
   训练时 → 直接从内存读取（已经是最快的）
            ↓
   不需要 → 后台IO、解压等耗时操作

2. num_workers的作用:
   - 创建N个子进程
   - 每个子进程独立运行__getitem__
   - 主进程与子进程通过队列通信（需要序列化/反序列化）

3. 预加载模式下使用workers的代价:
   主进程内存数据 → 序列化 → 队列 → 子进程 → 反序列化 → 返回主进程
   |_______________|_________________________________|
          数据复制                多进程通信开销

   结果：反而更慢！

4. 测试数据印证:
   - num_workers=0: 900 seg/s  ✅ 最快！
   - num_workers=2: 33 seg/s   ❌ 慢27倍
   - num_workers=4: 17 seg/s   ❌ 慢53倍
   - num_workers=8: 9 seg/s    ❌ 慢100倍

---

【什么时候需要多workers？】

只有当__getitem__有耗时操作时，才需要并行：

✅ 需要workers的场景:
- 从磁盘读取文件（IO密集）
- 图像解码、解压缩
- 复杂数据增强
- 动态计算特征

❌ 不需要workers的场景:
- 数据已在内存（预加载模式）
- 简单numpy操作
- 直接索引数组

---

【你的20个H5场景应该用多少workers？】

方案A：如果使用预加载模式（不推荐20个H5）
  num_workers = 0  # 必须用0

方案B：如果使用按需加载模式（推荐）
  num_workers = 4-8  # IO密集，需要并行

方案C：如果使用LRU缓存模式（最佳）
  num_workers = 2-4  # 中等并行度
    """
    
    print(explanation)


def recommend_for_20_h5_files():
    """针对20个H5文件的推荐方案"""
    
    print("\n" + "="*70)
    print("20个H5文件的最佳方案")
    print("="*70)
    
    recommendation = """
【场景分析】
- 20个H5文件
- 每个约500MB
- 总计约10GB
- 需要跨H5随机采样

【方案对比】

方案1：全预加载（不推荐）
  代码: preload=True
  内存: 10GB
  速度: 900 seg/s
  问题: ❌ 内存占用过大
        ❌ 初始化慢（20-40秒）
        ❌ 其他程序可能OOM

方案2：按需加载（可行，但慢）
  代码: preload=False, num_workers=4
  内存: <100MB
  速度: 1.5 seg/s
  问题: ❌ 训练速度慢
        ⚠️ 频繁磁盘IO

方案3：LRU缓存加载（推荐）✅
  代码: 需要实现Multi-H5 Dataset with LRU cache
  内存: 可控（例如缓存5个H5 = 2.5GB）
  速度: 缓存命中时~900 seg/s，未命中~10 seg/s
  优点: ✅ 内存可控
        ✅ 速度快（假设80%命中率 → 平均700+ seg/s）
        ✅ 自动管理

方案4：分文件训练（简单方案）
  思路: 每个epoch只使用5个H5，多个epoch覆盖所有数据
  代码: 
    epoch_files = all_files[epoch % 4 * 5 : (epoch % 4 + 1) * 5]
    dataset = Multi-H5-Dataset(epoch_files, preload=True)
  内存: 2.5GB（5个H5）
  速度: 900 seg/s
  优点: ✅ 简单易实现
        ✅ 速度快
        ✅ 内存可控
  缺点: ⚠️ 每个batch只来自5个H5（随机性稍差）

---

【立即可用的推荐】

如果急着训练（最简单）:
  → 使用方案4（分文件训练）

如果追求完美（需要写代码）:
  → 使用方案3（LRU缓存）
  → 我可以帮你实现

如果内存充足（32GB+）:
  → 使用方案1（全预加载）
    """
    
    print(recommendation)


if __name__ == "__main__":
    # 1. 分析内存需求
    memory_per_h5 = analyze_memory_requirements()
    
    # 2. 解释加载策略
    explain_loading_strategies()
    
    # 3. 解释num_workers
    explain_num_workers()
    
    # 4. 推荐方案
    recommend_for_20_h5_files()
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("""
关键点:
1. ✅ 预加载 = 整个文件读入内存
2. ✅ 20个H5全预加载 = 10GB内存（不推荐）
3. ✅ 预加载模式必须用num_workers=0（否则反而慢）
4. ✅ 按需加载模式应该用num_workers=4-8
5. ✅ 推荐：实现LRU缓存或分文件训练

下一步:
- 如果选择方案3（LRU缓存），告诉我，我帮你实现
- 如果选择方案4（分文件训练），告诉我，我帮你实现
- 如果选择方案2（按需加载），当前代码已支持
    """)
