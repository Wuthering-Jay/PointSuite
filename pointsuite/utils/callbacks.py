import torch
import numpy as np
import os
import glob
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import List, Any, Dict, Optional
from pathlib import Path
from collections import defaultdict

# 导入 laspy (您需要 'pip install laspy')
try:
    import laspy
except ImportError:
    print("警告: 'laspy' 库未安装。PredictionWriter 将无法保存 .las 文件。")
    print("请运行: pip install laspy")


class SegmentationWriter(BasePredictionWriter):
    """
    用于语义分割的 PredictionWriter 回调 (适配 bin+pkl 数据格式)

    专为 PointSuite 的 bin+pkl 数据结构设计，与 BinPklDataset 和 SemanticSegmentationTask 协同工作。

    数据流程:
    1. tile.py 在分割 LAS 文件时，为每个 segment 保存文件关联信息:
       - 'bin_file': bin 文件名
       - 'bin_path': 完整 bin 文件路径
       - 'pkl_path': 完整 pkl 文件路径
       
    2. BinPklDataset (test split) 加载数据时，将文件信息加入 data 字典:
       {'coord', 'feat', 'indices', 'bin_file', 'bin_path', 'pkl_path', ...}
       
    3. SemanticSegmentationTask.predict_step 返回时，传递文件信息:
       {'logits', 'indices', 'bin_file', 'bin_path', 'pkl_path', 'coord'}
       - 'logits': [N, C] 模型预测的类别 logits
       - 'indices': [N] 原始点索引
       - 'bin_file': bin 文件名（直接来自 dataset）
       - 'bin_path': bin 文件完整路径
       - 'pkl_path': pkl 文件完整路径
       - 'coord': [N, 3] 点坐标 (用于可视化)

    4. 本回调执行投票并保存:
       - 直接使用传递的文件路径信息，无需推断
       - 对于每个 bin 文件，累积所有 segment 的预测
       - 使用 logits 平均进行多次预测投票
       - 从原始 bin/pkl 加载完整点云和 LAS 头信息
       - 保存为 .las 文件，保留原始坐标系统和精度

    工作流程:
    1. write_on_batch_end: 将每个批次的预测流式写入临时文件 (.tmp)，防止 OOM
    2. on_predict_end: 预测结束后触发，对每个 bin 文件执行:
       a. 收集所有临时文件
       b. 按 bin 文件分组
       c. 执行投票累积 (logits 平均)
       d. 从原始 bin/pkl 加载坐标和 LAS 头
       e. 保存完整点云为 .las 文件
       f. 清理临时文件
    
    注意: 
    - 文件信息在整个数据流中显式传递，避免了推断的不确定性
    - 即使 batch 包含来自多个 segment 的点，它们必定来自同一个 bin 文件
    """
    
    def __init__(self, 
                 output_dir: str, 
                 write_interval: str = "batch", 
                 num_classes: int = -1,
                 save_logits: bool = False,
                 reverse_class_mapping: Optional[Dict[int, int]] = None):
        """
        Args:
            output_dir (str): 保存最终 .las 文件的目录
            write_interval (str): 必须是 "batch" 才能实现流式传输
            num_classes (int): 类的数量，用于创建投票数组
                              如果为 -1，将从 Task 的 head.out_channels 自动推断
            save_logits (bool): 是否同时保存 logits 到 .npz 文件 (用于后处理/集成)
            reverse_class_mapping (Optional[Dict[int, int]]): 将连续标签映射回原始标签
                                  例如: {0: 0, 1: 1, 2: 2, 3: 6, 4: 9}
                                  如果为 None，则不应用映射
        """
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.temp_dir = os.path.join(self.output_dir, "temp_predictions")
        self.num_classes = num_classes
        self.save_logits = save_logits
        self.reverse_class_mapping = reverse_class_mapping
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def _get_bin_pkl_path_from_indices(self, indices: torch.Tensor, trainer: 'pl.Trainer') -> Optional[str]:
        """
        从点索引反向推断对应的 bin/pkl 文件 (已弃用，保留作为后备方案)
        
        ⚠️ 已弃用: 现在文件信息直接从 dataset 传递，无需推断。
        此方法仅作为后备方案保留，用于处理旧格式数据。
        
        通过检查 dataset.data_list 来确定这些点属于哪个 bin 文件。
        
        Args:
            indices: 点的原始索引 [N]
            trainer: PyTorch Lightning Trainer
            
        Returns:
            bin 文件的基础名称 (不带扩展名)，如果无法确定则返回 None
        """
        try:
            dataset = trainer.predict_dataloaders.dataset
            
            # 获取第一个索引对应的样本信息
            # 假设一个 batch 的所有点来自同一个 bin 文件
            # (这在我们的数据结构中通常是成立的，因为每个 segment 来自一个 bin 文件)
            
            # 我们需要找到包含这些 indices 的 segment
            # 由于 segment 的 indices 是离散的，我们检查第一个索引
            first_idx = indices[0].item() if isinstance(indices[0], torch.Tensor) else indices[0]
            
            # 从 data_list 中找到包含该索引的 segment
            for sample_info in dataset.data_list:
                segment_indices = sample_info.get('indices', [])
                if first_idx in segment_indices:
                    bin_path = Path(sample_info['bin_path'])
                    return bin_path.stem  # 返回文件名 (不带扩展名)
            
            # 如果没找到，使用 batch_idx 作为后备方案
            return None
            
        except Exception as e:
            print(f"警告: 无法从 indices 推断 bin 文件: {e}")
            return None

    def _get_bin_info_from_sample(self, trainer: 'pl.Trainer', sample_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        从 dataset 获取 bin 文件信息
        
        Args:
            trainer: PyTorch Lightning Trainer
            sample_idx: 样本索引 (默认获取第一个)
            
        Returns:
            包含 bin_path, pkl_path, bin_basename 等信息的字典
        """
        try:
            dataset = trainer.predict_dataloaders.dataset
            
            if sample_idx >= len(dataset.data_list):
                sample_idx = 0
                
            sample_info = dataset.data_list[sample_idx]
            
            bin_path = Path(sample_info['bin_path'])
            pkl_path = Path(sample_info['pkl_path'])
            
            return {
                'bin_path': str(bin_path),
                'pkl_path': str(pkl_path),
                'bin_basename': bin_path.stem,
            }
            
        except Exception as e:
            print(f"错误: 无法从 dataset 获取 bin 文件信息: {e}")
            return None


    def write_on_batch_end(
        self, 
        trainer: 'pl.Trainer', 
        pl_module: 'pl.LightningModule', 
        prediction: Dict[str, torch.Tensor], 
        batch_indices: List[int], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int
    ):
        """ 
        在每个预测批次结束后，将结果流式写入临时文件
        
        prediction 字典应包含:
        - 'logits': [N, C] 模型预测
        - 'indices': [N] 原始 bin 文件中的点索引
        - 'bin_file': bin 文件名（列表，每个点对应的文件）
        - 'coord': [N, 3] 点坐标 (可选，用于调试)
        """
        
        if 'logits' not in prediction or 'indices' not in prediction:
            print(f"警告: predict_step 必须返回 'logits' 和 'indices'。跳过批次 {batch_idx}")
            return
        
        # 1. 🔥 直接从 prediction 获取 bin 文件信息（由 dataset 提供）
        if 'bin_file' in prediction and len(prediction['bin_file']) > 0:
            # bin_file 是一个列表（collate_fn 可能会 stack 或保持为列表）
            bin_files = prediction['bin_file']
            
            # 取第一个文件名（假设一个 batch 内的点来自同一个 bin 文件）
            if isinstance(bin_files, list):
                bin_basename = bin_files[0]
            elif isinstance(bin_files, torch.Tensor):
                # 如果被转换为 tensor，转回字符串
                bin_basename = str(bin_files[0].item())
            else:
                bin_basename = str(bin_files)
        else:
            # 后备方案：使用旧的推断方法（不推荐，但保持兼容性）
            indices = prediction['indices']
            bin_basename = self._get_bin_pkl_path_from_indices(indices, trainer)
            
            if bin_basename is None:
                bin_basename = f"unknown_batch_{batch_idx}"
                print(f"警告: 无法确定 batch {batch_idx} 对应的 bin 文件，使用 {bin_basename}")
        
        # 2. 定义临时文件名
        tmp_filename = f"{bin_basename}_batch_{batch_idx}.pred.tmp"
        save_path = os.path.join(self.temp_dir, tmp_filename)
        
        # 3. 保存预测结果到磁盘 (只保存必要信息)
        save_dict = {
            'logits': prediction['logits'].cpu(),      # [N, C]
            'indices': prediction['indices'].cpu(),    # [N]
            'bin_file': bin_basename,                  # 🔥 保存文件名
        }
        
        # 🔥 保存完整路径信息（如果可用）
        if 'bin_path' in prediction:
            save_dict['bin_path'] = prediction['bin_path']
        if 'pkl_path' in prediction:
            save_dict['pkl_path'] = prediction['pkl_path']
        
        # 可选: 保存坐标用于调试
        if 'coord' in prediction and self.save_logits:
            save_dict['coord'] = prediction['coord'].cpu()
        
        torch.save(save_dict, save_path)
        
        # 可选: 打印进度
        if batch_idx % 10 == 0:
            pl_module.print(f"[SegmentationWriter] 已保存 batch {batch_idx} 到 {tmp_filename}")

    def on_predict_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        """
        在整个预测结束后触发，对所有 bin 文件执行投票和保存
        
        此方法会:
        1. 按 bin 文件分组所有临时预测文件
        2. 对每个 bin 文件执行投票 (logits 平均)
        3. 从原始 bin/pkl 加载完整点云数据
        4. 保存为 .las 文件
        5. 清理临时文件
        """
        
        # 如果 num_classes 未指定，从 Task 推断
        if self.num_classes == -1:
            try:
                self.num_classes = pl_module.head.out_channels
                pl_module.print(f"[SegmentationWriter] 从模型推断类别数: {self.num_classes}")
            except Exception as e:
                print(f"错误: 无法从模型推断 num_classes: {e}")
                print("请在初始化 SegmentationWriter 时显式指定 num_classes")
                return

        pl_module.print(f"\n[SegmentationWriter] 预测完成，开始拼接和投票...")
        
        # 1. 查找所有临时文件
        tmp_files = sorted(glob.glob(os.path.join(self.temp_dir, "*.pred.tmp")))
        
        if not tmp_files:
            pl_module.print("[SegmentationWriter] 警告: 未找到临时预测文件")
            return
        
        pl_module.print(f"[SegmentationWriter] 找到 {len(tmp_files)} 个临时预测文件")
        
        # 2. 按 bin 文件分组临时文件
        # 文件名格式: {bin_basename}_batch_{batch_idx}.pred.tmp
        bin_file_groups = defaultdict(list)
        
        for tmp_file in tmp_files:
            filename = os.path.basename(tmp_file)
            # 提取 bin_basename (去除 _batch_xxx.pred.tmp 部分)
            bin_basename = filename.split('_batch_')[0]
            bin_file_groups[bin_basename].append(tmp_file)
        
        pl_module.print(f"[SegmentationWriter] 检测到 {len(bin_file_groups)} 个唯一 bin 文件")
        
        # 3. 对每个 bin 文件执行投票和保存
        for bin_basename, tmp_file_list in bin_file_groups.items():
            pl_module.print(f"\n[SegmentationWriter] 处理 bin 文件: {bin_basename} ({len(tmp_file_list)} 个批次)")
            
            try:
                self._process_single_bin_file(
                    bin_basename=bin_basename,
                    tmp_files=tmp_file_list,
                    trainer=trainer,
                    pl_module=pl_module
                )
            except Exception as e:
                pl_module.print(f"!!! 错误: 处理 {bin_basename} 时失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 4. 清理所有临时文件
        pl_module.print(f"\n[SegmentationWriter] 清理临时文件...")
        for tmp_file in tmp_files:
            try:
                os.remove(tmp_file)
            except Exception as e:
                print(f"警告: 无法删除临时文件 {tmp_file}: {e}")
        
        pl_module.print(f"[SegmentationWriter] 所有预测已保存到 {self.output_dir}")
        pl_module.print("="*70)
    
    def _process_single_bin_file(
        self,
        bin_basename: str,
        tmp_files: List[str],
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule'
    ):
        """
        处理单个 bin 文件的所有预测批次
        
        执行步骤:
        1. 从临时文件加载所有预测并执行投票
        2. 从原始 bin/pkl 文件加载完整点云数据
        3. 应用类别映射 (如果有)
        4. 保存为 .las 文件
        5. (可选) 保存 logits 到 .npz
        
        Args:
            bin_basename: bin 文件的基础名称 (不带扩展名)
            tmp_files: 该 bin 文件的所有临时预测文件列表
            trainer: PyTorch Lightning Trainer
            pl_module: PyTorch Lightning Module
        """
        
        # 1. 🔥 优先从临时文件中获取完整路径信息
        bin_path, pkl_path = None, None
        
        # 尝试从第一个临时文件中读取路径信息
        if len(tmp_files) > 0:
            try:
                first_tmp = torch.load(tmp_files[0])
                if 'bin_path' in first_tmp and 'pkl_path' in first_tmp:
                    # 从临时文件中直接获取路径
                    bin_path_list = first_tmp['bin_path']
                    pkl_path_list = first_tmp['pkl_path']
                    
                    # 处理列表情况（collate_fn 可能保持为列表）
                    if isinstance(bin_path_list, list):
                        bin_path = str(Path(bin_path_list[0]))
                        pkl_path = str(Path(pkl_path_list[0]))
                    else:
                        bin_path = str(Path(bin_path_list))
                        pkl_path = str(Path(pkl_path_list))
                    
                    pl_module.print(f"  - 从临时文件获取路径 ✓")
            except Exception as e:
                pl_module.print(f"  - 从临时文件获取路径失败: {e}")
        
        # 2. 如果没有从临时文件获取到，使用旧的查找方法（后备方案）
        if bin_path is None or pkl_path is None:
            pl_module.print(f"  - 使用后备方案查找文件...")
            bin_path, pkl_path = self._find_bin_pkl_paths(bin_basename, trainer)
        
        if bin_path is None or pkl_path is None:
            pl_module.print(f"错误: 无法找到 {bin_basename} 对应的 bin/pkl 文件")
            return
        
        pl_module.print(f"  - Bin 文件: {bin_path}")
        pl_module.print(f"  - Pkl 文件: {pkl_path}")
        
        # 2. 从 pkl 加载元数据
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 3. 使用 memmap 加载完整点云数据
        point_data = np.memmap(bin_path, dtype=metadata['dtype'], mode='r')
        num_total_points = len(point_data)
        
        pl_module.print(f"  - 总点数: {num_total_points:,}")
        
        # 4. 创建投票数组
        logits_sum = torch.zeros((num_total_points, self.num_classes), dtype=torch.float32)
        counts = torch.zeros(num_total_points, dtype=torch.int32)
        
        # 5. 加载所有临时文件并累积投票
        pl_module.print(f"  - 加载 {len(tmp_files)} 个预测批次...")
        
        for tmp_file in tmp_files:
            try:
                pred_data = torch.load(tmp_file)
                indices = pred_data['indices']  # [N]
                logits = pred_data['logits']    # [N, C]
                
                # 累积 logits
                logits_sum.index_add_(0, indices.long(), logits)
                counts.index_add_(0, indices.long(), torch.ones(len(indices), dtype=torch.int32))
                
            except Exception as e:
                pl_module.print(f"    警告: 加载 {tmp_file} 失败: {e}")
        
        # 6. 计算平均 logits 和最终预测
        pl_module.print(f"  - 计算最终预测...")
        
        # 处理未被预测的点
        unpredicted_mask = (counts == 0)
        counts[unpredicted_mask] = 1  # 避免除以 0
        
        # 平均 logits
        mean_logits = logits_sum / counts.unsqueeze(-1)
        
        # Argmax 获取类别
        final_preds = torch.argmax(mean_logits, dim=1).numpy().astype(np.uint8)
        
        # 对未预测的点赋值 (使用 0 或 ignore_label)
        if unpredicted_mask.any():
            num_unpredicted = unpredicted_mask.sum().item()
            pl_module.print(f"    警告: {num_unpredicted} 个点未被预测，将赋予标签 0")
            final_preds[unpredicted_mask.numpy()] = 0
        
        # 7. 应用反向类别映射 (如果有)
        if self.reverse_class_mapping is not None:
            pl_module.print(f"  - 应用反向类别映射...")
            final_preds_mapped = np.zeros_like(final_preds)
            for continuous_label, original_label in self.reverse_class_mapping.items():
                final_preds_mapped[final_preds == continuous_label] = original_label
            final_preds = final_preds_mapped
        
        # 8. 提取坐标 (XYZ)
        xyz = np.stack([
            point_data['X'],
            point_data['Y'],
            point_data['Z']
        ], axis=1).astype(np.float64)
        
        # 9. 保存为 .las 文件（包含所有原始属性）
        final_las_path = os.path.join(self.output_dir, f"{bin_basename}_predicted.las")
        
        try:
            # 将 bin_path 添加到 metadata 中，方便 _save_las_file 加载原始数据
            metadata['_bin_path'] = bin_path
            
            self._save_las_file(
                las_path=final_las_path,
                xyz=xyz,
                classification=final_preds,
                metadata=metadata,
                pl_module=pl_module
            )
            pl_module.print(f"  ✓ 已保存到: {final_las_path}")
            
        except Exception as e:
            pl_module.print(f"  !!! 错误: 保存 .las 文件失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 10. (可选) 保存 logits
        if self.save_logits:
            logits_path = os.path.join(self.output_dir, f"{bin_basename}_logits.npz")
            np.savez_compressed(
                logits_path,
                logits=mean_logits.numpy(),
                predictions=final_preds,
                counts=counts.numpy()
            )
            pl_module.print(f"  ✓ Logits 已保存到: {logits_path}")
    
    def _find_bin_pkl_paths(self, bin_basename: str, trainer: 'pl.Trainer') -> tuple:
        """
        根据 bin_basename 查找对应的 bin 和 pkl 文件路径
        
        Args:
            bin_basename: bin 文件的基础名称
            trainer: Trainer 对象
            
        Returns:
            (bin_path, pkl_path) 元组，如果找不到则返回 (None, None)
        """
        try:
            dataset = trainer.predict_dataloaders.dataset
            
            # 在 data_list 中查找匹配的文件
            for sample_info in dataset.data_list:
                bin_path = Path(sample_info['bin_path'])
                if bin_path.stem == bin_basename:
                    pkl_path = Path(sample_info['pkl_path'])
                    return str(bin_path), str(pkl_path)
            
            # 如果在 data_list 中没找到，尝试从 data_root 搜索
            data_root = Path(dataset.data_root) if not isinstance(dataset.data_root, (list, tuple)) else Path(dataset.data_root[0]).parent
            
            bin_path = data_root / f"{bin_basename}.bin"
            pkl_path = data_root / f"{bin_basename}.pkl"
            
            if bin_path.exists() and pkl_path.exists():
                return str(bin_path), str(pkl_path)
            
            return None, None
            
        except Exception as e:
            print(f"错误: 查找 bin/pkl 文件失败: {e}")
            return None, None
    
    def _save_las_file(
        self,
        las_path: str,
        xyz: np.ndarray,
        classification: np.ndarray,
        metadata: Dict[str, Any],
        pl_module: 'pl.LightningModule'
    ):
        """
        保存点云为 .las 文件，保留原始 LAS 头信息和所有点属性
        
        Args:
            las_path: 输出 .las 文件路径
            xyz: [N, 3] 点坐标
            classification: [N] 分类标签（预测结果）
            metadata: 从 pkl 文件加载的元数据 (包含 header_info)
            pl_module: PyTorch Lightning Module
        """
        
        try:
            # 1. 从 metadata 中恢复 LAS 头信息
            if 'header_info' in metadata:
                header_info = metadata['header_info']
                
                # 创建 LAS 头
                point_format = header_info.get('point_format', 3)
                version_str = header_info.get('version', '1.2')
                
                # 解析版本字符串
                if isinstance(version_str, str):
                    version_parts = version_str.split('.')
                    if len(version_parts) == 2:
                        major, minor = int(version_parts[0]), int(version_parts[1])
                    else:
                        major, minor = 1, 2
                else:
                    major, minor = 1, 2
                
                header = laspy.LasHeader(point_format=point_format, version=f"{major}.{minor}")
                
                # 设置缩放和偏移
                header.offsets = [
                    header_info.get('x_offset', 0),
                    header_info.get('y_offset', 0),
                    header_info.get('z_offset', 0)
                ]
                header.scales = [
                    header_info.get('x_scale', 0.01),
                    header_info.get('y_scale', 0.01),
                    header_info.get('z_scale', 0.01)
                ]
                
                # 恢复其他头信息
                if 'system_identifier' in header_info:
                    header.system_identifier = header_info['system_identifier']
                if 'generating_software' in header_info:
                    header.generating_software = header_info['generating_software']
                
                # 恢复 VLRs (Variable Length Records) - 包含坐标系信息
                if 'vlrs' in header_info and header_info['vlrs']:
                    for vlr_dict in header_info['vlrs']:
                        try:
                            vlr = laspy.VLR(
                                user_id=vlr_dict['user_id'],
                                record_id=vlr_dict['record_id'],
                                description=vlr_dict['description'],
                                record_data=vlr_dict.get('record_data', b'')
                            )
                            header.vlrs.append(vlr)
                        except Exception as e:
                            pl_module.print(f"    警告: 恢复 VLR 失败: {e}")
                
                pl_module.print(f"    - 使用原始 LAS 头信息 (format {point_format}, version {major}.{minor})")
                
            else:
                # 如果没有 header_info，使用默认值
                pl_module.print("    警告: 元数据中没有 header_info，使用默认值")
                header = laspy.LasHeader(point_format=3, version='1.2')
                header.offsets = xyz.min(axis=0)
                header.scales = np.array([0.001, 0.001, 0.001])
            
            # 2. 创建 LAS 数据
            las = laspy.LasData(header)
            
            # 3. 设置坐标 (laspy 会自动应用 scale 和 offset)
            las.x = xyz[:, 0]
            las.y = xyz[:, 1]
            las.z = xyz[:, 2]
            
            # 4. 🔥 从原始 bin 文件恢复所有可用属性
            if 'dtype' in metadata:
                dtype = metadata['dtype']
                
                # 获取 bin 文件路径（优先从 metadata 中的 _bin_path）
                bin_path = metadata.get('_bin_path', None)
                
                if bin_path and Path(bin_path).exists():
                    pl_module.print(f"    - 从原始 bin 文件恢复属性: {Path(bin_path).name}")
                    
                    # 使用 memmap 加载原始数据
                    point_data = np.memmap(bin_path, dtype=dtype, mode='r')
                    
                    # 恢复各个属性（根据 dtype 中的字段）
                    field_names = [name for name, _ in dtype]
                    
                    # 强度 (Intensity)
                    if 'intensity' in field_names:
                        las.intensity = point_data['intensity']
                        pl_module.print(f"      ✓ 恢复 intensity")
                    
                    # 回波信息 (Return Number, Number of Returns)
                    if 'return_number' in field_names:
                        las.return_number = point_data['return_number']
                        pl_module.print(f"      ✓ 恢复 return_number")
                    if 'number_of_returns' in field_names:
                        las.number_of_returns = point_data['number_of_returns']
                        pl_module.print(f"      ✓ 恢复 number_of_returns")
                    
                    # 扫描角度 (Scan Angle)
                    if 'scan_angle_rank' in field_names:
                        las.scan_angle_rank = point_data['scan_angle_rank']
                        pl_module.print(f"      ✓ 恢复 scan_angle_rank")
                    elif 'scan_angle' in field_names:
                        las.scan_angle = point_data['scan_angle']
                        pl_module.print(f"      ✓ 恢复 scan_angle")
                    
                    # 用户数据 (User Data)
                    if 'user_data' in field_names:
                        las.user_data = point_data['user_data']
                        pl_module.print(f"      ✓ 恢复 user_data")
                    
                    # 点源 ID (Point Source ID)
                    if 'point_source_id' in field_names:
                        las.point_source_id = point_data['point_source_id']
                        pl_module.print(f"      ✓ 恢复 point_source_id")
                    
                    # GPS 时间 (GPS Time)
                    if 'gps_time' in field_names:
                        las.gps_time = point_data['gps_time']
                        pl_module.print(f"      ✓ 恢复 gps_time")
                    
                    # RGB 颜色 (如果 point_format 支持)
                    if header.point_format.id in [2, 3, 5, 7, 8, 10]:
                        if 'red' in field_names and 'green' in field_names and 'blue' in field_names:
                            las.red = point_data['red']
                            las.green = point_data['green']
                            las.blue = point_data['blue']
                            pl_module.print(f"      ✓ 恢复 RGB 颜色")
                        
                        # NIR (近红外) - 如果支持
                        if 'nir' in field_names and header.point_format.id in [8, 10]:
                            las.nir = point_data['nir']
                            pl_module.print(f"      ✓ 恢复 NIR")
                    
                    # 其他可能的字段可以根据需要添加
                    
                else:
                    pl_module.print(f"    警告: 未找到原始 bin 文件: {bin_path}")
                    pl_module.print(f"    只保存坐标和分类标签")
            
            # 5. 设置预测的分类标签（覆盖原始分类）
            las.classification = classification
            pl_module.print(f"      ✓ 设置预测分类标签")
            
            # 6. 写入文件
            las.write(las_path)
            
        except Exception as e:
            pl_module.print(f"    错误: 保存 LAS 文件失败: {e}")
            import traceback
            traceback.print_exc()
            raise