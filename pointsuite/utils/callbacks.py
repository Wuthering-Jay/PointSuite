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
import time
from pytorch_lightning.callbacks import Callback


# 导入 laspy (您需要 'pip install laspy')
try:
    import laspy
except ImportError:
    print("警告: 'laspy' 库未安装。PredictionWriter 将无法保存 .las 文件。")
    print("请运行: pip install laspy")


# ============================================
# 辅助函数
# ============================================

def create_reverse_class_mapping(class_mapping: Dict[int, int]) -> Dict[int, int]:
    """
    从 class_mapping 创建 reverse_class_mapping
    
    Args:
        class_mapping: 原始标签 -> 连续标签的映射
                      例如: {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
    
    Returns:
        reverse_class_mapping: 连续标签 -> 原始标签的映射
                              例如: {0: 0, 1: 1, 2: 2, 3: 6, 4: 9}
    
    Example:
        >>> class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
        >>> reverse_mapping = create_reverse_class_mapping(class_mapping)
        >>> print(reverse_mapping)
        {0: 0, 1: 1, 2: 2, 3: 6, 4: 9}
    """
    return {v: k for k, v in class_mapping.items()}


class SemanticPredictLasWriter(BasePredictionWriter):
    """
    用于语义分割的 PredictionWriter 回调 (适配 bin+pkl 数据格式)
    
    重命名自 SegmentationWriter，专为 PointSuite 的 bin+pkl 数据结构设计。
    负责将模型预测结果流式写入临时文件，并在预测结束后合并、投票并保存为 LAS 文件。

    主要功能:
    1. 流式写入: 防止大规模点云预测时的 OOM。
    2. 投票机制: 对重叠预测进行 logits 平均投票。
    3. 完整性恢复: 从原始 bin/pkl 恢复坐标和属性。
    4. 格式保持: 保留原始 LAS 头信息和坐标系。
    """
    
    def __init__(self, 
                 output_dir: str, 
                 write_interval: str = "batch", 
                 num_classes: int = -1,
                 save_logits: bool = False,
                 reverse_class_mapping: Optional[Dict[int, int]] = None,
                 auto_infer_reverse_mapping: bool = True):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.temp_dir = os.path.join(self.output_dir, "temp_predictions")
        self.num_classes = num_classes
        self.save_logits = save_logits
        self.reverse_class_mapping = reverse_class_mapping
        self.auto_infer_reverse_mapping = auto_infer_reverse_mapping
        self._mapping_inferred = False
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def on_predict_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        """预测开始前的初始化工作，主要是推断类别映射"""
        self._infer_class_mapping(trainer, pl_module)

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
        """每个批次结束时，将预测结果写入临时文件"""
        if not self._validate_prediction(prediction, batch_idx):
            return
        
        # 1. 准备数据
        bin_files = prediction['bin_file']
        logits = prediction['logits'].cpu().float() # 确保 float32
        indices = prediction['indices'].cpu()
        
        bin_paths = prediction.get('bin_path', [None] * len(bin_files))
        pkl_paths = prediction.get('pkl_path', [None] * len(bin_files))
        
        # 获取 offset
        offsets = batch['offset'].cpu().numpy() if 'offset' in batch else [len(logits)]
        
        # 2. 按文件分组并保存
        self._save_batch_predictions(
            bin_files, logits, indices, bin_paths, pkl_paths, offsets, batch_idx, pl_module
        )

    def on_predict_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        """预测结束后的汇总处理"""
        self._ensure_num_classes(pl_module)
        pl_module.print(f"\n[SemanticPredictLasWriter] 预测完成，开始拼接和投票...")
        
        tmp_files = sorted(glob.glob(os.path.join(self.temp_dir, "*.pred.tmp")))
        if not tmp_files:
            pl_module.print("[SemanticPredictLasWriter] 警告: 未找到临时预测文件")
            return
            
        # 按 bin 文件分组
        bin_file_groups = self._group_temp_files(tmp_files)
        pl_module.print(f"[SemanticPredictLasWriter] 检测到 {len(bin_file_groups)} 个唯一 bin 文件")
        
        try:
            for bin_basename, file_list in bin_file_groups.items():
                pl_module.print(f"\n[SemanticPredictLasWriter] 处理 bin 文件: {bin_basename} ({len(file_list)} 个批次)")
                try:
                    self._process_single_bin_file(bin_basename, file_list, trainer, pl_module)
                except Exception as e:
                    pl_module.print(f"!!! 错误: 处理 {bin_basename} 时失败: {e}")
                    import traceback
                    traceback.print_exc()
        finally:
            self._cleanup_temp_files(tmp_files, pl_module)

    # ================= 内部辅助方法 =================

    def _infer_class_mapping(self, trainer, pl_module):
        """推断反向类别映射"""
        if self.reverse_class_mapping is not None:
            pl_module.print(f"[SemanticPredictLasWriter] 使用用户提供的 reverse_class_mapping: {self.reverse_class_mapping}")
            return
        
        if not self.auto_infer_reverse_mapping:
            return
            
        # 尝试从模型 checkpoint 获取
        try:
            if hasattr(pl_module, 'hparams') and hasattr(pl_module.hparams, 'class_mapping'):
                mapping = pl_module.hparams.class_mapping
                if mapping:
                    self.reverse_class_mapping = {v: k for k, v in mapping.items()}
                    self._mapping_inferred = True
                    pl_module.print(f"[SemanticPredictLasWriter] 从模型 checkpoint 加载 reverse_class_mapping")
                    return
        except Exception:
            pass
            
        # 尝试从 DataModule 获取
        try:
            datamodule = trainer.datamodule
            if hasattr(datamodule, 'class_mapping') and datamodule.class_mapping:
                self.reverse_class_mapping = {v: k for k, v in datamodule.class_mapping.items()}
                self._mapping_inferred = True
                pl_module.print(f"[SemanticPredictLasWriter] 从 DataModule 推断 reverse_class_mapping")
            else:
                pl_module.print(f"[SemanticPredictLasWriter] 未找到 class_mapping，使用连续标签")
        except Exception as e:
            pl_module.print(f"[SemanticPredictLasWriter] 警告: 无法推断 reverse_class_mapping: {e}")

    def _validate_prediction(self, prediction, batch_idx):
        if 'logits' not in prediction or 'indices' not in prediction:
            print(f"警告: predict_step 必须返回 'logits' 和 'indices'。跳过批次 {batch_idx}")
            return False
        if 'bin_file' not in prediction or len(prediction['bin_file']) == 0:
            print(f"警告: batch {batch_idx} 缺少 bin_file 信息，跳过")
            return False
        return True

    def _save_batch_predictions(self, bin_files, logits, indices, bin_paths, pkl_paths, offsets, batch_idx, pl_module):
        """将一个批次的预测按文件拆分并保存"""
        file_groups = defaultdict(lambda: {'logits': [], 'indices': [], 'bin_path': None, 'pkl_path': None})
        
        start_idx = 0
        for i, end_idx in enumerate(offsets):
            # 获取文件名
            f = bin_files[i]
            bin_basename = (f if isinstance(f, str) else str(f))
            if bin_basename.endswith('.bin'):
                bin_basename = bin_basename[:-4]
            
            # 切片
            file_groups[bin_basename]['logits'].append(logits[start_idx:end_idx])
            file_groups[bin_basename]['indices'].append(indices[start_idx:end_idx])
            
            # 路径
            if file_groups[bin_basename]['bin_path'] is None:
                if isinstance(bin_paths, list) and i < len(bin_paths):
                    file_groups[bin_basename]['bin_path'] = bin_paths[i]
                if isinstance(pkl_paths, list) and i < len(pkl_paths):
                    file_groups[bin_basename]['pkl_path'] = pkl_paths[i]
            
            start_idx = end_idx
            
        # 保存
        for bin_basename, data in file_groups.items():
            if not data['logits']: continue
            
            save_path = os.path.join(self.temp_dir, f"{bin_basename}_batch_{batch_idx}.pred.tmp")
            save_dict = {
                'logits': torch.cat(data['logits'], dim=0),
                'indices': torch.cat(data['indices'], dim=0),
                'bin_file': bin_basename,
            }
            if data['bin_path']: save_dict['bin_path'] = data['bin_path']
            if data['pkl_path']: save_dict['pkl_path'] = data['pkl_path']
            
            torch.save(save_dict, save_path)
            
        if batch_idx % 10 == 0:
            pl_module.print(f"[SemanticPredictLasWriter] Batch {batch_idx}: 保存了 {len(file_groups)} 个文件的预测")

    def _ensure_num_classes(self, pl_module):
        if self.num_classes != -1: return
        
        try:
            if hasattr(pl_module, 'head'):
                if hasattr(pl_module.head, 'out_channels'):
                    self.num_classes = pl_module.head.out_channels
                elif hasattr(pl_module.head, 'num_classes'):
                    self.num_classes = pl_module.head.num_classes
            elif hasattr(pl_module, 'num_classes'):
                self.num_classes = pl_module.num_classes
            pl_module.print(f"[SemanticPredictLasWriter] 从模型推断类别数: {self.num_classes}")
        except Exception:
            print("错误: 无法从模型推断 num_classes，请显式指定")

    def _group_temp_files(self, tmp_files):
        groups = defaultdict(list)
        for f in tmp_files:
            basename = os.path.basename(f).split('_batch_')[0]
            groups[basename].append(f)
        return groups

    def _cleanup_temp_files(self, tmp_files, pl_module):
        pl_module.print(f"\n[SemanticPredictLasWriter] 清理临时文件...")
        for f in tmp_files:
            try:
                if os.path.exists(f): os.remove(f)
            except Exception as e:
                print(f"警告: 无法删除 {f}: {e}")
        
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                pl_module.print(f"[SemanticPredictLasWriter] 已清理临时文件夹")
        except Exception as e:
            pl_module.print(f"警告: 清理文件夹失败: {e}")
        pl_module.print(f"[SemanticPredictLasWriter] 所有预测已保存到 {self.output_dir}")
        pl_module.print("="*70)

    def _process_single_bin_file(self, bin_basename, tmp_files, trainer, pl_module):
        # 1. 获取路径
        bin_path, pkl_path = self._get_file_paths(bin_basename, tmp_files, trainer, pl_module)
        if not bin_path or not pkl_path: return

        # 2. 加载元数据和点云
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        point_data = np.memmap(bin_path, dtype=metadata['dtype'], mode='r')
        num_points = len(point_data)
        
        # 3. 投票
        final_preds, mean_logits, counts = self._perform_voting(tmp_files, num_points, pl_module)
        
        # 4. 映射
        if self.reverse_class_mapping:
            final_preds = self._apply_mapping(final_preds, pl_module)
            
        # 5. 保存 LAS
        xyz = np.stack([point_data['X'], point_data['Y'], point_data['Z']], axis=1).astype(np.float64)
        metadata['_bin_path'] = bin_path
        
        self._save_las_file(
            os.path.join(self.output_dir, f"{bin_basename}.las"),
            xyz, final_preds, metadata, pl_module
        )
        
        # 6. 保存 Logits
        if self.save_logits:
            np.savez_compressed(
                os.path.join(self.output_dir, f"{bin_basename}_logits.npz"),
                logits=mean_logits.numpy(),
                predictions=final_preds,
                counts=counts.numpy()
            )

    def _get_file_paths(self, bin_basename, tmp_files, trainer, pl_module):
        # 尝试从临时文件获取
        if tmp_files:
            try:
                # 显式设置 weights_only=False 以消除 FutureWarning
                # 我们需要加载包含路径字符串的字典，这是安全的（因为是我们在 write_on_batch_end 中生成的）
                data = torch.load(tmp_files[0], weights_only=False)
                if 'bin_path' in data and 'pkl_path' in data:
                    bp = data['bin_path']
                    pp = data['pkl_path']
                    return (str(bp[0]) if isinstance(bp, list) else str(bp)), \
                           (str(pp[0]) if isinstance(pp, list) else str(pp))
            except Exception:
                pass
        
        # 后备方案
        return self._find_bin_pkl_paths(bin_basename, trainer)

    def _perform_voting(self, tmp_files, num_points, pl_module):
        logits_sum = torch.zeros((num_points, self.num_classes), dtype=torch.float32)
        counts = torch.zeros(num_points, dtype=torch.int32)
        
        for f in tmp_files:
            try:
                # 显式设置 weights_only=False 以消除 FutureWarning
                d = torch.load(f, weights_only=False)
                # 确保 float32
                logits_sum.index_add_(0, d['indices'].long(), d['logits'].float())
                counts.index_add_(0, d['indices'].long(), torch.ones(len(d['indices']), dtype=torch.int32))
            except Exception as e:
                pl_module.print(f"    警告: 加载 {f} 失败: {e}")
                
        # 计算平均
        mask = (counts == 0)
        counts[mask] = 1
        mean_logits = logits_sum / counts.unsqueeze(-1)
        
        # Argmax
        if mean_logits.ndim == 2 and mean_logits.size(1) > 1:
            preds = torch.argmax(mean_logits, dim=1).numpy().astype(np.uint8)
        else:
            preds = mean_logits.squeeze().numpy().astype(np.uint8)
            
        if mask.any():
            preds[mask.numpy()] = 0
            
        return preds, mean_logits, counts

    def _apply_mapping(self, preds, pl_module):
        pl_module.print(f"  - 应用反向类别映射")
        max_label = max(self.reverse_class_mapping.keys())
        mapping = np.arange(max_label + 1)
        for k, v in self.reverse_class_mapping.items():
            mapping[k] = v
        return mapping[preds]

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


class AutoEmptyCacheCallback(Callback):
    """
    自动显存清理回调函数
    
    逻辑：
    1. 定期清理：作为兜底。
    2. 智能检测：一旦发现当前 Batch 耗时异常（绝对或相对），立即清理。
    3. 无冷却期：只要慢，就尝试救，优先保证速度恢复。
    """
    def __init__(
        self, 
        slowdown_threshold: float = 3.0,   # 相对阈值
        absolute_threshold: float = None,  # 绝对阈值 (秒)
        clear_interval: int = 0,           # 定期清理
        warmup_steps: int = 50,            # 预热步数
        verbose: bool = True
    ):
        super().__init__()
        self.config = {
            'slowdown': slowdown_threshold,
            'absolute': absolute_threshold,
            'interval': clear_interval,
            'warmup': warmup_steps
        }
        self.verbose = verbose
        
        # 状态追踪
        self.states = {} 

    def _get_state(self, stage):
        if stage not in self.states:
            self.states[stage] = {
                'start_time': 0.0,
                'avg_time': 0.0
            }
        return self.states[stage]

    def _on_batch_start(self, stage):
        state = self._get_state(stage)
        state['start_time'] = time.time()

    def _on_batch_end(self, trainer, batch_idx, stage):
        state = self._get_state(stage)
        duration = time.time() - state['start_time']
        
        # Train阶段用 global_step, 其他阶段用 batch_idx (仅用于日志显示)
        current_step = trainer.global_step if stage == 'train' else batch_idx
        
        should_clear = False
        reason = ""

        # =========================================================
        # 核心逻辑
        # =========================================================

        # 1. 优先检查定期清理
        if self.config['interval'] > 0 and (batch_idx + 1) % self.config['interval'] == 0:
            should_clear = True
            reason = "periodic"
            
        # 2. 智能检测 (如果没有触发定期清理)
        else:
            # 判断是否已预热 (有历史平均值 或 超过预热步数)
            is_warmed_up = (state['avg_time'] > 0) or (batch_idx > self.config['warmup'])
            
            # A. 相对检测: 比平均值慢 N 倍
            if (is_warmed_up and 
                state['avg_time'] > 0 and 
                duration > state['avg_time'] * self.config['slowdown']):
                should_clear = True
                reason = f"slowdown ({duration:.2f}s vs avg {state['avg_time']:.2f}s)"
            
            # B. 绝对检测: 超过 N 秒
            elif (self.config['absolute'] is not None and 
                  duration > self.config['absolute']):
                should_clear = True
                reason = f"absolute limit ({duration:.2f}s > {self.config['absolute']}s)"

        # =========================================================
        # 执行动作
        # =========================================================
        if should_clear:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if self.verbose:
                    stage_name = stage.upper()
                    step_info = f"global_step={current_step}" if stage == 'train' else f"batch={batch_idx}"
                    trainer.print(f"\n[AutoCache][{stage_name}] 🧹 Cleared at {step_info}. Reason: {reason}")

        # 更新平均值 (EMA)
        # 策略：只有在【未触发清理】(即认为是正常Batch) 时才更新平均值
        # 这样可以防止异常慢的 Batch 污染平均值，保持检测的敏锐度
        if not should_clear:
            if state['avg_time'] == 0:
                state['avg_time'] = duration
            else:
                # alpha = 0.05
                state['avg_time'] = state['avg_time'] * 0.95 + duration * 0.05

    # ================= 钩子绑定 (保持不变) =================
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._on_batch_start('train')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._on_batch_end(trainer, batch_idx, 'train')

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._on_batch_start('val')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._on_batch_end(trainer, batch_idx, 'val')

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._on_batch_start('test')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._on_batch_end(trainer, batch_idx, 'test')

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._on_batch_start('predict')

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._on_batch_end(trainer, batch_idx, 'predict')