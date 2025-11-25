"""
自定义进度条回调

显示训练过程中的关键信息：
- 当前 batch size 和总点数
- 简化的指标名称
"""

from pytorch_lightning.callbacks import TQDMProgressBar
from typing import Dict, Any


class CustomProgressBar(TQDMProgressBar):
    """
    自定义进度条，显示 batch 信息和简化指标
    
    显示格式示例：
    Epoch 1: 100%|████| 50/50 [01:23<00:00, bs=4, pts=143K, loss=0.52, mIoU=0.78]
    """
    
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        """
        初始化自定义进度条
        
        参数：
            refresh_rate: 刷新频率
            process_position: 进程位置
        """
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
    
    def get_metrics(self, trainer, pl_module) -> Dict[str, Any]:
        """
        获取要在进度条中显示的指标
        
        简化指标名称：
        - train_loss -> loss
        - mean_iou -> mIoU
        - overall_accuracy -> OA
        - learning_rate -> lr
        - ce_loss_step -> CE
        - lovasz_loss_step -> Lov
        - total_loss_step -> loss
        """
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)  # 移除版本号
        
        # 简化指标名称
        simplified = {}
        
        # 优先处理 loss (优先显示实时 loss)
        # 使用 try-except 块防止任何属性访问错误导致训练崩溃
        try:
            # 1. 尝试从 trainer.live_loss 获取 (最可靠，由 training_step 直接写入)
            if hasattr(trainer, 'live_loss'):
                simplified["loss"] = trainer.live_loss
            # 2. 尝试从 pl_module.last_loss 获取
            elif hasattr(pl_module, 'last_loss'):
                simplified["loss"] = pl_module.last_loss
            # 3. 尝试从 trainer.lightning_module 获取 (以防 pl_module 是副本)
            elif hasattr(trainer, 'lightning_module') and hasattr(trainer.lightning_module, 'last_loss'):
                simplified["loss"] = trainer.lightning_module.last_loss
        except Exception:
            pass
            
        # 如果上述方法都失败，回退到 items 中的值
        if "loss" not in simplified:
            if "total_loss_step" in items:
                simplified["loss"] = items.pop("total_loss_step")
        elif "train_loss_step" in items:
            simplified["loss"] = items.pop("train_loss_step")
        elif "loss" in items:
            simplified["loss"] = items.pop("loss")
        elif "train_loss" in items:
            simplified["loss"] = items.pop("train_loss")
            
        for k, v in items.items():
            # 移除前缀和后缀
            k_clean = k.replace("train_", "").replace("val_", "").replace("_step", "").replace("_epoch", "")
            
            # 简化名称映射
            name_map = {
                "loss": "loss",
                "total_loss": "loss",
                "mean_iou": "mIoU",
                "overall_accuracy": "OA",
                "learning_rate": "lr",
                "ce_loss": "CE",
                "lovasz_loss": "LOV",
            }
            
            k_short = name_map.get(k_clean, k_clean)
            
            # 过滤掉不需要在进度条显示的指标 (如验证集的详细指标)
            # 只要不在 name_map 中的，或者显式排除的
            if k_short in ['mIoU', 'OA', 'precision', 'recall', 'f1']:
                continue
                
            simplified[k_short] = v
        
        # 添加 batch 信息（如果可用）
        if hasattr(pl_module, '_current_batch_info'):
            batch_info = pl_module._current_batch_info
            if 'batch_size' in batch_info:
                simplified['bs'] = batch_info['batch_size']
            if 'num_points' in batch_info:
                # 简化点数显示（K = 千，M = 百万）
                pts = batch_info['num_points']
                if pts >= 1_000_000:
                    simplified['pts'] = f"{pts/1_000_000:.1f}M"
                elif pts >= 1_000:
                    simplified['pts'] = f"{pts/1_000:.0f}K"
                else:
                    simplified['pts'] = str(pts)
        
        return simplified
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在训练 batch 结束时更新 batch 信息"""
        # 提取 batch 信息
        if isinstance(batch, dict):
            batch_size = len(batch.get('offset', [])) if 'offset' in batch else 1
            num_points = batch.get('coord', None)
            if num_points is not None:
                num_points = num_points.shape[0]
            
            # 保存到模块以供进度条使用
            pl_module._current_batch_info = {
                'batch_size': batch_size,
                'num_points': num_points,
            }
        
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """在验证 batch 结束时更新 batch 信息"""
        # 提取 batch 信息
        if isinstance(batch, dict):
            batch_size = len(batch.get('offset', [])) if 'offset' in batch else 1
            num_points = batch.get('coord', None)
            if num_points is not None:
                num_points = num_points.shape[0]
            
            # 保存到模块以供进度条使用
            pl_module._current_batch_info = {
                'batch_size': batch_size,
                'num_points': num_points,
            }
        
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
