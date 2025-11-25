import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import importlib
import yaml
from typing import List, Dict, Any

class BaseTask(pl.LightningModule):
    """
    一个抽象的任务基类 (LightningModule)。
    
    它负责处理所有任务共有的逻辑：
    1. 自动从 YAML 配置中实例化损失函数 (losses)。
    2. 自动从 YAML 配置中实例化指标 (metrics)。
    3. 自动在 validation/test epoch 结束时计算和记录所有指标。
    
    注意: 
    我们 *不* 在这里实现 `configure_optimizers`。
    PyTorch Lightning 的 `LightningCLI` 会自动读取您在
    `configs/schedules/` 目录中定义的 `optimizer` 和 `lr_scheduler` 
    配置，并自动为您配置它们。这保持了 Task 模块的简洁。
    """
    
    def __init__(self,
                 learning_rate: float = 1e-3,
                 loss_configs: List[Dict[str, Any]] = None,
                 metric_configs: List[Dict[str, Any]] = None,
                 class_mapping: Dict[int, int] = None,
                 class_names: List[str] = None,
                 ignore_label: int = -1):
        """
        Args:
            learning_rate (float): 学习率。
                                   我们在此处接收 learning_rate (而不是仅在优化器配置中)
                                   主要有两个原因:
                                   1. 日志记录: 'self.save_hyperparameters()' 会自动
                                      将 'learning_rate' 记录到 TensorBoard/Wandb。
                                   2. 灵活性: 允许在不使用 'LightningCLI' 的纯 Python 模式下
                                      轻松访问 'self.hparams.learning_rate' 来配置优化器。
                                   
                                   在 YAML 配置中，我们应将此 'learning_rate' 视为“单一事实来源”，
                                   并在 'optimizer' 配置中使用 YAML 链接 (例如:
                                   lr: ${model.init_args.learning_rate}) 来引用它。
                                   
            loss_configs (List[Dict]): 
                来自 YAML 的损失函数配置列表。
                示例: 
                - class_path: point_suite.models.losses.focal_loss.FocalLoss
                  init_args: { gamma: 2.0 }
                  weight: 1.0 # (可选) 损失的权重
                  
            metric_configs (List[Dict]): 
                来自 YAML 的指标配置列表。
                示例:
                - class_path: pointsuite.utils.metrics.OverallAccuracy
                  init_args: { num_classes: 8 }
                  
            class_mapping (Dict[int, int]): 
                原始类别标签 -> 连续类别标签的映射
                例如: {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
                此映射将被保存到 checkpoint，并在预测时自动加载到 SemanticPredictLasWriter
                如果为 None，表示不使用类别映射
                
            class_names (List[str]): 
                类别名称列表，用于验证时显示
                例如: ['Ground', 'Vegetation', 'Building']
                如果为 None，验证时显示 Class 0, Class 1, ...
        """
        super().__init__()
        # 将超参数保存到 checkpoint
        # 🔥 关键修改：保存所有参数，包括 loss_configs 和 metric_configs
        # 这样 load_from_checkpoint 才能正确重建 Task
        self.save_hyperparameters()
        
        # 保存 class_mapping 用于 SemanticPredictLasWriter
        self.class_mapping = class_mapping
        
        # 追踪最佳 mIoU
        self.best_miou = 0.0
        self.best_miou_epoch = -1
        
        # 🔥 自定义 hparams 保存钩子，确保中文正确显示
        self._custom_save_hparams()
        
        # --- 1. 动态实例化损失函数 ---
        self.losses = nn.ModuleDict()
        self.loss_weights = {}
        if loss_configs:
            for cfg in loss_configs:
                # 'loss_name' 是我们给这个损失起的名字，例如 'focal_loss'
                loss_name = cfg.get("name", cfg["class_path"].split('.')[-1].lower())
                loss_class = self._import_class(cfg["class_path"])
                init_args = cfg.get("init_args", {})
                
                self.losses[loss_name] = loss_class(**init_args)
                self.loss_weights[loss_name] = cfg.get("weight", 1.0)
                
        # --- 2. 动态实例化指标 ---
        # 我们使用 ModuleDict 来确保指标被正确移动到 GPU
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        if metric_configs:
            for cfg in metric_configs:
                metric_name = cfg.get("name", cfg["class_path"].split('.')[-1].lower())
                metric_class = self._import_class(cfg["class_path"])
                init_args = cfg.get("init_args", {})
                
                # 为 val 和 test 分别创建实例，以避免状态冲突
                self.val_metrics[metric_name] = metric_class(**init_args)
                self.test_metrics[metric_name] = metric_class(**init_args)
    
    def _custom_save_hparams(self):
        """
        自定义保存 hparams.yaml，确保中文字符正确显示
        覆盖 PyTorch Lightning 默认的 YAML dump 行为
        """
        try:
            import os
            # 获取 log_dir
            if hasattr(self.logger, 'log_dir') and self.logger.log_dir:
                hparams_file = os.path.join(self.logger.log_dir, 'hparams.yaml')
                # 延迟保存：在 trainer 完成设置后再保存
                # 这里只是标记，实际保存会在 on_train_start 中进行
                self._pending_hparams_save = True
        except Exception:
            pass  # 如果失败就使用默认行为
    
    def on_train_start(self):
        """训练开始时保存 hparams（确保中文正确显示）"""
        if hasattr(self, '_pending_hparams_save') and self._pending_hparams_save:
            try:
                import os
                if hasattr(self.logger, 'log_dir') and self.logger.log_dir:
                    hparams_file = os.path.join(self.logger.log_dir, 'hparams.yaml')
                    # 使用 allow_unicode=True 确保中文正确保存
                    with open(hparams_file, 'w', encoding='utf-8') as f:
                        yaml.dump(
                            dict(self.hparams), 
                            f, 
                            allow_unicode=True,  # 🔥 关键：允许 Unicode 字符
                            default_flow_style=False,
                            sort_keys=False
                        )
                self._pending_hparams_save = False
            except Exception as e:
                print(f"Warning: Could not save hparams with Chinese characters: {e}")

    def _import_class(self, class_path: str) -> type:
        """一个辅助函数，用于从字符串路径动态导入类"""
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _calculate_total_loss(self, preds: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        (子类可以覆盖)
        计算所有损失函数的加权总和。
        
        注意：Loss 计算强制在 FP32 下运行，以避免混合精度训练中的数值不稳定问题，
        特别是当使用 ignore_index=-1 时。
        
        Args:
            preds (Any): 模型的 forward() 输出。
            batch (Dict): 来自 DataLoader 的批次数据。
            
        Returns:
            Dict[str, torch.Tensor]: 包含 'total_loss' 和每个单独损失的字典。
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # 强制 Loss 计算在 FP32 下运行，避免混合精度问题
        # 注意：这里需要同时禁用 autocast 并将 tensors 转为 FP32
        with torch.amp.autocast('cuda', enabled=False):
            # 将 preds 转换为 FP32，但保留梯度
            if isinstance(preds, torch.Tensor):
                if preds.is_floating_point() and preds.dtype != torch.float32:
                    preds = preds.float()
            elif isinstance(preds, dict):
                preds_fp32 = {}
                for k, v in preds.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32:
                        preds_fp32[k] = v.float()
                    else:
                        preds_fp32[k] = v
                preds = preds_fp32
            
            # 同样处理 batch 中的 target (虽然通常是 long 类型)
            batch_fp32 = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32:
                    batch_fp32[k] = v.float()
                else:
                    batch_fp32[k] = v
            
            for name, loss_fn in self.losses.items():
                # 损失函数接收 (preds, batch)
                loss = loss_fn(preds, batch_fp32)
                # 确保 loss 也是 FP32
                if loss.dtype != torch.float32:
                    loss = loss.float()
                loss_dict[name] = loss
                total_loss += self.loss_weights[name] * loss
            
        loss_dict["total_loss"] = total_loss
        return loss_dict
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """
        从 batch 中推断 batch_size。
        
        适配我们项目的 collate_fn：
        - 如果有 'batch_index'，使用 max + 1
        - 如果有 'offset'，使用 len(offset)
        - 否则返回 1
        """
        if 'batch_index' in batch:
            return batch['batch_index'].max().item() + 1
        elif 'offset' in batch:
            return len(batch['offset'])
        else:
            return 1
    
    def postprocess_predictions(self, preds: Any) -> torch.Tensor:
        """
        后处理预测结果，将模型输出转换为标签或 logits
        
        这是一个可选的钩子方法，子类可以覆盖以支持复杂的输出处理。
        
        默认行为:
        - 如果 preds 是字典且包含 'logits' 键，返回 preds['logits']
        - 如果 preds 是字典且包含 'labels' 键，返回 preds['labels']
        - 否则假设 preds 就是 logits/labels，直接返回
        
        用途:
        1. Mask3D: 可以在这里实现 class_logits @ mask_logits
        2. 多任务模型: 可以提取特定任务的输出
        3. 后处理: argmax, softmax, sigmoid 等
        
        Args:
            preds: 模型的原始输出 (可以是 Tensor, Dict, Tuple 等)
            
        Returns:
            torch.Tensor: 
                - 对于验证/测试: 返回 logits [N, C] 用于 metrics 计算
                - 对于预测: 返回 logits [N, C] 或 labels [N] 用于保存
        
        Examples:
            >>> # 示例 1: 标准语义分割 (默认)
            >>> def postprocess_predictions(self, preds):
            >>>     return preds  # 直接返回 logits
            
            >>> # 示例 2: Mask3D
            >>> def postprocess_predictions(self, preds):
            >>>     class_logits = preds['class_logits']  # [N_queries, C]
            >>>     mask_logits = preds['mask_logits']    # [N_queries, N]
            >>>     point_logits = class_logits.T @ mask_logits  # [C, N]
            >>>     return point_logits.T  # [N, C]
            
            >>> # 示例 3: 直接返回标签
            >>> def postprocess_predictions(self, preds):
            >>>     if isinstance(preds, dict) and 'labels' in preds:
            >>>         return preds['labels']  # [N] - Metrics 会自动检测
            >>>     return torch.argmax(preds, dim=-1)  # 手动 argmax
        """
        # 默认实现：处理常见的字典格式
        if isinstance(preds, dict):
            if 'logits' in preds:
                return preds['logits']
            elif 'labels' in preds:
                return preds['labels']
            elif 'pred' in preds:
                return preds['pred']
            else:
                # 如果是字典但没有标准键，返回第一个值
                # 这可能需要子类覆盖
                return next(iter(preds.values()))
        
        # 如果不是字典，假设就是 logits/labels
        return preds
    
    # --- 训练 (Training) 逻辑 ---
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        训练步骤。
        """
        # 确保模型在训练模式（解决 eval mode 警告）
        if batch_idx == 0:
            self.train()
        
        # 调试信息：记录训练步数和数据统计
        if batch_idx % 50 == 0 or batch_idx > 160:  # 在问题区域附近更频繁记录
            coord = batch.get('coord', None)
            if coord is not None:
                print(f"\n[Step {self.global_step}] Batch {batch_idx}: "
                      f"points={len(coord)}, "
                      f"coord_range=[{coord.min(0)[0]}, {coord.max(0)[0]}]")
        
        # 前向传播
        try:
            preds = self(batch)
        except Exception as e:
            # 保存问题数据
            import pickle
            error_data_path = f'error_batch_{batch_idx}_step_{self.global_step}.pkl'
            with open(error_data_path, 'wb') as f:
                pickle.dump({
                    'batch_idx': batch_idx,
                    'global_step': self.global_step,
                    'batch': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()},
                    'error': str(e)
                }, f)
            print(f"\n{'='*80}")
            print(f"[ERROR] 在 batch_idx={batch_idx}, global_step={self.global_step} 时发生错误")
            print(f"问题数据已保存到: {error_data_path}")
            print(f"{'='*80}\n")
            raise
        
        # Loss 计算
        loss_dict = self._calculate_total_loss(preds, batch)
        total_loss = loss_dict["total_loss"]
        
        # 保存最新的 loss 到模块中，供 CustomProgressBar 直接读取
        # 避免 PL 默认进度条的平滑处理导致数值看起来“卡死”
        current_loss = total_loss.item()
        self.last_loss = current_loss
        # 🔥 强制更新 trainer 上的属性，确保 CustomProgressBar 能读取到最新值
        if self.trainer is not None:
            self.trainer.live_loss = current_loss
        
        # 记录损失
        batch_size = self._get_batch_size(batch)
        for name, loss_value in loss_dict.items():
            # 所有损失都显示在进度条，同时记录到 TensorBoard
            self.log(
                f"{name}_step",
                loss_value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=batch_size,
            )
            
        # 调试：每 100 步打印一次 loss，确认是否在变化
        if batch_idx % 100 == 0:
            print(f" [Step {self.global_step}] Loss: {total_loss.item():.6f}")
        
        return total_loss

    def on_train_epoch_end(self):
        """
        在训练 epoch 结束时调用，清理显存以便验证。
        """
        # 强制清理 CUDA 缓存，避免验证时 OOM
        if torch.cuda.is_available():
            import gc
            gc.collect()  # Python 垃圾回收
            torch.cuda.empty_cache()  # CUDA 缓存清理
            torch.cuda.synchronize()  # 同步 CUDA 操作
            # 输出显存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n[Memory] Epoch {self.current_epoch} training end: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            print(f"[Memory] Cleared cache and GC, starting validation...\n")
    
    def on_validation_start(self):
        """
        在验证开始前再次清理显存。
        """
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Memory] Validation start: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB\n")
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        # 1. 前向传播
        preds = self.forward(batch)
        
        # 2. 计算损失
        loss_dict = self._calculate_total_loss(preds, batch)
        
        # 3. 记录损失 (PL 会自动添加 'val/' 前缀)
        batch_size = self._get_batch_size(batch)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        # 4. 后处理预测结果 (支持 Mask3D 等复杂输出)
        processed_preds = self.postprocess_predictions(preds)
        
        # 5. 更新指标
        # 提取目标标签 (支持多种命名约定)
        target = batch.get('class', batch.get('label', batch.get('labels', batch.get('target'))))
        for metric in self.val_metrics.values():
            metric.update(processed_preds, target)

    def on_validation_epoch_end(self):
        # 5. 在 epoch 结束时，计算并记录所有指标
        metric_results = {}
        
        # 临时存储用于打印的指标
        print_metrics = {}
        
        for name, metric in self.val_metrics.items():
            val = metric.compute()
            
            # 处理返回字典的指标 (如 SegmentationMetrics)
            if isinstance(val, dict):
                # 记录到 metric_results (用于 log)
                for k, v in val.items():
                    # 过滤掉非标量值 (如 class_names 列表)
                    if isinstance(v, (torch.Tensor, float, int)):
                        # 确保 tensor 是标量
                        if isinstance(v, torch.Tensor) and v.numel() > 1:
                            continue
                        metric_results[k] = v
                
                # 保存完整结果用于打印
                print_metrics.update(val)
                
                # 如果包含混淆矩阵相关的详细信息，尝试提取
                if 'iou_per_class' in val:
                    print_metrics['per_class_iou'] = val['iou_per_class']
                if 'precision_per_class' in val:
                    print_metrics['per_class_precision'] = val['precision_per_class']
                if 'recall_per_class' in val:
                    print_metrics['per_class_recall'] = val['recall_per_class']
                if 'f1_per_class' in val:
                    print_metrics['per_class_f1'] = val['f1_per_class']
                
            else:
                metric_results[name] = val
                print_metrics[name] = val
            
            metric.reset() # 重置指标状态
        
        # 记录指标 (prog_bar=False 以避免污染进度条)
        self.log_dict(metric_results, on_step=False, on_epoch=True, prog_bar=False)
        
        # --- 打印详细指标 ---
        # 检查是否有 mIoU 信息 (无论是来自 SegmentationMetrics 还是单独的 MeanIoU)
        miou_key = 'mean_iou'
        if miou_key in print_metrics:
            try:
                current_miou = float(print_metrics[miou_key])
                
                # 更新最佳 mIoU
                if current_miou > self.best_miou:
                    self.best_miou = current_miou
                    self.best_miou_epoch = self.current_epoch
                
                # 获取其他指标
                overall_acc = print_metrics.get('overall_accuracy', None)
                
                # 准备每类指标
                per_class_iou = print_metrics.get('per_class_iou', None)
                per_class_precision = print_metrics.get('per_class_precision', None)
                per_class_recall = print_metrics.get('per_class_recall', None)
                per_class_f1 = print_metrics.get('per_class_f1', None)
                
                # 如果没有直接提供每类指标，尝试从旧的 MeanIoU metric 对象中获取 (兼容旧代码)
                if per_class_iou is None and 'mean_iou' in self.val_metrics:
                    metric = self.val_metrics['mean_iou']
                    if hasattr(metric, 'confusion_matrix'):
                        confmat = metric.confusion_matrix.cpu().numpy()
                        import numpy as np
                        intersection = np.diag(confmat)
                        union = confmat.sum(1) + confmat.sum(0) - np.diag(confmat)
                        per_class_iou = intersection / (union + 1e-10)
                        per_class_precision = intersection / (confmat.sum(0) + 1e-10)
                        per_class_recall = intersection / (confmat.sum(1) + 1e-10)
                        per_class_f1 = 2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall + 1e-10)

                # 输出标题和总体指标
                print(f"\n{'='*100}")
                print(f"Validation Epoch {self.current_epoch} - Metrics")
                print(f"{'='*100}")
                if overall_acc is not None:
                    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
                print(f"Mean IoU (current): {current_miou:.4f}")
                print(f"Mean IoU (best)   : {self.best_miou:.4f} (Epoch {self.best_miou_epoch})")
                if current_miou > self.best_miou - 1e-6:  # 当前是最佳
                    print(f"🎉 New best mIoU achieved!")
                print(f"{'='*100}")
                
                # 输出每个类别的详细指标
                if per_class_iou is not None:
                    print(f"  {'Class':15s}  {'IoU':>8s}  {'Precision':>10s}  {'Recall':>8s}  {'F1-Score':>10s}")
                    print(f"  {'-'*15}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
                    
                    # 获取类别名
                    class_names = print_metrics.get('class_names', None)
                    if class_names is None:
                        class_names = self.hparams.get('class_names', None) if hasattr(self, 'hparams') else None
                    
                    import numpy as np
                    # 确保是 numpy 数组
                    if isinstance(per_class_iou, torch.Tensor): per_class_iou = per_class_iou.cpu().numpy()
                    if isinstance(per_class_precision, torch.Tensor): per_class_precision = per_class_precision.cpu().numpy()
                    if isinstance(per_class_recall, torch.Tensor): per_class_recall = per_class_recall.cpu().numpy()
                    if isinstance(per_class_f1, torch.Tensor): per_class_f1 = per_class_f1.cpu().numpy()

                    num_classes = len(per_class_iou)
                    for i in range(num_classes):
                        c_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                        print(f"  {c_name:15s}  {per_class_iou[i]:8.4f}  {per_class_precision[i]:10.4f}  "
                              f"{per_class_recall[i]:8.4f}  {per_class_f1[i]:10.4f}")
                    
                    # 计算平均指标
                    mean_precision = np.nanmean(per_class_precision)
                    mean_recall = np.nanmean(per_class_recall)
                    mean_f1 = np.nanmean(per_class_f1)
                    
                    print(f"  {'-'*15}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
                    print(f"  {'Mean':15s}  {current_miou:8.4f}  {mean_precision:10.4f}  "
                          f"{mean_recall:8.4f}  {mean_f1:10.4f}")
                print(f"{'='*100}\n")
            except Exception as e:
                print(f"Warning: Could not print detailed metrics: {e}")
                import traceback
                traceback.print_exc()

    # --- 测试 (Test) 逻辑 ---
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        测试步骤：计算损失和指标
        """
        # 逻辑与 validation_step 相同
        preds = self.forward(batch)
        loss_dict = self._calculate_total_loss(preds, batch)
        batch_size = self._get_batch_size(batch)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # 后处理预测结果
        processed_preds = self.postprocess_predictions(preds)
        
        # 提取目标标签 (支持多种命名约定)
        target = batch.get('class', batch.get('label', batch.get('labels', batch.get('target'))))
        for metric in self.test_metrics.values():
            metric.update(processed_preds, target)

    def on_test_epoch_end(self):
        # 在 epoch 结束时，计算并记录所有指标
        metric_results = {}
        print_metrics = {}
        
        for name, metric in self.test_metrics.items():
            val = metric.compute()
            
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, (torch.Tensor, float, int)):
                        # 确保 tensor 是标量
                        if isinstance(v, torch.Tensor) and v.numel() > 1:
                            continue
                        metric_results[k] = v
                
                print_metrics.update(val)
                
                if 'iou_per_class' in val:
                    print_metrics['per_class_iou'] = val['iou_per_class']
                if 'precision_per_class' in val:
                    print_metrics['per_class_precision'] = val['precision_per_class']
                if 'recall_per_class' in val:
                    print_metrics['per_class_recall'] = val['recall_per_class']
                if 'f1_per_class' in val:
                    print_metrics['per_class_f1'] = val['f1_per_class']
            else:
                metric_results[name] = val
                print_metrics[name] = val
            
            metric.reset()
            
        self.log_dict(metric_results, on_step=False, on_epoch=True)
        
        # --- 打印详细指标 ---
        miou_key = 'mean_iou'
        if miou_key in print_metrics:
            try:
                current_miou = float(print_metrics[miou_key])
                overall_acc = print_metrics.get('overall_accuracy', None)
                
                per_class_iou = print_metrics.get('per_class_iou', None)
                per_class_precision = print_metrics.get('per_class_precision', None)
                per_class_recall = print_metrics.get('per_class_recall', None)
                per_class_f1 = print_metrics.get('per_class_f1', None)
                
                # 兼容旧代码
                if per_class_iou is None and 'mean_iou' in self.test_metrics:
                    metric = self.test_metrics['mean_iou']
                    if hasattr(metric, 'confusion_matrix'):
                        confmat = metric.confusion_matrix.cpu().numpy()
                        import numpy as np
                        intersection = np.diag(confmat)
                        union = confmat.sum(1) + confmat.sum(0) - np.diag(confmat)
                        per_class_iou = intersection / (union + 1e-10)
                        per_class_precision = intersection / (confmat.sum(0) + 1e-10)
                        per_class_recall = intersection / (confmat.sum(1) + 1e-10)
                        per_class_f1 = 2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall + 1e-10)

                print(f"\n{'='*100}")
                print(f"Test Results - Metrics")
                print(f"{'='*100}")
                if overall_acc is not None:
                    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
                print(f"Mean IoU: {current_miou:.4f}")
                print(f"{'='*100}")
                
                if per_class_iou is not None:
                    print(f"  {'Class':15s}  {'IoU':>8s}  {'Precision':>10s}  {'Recall':>8s}  {'F1-Score':>10s}")
                    print(f"  {'-'*15}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
                    
                    class_names = print_metrics.get('class_names', None)
                    if class_names is None:
                        class_names = self.hparams.get('class_names', None) if hasattr(self, 'hparams') else None
                    
                    import numpy as np
                    if isinstance(per_class_iou, torch.Tensor): per_class_iou = per_class_iou.cpu().numpy()
                    if isinstance(per_class_precision, torch.Tensor): per_class_precision = per_class_precision.cpu().numpy()
                    if isinstance(per_class_recall, torch.Tensor): per_class_recall = per_class_recall.cpu().numpy()
                    if isinstance(per_class_f1, torch.Tensor): per_class_f1 = per_class_f1.cpu().numpy()

                    num_classes = len(per_class_iou)
                    for i in range(num_classes):
                        c_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                        print(f"  {c_name:15s}  {per_class_iou[i]:8.4f}  {per_class_precision[i]:10.4f}  "
                              f"{per_class_recall[i]:8.4f}  {per_class_f1[i]:10.4f}")
                    
                    mean_precision = np.nanmean(per_class_precision)
                    mean_recall = np.nanmean(per_class_recall)
                    mean_f1 = np.nanmean(per_class_f1)
                    
                    print(f"  {'-'*15}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
                    print(f"  {'Mean':15s}  {current_miou:8.4f}  {mean_precision:10.4f}  {mean_recall:8.4f}  {mean_f1:10.4f}")
                print(f"{'='*100}\n")
            except Exception as e:
                print(f"警告: 无法打印详细指标: {e}")