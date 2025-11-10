from modules.point_wise import PointBatchNorm, PointLayerNorm
import torch.nn as nn
import torch


class SegHead(nn.Module):
    """
    常规语义分割头，将点特征映射到类别 logits
    """
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            norm_layer: nn.Module = PointLayerNorm,
    ):
        super().__init__()
        self.seg_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入点特征，形状为 [N, C_in]
        
        Returns:
            torch.Tensor: 输出类别 logits，形状为 [N, num_classes]
        """
        logits = self.seg_head(x)
        return logits