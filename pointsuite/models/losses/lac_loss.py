# 要解决 pointops 与 DataLoader 多进程冲突的问题，要在程序入口处设置环境变量：
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['OMP_NUM_THREADS'] = '4'


import torch
import torch.nn as nn
import torch.nn.functional as F

class LACLoss(nn.Module):
    """
    标签感知一致性损失 (受超点/监督对比损失启发)
    
    核心思想: 
    只在那些“几何相近”且“真实语义标签相同”的点对之间, 
    强制要求它们的“预测概率”保持一致。
    
    这避免了在不同物体的边界处进行不必要的平滑，从而保护了边界的清晰度。
    """
    def __init__(self, k_neighbors=16, ignore_index=-1, loss_weight=1.0):
        """
        Args:
            k_neighbors (int): 用于计算一致性损失的邻居数量
            ignore_index (int, optional): 训练时要忽略的标签索引
            loss_weight (float): 此损失的权重
        """
        super(LACLoss, self).__init__()
        self.k_neighbors = k_neighbors
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        
    def forward(self, pred, data_dict):
        """
        Args:
            pred (torch.Tensor): 网络的原始输出 (seg_logits), shape [N, num_classes]
            data_dict (dict): 包含所有需要信息的字典, 必须包含:
                              - 'coord' (torch.Tensor): 点的坐标, shape [N, 3]
                              - 'offset' (torch.Tensor): 批次信息, shape [B]
                              - 'segment' (torch.Tensor): 真实标签, shape [N]
        Returns:
            torch.Tensor: 一个标量的标签感知平滑损失值
        """
        # Lazy import to avoid conflicts in DataLoader workers
        import pointops
        
        # 1. 从 pred 和 data_dict 中提取所有需要的数据
        seg_logits = pred
        coord = data_dict["coord"]
        offset = data_dict["offset"].int()
        labels = data_dict["class"] 
        
        # 2. 计算预测概率
        # 这个计算需要保留梯度，因此在 no_grad() 之外
        probs = F.softmax(seg_logits, dim=1) # [N, C]

        # 3. 寻找邻居并计算所有掩码 (不需要梯度)
        with torch.no_grad():
            # 3.1 寻找K近邻
            reference_index, _ = pointops.knn_query(
                self.k_neighbors, coord, offset
            ) # [N, k]
            
            # 3.2 创建有效的KNN掩码 (无效邻居索引为 -1)
            knn_mask = torch.sign(reference_index + 1).bool() # [N, k]

            # 3.3 分组邻居的真实标签
            # grouping 需要 [N, C] 格式, 所以先 unsqueeze
            neighbor_labels = pointops.grouping(
                reference_index, labels.unsqueeze(1).float(), coord, with_xyz=False
            ) # [N, k, 1]
            neighbor_labels = neighbor_labels.squeeze(-1).long() # [N, k]

            # 3.4 语义掩码: 中心点和邻居点是否具有相同的真实标签?
            # labels.unsqueeze(1) -> [N, 1]
            same_label_mask = (labels.unsqueeze(1) == neighbor_labels) # [N, k]

            # 3.5 忽略掩码: 中心点和邻居点是否都不是 ignore_index?
            center_valid_mask = (labels != self.ignore_index) # [N]
            neighbor_valid_mask = (neighbor_labels != self.ignore_index) # [N, k]
            
            # 3.6 最终掩码: 我们只关心那些...
            # (1) 是有效邻居 (knn_mask)
            # (2) 中心点不是 ignore (center_valid_mask)
            # (3) 邻居点不是 ignore (neighbor_valid_mask)
            # (4) 中心点和邻居点标签相同 (same_label_mask)
            total_mask = (
                knn_mask & 
                center_valid_mask.unsqueeze(1) & 
                neighbor_valid_mask & 
                same_label_mask
            ) # [N, k]

        # 4. 分组邻居的预测概率 (这步需要梯度)
        neighbor_probs = pointops.grouping(
            reference_index, probs, coord, with_xyz=False
        ) # [N, k, C]
        
        # 5. 计算损失
        
        # 5.1 计算中心点和邻居点的概率分布差异 (L2 距离的平方)
        # center_probs -> [N, 1, C]
        center_probs = probs.unsqueeze(1)
        # prob_dist_sq -> [N, k, C] -> [N, k]
        prob_dist_sq = ((center_probs - neighbor_probs)**2).sum(dim=2)

        # 5.2 应用最终的 "标签感知" 掩码
        masked_prob_dist_sq = prob_dist_sq * total_mask.float() # [N, k]
        
        # 5.3 计算平均损失 (只对所有有效的点对 "pairs" 求均值)
        sum_loss = masked_prob_dist_sq.sum()
        num_valid_pairs = total_mask.sum().clamp(min=1.0)
        
        mean_loss = sum_loss / num_valid_pairs
            
        return mean_loss * self.loss_weight