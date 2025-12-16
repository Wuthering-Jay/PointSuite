import torch
import torch.nn as nn
import torch.nn.functional as F

class SACBLoss(nn.Module):
    """
    结构感知对比边界细化损失 (Structure-Aware Contrastive Boundary Loss)
    
    Paper Innovation Point C: 
    显式地优化特征空间中的决策边界。
    - 对同类邻居 (Positive): 拉近特征距离 (Consistency)
    - 对异类邻居 (Negative): 推远特征距离 (Boundary Sharpening)
    """
    def __init__(self, k_neighbors=16, ignore_index=-1, loss_weight=1.0, push_margin=0.5, temperature=0.1):
        super(SACBLoss, self).__init__()
        self.k_neighbors = k_neighbors
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.push_margin = push_margin # 异类特征之间的最小余弦距离
        self.temperature = temperature # 控制对比学习的平滑度

    def forward(self, features, data_dict):
        """
        Args:
            features (torch.Tensor): 网络的倒数第二层特征 (Penultimate Layer), shape [N, D]
                                     注意：不要传 logits，要传进入分类头之前的 feature
            data_dict (dict): 包含 coord, offset, segment/class (labels)
        """
        import pointops
        
        # 1. 准备数据
        coord = data_dict["coord"]
        offset = data_dict["offset"].int()
        # 支持多种标签键名
        labels = data_dict.get("class")
        
        # 对特征进行归一化，以便计算余弦相似度
        features = F.normalize(features, p=2, dim=1) # [N, D]

        # 2. 寻找邻居 (No Grad)
        with torch.no_grad():
            idx, _ = pointops.knn_query(self.k_neighbors, coord, offset) # [N, K]
            
            # 构建掩码
            knn_mask = (idx != -1) # [N, K]
            
            # 获取邻居标签
            neighbor_labels = pointops.grouping(
                idx, labels.unsqueeze(1).float(), coord, with_xyz=False
            ).squeeze(-1).long() # [N, K]
            
            # Mask 1: 有效点 (非 ignore)
            valid_mask = (labels != self.ignore_index).unsqueeze(1) & \
                         (neighbor_labels != self.ignore_index) & knn_mask # [N, K]

            # Mask 2: 正样本对 (同类) -> 用于平滑内部
            pos_mask = (labels.unsqueeze(1) == neighbor_labels) & valid_mask

            # Mask 3: 负样本对 (异类) -> 用于锐化边界
            # 只有当中心点在边界附近时，才会有异类邻居
            neg_mask = (labels.unsqueeze(1) != neighbor_labels) & valid_mask

        # 3. 获取邻居特征 (Grad required)
        # features: [N, D] -> neighbor_features: [N, K, D]
        neighbor_features = pointops.grouping(idx, features, coord, with_xyz=False)
        
        # 4. 计算余弦相似度
        # center: [N, 1, D], neighbor: [N, K, D] -> dot product -> [N, K]
        cosine_sim = (features.unsqueeze(1) * neighbor_features).sum(dim=-1)
        
        # 5. 计算损失
        
        # Part A: Positive Loss (Pull) - 让同类更像
        # Sim 越接近 1, Loss 越小
        pos_loss = (1.0 - cosine_sim) * pos_mask.float()
        pos_loss = pos_loss.sum() / (pos_mask.sum().clamp(min=1.0))
        
        # Part B: Negative Loss (Push) - 让异类更不像
        # Sim 应该小于 margin (比如 0.2)。如果 Sim > margin，则产生 Loss
        # ReLU(sim - margin)
        neg_loss = F.relu(cosine_sim - self.push_margin) * neg_mask.float()
        
        # 这里的负样本对可能很少（只有边界点有），所以要小心除数为0
        num_neg_pairs = neg_mask.sum()
        if num_neg_pairs > 0:
            neg_loss = neg_loss.sum() / num_neg_pairs
        else:
            neg_loss = 0.0

        # 6. 总损失
        # 通常 Negative Loss 权重给大一点，因为边界点稀少但重要
        total_loss = pos_loss + 0.5 * neg_loss
        
        return total_loss * self.loss_weight