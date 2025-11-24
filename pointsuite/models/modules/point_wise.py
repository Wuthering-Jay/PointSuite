import torch
import torch.nn as nn
import einops
import pointops
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
from pointsuite.models.utils import offset2batch, batch2offset

# ———— Normalization 层 ————
class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    对形状为[n, c], [n, l, c]的点云数据进行批量归一化
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class PointLayerNorm(nn.Module):
    """
    Layer Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    对形状为[n, c], [n, l, c]的点云数据进行层归一化
    """
    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.LayerNorm(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input)
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError
        

# ———— 局部特征聚合层 ————
class GroupedVectorAttention(nn.Module):
    """
    分组向量注意力机制
    Args:
        embed_channesl: 输入输出维度
        groups: 分组数量
        attn_drop_rate: drop比例
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        norm_layer: 归一化层类型
    """
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        norm_layer=PointLayerNorm,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias
        self.norm_layer = norm_layer

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            self.norm_layer(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            self.norm_layer(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                self.norm_layer(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                self.norm_layer(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            self.norm_layer(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        """
        input: feat: [n, c], coord: [n, 3], reference_index: [n, k]
        output: feat: [n, c]
        """
        query, key, value = (
            self.linear_q(feat), # [n, c]
            self.linear_k(feat), # [n, c]
            self.linear_v(feat), # [n, c]
        )
        
        # pointops.grouping 只支持 FP32 输入，输出让 autocast 管理
        key = pointops.grouping(reference_index, key.float(), coord.float(), with_xyz=True) # [n, k, 3+c]
        value = pointops.grouping(reference_index, value.float(), coord.float(), with_xyz=False) # [n, k, c]
        pos, key = key[:, :, 0:3], key[:, :, 3:] # [n, k, 3], [n, k, c]
        relation_qk = key - query.unsqueeze(1) # [n, k ,c], 邻域内与中心点的相对位置, 用于相对位置编码
        if self.pe_multiplier: # 乘性因子
            pem = self.linear_p_multiplier(pos) # [n, k, c]
            relation_qk = relation_qk * pem # [n, k, c]
        if self.pe_bias: # 偏置因子
            peb = self.linear_p_bias(pos) # [n, k, c]
            relation_qk = relation_qk + peb # [n, k, c]
            value = value + peb # [n, k, c]

        weight = self.weight_encoding(relation_qk) # [n, k, g]
        weight = self.attn_drop(self.softmax(weight)) # [n, k, g]

        # mask 操作不需要梯度
        with torch.no_grad():
            mask = torch.sign(reference_index + 1) # [n, k], 无效邻域点标记为0
        
        weight = torch.einsum("n s g, n s -> n s g", weight, mask) # [n, k, g]
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups) # [n, k, g, i]
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight) # [n, g, i]
        feat = einops.rearrange(feat, "n g i -> n (g i)") # [n, c]
        return feat # [n, c]
    

class PointNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=PointLayerNorm, k_neighbors=16):
        """
        使用 Linear 层和 PointBatchNorm 的 PointNet 层实现
        
        参数:
            in_channels: 输入特征维度 c1
            out_channels: 输出特征维度 c2
            k_neighbors: KNN近邻数
            norm_layer: 归一化层类型
        """
        super().__init__()
        self.k_neighbors = k_neighbors
        self.norm_layer = norm_layer

        # 计算中间层维度
        mid_channels = max(in_channels, out_channels // 2)
        
        # 输入特征维度调整（如果包含xyz坐标）
        mlp_in_channels = in_channels
        
        # 定义共享MLP网络（使用Linear层）
        self.shared_mlp = nn.Sequential(
            nn.Linear(mlp_in_channels, mid_channels),
            self.norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels),
            self.norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, points):
        coord, feat, offset = points
        
        # KNN 查询需要 FP32 坐标
        with torch.no_grad():
            reference_index, _ = pointops.knn_query(self.k_neighbors, coord.float(), offset)
        
        # pointops.grouping 只支持 FP32 输入，输出让 autocast 管理
        grouped_features = pointops.grouping(reference_index, feat.float(), coord.float(), with_xyz=False)
        
        n_points = grouped_features.shape[0]
        grouped_features = grouped_features.reshape(-1, grouped_features.shape[-1])
        out = self.shared_mlp(grouped_features)
        out = out.reshape(n_points, self.k_neighbors, -1)
        out = out.max(dim=1)[0]
        
        return out
    

 # ———— 池化层 ————

 # ... (之前的导入保持不变)

class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """
    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointLayerNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], start: [b, 3]
        output: points: [pxo], [[v,3],[v,c],[b]], cluster: [n]
        """
        coord, feat, offset = points
        batch = offset2batch(offset)
        
        # 1. 确保 feat 计算在自动混合精度下进行
        feat = self.act(self.norm(self.fc(feat)))
        
        # 2. 坐标计算必须使用 FP32
        coord_fp32 = coord.float()
        
        with torch.no_grad():
            start = (
                segment_csr(
                    coord_fp32,
                    torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                    reduce="min",
                )
                if start is None
                else start
            )
        
        cluster = voxel_grid(
            pos=coord_fp32 - start[batch], size=self.grid_size, batch=batch, start=0
        )
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        
        # 获取排序索引
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        # 3. 池化操作
        coord = segment_csr(coord_fp32[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        
        # 🔥 [修复 1] 正确更新 batch 索引 (先重排再切片)
        batch = batch[sorted_cluster_indices][idx_ptr[:-1]]
        
        # 🔥 [修复 2] 强制 offset 为 int32 !!! (这是最关键的一步)
        offset = batch2offset(batch).int()
        
        return [coord, feat, offset], cluster

        
class GridPool1(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    格网池化，基于体素划分进行池化下采样，体素内坐标平均池化，特征最大池化，得到新的pxo，同时输出体素索引
    Args:
        in_channels: 输入维度
        out_channels: 输出维度
        grid_size: 体素大小
        bias: fc层偏置
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointLayerNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], start: [b, 3]
        output: points: [pxo], [[v,3],[v,c],[b]], cluster: [n]
        """
        coord, feat, offset = points # [n, 3] [n, c] [b]
        batch = offset2batch(offset) # [b] -> [n]
        feat = self.act(self.norm(self.fc(feat))) # [n, c]
        
        # 坐标操作需要 FP32 精度，但不禁用 autocast
        # 只在需要时转换 coord 到 FP32，让 feat 保持 autocast 的类型
        coord_fp32 = coord.float()
        
        # 计算 start (batch 内每个样本的最小坐标)
        with torch.no_grad():
            start = (
                segment_csr(
                    coord_fp32,
                    torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                    reduce="min",
                )
                if start is None
                else start
            )
        
        # voxel_grid 和后续操作不能在 no_grad 中，因为需要跟踪梯度
        cluster = voxel_grid(
            pos=coord_fp32 - start[batch], size=self.grid_size, batch=batch, start=0
        )
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        # 输出坐标用 FP32，feat 保持原始类型
        coord = segment_csr(coord_fp32[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster  # 不 detach，让 PyTorch 管理梯度
    
class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    带有跳跃连接的上采样
    Args:
        in_channels: 输入维度
        out_channels: 输出维度
        skip_channels: 跳跃连接维度
        bias: fc层偏置
        skip: 是否使用跳跃连接
        backend: 上采样方式，'map' or 'interp'
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bias=True,
        skip=True,
        backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointLayerNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointLayerNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], skip_points: [pxo], [[ns,3],[ns,c],[b]], cluster: [ns]
        output: points: [pxo], [[ns,3],[ns,c],[b]]
        """
        coord, feat, offset = points 
        skip_coord, skip_feat, skip_offset = skip_points 
        
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster] 
        else:
            feat_proj = self.proj(feat)
            # 🔥 [修复 3] 防御性编程：确保 interpolation 接收 int32 offset
            if offset.dtype != torch.int32:
                offset = offset.int()
            if skip_offset.dtype != torch.int32:
                skip_offset = skip_offset.int()
                
            feat = pointops.interpolation(
                coord.float(), skip_coord.float(), feat_proj.float(), offset, skip_offset
            ) 
        
        if self.skip: 
            feat = feat + self.proj_skip(skip_feat) 
        return [skip_coord, feat, skip_offset]