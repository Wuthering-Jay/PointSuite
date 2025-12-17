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
import torch
import torch.nn as nn
import torch.nn.functional as F
import pointops
import einops
import numpy as np

# ==========================================
# 1. 创新点核心模块：各向异性频谱几何编码 (Anisotropic SG-RPE)
# ==========================================
class AnisotropicSG_RPE(nn.Module):
    """
    针对机载点云优化：
    1. 显式解耦 Z 轴 (Elevation) 与 XY 平面 (Range)，因为 DALES 中高度包含极其重要的语义(灌木/树/地面)。
    2. 引入 NeRF 风格的傅里叶高频映射，捕捉微小的几何纹理差异。
    """
    def __init__(self, embed_dim=32, num_freqs=4):
        super().__init__()
        self.num_freqs = num_freqs
        self.embed_dim = embed_dim
        
        # 定义频率带: 2^0, 2^1, ... (Log-spaced)
        # num_freqs=4 意味着能覆盖从 0.1m 到 1.6m 的多尺度细节
        self.freq_bands = 2.0 ** torch.linspace(0., num_freqs - 1, num_freqs)
        
        # 输入维度计算:
        # 原始 delta_xyz (3) + 欧氏距离 (1)
        # + XY 频率特征 (2 dim * num_freqs * 2 sin/cos)
        # + Z  频率特征 (1 dim * num_freqs * 2 sin/cos)
        self.input_feat_dim = 3 + 1 + (2 * num_freqs * 2) + (1 * num_freqs * 2)
        
        # 映射网络
        self.net = nn.Sequential(
            nn.Linear(self.input_feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, delta_xyz):
        """
        delta_xyz: [N, K, 3] (Neighbor - Center)
        """
        N, K, C = delta_xyz.shape
        x = delta_xyz.view(-1, C) # Flatten [M, 3]
        
        # 1. 基础几何特征
        d_xy = x[:, 0:2] # [M, 2]
        d_z  = x[:, 2:3] # [M, 1] (Z轴独立)
        dist = torch.norm(x, p=2, dim=1, keepdim=True) # [M, 1]
        
        features = [x, dist] # 包含原始低频信息
        
        # 2. 频谱映射 (Spectral Mapping)
        # 确保频率在同一设备上
        if x.device != self.freq_bands.device:
            self.freq_bands = self.freq_bands.to(x.device)

        # (a) XY 平面编码 (捕捉水平轮廓，如建筑边缘)
        for freq in self.freq_bands:
            features.append(torch.sin(d_xy * freq * np.pi))
            features.append(torch.cos(d_xy * freq * np.pi))
            
        # (b) Z 轴编码 (捕捉高度纹理，如灌木噪点)
        # Z轴通常需要更高的敏感度，这里我们复用相同的频率，但它是独立通道
        for freq in self.freq_bands:
            features.append(torch.sin(d_z * freq * np.pi))
            features.append(torch.cos(d_z * freq * np.pi))
            
        # 3. 拼接与融合
        x_enc = torch.cat(features, dim=1) # [M, input_feat_dim]
        out = self.net(x_enc)
        
        return out.view(N, K, self.embed_dim)

# ==========================================
# 2. 辅助函数：全局池化
# ==========================================
def global_max_pool_with_offset(x, offset):
    if offset is None:
        return x.max(dim=0, keepdim=True)[0].expand_as(x)
        print("Warning: offset is None in global_max_pool_with_offset!")
    batch_size = offset.shape[0]
    global_feats = []
    start = 0
    for i in range(batch_size):
        end = offset[i].item()
        if end > start:
            batch_max = x[start:end].max(dim=0, keepdim=True)[0]
            global_feats.append(batch_max.expand(end - start, -1))
        else:
            global_feats.append(x[start:end]) 
        start = end
    return torch.cat(global_feats, dim=0)

# ==========================================
# 3. 多尺度动态门控注意力 (Multi-Scale Dynamic Gated Attention)
# ==========================================
class DynamicGatedAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        # 兼容性参数，不再使用
        pe_multiplier=False, 
        pe_bias=True,        
        norm_layer=nn.BatchNorm1d,
        scaling_factor=20.0 # 此处其实不需要了，因为我们换用了 SG-RPE
    ):
        super(DynamicGatedAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        # 1. 基础线性投影
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

        # 2. [创新点 B] Anisotropic SG-RPE
        # 替代了原来的 linear_p_bias (MLP)
        self.pos_encoder = AnisotropicSG_RPE(
            embed_dim=embed_channels,
            num_freqs=4 
        )

        # 3. 几何权重编码 (MLP)
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups), 
            self.norm_layer(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )

        # 4. [创新点 A] 多尺度动态门控
        # 输入维度: Local(C) + Neighborhood(C) + Global(C) = 3*C
        # 这是解决 "卡车 vs 建筑" 尺度混淆的关键
        self.context_gate = nn.Sequential(
            nn.Linear(embed_channels * 3, embed_channels),
            self.norm_layer(embed_channels),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channels, groups),
            nn.Sigmoid() 
        )

        # [初始化修正] 保证初始全开，防止掉点
        nn.init.constant_(self.context_gate[3].bias, 3.0)
        nn.init.normal_(self.context_gate[3].weight, std=0.001)

        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index, offset=None):
        """
        Args:
            feat: [N, C]
            coord: [N, 3]
            reference_index: [N, K]
            offset: [B]
        """
        # A. 基础变换
        query_raw, key_raw, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )

        # B. 准备数据 & Grouping
        # key_grouped: [N, K, C]
        # neighbor_xyz: [N, K, 3]
        key_grouped = pointops.grouping(reference_index.detach(), key_raw.float(), coord.float(), with_xyz=True) 
        neighbor_xyz = key_grouped[:, :, 0:3]
        key_feat = key_grouped[:, :, 3:]
        
        value_grouped = pointops.grouping(reference_index.detach(), value.float(), coord.float(), with_xyz=False)

        # C. [创新点 B 应用] 显式相对位置编码
        center_xyz = coord.unsqueeze(1) # [N, 1, 3]
        delta_xyz = neighbor_xyz - center_xyz # [N, K, 3]
        
        # 调用 SG-RPE 模块
        # 输出包含了高频纹理感知的位置特征，专门对付灌木
        relative_pos_enc = self.pos_encoder(delta_xyz) # [N, K, C]

        # D. 核心 Vector Attention 计算
        # Relation = (Key - Query) + SG-RPE
        # 保持了 "直观相减" 的逻辑，但 PosEnc 变强了
        relation = (key_feat - query_raw.unsqueeze(1)) + relative_pos_enc
        
        # 计算基础几何权重
        geom_weights = self.weight_encoding(relation) # [N, K, Groups]

        # E. [创新点 A 应用] 多尺度上下文门控
        # 1. Local Context: [N, C]
        local_ctx = query_raw
        
        # 2. Neighborhood Context: [N, C] (感知局部体量，区分卡车/建筑)
        # 将 query_raw 聚合到邻域
        # 注意：这里我们想知道中心点周围的语义环境，所以用 max pool
        neighbor_ctx_grouped = pointops.grouping(reference_index.detach(), query_raw.float(), coord.float(), with_xyz=False) # [N, K, C]
        neighbor_ctx = neighbor_ctx_grouped.max(dim=1)[0] # [N, C]
        
        # 3. Global Context: [N, C] (感知场景类型)
        global_ctx = global_max_pool_with_offset(query_raw, offset)
        
        # 4. 融合与门控
        gate_input = torch.cat([local_ctx, neighbor_ctx, global_ctx], dim=-1) # [N, 3C]
        gate_coeff = self.context_gate(gate_input).unsqueeze(1) # [N, 1, Groups]

        # F. 调制
        refined_weights = geom_weights * gate_coeff 

        # G. 聚合
        attn_weights = self.softmax(refined_weights)
        attn_weights = self.attn_drop(attn_weights)

        with torch.no_grad():
            mask = torch.sign(reference_index + 1)
        attn_weights = torch.einsum("n k g, n k -> n k g", attn_weights, mask)

        value_grouped = einops.rearrange(value_grouped, "n k (g i) -> n k g i", g=self.groups)
        feat_out = torch.einsum("n k g i, n k g -> n g i", value_grouped, attn_weights)
        feat_out = einops.rearrange(feat_out, "n g i -> n (g i)")

        return feat_out

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
        norm_layer=PointBatchNorm,
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
        # self.rope = Continuous3DRoPE(embed_channels, scaling_factor=10.0)

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

        # 3. [升级] 多尺度动态门控 (Multi-Scale Context Gate)
        # 输入维度: Local(C) + Neighborhood(C) + Global(C) = 3 * C
        # 这种设计能让门控感知到"物体大小"和"场景类型"
        self.context_gate = nn.Sequential(
            nn.Linear(embed_channels * 3, embed_channels), # 输入变大了
            self.norm_layer(embed_channels),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channels, groups),
            nn.Sigmoid() 
        )

        # 4. [初始化] 依然保持 Bias=3.0 的策略
        nn.init.constant_(self.context_gate[3].bias, 3.0)
        nn.init.normal_(self.context_gate[3].weight, std=0.001)

        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, offset, reference_index):
        """
        input: feat: [n, c], coord: [n, 3], offset: [b], reference_index: [n, k]
        output: feat: [n, c]
        """
        query, key, value = (
            self.linear_q(feat), # [n, c]
            self.linear_k(feat), # [n, c]
            self.linear_v(feat), # [n, c]
        )
        
        # pointops.grouping 只支持 FP32 输入，输出让 autocast 管理
        key = pointops.grouping(reference_index.detach(), key.float(), coord.float(), with_xyz=True) # [n, k, 3+c]
        value = pointops.grouping(reference_index.detach(), value.float(), coord.float(), with_xyz=False) # [n, k, c]
        pos, key = key[:, :, 0:3], key[:, :, 3:] # [n, k, 3], [n, k, c]
        relation_qk = key - query.unsqueeze(1) # [n, k ,c], 邻域内与中心点的相对位置, 用于相对位置编码
        if self.pe_multiplier: # 乘性因子
            pem = self.linear_p_multiplier(pos) # [n, k, c]
            relation_qk = relation_qk * pem # [n, k, c]
        if self.pe_bias: # 偏置因子
            peb = self.linear_p_bias(pos) # [n, k, c]
            relation_qk = relation_qk + peb # [n, k, c]
            value = value + peb # [n, k, c]

        # A. 基础几何权重
        weight = self.weight_encoding(relation_qk) 

        # =======================================================
        # B. [升级核心] 构建多尺度上下文 (Multi-Scale Context)
        # =======================================================
        
        # 1. Local (中心点语义)
        local_ctx = query # [N, C]
        
        # 2. Neighborhood (邻域语义 - 感知局部边缘/体量)
        # 重新 group 一次 query 的原始特征 (或者用 linear_q 后的特征)
        # 使用 linear_q 后的 query 来做 grouping 比较省事
        # Max Pooling 能捕捉到邻域内最显著的特征 (比如卡车的棱角)
        neighbor_ctx_grouped = pointops.grouping(reference_index.detach(), query.float(), coord.float(), with_xyz=False) # [N, K, C]
        neighbor_ctx = neighbor_ctx_grouped.max(dim=1)[0] # [N, C]
        
        # 3. Global (场景语义 - 感知宏观环境)
        global_ctx = global_max_pool_with_offset(query, offset) # [N, C]
        
        # 4. 融合输入 Gate
        # [N, C] + [N, C] + [N, C] -> [N, 3C]
        gate_input = torch.cat([local_ctx, neighbor_ctx, global_ctx], dim=-1)
        
        # 5. 生成门控系数
        gate_coeff = self.context_gate(gate_input) # [N, Groups]
        
        # =======================================================

        # 调制权重
        weight = weight * gate_coeff.unsqueeze(1)

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
    def __init__(self, in_channels, out_channels, norm_layer=PointBatchNorm, k_neighbors=16):
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
class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    格网池化，基于体素划分进行池化下采样，体素内坐标平均池化，特征最大池化，得到新的pxo，同时输出体素索引
    Args:
        in_channels: 输入维度
        out_channels: 输出维度
        grid_size: 体素大小
        bias: fc层偏置
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False, norm_layer=PointBatchNorm):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], start: [b, 3]
        output: points: [pxo], [[v,3],[v,c],[b]], cluster: [n]
        """
        coord, feat, offset = points # [n, 3] [n, c] [b]
        batch = offset2batch(offset) # [b] -> [n]
        feat = self.act(self.norm(self.fc(feat))) # [n, c]
        
        # 这些操作不需要梯度
        with torch.no_grad():
            start = (
                segment_csr(
                    coord,
                    torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                    reduce="min",
                )
                if start is None
                else start
            )
            cluster = voxel_grid(
                pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
            )
            unique, cluster, counts = torch.unique(
                cluster, sorted=True, return_inverse=True, return_counts=True
            )
            _, sorted_cluster_indices = torch.sort(cluster)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        # 使用 detach 后的索引
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster.detach()  # cluster 不需要梯度
    

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
        norm_layer=PointBatchNorm,
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
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], skip_points: [pxo], [[ns,3],[ns,c],[b]], cluster: [ns]
        output: points: [pxo], [[ns,3],[ns,c],[b]]
        """
        coord, feat, offset = points # [n, 3] [n, c] [b]
        skip_coord, skip_feat, skip_offset = skip_points # [ns, 3] [ns, c] [b]
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster] # [n, c] -> [ns, c], 投影上采样
        else:
            feat = pointops.interpolation(
                coord, skip_coord, self.proj(feat), offset, skip_offset
            ) # [n, c] -> [ns, c], 插值上采样
        if self.skip: # 跳跃连接，特征融合
            feat = feat + self.proj_skip(skip_feat) # [ns, c]
        return [skip_coord, feat, skip_offset] # [ns, 3] [ns, c] [b]