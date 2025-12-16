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

class Cayley3DRoPE(nn.Module):
    def __init__(self, embed_dim, base=10000.0, scaling_factor=20.0):
        """
        Cayley-STRING 3D RoPE 实现
        
        Args:
            embed_dim: 输入特征维度
            scaling_factor: 坐标缩放因子 (针对户外点云)
        """
        super().__init__()
        # 复用之前的 Continuous3DRoPE 逻辑作为内核
        self.rope_kernel = Continuous3DRoPE(embed_dim, base, scaling_factor)
        
        # --- Cayley-STRING 核心部分 ---
        # 我们需要学习一个反对称矩阵 S (Skew-symmetric matrix)
        # 参数量: d*(d-1)/2，非常小
        # S 的定义：S = A - A.T，其中 A 是可学习参数
        self.S_param = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        
        self.embed_dim = embed_dim

    def _get_orthogonal_matrix_P(self):
        """
        通过 Cayley Transform 计算正交矩阵 P
        P = (I - S)(I + S)^-1
        """
        # 1. 构造反对称矩阵 S
        S = torch.triu(self.S_param, diagonal=1) # 取上三角
        S = S - S.t() # S = A - A.T，保证反对称性
        
        # 2. 计算 P
        I = torch.eye(self.embed_dim, device=S.device, dtype=S.dtype)
        
        # P = (I - S) @ (I + S)^-1
        # 在 PyTorch 中，用 solve 求解线性方程组比求逆更稳健
        # (I + S) @ P.T = (I - S).T -> P @ (I + S).T = (I - S)
        # 这里直接用数学定义计算
        denom = torch.linalg.solve(I + S, I - S) 
        
        # 实际上 P = denom。因为 (I+S)P = (I-S)
        return denom

    def forward(self, x, coord):
        """
        x: [N, C] 特征
        coord: [N, 3] 坐标
        """
        # 1. 计算当前的正交变换矩阵 P [C, C]
        P = self._get_orthogonal_matrix_P()
        
        # 2. Basis Change (基变换): x' = x @ P.T
        # 论文公式 R(r) = P * RoPE * P.T
        # 作用在向量 z 上: z_new = R(r) * z = P * RoPE * (P.T * z)
        # 所以第一步是乘 P.T
        x_rotated_basis = torch.matmul(x, P.t()) 
        
        # 3. Apply Standard 3D RoPE (在优化后的基上旋转)
        x_rope = self.rope_kernel(x_rotated_basis, coord)
        
        # 4. Inverse Basis Change (逆变换): x_out = x_rope @ P
        # 变回原来的语义空间
        x_out = torch.matmul(x_rope, P)
        
        return x_out

# ==========================================
# 1. 连续 3D 旋转位置编码 (RoPE) 模块
# ==========================================
class Continuous3DRoPE(nn.Module):
    def __init__(self, embed_dim, base=10000.0, scaling_factor=20.0):
        """
        Args:
            embed_dim (int): 输入维度 (e.g., 32, 64)
            scaling_factor (float): 推荐 10.0 - 50.0 用于户外点云
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.base = base
        self.scaling_factor = scaling_factor
        
        # 自动分配维度: 确保 xyz 分到的维度是偶数
        dim_per_axis = (embed_dim // 3) // 2 * 2 
        self.dim_per_axis = dim_per_axis
        self.dim_rotated = dim_per_axis * 3
        self.dim_remain = embed_dim - self.dim_rotated
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, coord):
        # x: [N, C], coord: [N, 3]
        coord = coord * self.scaling_factor
        
        x_rotated_part = x[..., :self.dim_rotated]
        x_remain_part = x[..., self.dim_rotated:]
        
        x_feat, y_feat, z_feat = torch.split(x_rotated_part, self.dim_per_axis, dim=-1)
        
        x_out = self._apply_rope_1d(x_feat, coord[..., 0])
        y_out = self._apply_rope_1d(y_feat, coord[..., 1])
        z_out = self._apply_rope_1d(z_feat, coord[..., 2])
        
        return torch.cat([x_out, y_out, z_out, x_remain_part], dim=-1)

    def _apply_rope_1d(self, feat, coord_1d):
        angles = torch.outer(coord_1d.view(-1), self.inv_freq).view(*coord_1d.shape, -1)
        pos_sin = torch.sin(angles)
        pos_cos = torch.cos(angles)
        feat_r, feat_i = feat.chunk(2, dim=-1)
        feat_out_r = feat_r * pos_cos - feat_i * pos_sin
        feat_out_i = feat_r * pos_sin + feat_i * pos_cos
        return torch.cat([feat_out_r, feat_out_i], dim=-1)

# ==========================================
# 2. 辅助函数：处理 Batch 的全局池化
# ==========================================
def global_max_pool_with_offset(x, offset):
    """
    对 Pointops 格式的 Flatten 数据进行全局最大池化。
    x: [N_total, C]
    offset: [B], e.g. [1000, 2000, 3000...] 表示每个 batch 的结束索引
    Returns:
        global_feat_expanded: [N_total, C] (每个点都赋上了它所属点云的全局特征)
    """
    if offset is None:
        # 如果没有 offset，假设整个输入就是一个 batch
        # [1, C] -> [N, C]
        return x.max(dim=0, keepdim=True)[0].expand_as(x)
    
    batch_size = offset.shape[0]
    global_feats = []
    start = 0
    for i in range(batch_size):
        end = offset[i].item()
        # 对当前 batch 的点取 max
        # feat_batch: [N_i, C] -> max -> [1, C]
        batch_max = x[start:end].max(dim=0, keepdim=True)[0]
        # 扩展回该 batch 的点数
        global_feats.append(batch_max.expand(end - start, -1))
        start = end
    
    return torch.cat(global_feats, dim=0)

# ==========================================
# 3. 动态门控注意力主类 (Dynamic Gated Attention)
# ==========================================
class DynamicGatedAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        # --- 兼容性保留参数 (虽然 RoPE 不需要它们，但保留以防报错) ---
        pe_multiplier=False, 
        pe_bias=True,        
        # -------------------------------------------------------
        norm_layer=PointBatchNorm, # 保持你原本的 norm_layer
        scaling_factor=2.0        # [新增] 仅需增加这一个参数，且给定默认值
    ):
        super(DynamicGatedAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        # 1. 基础线性层
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

        # 2. [RoPE 替代了原本的 pe_multiplier/bias]
        # 虽然参数里传进了 pe_bias，但我们这里不使用它，而是初始化 RoPE
        # self.rope = Continuous3DRoPE(embed_channels, scaling_factor=scaling_factor)
        self.rope = Cayley3DRoPE(embed_channels, scaling_factor=scaling_factor)

        # 3. 几何权重编码
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            self.norm_layer(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )

        # 4. [新增] 动态语义门控 (context_gate)
        # 输入维度: Local Query (C) + Global Context (C) = 2*C
        self.context_gate = nn.Sequential(
            nn.Linear(embed_channels * 2, embed_channels),
            self.norm_layer(embed_channels),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channels, groups),
            nn.Sigmoid() 
        )
        nn.init.constant_(self.context_gate[3].bias, 3.0) 
        nn.init.normal_(self.context_gate[3].weight, std=0.001)

        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index, offset=None):
        """
        Args:
            feat: [N, C] 输入特征
            coord: [N, 3] 物理坐标
            reference_index: [N, K] 邻居索引
            offset: [B] (可选) Batch 偏移量，用于计算正确的全局池化。
                    如果你的 DataLoader 不传 offset 给 Block，
                    请确保 feat 是 [B, N, C] 格式并自行调整代码，
                    或者在这里传入 offset。
        """
        # 1. 基础线性变换
        query_raw, key_raw, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )

        # 2. [RoPE] 注入位置信息到 Q 和 K
        query = self.rope(query_raw, coord)
        key = self.rope(key_raw, coord)

        # 3. Grouping 邻域特征聚合
        # key_grouped: [N, K, C]
        key_grouped = pointops.grouping(reference_index.detach(), key.float(), coord.float(), with_xyz=False) 
        value_grouped = pointops.grouping(reference_index.detach(), value.float(), coord.float(), with_xyz=False)

        # 4. 计算相对几何特征 (Relation)
        # RoPE 使得 key - query 包含了旋转相对位置信息
        relation_qk = key_grouped - query.unsqueeze(1) # [N, K, C]

        # 5. 计算基础几何权重
        geom_weights = self.weight_encoding(relation_qk) # [N, K, Groups]

        # ==========================================
        # [关键部分] 计算全局增强的动态门控
        # ==========================================
        
        # 5.1 计算 Global Max Pooling (Scene Context)
        # 使用 query_raw (纯语义) 而不是 query (含位置) 来做池化效果更好
        global_feat = global_max_pool_with_offset(query_raw, offset) # [N, C]
        
        # 5.2 拼接 Local Context + Global Context
        # gate_input: [N, 2C]
        gate_input = torch.cat([query_raw, global_feat], dim=-1)
        
        # 5.3 生成门控系数
        # gate_coeff: [N, Groups] -> [N, 1, Groups]
        gate_coeff = self.context_gate(gate_input).unsqueeze(1) 

        # 6. 门控调制 (Modulation)
        # 语义上下文 (Local+Global) 决定激活哪些几何专家
        refined_weights = geom_weights * (1.0 + gate_coeff)

        # 7. Softmax & Aggregation
        attn_weights = self.softmax(refined_weights) # [N, K, Groups]
        attn_weights = self.attn_drop(attn_weights)

        # Mask 无效邻居
        with torch.no_grad():
            mask = torch.sign(reference_index + 1) # [N, K]
        
        # Apply mask: [N, K, G] * [N, K, 1]
        attn_weights = torch.einsum("n k g, n k -> n k g", attn_weights, mask)

        # 8. 加权求和
        # Value shape transform: [N, K, C] -> [N, K, G, C/G]
        value_grouped = einops.rearrange(value_grouped, "n k (g i) -> n k g i", g=self.groups)
        
        # Sum over K: [N, K, G, i] * [N, K, G] -> [N, G, i]
        feat_out = torch.einsum("n k g i, n k g -> n g i", value_grouped, attn_weights)
        
        # Merge groups back
        feat_out = einops.rearrange(feat_out, "n g i -> n (g i)") # [N, C]

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
        # query = self.rope(query, coord) # [n, c]
        # key = self.rope(key, coord)     # [n, c]
        
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