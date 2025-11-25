from ..modules.point_wise import *
from timm.layers import DropPath
from copy import deepcopy


class Block(nn.Module):
    """
    网络模块单位，结合GVA和BottleNeck对pxo进行处理，不改变数据维度
    Args:
        embed_channesl: 输入输出维度
        groups: 分组数量
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
    """
    def __init__(
        self,
        embed_channels,
        groups,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=PointBatchNorm,
    ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            norm_layer=norm_layer,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = norm_layer(embed_channels)
        self.norm2 = norm_layer(embed_channels)
        self.norm3 = norm_layer(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop_path = (DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity())

    def forward(self, points, reference_index):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], reference_index: [n, k]
        output: [pxo], [[n,3],[n,c],[b]], 不改变维度
        """
        coord, feat, offset = points # [n,3], [n,c], [b]
        identity = feat # [n, c]
        feat = self.act(self.norm1(self.fc1(feat))) # [n, c]
        feat = self.attn(feat, coord, reference_index) # [n, c]
        feat = self.act(self.norm2(feat)) # [n, c]
        feat = self.norm3(self.fc3(feat)) # [n, c]
        feat = identity + self.drop_path(feat) # [n, c], bottleneck设计
        feat = self.act(feat) # [n, c]
        return [coord, feat, offset] # [[n,3],[n,c],[b]]
    
class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=PointBatchNorm,
    ):
        super(BlockSequence, self).__init__()

        # 确保 drop_path_rates 为 list
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        # 多个 Block 堆叠
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                norm_layer=norm_layer,
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points 
        with torch.no_grad():
            reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        
        for block in self.blocks:
            points = block(points, reference_index)
        return points
    
class Encoder(nn.Module):
    """
    Encoder for Point Transformer V2, 先进行格网池化, 再进行BlockSequence处理
    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        groups: 分组数量
        grid_size: 体素大小
        neighbours: 邻域大小
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
    """
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        norm_layer=PointBatchNorm,
    ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
            norm_layer=norm_layer,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            norm_layer=norm_layer,
        )

    def forward(self, points):
        """
        input: points: [pxo], [[n,3],[n,c],[b]]
        output: points: [pxo], [[ns,3],[ns,c],[b]], cluster: [n]
        """
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    """
    Decoder for Point Transformer V2, 先进行上采样, 再进行BlockSequence处理
    Args:
        in_channels: 输入维度
        skip_channels: 跳跃连接维度
        embed_channels: 输出维度
        groups: 分组数量
        depth: 解码器深度
        neighbours: 邻域大小
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        unpool_backend: 上采样方式，'map' or 'interp'
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        groups,
        depth,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        unpool_backend="map",
        norm_layer=PointBatchNorm,
    ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
            norm_layer=norm_layer,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            norm_layer=norm_layer,
        )

    def forward(self, points, skip_points, cluster):
        """
        input: points: [pxo], [[ns,3],[ns,c],[b]], skip_points: [pxo], [[n,3],[n,c],[b]], cluster: [n]
        output: points: [pxo], [[n,3],[n,c],[b]]
        """
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    """
    Patch Embedding for Point Transformer V2
    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        groups: 分组数量
        neighbours: 邻域大小
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
    """
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=PointBatchNorm,
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = embed_channels // 2
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, self.mid_channels, bias=False),
            norm_layer(self.mid_channels),
            nn.ReLU(inplace=True),
        )
        self.pointnet = PointNetLayer(in_channels, embed_channels-self.mid_channels, norm_layer=norm_layer, k_neighbors=neighbours)
        
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )

    def forward(self, points):
        """
        input: points: [pxo], [[n,3],[n,c],[b]]
        output: points: [pxo], [[n,3],[n,c],[b]]
        """
        coord, feat, offset = points
        feat1 = self.proj(feat)
        feat2 = self.pointnet(points)
        feat = torch.cat([feat1,feat2],dim=1)
        return self.blocks([coord, feat, offset])
    

class PointTransformerV2(nn.Module):
    """
    Point Transformer V2
    Args:
        in_channels: 输入维度
        patch_embed_depth: Patch Embedding深度
        patch_embed_channels: Patch Embedding输出维度
        patch_embed_groups: Patch Embedding分组数量
        patch_embed_neighbours: Patch Embedding邻域大小
        enc_depths: 编码器深度
        enc_channels: 编码器输出维度
        enc_groups: 编码器分组数量
        enc_neighbours: 编码器邻域大小
        dec_depths: 解码器深度
        dec_channels: 解码器输出维度
        dec_groups: 解码器分组数量
        dec_neighbours: 解码器邻域大小
        grid_sizes: 体素大小
        attn_qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        unpool_backend: 上采样方式，'map' or 'interp'
    """
    def __init__(
        self,
        in_channels,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        unpool_backend="map",
        norm_layer=PointBatchNorm,
    ):
        super(PointTransformerV2, self).__init__()
        self.in_channels = in_channels
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        # 点云嵌入层
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
        )
        # bottleneck的drop率逐渐提高
        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]
        # 前一层的输出维度作为下一层的输入维度
        enc_channels = [patch_embed_channels] + list(enc_channels) # [48, 96, 192, 384, 512]
        dec_channels = list(dec_channels) + [enc_channels[-1]] # [48, 96, 192, 384, 512]
        # 编码器与解码器
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]) : sum(enc_depths[: i + 1])
                ],
                norm_layer=norm_layer,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[
                    sum(dec_depths[:i]) : sum(dec_depths[: i + 1])
                ],
                unpool_backend=unpool_backend,
                norm_layer=norm_layer,
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)

    def forward(self, data_dict):
        """
        input: data_dict: {"coord": [n, 3], "feat": [n, c], "offset": [b]}
        output: seg_logits: [n, num_classes]
        """
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]] # 便于添加cluster
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling, 记录池化时的格网索引
            skips.append([points])  # record points info of current stage, 记录池化后的当前点云信息
        # 此时skips共五层，最后一层不带有cluster信息
        # 取出最后一层的点云信息
        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster) # 上采样
        coord, feat, offset = points
        return feat # [n, C]
