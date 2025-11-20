"""
调试 GridPool 中的索引问题 - 简化版本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pointsuite.data.datasets.dataset_bin import BinPklDataset
from pointsuite.data.datasets.collate import collate_fn
from pointsuite.models.backbones.point_transformer_v2m5 import PointTransformerV2
from pointsuite.models.heads.seg_head import SegHead

def test_gridpool_simple():
    """使用简化的数据加载测试 GridPool"""
    print("\n" + "="*80)
    print("简化测试: 直接加载数据测试 GridPool")
    print("="*80)
    
    # 创建数据集
    dataset = BinPklDataset(
        data_root="E:\\data\\DALES\\dales_las\\bin\\train",
        assets=["coord", "echo", "class"],
        loop=1,
        cache_data=False,
    )
    
    # 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # 创建模型
    backbone = PointTransformerV2(
        in_channels=5,
        patch_embed_depth=2,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=16,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(1, 2.5, 7.5, 15),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        unpool_backend="interp",
    )
    
    head = SegHead(in_channels=48, num_classes=8)
    model = nn.Sequential(backbone, head).cuda()
    
    # 测试前几个 batch
    print(f"\n数据集大小: {len(dataset)}")
    print(f"测试前 3 个 batch:")
    
    for i, batch in enumerate(loader):
        if i >= 3:
            break
            
        print(f"\n{'='*80}")
        print(f"Batch {i+1}")
        print(f"{'='*80}")
        
        # 准备输入
        coord = batch['coord'].cuda()
        echo = batch['echo']
        if isinstance(echo, list):
            echo = torch.tensor(echo, dtype=torch.float32).unsqueeze(1).cuda()
        else:
            echo = echo.unsqueeze(1).cuda()
        
        h_norm = batch.get('h_norm')
        if h_norm is None:
            h_norm = torch.zeros(coord.shape[0], 1).cuda()
        elif isinstance(h_norm, list):
            h_norm = torch.tensor(h_norm, dtype=torch.float32).unsqueeze(1).cuda()
        else:
            h_norm = h_norm.unsqueeze(1).cuda()
            
        feat = torch.cat([coord, echo, h_norm], dim=1)
        offset = batch['offset'].cuda()
        
        print(f"输入:")
        print(f"  coord: {coord.shape}, range=[{coord.min().item():.2f}, {coord.max().item():.2f}]")
        print(f"  feat: {feat.shape}")
        print(f"  offset: {offset}")
        print(f"  总点数: {coord.shape[0]}")
        
        data_dict = {
            'coord': coord,
            'feat': feat,
            'offset': offset,
        }
        
        try:
            # FP32 测试
            print(f"\n[FP32 前向传播]")
            model.eval()
            with torch.no_grad():
                output = model(data_dict)
            print(f"  ✅ 成功: output shape = {output.shape}")
            
            # FP16 测试
            print(f"\n[FP16 前向传播]")
            data_dict_fp16 = {
                'coord': coord,
                'feat': feat.half(),
                'offset': offset,
            }
            
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    output_fp16 = model(data_dict_fp16)
            print(f"  ✅ 成功: output shape = {output_fp16.shape}, dtype = {output_fp16.dtype}")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            print(f"\n详细错误:")
            import traceback
            traceback.print_exc()
            print(f"\n在 Batch {i+1} 失败")
            return False
    
    print(f"\n" + "="*80)
    print(f"✅ 所有测试通过!")
    print(f"="*80)
    return True

if __name__ == "__main__":
    success = test_gridpool_simple()
    if not success:
        print("\n发现问题！")

