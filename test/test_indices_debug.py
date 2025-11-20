import torch

# 模拟 FP16 环境下的计算
print("测试 indices 的 dtype")

num_classes = 8

# 测试1: 正常情况
print("\n测试1: 正常计算")
target = torch.tensor([0, 1, 2, 3]).cuda().long()
pred = torch.tensor([1, 2, 3, 0]).cuda().long()
print(f"  target dtype: {target.dtype}")
print(f"  pred dtype: {pred.dtype}")
indices = num_classes * target + pred
print(f"  indices before .long(): {indices}, dtype: {indices.dtype}")
indices_long = indices.long()
print(f"  indices after .long(): {indices_long}, dtype: {indices_long.dtype}")
cm = torch.bincount(indices_long, minlength=64)
print(f"  bincount success!")

# 测试2: 在autocast中
print("\n测试2: 在autocast中")
with torch.cuda.amp.autocast():
    target2 = torch.tensor([0, 1, 2, 3]).cuda().long()
    pred2 = torch.tensor([1, 2, 3, 0]).cuda().long()
    print(f"  target dtype: {target2.dtype}")
    print(f"  pred dtype: {pred2.dtype}")
    indices2 = num_classes * target2 + pred2
    print(f"  indices before .long(): {indices2}, dtype: {indices2.dtype}")
    indices2_long = indices2.long()
    print(f"  indices after .long(): {indices2_long}, dtype: {indices2_long.dtype}")
    cm2 = torch.bincount(indices2_long, minlength=64)
    print(f"  bincount success!")

# 测试3: 模拟_convert_preds_to_labels返回的long tensor
print("\n测试3: 模拟metrics场景")
with torch.cuda.amp.autocast():
    # 模拟logits
    logits = torch.randn(100, 8).cuda()
    labels = torch.randint(0, 8, (100,)).cuda()
    
    # 模拟_convert_preds_to_labels
    pred_labels = torch.argmax(logits, dim=1).long()
    print(f"  pred_labels dtype: {pred_labels.dtype}")
    
    # 过滤
    mask = labels != -1
    pred_labels = pred_labels[mask]
    target = labels[mask]
    
    # 再次确保long
    target = target.long()
    pred_labels = pred_labels.long()
    
    print(f"  After filtering - target dtype: {target.dtype}, pred dtype: {pred_labels.dtype}")
    
    indices3 = (num_classes * target + pred_labels).long()
    print(f"  indices dtype: {indices3.dtype}")
    cm3 = torch.bincount(indices3, minlength=64)
    print(f"  bincount success!")

print("\n所有测试通过！")
