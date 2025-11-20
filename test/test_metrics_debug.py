import torch

print("测试 metrics 过滤问题")

# 测试1: 不在autocast中
target = torch.tensor([0, 1, 2, -1, 3, -1, 4]).cuda()
print(f"\n测试1: 不在autocast中")
print(f"  target dtype: {target.dtype}")
target_long = target.long()
print(f"  target_long dtype: {target_long.dtype}")
mask = (target_long != -1)
print(f"  mask: {mask}")
filtered = target_long[mask]
print(f"  filtered: {filtered}")
print(f"  has -1? {(filtered == -1).any()}")

# 测试2: 在autocast中
print(f"\n测试2: 在autocast中")
with torch.cuda.amp.autocast():
    target2 = torch.tensor([0, 1, 2, -1, 3, -1, 4]).cuda()
    print(f"  target2 dtype: {target2.dtype}")
    target2_long = target2.long()
    print(f"  target2_long dtype: {target2_long.dtype}")
    mask2 = (target2_long != -1)
    print(f"  mask2: {mask2}")
    filtered2 = target2_long[mask2]
    print(f"  filtered2: {filtered2}")
    print(f"  has -1? {(filtered2 == -1).any()}")

# 测试3: target传入autocast之前就是long
print(f"\n测试3: target在autocast外创建，然后在autocast内使用")
target3 = torch.tensor([0, 1, 2, -1, 3, -1, 4], dtype=torch.long).cuda()
print(f"  target3 dtype before autocast: {target3.dtype}")
with torch.cuda.amp.autocast():
    print(f"  target3 dtype inside autocast: {target3.dtype}")
    target3_long = target3.long()
    print(f"  target3_long dtype: {target3_long.dtype}")
    mask3 = (target3_long != -1)
    print(f"  mask3: {mask3}")
    filtered3 = target3_long[mask3]
    print(f"  filtered3: {filtered3}")
    print(f"  has -1? {(filtered3 == -1).any()}")
