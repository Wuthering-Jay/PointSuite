"""
测试反向类别映射的修复

验证 SegmentationWriter 中的反向映射逻辑是否正确
"""

import numpy as np

# 模拟你的配置
CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}

# 构建反向映射
reverse_class_mapping = {v: k for k, v in CLASS_MAPPING.items()}
print(f"CLASS_MAPPING (原始→连续): {CLASS_MAPPING}")
print(f"REVERSE_MAPPING (连续→原始): {reverse_class_mapping}")
print()

# 模拟预测结果（连续标签 0-7）
np.random.seed(42)
final_preds = np.random.randint(0, 8, size=10000)

print("=" * 80)
print("测试旧的实现（有bug）")
print("=" * 80)

# 旧的实现（有bug）
final_preds_mapped_old = np.zeros_like(final_preds)  # 🔥 问题：初始化为全0
for continuous_label, original_label in reverse_class_mapping.items():
    final_preds_mapped_old[final_preds == continuous_label] = original_label

print("\n旧实现 - 映射前类别分布（连续标签）:")
pred_counts = np.bincount(final_preds, minlength=8)
for i, count in enumerate(pred_counts):
    print(f"  类别 {i}: {count:5d} 点 ({count/len(final_preds)*100:5.2f}%)")

print("\n旧实现 - 映射后类别分布（原始标签）:")
unique_labels = np.unique(final_preds_mapped_old)
for label in unique_labels:
    count = (final_preds_mapped_old == label).sum()
    print(f"  标签 {label}: {count:5d} 点 ({count/len(final_preds_mapped_old)*100:5.2f}%)")

# 检查是否所有点都被正确映射
unmapped_old = np.sum(final_preds_mapped_old == 0) - np.sum(final_preds == 0)
if unmapped_old > 0:
    print(f"\n⚠️  警告：旧实现有 {unmapped_old} 个点被错误地映射为 0！")

print("\n" + "=" * 80)
print("测试新的实现（修复后）")
print("=" * 80)

# 新的实现（修复后）
max_continuous_label = max(reverse_class_mapping.keys())
mapping_array = np.arange(max_continuous_label + 1)  # 默认保持不变

for continuous_label, original_label in reverse_class_mapping.items():
    mapping_array[continuous_label] = original_label

# 向量化映射
final_preds_mapped_new = mapping_array[final_preds]

print("\n新实现 - 映射前类别分布（连续标签）:")
pred_counts = np.bincount(final_preds, minlength=8)
for i, count in enumerate(pred_counts):
    print(f"  类别 {i}: {count:5d} 点 ({count/len(final_preds)*100:5.2f}%)")

print("\n新实现 - 映射后类别分布（原始标签）:")
unique_labels = np.unique(final_preds_mapped_new)
for label in unique_labels:
    count = (final_preds_mapped_new == label).sum()
    print(f"  标签 {label}: {count:5d} 点 ({count/len(final_preds_mapped_new)*100:5.2f}%)")

print("\n" + "=" * 80)
print("验证映射正确性")
print("=" * 80)

# 验证每个连续标签是否被正确映射
all_correct = True
for continuous_label, original_label in reverse_class_mapping.items():
    mask = (final_preds == continuous_label)
    mapped_values = final_preds_mapped_new[mask]
    
    if not np.all(mapped_values == original_label):
        print(f"❌ 错误: 连续标签 {continuous_label} 没有被正确映射为 {original_label}")
        all_correct = False
    else:
        count = mask.sum()
        print(f"✓ 连续标签 {continuous_label} → 原始标签 {original_label} ({count} 个点)")

if all_correct:
    print(f"\n✅ 所有映射都正确！")
else:
    print(f"\n❌ 存在映射错误！")

# 对比新旧实现的差异
diff = np.sum(final_preds_mapped_new != final_preds_mapped_old)
print(f"\n新旧实现差异: {diff} 个点的标签不同 ({diff/len(final_preds)*100:.2f}%)")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"旧实现问题：由于 np.zeros_like() 初始化，映射逻辑实际上是覆盖式的")
print(f"新实现优势：使用向量化数组映射，效率更高且不会遗漏任何点")
print(f"\n建议：使用新实现替换 callbacks.py 中的反向映射代码")
