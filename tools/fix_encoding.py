"""修复 transforms.py 文件的编码问题"""
import re

# 读取文件
with open(r'e:\code\python\PointSuite\pointsuite\data\transforms.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 使用正则表达式找到所有乱码注释行并替换
# 匹配包含乱码字符的注释行
def clean_garbled_comments(text):
    """清理所有乱码注释"""
    lines = text.split('\n')
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        # 检查是否是乱码注释（包含非 ASCII 的奇怪字符组合）
        if stripped.startswith('#') and any(ord(c) > 127 for c in stripped):
            # 保留缩进
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            
            # 尝试识别并替换常见模式
            if 'ââââ' in stripped or '————' in stripped:
                # 章节分隔符，保留但清理
                new_lines.append(f'{spaces}# ----------------------------------------------------')
            elif 'ð¥' in stripped or '🔥' in stripped:
                # 带火焰 emoji 的注释，清理
                new_lines.append(f'{spaces}# 注意: 关键修改')
            else:
                # 其他乱码注释，使用通用占位符或跳过
                # 保留空行作为分隔
                pass
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)

# 新的文件头
new_header = '''"""
3D 点云数据增强变换模块

本模块提供丰富的点云数据增强变换操作，包括:
- 通用操作: Compose, Collect, ToTensor, Update
- 坐标变换: NormalizeCoord, CenterShift, CentroidShift, RandomShift
- 随机增强: RandomRotate, RandomFlip, RandomScale, RandomJitter, RandomDropout
- 采样操作: GridSample, FarthestPointSample, RandomSample

使用示例:
    >>> from pointsuite.data.transforms import Compose, RandomRotate, ToTensor
    >>> transforms = Compose([
    ...     RandomRotate(angle=[-1, 1], axis='z'),
    ...     ToTensor(),
    ... ])
    >>> data = transforms(data_dict)

继承开发指南:
    所有变换类需要实现 __call__(self, data_dict) 方法:
    - 输入: data_dict (Dict) - 包含点云数据的字典
    - 输出: data_dict (Dict) - 变换后的数据字典
    - 返回 None 表示丢弃该样本
"""

'''

# 找到 import random 的位置
import_pos = content.find('import random')
if import_pos > 0:
    # 保留从 import random 开始的内容
    rest_content = content[import_pos:]
    
    # 清理乱码注释
    rest_content = clean_garbled_comments(rest_content)
    
    # 写入新文件
    new_content = new_header + rest_content
    with open(r'e:\code\python\PointSuite\pointsuite\data\transforms.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print('File fixed successfully!')
else:
    print('Could not find import statement')
