"""测试 collate_fn 名称显示"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointsuite.data.datamodule_bin import BinPklDataModule

data_root = r"E:\data\Dales\dales_las\bin\train"

print("创建 DataModule...")
datamodule = BinPklDataModule(
    data_root=data_root,
    batch_size=4,
    num_workers=0,
    assets=['coord', 'class']
)

print("\nSetup DataModule...")
datamodule.setup()

print("\n打印 DataModule 信息...")
datamodule.print_info()
