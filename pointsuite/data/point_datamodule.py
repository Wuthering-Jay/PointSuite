"""
点云数据的 PyTorch Lightning DataModule

本模块通过将 BinPklDataModule 重新导出为 PointDataModule 来提供向后兼容性
新代码应直接使用 BinPklDataModule 或 DataModuleBase

已弃用：此模块保留用于向后兼容性
请改用 datamodule_binpkl.BinPklDataModule

示例（旧方式 - 仍然有效）:
    from pointsuite.data.point_datamodule import PointDataModule
    datamodule = PointDataModule(...)

示例（新方式 - 推荐）:
    from pointsuite.data.datamodule_binpkl import BinPklDataModule
    datamodule = BinPklDataModule(...)
    
示例（为自定义数据集使用基类）:
    from pointsuite.data.datamodule_base import DataModuleBase
    
    class MyDataModule(DataModuleBase):
        def _create_dataset(self, data_paths, split, transforms):
            return MyCustomDataset(data_paths, split=split, transform=transforms)
"""

# 导入新的实现
from .datamodule_binpkl import BinPklDataModule

# 为向后兼容性重新导出
PointDataModule = BinPklDataModule

# Convenience re-exports
__all__ = ['PointDataModule', 'BinPklDataModule']