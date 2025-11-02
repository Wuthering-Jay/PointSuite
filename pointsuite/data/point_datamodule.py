"""
PyTorch Lightning DataModule for Point Cloud Data

This module provides backward compatibility by re-exporting BinPklDataModule
as PointDataModule. New code should use BinPklDataModule or DataModuleBase directly.

DEPRECATED: This module is kept for backward compatibility.
Please use datamodule_binpkl.BinPklDataModule instead.

Example (old way - still works):
    from pointsuite.data.point_datamodule import PointDataModule
    datamodule = PointDataModule(...)

Example (new way - recommended):
    from pointsuite.data.datamodule_binpkl import BinPklDataModule
    datamodule = BinPklDataModule(...)
    
Example (using base class for custom datasets):
    from pointsuite.data.datamodule_base import DataModuleBase
    
    class MyDataModule(DataModuleBase):
        def _create_dataset(self, data_paths, split, transforms):
            return MyCustomDataset(data_paths, split=split, transform=transforms)
"""

# Import the new implementation
from .datamodule_binpkl import BinPklDataModule

# Re-export for backward compatibility
PointDataModule = BinPklDataModule

# Convenience re-exports
__all__ = ['PointDataModule', 'BinPklDataModule']