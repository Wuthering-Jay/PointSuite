# PointDataModule Documentation

`PointDataModule` is a PyTorch Lightning compatible DataModule designed for efficient point cloud data loading and processing using the BinPklDataset format.

## Features

✅ **PyTorch Lightning Compatible**: Seamless integration with Lightning Trainer  
✅ **Flexible Data Loading**: Support for train/val/test splits  
✅ **Custom Transforms**: Easy integration of data augmentation pipelines  
✅ **Memory Management**: Built-in caching and point limit controls  
✅ **Class Mapping**: Automatic label remapping for non-continuous classes  
✅ **Multi-GPU Ready**: Works with DDP and other distributed strategies  
✅ **Configurable Workers**: Optimized data loading with persistent workers

## Quick Start

### Basic Usage

```python
from pointsuite.data.point_datamodule import PointDataModule
import pytorch_lightning as pl

# Create DataModule
datamodule = PointDataModule(
    data_root='path/to/data',
    train_files=['train.pkl'],
    val_files=['val.pkl'],
    batch_size=8,
    num_workers=4
)

# Setup and get dataloaders
datamodule.setup('fit')
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# Or use directly with Trainer
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, datamodule)
```

### With Data Augmentation

```python
from pointsuite.data.transforms import RandomRotate, RandomScale, Normalize

train_transforms = [
    RandomRotate(angle=[-1, 1], axis='z'),
    RandomScale(scale=[0.9, 1.1]),
    Normalize(),
]

datamodule = PointDataModule(
    data_root='path/to/data',
    batch_size=8,
    num_workers=4,
    train_transforms=train_transforms,
    loop=2  # Loop through training data twice per epoch
)
```

### With Class Mapping

```python
# Map discontinuous class labels to continuous indices
class_mapping = {
    0: 0,  # Unclassified
    1: 1,  # Ground
    2: 2,  # Vegetation
    6: 3,  # Building
    9: 4,  # Water
}

datamodule = PointDataModule(
    data_root='path/to/data',
    batch_size=8,
    num_workers=4,
    class_mapping=class_mapping
)
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_root` | str | **Required** | Root directory containing data files |
| `train_files` | List[str] | None | Training file names (auto-discover if None) |
| `val_files` | List[str] | None | Validation file names (auto-discover if None) |
| `test_files` | List[str] | None | Test file names (auto-discover if None) |
| `batch_size` | int | 8 | Number of samples per batch |
| `num_workers` | int | 4 | Number of data loading workers |

### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `assets` | List[str] | `['coord', 'intensity', 'classification']` | Data attributes to load |
| `train_transforms` | List | None | Transforms for training data |
| `val_transforms` | List | None | Transforms for validation data |
| `test_transforms` | List | None | Transforms for test data |
| `ignore_label` | int | -1 | Label to ignore in loss computation |
| `loop` | int | 1 | Number of times to loop through training data |

### Memory & Performance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_data` | bool | False | Cache loaded data in memory |
| `pin_memory` | bool | True | Use pinned memory for faster GPU transfer |
| `persistent_workers` | bool | False | Keep workers alive between epochs |
| `prefetch_factor` | int | 2 | Number of batches to prefetch per worker |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_mapping` | Dict[int, int] | None | Map original labels to continuous indices |
| `collate_fn_type` | str | 'default' | Collate function type ('default' or 'limited') |
| `max_points` | int | 500000 | Max points per batch (with 'limited' collate) |

## Available Assets

The `assets` parameter controls which point cloud attributes to load:

- **`coord`**: XYZ coordinates (always required)
- **`intensity`**: LiDAR intensity values
- **`color`**: RGB color information
- **`normal`**: Surface normal vectors
- **`classification`**: Semantic labels
- **`return_number`**: LiDAR return number
- **`number_of_returns`**: Total number of returns

## Collate Functions

### Default Collate Function

Concatenates all points from batch samples into a single point cloud with offset markers:

```python
datamodule = PointDataModule(
    data_root='path/to/data',
    collate_fn_type='default'
)
```

**Output batch structure:**
```python
{
    'coord': Tensor[N, 3],           # All points concatenated
    'feature': Tensor[N, C],         # All features concatenated
    'classification': Tensor[N],     # All labels concatenated
    'offset': Tensor[B],             # Cumulative point counts per sample
}
```

### Limited Points Collate Function

Dynamically adjusts batch size to stay within memory limits:

```python
datamodule = PointDataModule(
    data_root='path/to/data',
    collate_fn_type='limited',
    max_points=500000  # Maximum 500k points per batch
)
```

## Multi-GPU Training

PointDataModule works seamlessly with PyTorch Lightning's distributed training:

```python
datamodule = PointDataModule(
    data_root='path/to/data',
    batch_size=4,  # Per-GPU batch size
    num_workers=8,
    persistent_workers=True
)

trainer = pl.Trainer(
    max_epochs=50,
    accelerator='gpu',
    devices=4,              # Use 4 GPUs
    strategy='ddp',         # Distributed Data Parallel
    precision=16,           # Mixed precision
    sync_batchnorm=True
)

trainer.fit(model, datamodule)
```

## Methods

### Setup Methods

- **`setup(stage)`**: Initialize datasets for given stage ('fit', 'validate', 'test', or None)
- **`prepare_data()`**: Called once for data preparation (download, tokenize, etc.)

### DataLoader Methods

- **`train_dataloader()`**: Returns training DataLoader
- **`val_dataloader()`**: Returns validation DataLoader
- **`test_dataloader()`**: Returns test DataLoader
- **`predict_dataloader()`**: Returns prediction DataLoader

### Utility Methods

- **`get_dataset_info(split)`**: Get information about a dataset split
- **`print_info()`**: Print detailed information about all datasets
- **`teardown(stage)`**: Clean up resources after training/testing

## Complete Training Example

```python
import pytorch_lightning as pl
from pointsuite.data.point_datamodule import PointDataModule
from your_model import YourPointCloudModel

# 1. Create DataModule
datamodule = PointDataModule(
    data_root='data/processed',
    train_files=['area1_train.pkl', 'area2_train.pkl'],
    val_files=['area3_val.pkl'],
    test_files=['area4_test.pkl'],
    batch_size=8,
    num_workers=8,
    assets=['coord', 'intensity', 'color', 'classification'],
    loop=2,
    cache_data=False,
    pin_memory=True,
    persistent_workers=True,
    class_mapping={0: 0, 1: 1, 2: 2, 6: 3, 9: 4},
    collate_fn_type='limited',
    max_points=500000
)

# 2. Create model
model = YourPointCloudModel(
    in_channels=7,  # coord(3) + intensity(1) + color(3)
    num_classes=5,
    learning_rate=0.001
)

# 3. Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    precision=16,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min'
        ),
    ]
)

# 4. Train
trainer.fit(model, datamodule)

# 5. Test
trainer.test(model, datamodule)
```

## Performance Tuning

### For Small Datasets (< 10GB)

```python
datamodule = PointDataModule(
    data_root='path/to/data',
    batch_size=16,
    num_workers=4,
    cache_data=True,        # Cache all data in RAM
    persistent_workers=True,
    prefetch_factor=4
)
```

### For Large Datasets (> 50GB)

```python
datamodule = PointDataModule(
    data_root='path/to/data',
    batch_size=4,
    num_workers=8,
    cache_data=False,       # Don't cache (use memmap)
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn_type='limited',
    max_points=500000
)
```

### For Multi-GPU Training

```python
datamodule = PointDataModule(
    data_root='path/to/data',
    batch_size=2,           # Per-GPU batch size
    num_workers=8,          # Workers per GPU
    persistent_workers=True,
    pin_memory=True
)
```

## Troubleshooting

### Out of Memory Errors

1. Reduce `batch_size`
2. Use `collate_fn_type='limited'` with lower `max_points`
3. Set `cache_data=False`
4. Use mixed precision training (`precision=16`)

### Slow Data Loading

1. Increase `num_workers`
2. Set `persistent_workers=True`
3. Increase `prefetch_factor`
4. Enable `cache_data=True` if you have enough RAM
5. Use SSD storage instead of HDD

### DataLoader Hanging

1. Set `num_workers=0` to debug
2. Check if transforms are thread-safe
3. Ensure data files are accessible
4. Try `persistent_workers=False`

## Related Documentation

- [BinPklDataset Documentation](datasets/README.md)
- [Transform Documentation](transforms.md)
- [Collate Functions](datasets/collate.py)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)

## License

This module is part of the PointSuite project.
