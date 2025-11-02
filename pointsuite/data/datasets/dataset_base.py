import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from collections.abc import Sequence
from .transforms import Compose


class DatasetBase(Dataset, ABC):
    """
    Abstract base dataset class for point cloud data.
    
    This serves as the foundation for all dataset implementations.
    Subclasses should implement the abstract methods to handle specific data formats.
    """

    VALID_ASSETS = [
        "coord",  # XYZ coordinates (required)
        "color",  # RGB color
        "normal",  # Normal vectors
        "intensity",  # Intensity
        "return_number",  # Return number
        "number_of_returns",  # Number of returns
        "classification",  # Classification labels
    ]

    def __init__(
            self,
            data_root,
            split: str = 'train',
            assets: Optional[List[str]] = None,
            transform: Optional[List] = None,
            ignore_label: int = -1,
            loop: int = 1,
            cache_data: bool = False,
            class_mapping: Optional[Dict[int, int]] = None,
            **kwargs
    ):
        """
        Initialize base dataset.
        
        Args:
            data_root: Root directory, single file path, or list of file paths
            split: Dataset split ('train', 'val', 'test')
            assets: List of data attributes to load (None for default)
            transform: Data transforms to apply
            ignore_label: Label to ignore in training/evaluation
            loop: Number of times to loop through dataset (for training augmentation)
            cache_data: Whether to cache loaded data in memory
            class_mapping: Dict mapping original class labels to continuous labels.
                          Example: {0: 0, 1: 1, 2: 2, 6: 3, 9: 4}
                          If None, no mapping is applied.
            **kwargs: Additional arguments for subclasses
        """
        super().__init__()
        
        # Handle different data_root types
        if isinstance(data_root, (list, tuple)):
            # List of paths
            self.data_root = data_root
        else:
            # Single path
            self.data_root = Path(data_root)
        
        self.split = split
        self.assets = assets if assets is not None else self.VALID_ASSETS.copy()
        self.transform = Compose(transform) if transform is not None else None
        self.ignore_label = ignore_label
        self.loop = (loop if split == 'train' else 1)
        self.cache_data = cache_data
        self.class_mapping = class_mapping
        
        # Cache for data if enabled
        self.data_cache = {} if cache_data else None
        
        # Validate data root (skip validation for list type, subclass will handle)
        if not isinstance(self.data_root, (list, tuple)) and not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")
        
        # Load data list (implemented by subclass)
        self.data_list = self._load_data_list()
        
        # Print initialization info
        self._print_init_info()
    
    def _print_init_info(self):
        """Print dataset initialization information."""
        print(f"==> {self.__class__.__name__} ({self.split}) initialized:")
        print(f"    - Data root: {self.data_root}")
        print(f"    - Total samples: {len(self.data_list)}")
        print(f"    - Assets: {self.assets}")
        print(f"    - Loop: {self.loop}")
        print(f"    - Cache: {'Enabled' if self.cache_data else 'Disabled'}")
    
    @abstractmethod
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """
        Load list of all data samples.
        Must be implemented by subclass.
        
        Returns:
            List of dicts containing sample information
        """
        raise NotImplementedError("Subclass must implement _load_data_list()")
    
    @abstractmethod
    def _load_data(self, idx: int) -> Dict[str, Any]:
        """
        Load point cloud data for a given index.
        Must be implemented by subclass.
        
        Args:
            idx: Data index
            
        Returns:
            Dict containing loaded data (coord, labels, etc.)
        """
        raise NotImplementedError("Subclass must implement _load_data()")
    
    def __len__(self) -> int:
        """Return dataset length considering loop factor."""
        return len(self.data_list) * self.loop
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return a data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing point cloud data and labels
        """
        # Handle loop
        data_idx = idx % len(self.data_list)
        
        # Check cache
        if self.cache_data and data_idx in self.data_cache:
            data_dict = self.data_cache[data_idx].copy()
        else:
            # Load data (implemented by subclass)
            data_dict = self._load_data(data_idx)
            
            # Cache if enabled
            if self.cache_data:
                self.data_cache[data_idx] = data_dict.copy()
        
        # Apply transforms
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        
        return data_dict
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get sample information without loading data.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample info dict
        """
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]





