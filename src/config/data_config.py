from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from . import DATA_DIR

@dataclass
class DataConfig:
    """Configuration for MARIDA dataset"""
    
    # Paths
    data_dir: Path = DATA_DIR
    patches_dir: Path = DATA_DIR / "patches"
    splits_dir: Path = DATA_DIR / "splits"
    
    # Dataset parameters
    image_size: int = 256
    num_classes: int = 15
    
    # Sentinel-2 bands configuration
    bands: List[str] = None
    
    def __post_init__(self):
        if self.bands is None:
            # Default Sentinel-2 bands (Blue, Green, Red, NIR)
            self.bands = ['B02', 'B03', 'B04', 'B08']
            
        # Verify paths exist
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.patches_dir.exists():
            raise FileNotFoundError(f"Patches directory not found: {self.patches_dir}")
        if not self.splits_dir.exists():
            raise FileNotFoundError(f"Splits directory not found: {self.splits_dir}")