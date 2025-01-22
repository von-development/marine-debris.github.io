import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from .dataset import MaridaDataset
from ..config.data_config import DataConfig

class MaridaDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for MARIDA dataset"""
    
    def __init__(
        self,
        data_config: DataConfig,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False
    ):
        """Initialize data module
        
        Args:
            data_config: Dataset configuration
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU transfer
            persistent_workers: Whether to maintain worker processes between batches
        """
        super().__init__()
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        self.train_dataset: Optional[MaridaDataset] = None
        self.val_dataset: Optional[MaridaDataset] = None
        self.test_dataset: Optional[MaridaDataset] = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation and testing"""
        if stage == 'fit' or stage is None:
            self.train_dataset = MaridaDataset(
                config=self.data_config,
                split='train'
            )
            
            self.val_dataset = MaridaDataset(
                config=self.data_config,
                split='val'
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = MaridaDataset(
                config=self.data_config,
                split='test'
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        ) 