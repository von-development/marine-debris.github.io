import pytorch_lightning as pl
from ultralytics import YOLO
import torch
import torch.nn as nn
from typing import Dict, Any
from .config import YOLOConfig
from pathlib import Path
import os

class MarineDebrisYOLO(pl.LightningModule):
    """YOLOv8 adaptado para detecção de detritos marinhos"""
    
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Configurar caminhos
        project_root = Path(__file__).parent.parent.parent.parent
        os.environ['PROJECT_ROOT'] = str(project_root)
        
        # Inicializar YOLO
        self.model = YOLO(self.config.version)
        
        # Configurar dataset
        data_yaml_path = Path(__file__).parent / 'data.yaml'
        self.model.overrides['data'] = str(data_yaml_path)
        self.model.overrides['task'] = 'detect'
        self.model.overrides['imgsz'] = self.config.image_size
        
        # Adaptar para 4 canais
        self.model.model.model[0].conv = nn.Conv2d(
            self.config.input_channels,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
        # Configurar número de classes
        self.model.model.model[-1].nc = self.config.num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels, confidence = batch
        
        # Forward pass
        outputs = self(images)
        
        # Calcular loss com pesos de confiança
        loss = self.model.loss_fn(outputs, labels, confidence)
        
        # Logging
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        images, labels, confidence = batch
        outputs = self(images)
        
        # Calcular métricas
        val_loss = self.model.loss_fn(outputs, labels, confidence)
        self.log('val_loss', val_loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate
        ) 