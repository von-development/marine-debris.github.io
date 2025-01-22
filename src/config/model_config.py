from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuração do modelo YOLOv8 para detecção de detritos marinhos"""
    
    # Parâmetros do modelo
    num_classes: int = 15
    input_channels: int = 4  # B02, B03, B04, B08
    image_size: int = 256
    
    # Configurações de treinamento
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    
    # Pesos e checkpoints
    weights_path: Optional[Path] = None
    save_dir: Path = Path("checkpoints")
    
    # Configurações específicas YOLOv8
    backbone: str = "yolov8m"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45 