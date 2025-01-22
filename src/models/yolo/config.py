from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class YOLOConfig:
    """Configurações específicas para o modelo YOLOv8"""
    
    # Modelo base
    version: str = "yolov8m"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    pretrained: bool = True
    
    # Parâmetros da arquitetura
    input_channels: int = 4    # B02, B03, B04, B08
    num_classes: int = 15
    image_size: int = 256
    
    # Parâmetros de detecção
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Treinamento
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 3
    weight_decay: float = 0.0005
    
    # Multi-label específico
    label_smoothing: float = 0.1
    pos_weight: Optional[List[float]] = None  # Pesos por classe
    
    # Caminhos e logs
    weights_dir: Path = Path("weights")
    save_dir: Path = Path("runs/train")
    log_dir: Path = Path("logs")
    
    def __post_init__(self):
        """Validar e criar diretórios"""
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    