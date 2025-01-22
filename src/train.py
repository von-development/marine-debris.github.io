import os
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import argparse

from src.models.yolo.config import YOLOConfig
from src.models.yolo.model import MarineDebrisYOLO
from src.data_processing.datamodule import MaridaDataModule
from src.training.metrics import calculate_metrics
from src.config.data_config import DataConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Treinar modelo YOLO para detecção de detritos marinhos')
    
    # Argumentos básicos
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    parser.add_argument('--max_epochs', type=int, default=100, help='Número máximo de épocas')
    parser.add_argument('--limit_train_batches', type=int, help='Limitar batches de treino (debug)')
    
    # Configurações do modelo
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--model_version', type=str, default='yolov8m')
    
    # Caminhos
    parser.add_argument('--data_dir', type=Path, default=Path('data'))
    parser.add_argument('--output_dir', type=Path, default=Path('outputs'))
    
    return parser.parse_args()

def setup_wandb_logger(config: YOLOConfig, debug: bool = False):
    """Configurar logger do Weights & Biases"""
    if debug:
        return None
    
    return WandbLogger(
        project='marine-debris',
        name=f'yolov8-{config.version}',
        save_dir=config.log_dir,
        config=vars(config)
    )

def setup_callbacks(config: YOLOConfig):
    """Configurar callbacks para treinamento"""
    callbacks = [
        # Salvar melhores modelos
        ModelCheckpoint(
            dirpath=config.save_dir,
            filename='{epoch}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        ),
        
        # Monitorar learning rate
        LearningRateMonitor(logging_interval='step')
    ]
    
    return callbacks

def setup_loggers(config: YOLOConfig, debug: bool = False):
    """Configurar loggers para treinamento"""
    if debug:
        return None
    
    loggers = []
    
    # WandB Logger
    wandb_logger = WandbLogger(
        project='marine-debris',
        name=f'yolov8-{config.version}',
        save_dir=config.log_dir,
        config=vars(config)
    )
    loggers.append(wandb_logger)
    
    # TensorBoard Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name='tensorboard'
    )
    loggers.append(tensorboard_logger)
    
    return loggers

def train(args):
    """Função principal de treinamento"""
    
    # Configurações
    yolo_config = YOLOConfig(
        version=args.model_version,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    data_config = DataConfig(
        data_dir=args.data_dir,
        image_size=yolo_config.image_size
    )
    
    # Dataset
    datamodule = MaridaDataModule(
        data_config=data_config,
        batch_size=yolo_config.batch_size,
        num_workers=4,
        persistent_workers=True
    )
    
    # Modelo
    model = MarineDebrisYOLO(yolo_config)
    
    # Loggers
    loggers = setup_loggers(yolo_config, args.debug)
    
    # Callbacks
    callbacks = setup_callbacks(yolo_config)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        precision='16-mixed',
        limit_train_batches=args.limit_train_batches if args.debug else None,
        limit_val_batches=10 if args.debug else None,
        log_every_n_steps=1 if args.debug else 50,
        detect_anomaly=args.debug,
        strategy='auto'
    )
    
    print("\nConfiguração do Treinamento:")
    print("===========================")
    print(f"Modo: {'Debug' if args.debug else 'Treinamento completo'}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Batch Size: {yolo_config.batch_size}")
    print(f"Learning Rate: {yolo_config.learning_rate}")
    print(f"Modelo: {yolo_config.version}")
    print(f"Épocas: {args.max_epochs}")
    if args.debug:
        print(f"Batches por época: {args.limit_train_batches}")
    print("\nIniciando treinamento...\n")
    
    # Treinar
    trainer.fit(model, datamodule)
    
    # Testar
    if not args.debug:
        trainer.test(model, datamodule)

if __name__ == "__main__":
    # Configurar reprodutibilidade
    pl.seed_everything(42)
    
    # Parsear argumentos
    args = parse_args()
    
    # Criar diretórios necessários
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Iniciar treinamento
    train(args) 