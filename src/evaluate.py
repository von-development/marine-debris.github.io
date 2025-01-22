import torch
from pathlib import Path
from src.models.yolo.config import YOLOConfig
from src.models.yolo.model import MarineDebrisYOLO
from src.data_processing.datamodule import MaridaDataModule
from src.training.metrics import calculate_metrics  # Importando função de métricas
from src.config.data_config import DataConfig  # Adicionar este import

def evaluate(checkpoint_path: str):
    """Avaliar modelo treinado
    
    Args:
        checkpoint_path: Caminho para o checkpoint do modelo
    """
    # Configurações
    yolo_config = YOLOConfig()
    data_config = DataConfig()
    
    # Carregar modelo
    model = MarineDebrisYOLO.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Dataset
    datamodule = MaridaDataModule(data_config=data_config)
    datamodule.setup('test')
    
    # Métricas
    results = []
    
    # Avaliar
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            images, labels = batch
            predictions = model(images)
            
            # Calcular métricas
            batch_results = {
                'predictions': predictions,
                'targets': labels
            }
            results.append(batch_results)
    
    # Calcular métricas finais
    metrics = calculate_metrics(results)
    
    # Imprimir resultados
    print("\nResultados da Avaliação:")
    print("========================")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='Caminho para o checkpoint do modelo')
    args = parser.parse_args()
    
    evaluate(args.checkpoint_path) 