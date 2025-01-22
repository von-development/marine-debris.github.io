from pathlib import Path
import torch
from src.config.data_config import DataConfig
from src.data_processing.dataset import MaridaDataset
import matplotlib.pyplot as plt
import numpy as np

def visualize_sample(image, labels, confidence=None, title=None):
    """Visualizar uma amostra do dataset"""
    # Normalizar imagem para visualização
    img = image.numpy()  # Converter tensor para numpy
    img = img.transpose(1, 2, 0)  # Mudar de (C,H,W) para (H,W,C)
    
    # Criar composição RGB usando 3 bandas
    rgb = np.stack([
        img[:,:,0],  # B02 (Blue)
        img[:,:,1],  # B03 (Green)
        img[:,:,2],  # B04 (Red)
    ], axis=-1)
    
    # Normalizar para visualização
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    # Plotar
    plt.figure(figsize=(15, 5))
    
    # Imagem
    plt.subplot(131)
    plt.imshow(rgb)
    plt.title('RGB Image')
    plt.axis('off')
    
    # Labels
    plt.subplot(132)
    plt.bar(range(len(labels)), labels.numpy())
    plt.title('Labels')
    plt.xticks(range(len(labels)), [
        'marine_debris', 'dense_plastic', 'sparse_plastic',
        'dense_sargassum', 'sparse_sargassum', 'natural_organic',
        'ship', 'cloud', 'cloud_shadow', 'water', 'water_turbid',
        'water_sediment', 'land', 'floating_algae', 'other'
    ], rotation=45, ha='right')
    
    # Confidence (se disponível)
    if confidence is not None:
        plt.subplot(133)
        plt.imshow(confidence.squeeze(), cmap='viridis')
        plt.title('Confidence Map')
        plt.colorbar()
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def test_dataset():
    """Testar e visualizar o dataset"""
    config = DataConfig()
    
    # Criar dataset
    train_dataset = MaridaDataset(config, split='train')
    
    # Debug info
    train_dataset._debug_info()
    
    print(f"\nTestando carregamento de amostras:")
    for i in range(3):
        try:
            sample = train_dataset[i]
            print(f"\nAmostra {i} carregada com sucesso:")
            print(f"Imagem shape: {sample['image'].shape}")
            print(f"Labels: {sample['labels'].numpy()}")
            
            # Visualizar
            visualize_sample(
                sample['image'],
                sample['labels'],
                sample['confidence'],
                f"Sample {i}: {sample['filename']}"
            )
        except Exception as e:
            print(f"Erro ao carregar amostra {i}:")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem: {str(e)}")
            raise

if __name__ == "__main__":
    test_dataset() 