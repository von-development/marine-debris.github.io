from pathlib import Path
import yaml
import os
from src.config.data_config import DataConfig

def verify_dataset_structure():
    """Verificar estrutura do dataset"""
    config = DataConfig()
    
    print("\nVerificando estrutura do dataset:")
    print(f"Data dir: {config.data_dir}")
    print(f"Patches dir: {config.patches_dir}")
    
    # Verificar diretórios
    if not config.patches_dir.exists():
        print("ERRO: Diretório patches não encontrado!")
        return
    
    # Verificar splits
    splits_dir = config.data_dir / 'splits'
    print(f"\nVerificando splits em: {splits_dir}")
    for split in ['train_X.txt', 'val_X.txt', 'test_X.txt']:
        split_path = splits_dir / split
        if split_path.exists():
            with open(split_path) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                print(f"- {split}: {len(lines)} imagens")
    
    # Verificar labels
    labels_path = config.data_dir / 'labels_mapping.txt'
    print(f"\nVerificando labels em: {labels_path}")
    if labels_path.exists():
        with open(labels_path) as f:
            labels = yaml.safe_load(f)
            print(f"Total de labels: {len(labels)}")
    
    # Listar diretórios
    print("\nDiretórios em patches:")
    if config.patches_dir.exists():
        for d in config.patches_dir.iterdir():
            if d.is_dir():
                files = list(d.glob("*.tif"))
                print(f"\n- {d.name} ({len(files)} arquivos)")
                # Mostrar alguns exemplos
                for f in files[:3]:
                    print(f"  - {f.name}")
                if len(files) > 3:
                    print("  ...")

def analyze_dataset_structure():
    """Analisar estrutura detalhada do dataset"""
    config = DataConfig()
    
    # Carregar splits
    splits = {}
    for split in ['train', 'val', 'test']:
        split_path = config.splits_dir / f'{split}_X.txt'
        if split_path.exists():
            with open(split_path) as f:
                splits[split] = [l.strip() for l in f.readlines() if l.strip()]
    
    # Carregar labels
    labels_path = config.data_dir / 'labels_mapping.txt'
    if labels_path.exists():
        with open(labels_path) as f:
            labels = yaml.safe_load(f)
    
    # Análise
    print("\nAnálise do Dataset:")
    print("\nDistribuição por split:")
    for split, images in splits.items():
        print(f"- {split}: {len(images)} imagens")
    
    print("\nVerificando consistência:")
    for split, images in splits.items():
        print(f"\nSplit: {split}")
        for img_id in images[:3]:  # Verificar primeiras 3 imagens
            print(f"\n- Imagem: {img_id}")
            
            # Verificar label
            label_key = f"S2_{img_id}.tif"
            if label_key in labels:
                label = labels[label_key]
                print(f"  Label encontrado: {sum(label)} classes positivas")
            else:
                print("  ERRO: Label não encontrado!")
            
            # Verificar arquivos
            parts = img_id.split('_')
            dir_name = f"S2_{parts[0]}_{parts[1]}"
            img_dir = config.patches_dir / dir_name
            
            base_name = f"S2_{img_id}"
            files = [
                img_dir / f"{base_name}.tif",
                img_dir / f"{base_name}_cl.tif",
                img_dir / f"{base_name}_conf.tif"
            ]
            
            for f in files:
                print(f"  {f.name}: {'✓' if f.exists() else '✗'}")

if __name__ == "__main__":
    print("=== Verificação Básica ===")
    verify_dataset_structure()
    print("\n=== Análise Detalhada ===")
    analyze_dataset_structure() 