import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional
from ..config.data_config import DataConfig

class MaridaDataset(Dataset):
    """MARIDA (MARIne Debris Archive) Dataset"""
    
    def __init__(
        self, 
        config: DataConfig, 
        split: str = 'train', 
        transform = None,
        load_confidence: bool = True
    ):
        self.config = config
        self.split = split
        self.transform = transform
        self.load_confidence = load_confidence
        
        # Load dataset files and labels
        self.image_files = self._load_split_files()
        self.labels = self._load_labels()
    
    def _load_split_files(self) -> List[str]:
        """Load files for specific split"""
        split_file = self.config.splits_dir / f'{self.split}_X.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    
    def _load_labels(self) -> Dict[str, List[int]]:
        """Load label mapping"""
        label_file = self.config.data_dir / 'labels_mapping.txt'
        if not label_file.exists():
            raise FileNotFoundError(f"Labels file not found: {label_file}")
            
        with open(label_file, 'r') as f:
            return json.load(f)
    
    def _get_image_path(self, image_id: str) -> Tuple[Path, Path, Path]:
        """Get paths for image, label and confidence files"""
        # Construir o diretório da imagem
        # Exemplo: image_id = "1-12-19_48MYU_0"
        parts = image_id.split('_')  # ['1-12-19', '48MYU', '0']
        dir_name = f"S2_{parts[0]}_{parts[1]}"  # "S2_1-12-19_48MYU"
        img_dir = self.config.patches_dir / dir_name
        
        # Construir caminhos completos
        base_name = f"S2_{image_id}"  # "S2_1-12-19_48MYU_0"
        image_path = img_dir / f"{base_name}.tif"
        label_path = img_dir / f"{base_name}_cl.tif"
        conf_path = img_dir / f"{base_name}_conf.tif"
        
        return image_path, label_path, conf_path
    
    def _load_image(self, image_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load image, label and confidence mask"""
        image_path, label_path, conf_path = self._get_image_path(image_id)
        
        # Load image
        with rasterio.open(image_path) as src:
            image = np.stack([
                src.read(i) for i in [1, 2, 3, 7]  # B02, B03, B04, B08
            ]).astype(np.float32)
        
        # Load label
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.float32)
        
        # Load confidence
        with rasterio.open(conf_path) as src:
            confidence = src.read(1).astype(np.float32)
        
        return image, label, confidence
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get image ID
        image_id = self.image_files[idx]
        
        # Load data
        image, label, confidence = self._load_image(image_id)
        
        # Get labels from mapping
        labels = torch.tensor(self.labels[f"S2_{image_id}.tif"], dtype=torch.float32)
        
        # Prepare output
        output = {
            'image': torch.from_numpy(image),
            'labels': labels,
            'confidence': torch.from_numpy(confidence),
            'filename': image_id
        }
        
        # Apply transforms
        if self.transform:
            output = self.transform(output)
        
        return output
    
    def _debug_info(self):
        """Print debug information"""
        print(f"\nDataset Debug Info:")
        print(f"Split: {self.split}")
        print(f"Number of images: {len(self.image_files)}")
        print(f"Data dir: {self.config.data_dir}")
        print(f"Patches dir: {self.config.patches_dir}")
        
        # Listar alguns diretórios em patches
        print("\nDiretórios em patches:")
        if self.config.patches_dir.exists():
            dirs = list(self.config.patches_dir.iterdir())[:5]
            for d in dirs:
                print(f"- {d.name}")
        
        print(f"\nFirst few image IDs:")
        for img_id in self.image_files[:3]:
            print(f"\n- {img_id}")
            image_path, label_path, conf_path = self._get_image_path(img_id)
            print(f"  Image path: {image_path}")
            print(f"  Image exists: {image_path.exists()}")
            print(f"  Label exists: {label_path.exists()}")
            print(f"  Conf exists: {conf_path.exists()}")
            
            # Se o diretório pai existir, listar arquivos
            if image_path.parent.exists():
                print(f"  Files in directory:")
                for f in image_path.parent.glob("*.tif"):
                    print(f"    - {f.name}")