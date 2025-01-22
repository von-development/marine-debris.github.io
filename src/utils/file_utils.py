from pathlib import Path
from typing import Tuple

def get_file_paths(base_dir: Path, image_id: str) -> Tuple[Path, Path, Path]:
    """Get paths for image, label and confidence files
    
    Args:
        base_dir: Base directory containing the data
        image_id: Image identifier (e.g., '1-12-19_48MYU_0')
    
    Returns:
        Tuple containing paths for (image, label, confidence) files
    """
    # Construir o diretório da imagem
    parts = image_id.split('_')
    dir_name = f"S2_{parts[0]}_{parts[1]}_{parts[2]}"
    img_dir = base_dir / 'patches' / dir_name
    
    # Construir caminhos completos
    image_path = img_dir / f"S2_{image_id}.tif"
    label_path = img_dir / f"S2_{image_id}_cl.tif"
    conf_path = img_dir / f"S2_{image_id}_conf.tif"
    
    return image_path, label_path, conf_path 