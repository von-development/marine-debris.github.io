from typing import Dict, Any
import albumentations as A
import numpy as np

class YOLOAugmentation:
    """Augmentações específicas para treinamento do YOLO"""
    
    @staticmethod
    def get_transforms(train: bool = True) -> A.Compose:
        """
        Args:
            train: Se True, retorna transforms de treino
        """
        if train:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                
                A.Normalize(
                    mean=[1365.4, 1164.7, 939.3, 816.8],
                    std=[1087.4, 705.3, 574.5, 544.7],
                    max_pixel_value=10000.0
                ),
            ])
        else:
            return A.Compose([
                A.Normalize(
                    mean=[1365.4, 1164.7, 939.3, 816.8],
                    std=[1087.4, 705.3, 574.5, 544.7],
                    max_pixel_value=10000.0
                ),
            ]) 