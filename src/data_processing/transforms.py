import numpy as np
import albumentations as A
from typing import Dict, Any
import torch

class MaridaTransforms:
    """Transformations for MARIDA satellite imagery"""
    
    def __init__(self, is_training: bool = True):
        """Initialize transforms pipeline
        
        Args:
            is_training (bool): Whether to use training or validation transforms
        """
        self.is_training = is_training
        self.transforms = self._get_transforms()
    
    def _get_transforms(self) -> A.Compose:
        """Get transformation pipeline"""
        if self.is_training:
            return A.Compose([
                # Data augmentation for training
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                
                # Normalization (using Sentinel-2 standard values)
                A.Normalize(
                    mean=[1365.4, 1164.7, 939.3, 816.8],  # B02, B03, B04, B08
                    std=[1087.4, 705.3, 574.5, 544.7],
                    max_pixel_value=10000.0
                ),
            ], additional_targets={'confidence': 'mask'})
        else:
            return A.Compose([
                # Only normalization for validation/testing
                A.Normalize(
                    mean=[1365.4, 1164.7, 939.3, 816.8],
                    std=[1087.4, 705.3, 574.5, 544.7],
                    max_pixel_value=10000.0
                ),
            ], additional_targets={'confidence': 'mask'})
    
    def __call__(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Apply transforms to image and masks"""
        # Ensure image is numpy array and in correct format (HWC)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Move channels to last dimension for albumentations
        image = np.transpose(image, (1, 2, 0))
        
        # Prepare transform inputs
        transform_inputs = {'image': image}
        
        # Add confidence mask if present
        if 'confidence' in kwargs:
            confidence = kwargs['confidence']
            if isinstance(confidence, torch.Tensor):
                confidence = confidence.numpy()
            # Move channel to last dimension
            confidence = np.transpose(confidence, (1, 2, 0))
            transform_inputs['confidence'] = confidence
        
        # Apply transforms
        transformed = self.transforms(**transform_inputs)
        
        # Move channels back to first dimension
        transformed['image'] = np.transpose(transformed['image'], (2, 0, 1))
        
        # Prepare output
        output = {
            'image': transformed['image'],
            **kwargs  # Keep other inputs unchanged
        }
        
        # Update confidence mask if it was transformed
        if 'confidence' in transformed:
            output['confidence'] = np.transpose(transformed['confidence'], (2, 0, 1))
        
        return output 