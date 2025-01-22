import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class WeightedYOLOLoss(nn.Module):
    """Loss function ponderada pela confiança das anotações"""
    
    def __init__(self, num_classes: int, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.pos_weight = pos_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: Predições do modelo
            targets: Ground truth
            confidence: Pesos de confiança para cada amostra
        """
        # BCE Loss com pesos
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Aplicar pesos de confiança
        weighted_loss = bce_loss * confidence.unsqueeze(-1)
        
        return weighted_loss.mean() 