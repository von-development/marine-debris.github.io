import torch
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import average_precision_score, precision_recall_curve

def calculate_metrics(results: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Calcular métricas de avaliação para detecção de detritos marinhos
    
    Args:
        results: Lista de dicionários contendo predições e alvos
        
    Returns:
        Dict com métricas calculadas
    """
    all_predictions = []
    all_targets = []
    
    # Concatenar resultados
    for batch in results:
        all_predictions.append(batch['predictions'].cpu().numpy())
        all_targets.append(batch['targets'].cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calcular métricas
    metrics = {}
    
    # mAP (mean Average Precision)
    ap_per_class = []
    for i in range(targets.shape[1]):
        ap = average_precision_score(targets[:, i], predictions[:, i])
        ap_per_class.append(ap)
        metrics[f'AP_class_{i}'] = float(ap)
    
    metrics['mAP'] = float(np.mean(ap_per_class))
    
    # Outras métricas
    metrics['accuracy'] = float(
        np.mean((predictions > 0.5) == targets)
    )
    
    # F1-Score
    precision = np.mean(
        [precision_recall_curve(targets[:, i], predictions[:, i])[0]
         for i in range(targets.shape[1])]
    )
    recall = np.mean(
        [precision_recall_curve(targets[:, i], predictions[:, i])[1]
         for i in range(targets.shape[1])]
    )
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    metrics['f1_score'] = float(f1)
    
    return metrics