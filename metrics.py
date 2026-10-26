"""
Evaluation Metrics for TSDiff Trajectory Forecasting & Refinement
Student Roll: 2303A52440 | Course: AR/VR | Project: TSDiff

Computes MAE, MSE, RMSE, and Continuous Ranked Probability Score (CRPS).
"""

import numpy as np
import torch

def compute_point_metrics(pred, target):
    """
    Computes deterministic point forecasting errors.
    pred: [B, C, L] or [C, L] numpy array or torch Tensor
    target: [B, C, L] or [C, L] numpy array or torch Tensor
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    diff = pred - target
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse)}


def compute_crps(samples, target):
    """
    Computes Continuous Ranked Probability Score (CRPS) across probabilistic samples:
    CRPS(F, y) = E_F |X - y| - 0.5 * E_F |X - X'|
    samples: [Num_Samples, C, Pred_Len]
    target: [C, Pred_Len]
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    N, C, L = samples.shape
    # Term 1: E|X - y|
    term1 = np.mean(np.abs(samples - target[None, :, :]), axis=0) # [C, L]
    
    # Term 2: 0.5 * E|X - X'|
    diff_pairs = np.abs(samples[:, None, :, :] - samples[None, :, :, :]) # [N, N, C, L]
    term2 = 0.5 * np.mean(diff_pairs, axis=(0, 1)) # [C, L]
    
    crps_per_step = term1 - term2
    return float(np.mean(crps_per_step))
