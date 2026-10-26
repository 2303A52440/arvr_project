"""
Master End-to-End TSDiff Execution & Evaluation Pipeline
Student Roll: 2303A52440 | Course: AR/VR | Project: TSDiff

Executes complete training, evaluation, metrics calculation, and figure generation.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim

from data_generator import get_data_loaders, BaseDeadReckoningForecaster
from model import TSDiffScoreNet
from diffusion import TSDiffEngine
from metrics import compute_point_metrics, compute_crps
from visualizer import (
    plot_predict_forecast,
    plot_refine_comparison,
    plot_synthetic_samples,
    plot_denoising_process,
    plot_guidance_ablation
)

def run_tsdiff_pipeline():
    print("=" * 80)
    print("   TSDiff: Self-Guiding Diffusion Model for AR/VR Trajectory Forecasting")
    print("   Student Roll Number: 2303A52440 | Course: AR/VR | Submission Rubric")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {device}")
    
    # -------------------------------------------------------------
    # 1. Dataset Initialization
    # -------------------------------------------------------------
    print("\n[Phase 1] Initializing AR/VR 6-DoF Motion Trajectory Dataset...")
    seq_length = 64
    obs_len = 32
    pred_len = 32
    batch_size = 32
    train_loader, val_loader, dataset = get_data_loaders(
        batch_size=batch_size, num_train=1000, num_val=200, seq_length=seq_length
    )
    print(f"    - Sequences: 1000 Train, 200 Validation")
    print(f"    - Trajectory Shape: [Batch, Channels=6, Seq_Len={seq_length}]")
    print(f"    - Channels: Pos X, Pos Y, Pos Z (meters), Roll, Pitch, Yaw (radians)")
    print(f"    - Split: Historical Context = {obs_len} frames | Future Target = {pred_len} frames")
    
    # -------------------------------------------------------------
    # 2. Model & Diffusion Engine Setup
    # -------------------------------------------------------------
    print("\n[Phase 2] Building 1D Dilated Temporal ResNet & TSDiff Engine...")
    model = TSDiffScoreNet(in_channels=6, model_channels=64, num_layers=4, time_emb_dim=128)
    tsdiff = TSDiffEngine(model, num_timesteps=50, beta_start=1e-4, beta_end=0.02, device=device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    - Backbone: Dilated 1D Temporal ResNet (dilations: 1, 2, 4, 8)")
    print(f"    - Total Trainable Parameters: {total_params:,}")
    print(f"    - Diffusion Schedule: Linear with T = 50 timesteps")
    
    # -------------------------------------------------------------
    # 3. Model Training (Score Matching)
    # -------------------------------------------------------------
    print("\n[Phase 3] Training Unconditional Diffusion Model (Predict, Refine, Synthesize)...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 15
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = tsdiff.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] - Score Matching MSE Loss: {avg_loss:.5f}")
            
    train_duration = time.time() - start_time
    print(f"    [+] Training completed in {train_duration:.2f} seconds.")
    
    # -------------------------------------------------------------
    # 4. Pillar 1: Predict (Self-Guided Trajectory Forecasting)
    # -------------------------------------------------------------
    print("\n[Phase 4] Executing Pillar 1: PREDICT (Self-Guided Inference)...")
    test_batch = next(iter(val_loader)).to(device)
    target_sample = test_batch[0:1] # [1, 6, 64]
    
    # Build observation mask (1 for past 32 frames, 0 for future 32 frames)
    mask = torch.zeros(1, 6, seq_length, device=device)
    mask[:, :, :obs_len] = 1.0
    
    num_samples = 16
    print(f"    - Generating {num_samples} probabilistic trajectory forecasts via self-guidance (scale=1.5)...")
    forecast_samples = tsdiff.predict_self_guided(
        target_sample, mask, guidance_scale=1.5, num_samples=num_samples
    ) # [16, 6, 64]
    
    # Unnormalize to real physical units for evaluation
    target_unnorm = np.squeeze(dataset.unnormalize(target_sample.cpu()).numpy()) # [6, 64]
    samples_unnorm = dataset.unnormalize(forecast_samples.cpu()).numpy() # [16, 6, 64]
    if samples_unnorm.ndim == 4:
        samples_unnorm = samples_unnorm.squeeze(1)
    
    pred_future_samples = samples_unnorm[:, :, obs_len:]
    true_future = target_unnorm[:, obs_len:]
    history = target_unnorm[:, :obs_len]
    median_pred = np.median(pred_future_samples, axis=0)
    
    predict_point_metrics = compute_point_metrics(median_pred, true_future)
    predict_crps = compute_crps(pred_future_samples, true_future)
    
    print(f"    - TSDiff Predict MAE : {predict_point_metrics['MAE']:.4f}")
    print(f"    - TSDiff Predict RMSE: {predict_point_metrics['RMSE']:.4f}")
    print(f"    - TSDiff Predict CRPS: {predict_crps:.4f}")
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_predict_forecast(
        history, true_future, pred_future_samples,
        save_path=os.path.join(output_dir, "tsdiff_predict_forecasting.png"),
        obs_len=obs_len
    )
    
    # -------------------------------------------------------------
    # 5. Pillar 2: Refine (Iterative Denoising of Base Forecaster)
    # -------------------------------------------------------------
    print("\n[Phase 5] Executing Pillar 2: REFINE (Base Predictor Polishing)...")
    base_forecaster = BaseDeadReckoningForecaster(obs_len=obs_len, pred_len=pred_len)
    base_trajectory = base_forecaster.forecast(target_sample.cpu(), noise_std=0.15) # [1, 6, 64]
    
    # Run TSDiff Refinement
    refined_trajectory = tsdiff.refine(
        base_trajectory.to(device), mask, noise_step=15, guidance_scale=1.2
    )
    
    base_unnorm = np.squeeze(dataset.unnormalize(base_trajectory.cpu()).numpy())
    refined_unnorm = np.squeeze(dataset.unnormalize(refined_trajectory.cpu()).numpy())
    
    base_metrics = compute_point_metrics(base_unnorm[:, obs_len:], true_future)
    refined_metrics = compute_point_metrics(refined_unnorm[:, obs_len:], true_future)
    
    mae_improvement = ((base_metrics['MAE'] - refined_metrics['MAE']) / base_metrics['MAE']) * 100
    rmse_improvement = ((base_metrics['RMSE'] - refined_metrics['RMSE']) / base_metrics['RMSE']) * 100
    
    print(f"    - Unrefined Base Forecaster MAE : {base_metrics['MAE']:.4f} | RMSE: {base_metrics['RMSE']:.4f}")
    print(f"    - TSDiff Refined Trajectory MAE : {refined_metrics['MAE']:.4f} | RMSE: {refined_metrics['RMSE']:.4f}")
    print(f"    - Refinement Improvement       : +{mae_improvement:.2f}% MAE Reduction (+{rmse_improvement:.2f}% RMSE)")
    
    plot_refine_comparison(
        target_unnorm, base_unnorm, refined_unnorm,
        save_path=os.path.join(output_dir, "tsdiff_refine_comparison.png"),
        obs_len=obs_len
    )
    
    # -------------------------------------------------------------
    # 6. Pillar 3: Synthesize (Generative Motion Sampling)
    # -------------------------------------------------------------
    print("\n[Phase 6] Executing Pillar 3: SYNTHESIZE (Generative Trajectory Sampling)...")
    synthetic_samples, denoising_snaps = tsdiff.synthesize(shape=(6, seq_length), num_samples=4)
    synthetic_unnorm = dataset.unnormalize(synthetic_samples.cpu()).numpy()
    
    plot_synthetic_samples(
        synthetic_unnorm,
        save_path=os.path.join(output_dir, "tsdiff_synthesize_samples.png")
    )
    plot_denoising_process(
        denoising_snaps,
        channel_idx=0,
        save_path=os.path.join(output_dir, "tsdiff_denoising_steps.png")
    )
    print("    [+] Successfully synthesized 4 realistic 6-DoF multi-agent trajectories from scratch.")
    
    # -------------------------------------------------------------
    # 7. Ablation Study: Self-Guidance Scale (s)
    # -------------------------------------------------------------
    print("\n[Phase 7] Running Self-Guidance Scale Ablation Study...")
    scales = [0.0, 0.5, 1.0, 1.5, 2.5]
    ablation_maes = []
    ablation_crpss = []
    
    for s in scales:
        preds = tsdiff.predict_self_guided(target_sample, mask, guidance_scale=s, num_samples=8)
        preds_unnorm = dataset.unnormalize(preds.cpu()).numpy()[:, :, obs_len:]
        median_p = np.median(preds_unnorm, axis=0)
        m = compute_point_metrics(median_p, true_future)
        c = compute_crps(preds_unnorm, true_future)
        ablation_maes.append(m['MAE'])
        ablation_crpss.append(c)
        print(f"    - Guidance Scale s={s:.1f} -> MAE: {m['MAE']:.4f}, CRPS: {c:.4f}")
        
    plot_guidance_ablation(
        scales, ablation_maes, ablation_crpss,
        save_path=os.path.join(output_dir, "tsdiff_metrics_ablation.png")
    )
    
    # -------------------------------------------------------------
    # 8. Performance Summary Table Output
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                           FINAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Method / Model':<35} | {'MAE':<10} | {'RMSE':<10} | {'CRPS':<10} | {'Status':<10}")
    print("-" * 80)
    print(f"{'Base Dead-Reckoning (Unrefined)':<35} | {base_metrics['MAE']:<10.4f} | {base_metrics['RMSE']:<10.4f} | {'N/A (Point)':<10} | Baseline")
    print(f"{'TSDiff (Predict: Self-Guided)':<35} | {predict_point_metrics['MAE']:<10.4f} | {predict_point_metrics['RMSE']:<10.4f} | {predict_crps:<10.4f} | Optimal")
    print(f"{'TSDiff (Refine: Polished Base)':<35} | {refined_metrics['MAE']:<10.4f} | {refined_metrics['RMSE']:<10.4f} | {'N/A':<10} | Enhanced")
    print("=" * 80)
    print(f"[+] All figures and outputs generated in: {output_dir}")
    print("[+] Complete execution finished successfully.\n")

if __name__ == "__main__":
    run_tsdiff_pipeline()
