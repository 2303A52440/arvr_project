"""
Publication-Quality Visualization Suite for TSDiff AR/VR Evaluation
Student Roll: 2303A52440 | Course: AR/VR | Project: TSDiff
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Clean publication aesthetic styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.35

CHANNEL_NAMES = ['Pos X (m)', 'Pos Y (m)', 'Pos Z (m)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']

def plot_predict_forecast(history, true_future, samples_future, save_path="tsdiff_predict_forecasting.png", obs_len=32):
    """
    Plots historical observation, true future, predicted mean, 50% & 90% confidence bands.
    """
    # shapes: history [C, obs_len], true_future [C, pred_len], samples_future [N, C, pred_len]
    N, C, pred_len = samples_future.shape
    total_len = obs_len + pred_len
    t_hist = np.arange(obs_len)
    t_pred = np.arange(obs_len, total_len)
    
    mean_pred = np.mean(samples_future, axis=0)
    median_pred = np.median(samples_future, axis=0)
    p5 = np.percentile(samples_future, 5, axis=0)
    p95 = np.percentile(samples_future, 95, axis=0)
    p25 = np.percentile(samples_future, 25, axis=0)
    p75 = np.percentile(samples_future, 75, axis=0)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for i in range(6):
        ax = axes[i]
        # Past ground truth
        ax.plot(t_hist, history[i], color='#1f77b4', linewidth=2.0, label='Observed History (32 frames)')
        # Future ground truth
        ax.plot(t_pred, true_future[i], color='#2ca02c', linewidth=2.0, linestyle='--', label='Ground Truth Future')
        # Probabilistic Diffusion Median & Mean
        ax.plot(t_pred, median_pred[i], color='#d62728', linewidth=2.0, label='TSDiff Forecast (Median)')
        # Uncertainty intervals
        ax.fill_between(t_pred, p5[i], p95[i], color='#ff9896', alpha=0.35, label='90% Credible Band')
        ax.fill_between(t_pred, p25[i], p75[i], color='#d62728', alpha=0.25, label='50% Credible Band')
        
        ax.axvline(x=obs_len - 1, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_title(f'6-DoF Channel: {CHANNEL_NAMES[i]}', fontweight='bold', fontsize=11)
        ax.grid(True, linestyle='--')
        if i == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
    fig.suptitle('TSDiff Pillar 1: Self-Guided Trajectory Prediction (Predict)\nAR/VR 6-DoF Head Pose Forecasting under Observation Guidance', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualizer] Saved prediction figure to: {save_path}")


def plot_refine_comparison(gt_seq, base_seq, refined_seq, save_path="tsdiff_refine_comparison.png", obs_len=32):
    """
    Visualizes TSDiff Refine: compares unrefined base forecaster vs TSDiff refined trajectory vs ground truth.
    """
    total_len = gt_seq.shape[1]
    t = np.arange(total_len)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for i in range(6):
        ax = axes[i]
        ax.plot(t, gt_seq[i], color='#2ca02c', linewidth=2.2, label='Ground Truth Motion')
        ax.plot(t, base_seq[i], color='#7f7f7f', linestyle=':', linewidth=1.8, label='Base Forecaster (Dead Reckoning + Drift)')
        ax.plot(t, refined_seq[i], color='#1f77b4', linewidth=2.0, label='TSDiff Refined (Score Polished)')
        
        ax.axvline(x=obs_len - 1, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.set_title(f'Channel: {CHANNEL_NAMES[i]}', fontweight='bold', fontsize=11)
        ax.grid(True, linestyle='--')
        if i == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
    fig.suptitle('TSDiff Pillar 2: Trajectory Denoising & Refinement (Refine)\nMitigating Sensor Jitter & Extrapolation Drift in VR Head Pose', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualizer] Saved refinement figure to: {save_path}")


def plot_synthetic_samples(synthetic_trajs, save_path="tsdiff_synthesize_samples.png"):
    """
    Visualizes TSDiff Synthesize: shows 4 novel generated 6-DoF trajectories sampled from pure Gaussian noise.
    """
    # synthetic_trajs: [4, 6, seq_len]
    num_samples = min(4, synthetic_trajs.shape[0])
    seq_len = synthetic_trajs.shape[2]
    t = np.arange(seq_len)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    for c in range(6):
        ax = axes[c]
        for s in range(num_samples):
            ax.plot(t, synthetic_trajs[s, c], color=colors[s], linewidth=1.8, label=f'Synthesized Agent {s+1}')
        ax.set_title(f'{CHANNEL_NAMES[c]} Synthesized Variations', fontweight='bold', fontsize=11)
        ax.grid(True, linestyle='--')
        if c == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
    fig.suptitle('TSDiff Pillar 3: Generative Motion Synthesis (Synthesize)\nUnconditional Generation of Multi-Agent Spatial Trajectories from Gaussian Noise', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualizer] Saved synthesis figure to: {save_path}")


def plot_denoising_process(denoising_history, channel_idx=0, save_path="tsdiff_denoising_steps.png"):
    """
    Visualizes the step-by-step reverse diffusion trajectory from pure noise to clean trajectory.
    """
    # denoising_history is a list of tensors [B, C, L]
    fig, ax = plt.subplots(figsize=(10, 5))
    steps_total = len(denoising_history)
    alphas = np.linspace(0.2, 1.0, steps_total)
    
    for idx, snap in enumerate(denoising_history):
        data = snap[0, channel_idx].detach().cpu().numpy()
        step_label = f"Step {idx * (50 // (steps_total - 1))}" if idx < steps_total - 1 else "Final x_0"
        color = plt.cm.viridis(idx / (steps_total - 1))
        lw = 2.2 if idx == steps_total - 1 else 1.2
        ax.plot(data, color=color, alpha=alphas[idx], linewidth=lw, label=step_label)
        
    ax.set_title(f'TSDiff Reverse Diffusion Process: Denoising Step Progression ({CHANNEL_NAMES[channel_idx]})', fontweight='bold')
    ax.set_xlabel('Sequence Time Steps (Frames)', fontweight='bold')
    ax.set_ylabel('Amplitude (Metric Units)', fontweight='bold')
    ax.grid(True, linestyle='--')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualizer] Saved denoising steps figure to: {save_path}")


def plot_guidance_ablation(scales, maes, crpss, save_path="tsdiff_metrics_ablation.png"):
    """
    Visualizes the effect of the Self-Guidance Scale 's' on point accuracy (MAE) and probabilistic calibration (CRPS).
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = '#1f77b4'
    ax1.set_xlabel('Observation Self-Guidance Scale (s)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Mean Absolute Error (MAE)', color=color1, fontweight='bold', fontsize=11)
    line1 = ax1.plot(scales, maes, marker='o', linewidth=2.2, color=color1, label='MAE (Point Accuracy)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--')
    
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('Continuous Ranked Probability Score (CRPS)', color=color2, fontweight='bold', fontsize=11)
    line2 = ax2.plot(scales, crpss, marker='s', linewidth=2.2, linestyle='--', color=color2, label='CRPS (Probabilistic Fit)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9)
    
    plt.title('Self-Guidance Parameter Sensitivity (Ablation Study)\nTrade-Off Between Condition Fidelity and Prior Distribution Realism', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Visualizer] Saved ablation figure to: {save_path}")
