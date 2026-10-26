"""
AR/VR 6-DoF Trajectory Data Generator
Student Roll: 2303A52440 | Course: AR/VR | Project: TSDiff

Generates realistic 6-Degrees-of-Freedom (6-DoF) spatial-temporal motion trajectories
simulating Virtual Reality (VR) headset tracking and Augmented Reality (AR) spatial anchors.
Channels: [X, Y, Z (meters), Roll, Pitch, Yaw (radians)]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ARVRTrajectoryDataset(Dataset):
    """
    Simulated 6-DoF AR/VR Headset and Controller Motion Dataset.
    Synthesizes realistic human head motion patterns:
      - Continuous locomotion (walking / smooth translation)
      - Natural head turns (saccades, smooth pursuit, yaw/pitch scans)
      - Micro-jitter & vestibular tremor
    """
    def __init__(self, num_samples=1200, seq_length=64, dt=0.02, seed=42):
        super().__init__()
        np.random.seed(seed)
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.dt = dt
        self.data = self._generate_dataset()

    def _generate_single_trajectory(self):
        t = np.linspace(0, self.seq_length * self.dt, self.seq_length)
        
        # 1. Translation (X, Y, Z) in meters
        # Normal walking speed ~ 1.2 m/s with natural sway
        speed = np.random.uniform(0.5, 2.0)
        heading_angle = np.random.uniform(-np.pi, np.pi)
        
        # X: lateral movement with sway
        x_base = speed * np.cos(heading_angle) * t
        x_sway = 0.05 * np.sin(2 * np.pi * 1.8 * t)  # 1.8 Hz step frequency
        x = x_base + x_sway + np.random.normal(0, 0.005, self.seq_length)
        
        # Y: vertical eye height (1.6m - 1.75m standing height) + vertical bobbing
        y_base = 1.65 + np.random.uniform(-0.1, 0.1)
        y_bob = 0.03 * np.abs(np.sin(2 * np.pi * 0.9 * t))  # step cadence
        y = y_base + y_bob + np.random.normal(0, 0.003, self.seq_length)
        
        # Z: forward movement
        z_base = speed * np.sin(heading_angle) * t
        z = z_base + np.random.normal(0, 0.005, self.seq_length)
        
        # 2. Rotation: Roll (phi), Pitch (theta), Yaw (psi) in radians
        # Head roll: subtle tilt during turns
        roll = 0.04 * np.sin(2 * np.pi * 0.8 * t + np.random.uniform(0, np.pi))
        
        # Head pitch: slight vertical head scanning
        pitch_freq = np.random.uniform(0.3, 1.2)
        pitch = 0.15 * np.sin(2 * np.pi * pitch_freq * t) + np.random.uniform(-0.1, 0.1)
        
        # Head yaw: dominant turning behavior
        yaw_freq = np.random.uniform(0.2, 0.8)
        yaw_amp = np.random.uniform(0.2, 0.6)
        yaw = yaw_amp * np.sin(2 * np.pi * yaw_freq * t + heading_angle)
        
        traj = np.stack([x, y, z, roll, pitch, yaw], axis=0) # [6, seq_length]
        return traj.astype(np.float32)

    def _generate_dataset(self):
        trajectories = []
        for _ in range(self.num_samples):
            trajectories.append(self._generate_single_trajectory())
        trajectories = np.array(trajectories) # [num_samples, 6, seq_length]
        
        # Standardize data per channel for stable diffusion training
        self.mean = np.mean(trajectories, axis=(0, 2), keepdims=True)
        self.std = np.std(trajectories, axis=(0, 2), keepdims=True) + 1e-6
        normalized = (trajectories - self.mean) / self.std
        return torch.tensor(normalized, dtype=torch.float32)

    def unnormalize(self, x_norm):
        """Reverts normalized tensor or array back to physical metric units [meters / radians]."""
        if isinstance(x_norm, torch.Tensor):
            mean = torch.tensor(self.mean, device=x_norm.device, dtype=x_norm.dtype)
            std = torch.tensor(self.std, device=x_norm.device, dtype=x_norm.dtype)
            res = x_norm * std + mean
            if x_norm.dim() == 2 and res.dim() == 3:
                res = res.squeeze(0)
            return res
        res = x_norm * self.std + self.mean
        if x_norm.ndim == 2 and res.ndim == 3:
            res = res.squeeze(0)
        return res

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


def get_data_loaders(batch_size=32, num_train=1000, num_val=200, seq_length=64):
    """Factory helper to retrieve train and test PyTorch dataloaders."""
    train_dataset = ARVRTrajectoryDataset(num_samples=num_train, seq_length=seq_length, seed=42)
    val_dataset = ARVRTrajectoryDataset(num_samples=num_val, seq_length=seq_length, seed=100)
    
    # Share normalization constants
    val_dataset.mean = train_dataset.mean
    val_dataset.std = train_dataset.std
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset


class BaseDeadReckoningForecaster:
    """
    Implements a classic AR/VR Dead-Reckoning Extrapolation baseline (constant velocity + jitter).
    Used as the unrefined baseline input for TSDiff's 'Refine' stage.
    """
    def __init__(self, obs_len=32, pred_len=32):
        self.obs_len = obs_len
        self.pred_len = pred_len

    def forecast(self, x_seq, noise_std=0.08):
        """
        Takes full sequence [B, C, L], extracts observation [0:obs_len],
        and extrapolates forward with velocity and cumulative drift.
        """
        B, C, L = x_seq.shape
        x_obs = x_seq[:, :, :self.obs_len].clone()
        
        # Estimate last velocity from past 3 frames
        vel = (x_obs[:, :, -1:] - x_obs[:, :, -3:-2]) / 2.0
        
        preds = []
        curr = x_obs[:, :, -1:].clone()
        for step in range(self.pred_len):
            drift = torch.randn_like(curr) * noise_std * np.sqrt(step + 1) * 0.25
            curr = curr + vel + drift
            preds.append(curr)
            
        x_pred_part = torch.cat(preds, dim=-1)
        # Full trajectory combining observation and rough forecast
        full_extrapolated = torch.cat([x_obs, x_pred_part], dim=-1)
        return full_extrapolated
