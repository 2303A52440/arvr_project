"""
TSDiff Score / Noise Estimator Network (1D Dilated Temporal ResNet)
Student Roll: 2303A52440 | Course: AR/VR | Project: TSDiff

Implements the unconditional score-based noise predictor epsilon_theta(x_t, t)
using multi-scale dilated 1D temporal convolutions, SiLU activations, and
sinusoidal diffusion timestep conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Standard sinusoidal embedding mapping continuous diffusion timestep t
    into a high-dimensional harmonic feature vector.
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        """
        timesteps: 1D Tensor of shape [B]
        returns: Tensor of shape [B, dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1, 0, 0))
        return embedding


class ResidualTemporalBlock1D(nn.Module):
    """
    Residual Convolutional block with dilated 1D receptive field and
    diffusion time-embedding affine projection.
    """
    def __init__(self, channels, time_emb_dim, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, dilation=1
        )
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=channels)
        
        # Linear projection for time embedding conditioning
        self.time_proj = nn.Linear(time_emb_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x, time_emb):
        """
        x: [B, channels, seq_len]
        time_emb: [B, time_emb_dim]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Inject broadcasted time embedding
        t_proj = self.time_proj(time_emb).unsqueeze(-1) # [B, channels, 1]
        h = h + t_proj
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return x + h  # Residual skip connection


class TSDiffScoreNet(nn.Module):
    """
    Complete TSDiff Noise Estimation Neural Network.
    Maps [B, in_channels, seq_len] + [B] timesteps -> [B, in_channels, seq_len] (epsilon_pred).
    """
    def __init__(self, in_channels=6, model_channels=64, num_layers=4, time_emb_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # 1. Diffusion Timestep Encoder MLP
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 2. Input Projection
        self.input_conv = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 3. Dilated Residual Temporal Backbone
        dilations = [1, 2, 4, 8]
        self.res_blocks = nn.ModuleList([
            ResidualTemporalBlock1D(model_channels, time_emb_dim, dilation=dilations[i % len(dilations)])
            for i in range(num_layers)
        ])
        
        # 4. Output Projection
        self.out_norm = nn.GroupNorm(num_groups=4, num_channels=model_channels)
        self.out_act = nn.SiLU()
        self.output_conv = nn.Conv1d(model_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, timesteps):
        """
        Forward pass for score/noise estimation.
        x: Noisy sequence [B, C, L]
        timesteps: Integer step tensor [B]
        returns: Predicted noise epsilon_theta [B, C, L]
        """
        # Encode diffusion timesteps
        t_emb = self.time_embed(timesteps)
        
        # Project input to hidden dimension
        h = self.input_conv(x)
        
        # Process through dilated residual blocks
        for block in self.res_blocks:
            h = block(h, t_emb)
            
        # Final projection to match original data dimensionality
        h = self.out_norm(h)
        h = self.out_act(h)
        eps_pred = self.output_conv(h)
        
        return eps_pred
