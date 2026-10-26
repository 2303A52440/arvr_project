"""
TSDiff: Self-Guiding Diffusion Framework for Time Series & AR/VR Trajectories
Student Roll: 2303A52440 | Course: AR/VR | Project: TSDiff

Core implementation of:
  1. Forward Diffusion Process (Gaussian perturbation schedule)
  2. Training Loss (Simplified Score Matching)
  3. Predict (Observation Self-Guided In-Painting / Forecasting)
  4. Refine (Manifold projection of base dead-reckoning predictions)
  5. Synthesize (Unconditional generative sampling from noise)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TSDiffEngine(nn.Module):
    def __init__(self, model, num_timesteps=50, beta_start=1e-4, beta_end=0.02, schedule="linear", device="cpu"):
        super().__init__()
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 1. Variance schedule setup
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            steps = num_timesteps + 1
            s = 0.008
            x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register diffusion buffers
        self.register_buffer("betas", betas.to(device))
        self.register_buffer("alphas", alphas.to(device))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(device))
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.to(device))
        
        # Precompute diffusion coefficients for fast sampling
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(device))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).to(device))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas).to(device))
        
        # Posterior variance q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.to(device))

    def q_sample(self, x_0, t, noise=None):
        """
        Forward noising step: samples x_t given clean x_0 and step t:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_0):
        """
        Calculates MSE score matching loss for unconditional diffusion training:
        L_simple = E_{t, x_0, eps} [ || eps - eps_theta(x_t, t) ||^2 ]
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)
        
        eps_pred = self.model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)
        return loss

    def predict_x0_from_xt(self, x_t, t, eps_pred):
        """
        Tweedie's reconstruction: calculates predicted clean x_0 given noisy x_t and noise estimate.
        x_0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
        """
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return (x_t - sqrt_one_minus_alpha_bar * eps_pred) / torch.clamp(sqrt_alpha_bar, min=1e-5)

    @torch.no_grad()
    def p_sample(self, x_t, t, eps_pred=None):
        """
        Single reverse denoising step p_theta(x_{t-1} | x_t).
        """
        if eps_pred is None:
            eps_pred = self.model(x_t, t)
            
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        
        # Mean calculation
        mean = sqrt_recip_alpha * (x_t - (beta_t / torch.clamp(sqrt_one_minus_alpha_bar, min=1e-5)) * eps_pred)
        
        if (t == 0).all():
            return mean
        else:
            var = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def synthesize(self, shape, num_samples=8):
        """
        PILLAR 3: SYNTHESIZE
        Unconditionally generates novel, high-fidelity 6-DoF AR/VR motion trajectories
        starting from pure standard Gaussian noise x_T ~ N(0, I).
        """
        self.model.eval()
        B, C, L = num_samples, shape[0], shape[1]
        x_t = torch.randn(B, C, L, device=self.device)
        
        traj_history = [x_t.clone()]
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((B,), step, device=self.device, dtype=torch.long)
            eps_pred = self.model(x_t, t)
            x_t = self.p_sample(x_t, t, eps_pred)
            if step % (self.num_timesteps // 5) == 0 or step == 0:
                traj_history.append(x_t.clone())
                
        return x_t, traj_history

    def predict_self_guided(self, x_obs, mask, guidance_scale=1.5, num_samples=16):
        """
        PILLAR 1: PREDICT (Self-Guided Trajectory Forecasting)
        Leverages the unconditionally trained diffusion model to forecast future
        timesteps given partial historical observations, guided by observation gradient:
        eps_guided = eps_theta + s * sqrt(1 - alpha_bar_t) * grad_{x_t}( || m * (x_0_hat - x_obs) ||^2 )
        """
        self.model.eval()
        B, C, L = num_samples, x_obs.shape[1], x_obs.shape[2]
        # Tile single observation across batch
        x_obs_tiled = x_obs.repeat(num_samples, 1, 1).to(self.device)
        mask_tiled = mask.repeat(num_samples, 1, 1).to(self.device)
        
        # Initialize with Gaussian noise
        x_t = torch.randn(B, C, L, device=self.device)
        
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((B,), step, device=self.device, dtype=torch.long)
            
            # Enable gradient computation for self-guidance step
            x_t_in = x_t.detach().requires_grad_(True)
            eps_pred = self.model(x_t_in, t)
            
            # Predict x_0 under current parameters
            x_0_hat = self.predict_x0_from_xt(x_t_in, t, eps_pred)
            
            # Self-guidance loss on known historical observations
            obs_loss = F.mse_loss(mask_tiled * x_0_hat, mask_tiled * x_obs_tiled, reduction="sum")
            
            if guidance_scale > 0.0 and step > 0:
                # Compute gradient w.r.t x_t
                grad = torch.autograd.grad(obs_loss, x_t_in)[0]
                sqrt_one_minus_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
                eps_guided = eps_pred + guidance_scale * sqrt_one_minus_bar * grad
            else:
                eps_guided = eps_pred
                
            # Perform reverse step with guided score
            with torch.no_grad():
                x_prev = self.p_sample(x_t, t, eps_guided.detach())
                
                # Observation replacement step for exact past adherence
                if step > 0:
                    q_obs = self.q_sample(x_obs_tiled, t - 1)
                    x_t = mask_tiled * q_obs + (1.0 - mask_tiled) * x_prev
                else:
                    x_t = mask_tiled * x_obs_tiled + (1.0 - mask_tiled) * x_prev
                    
        return x_t

    def refine(self, x_base, mask, noise_step=15, guidance_scale=1.0):
        """
        PILLAR 2: REFINE (Trajectory Polishing & Denoising)
        Takes noisy / drifting predictions from a simple base forecaster (e.g., dead reckoning),
        adds partial noise at step noise_step, and runs self-guided reverse denoising to project
        the trajectory back onto the realistic physical manifold of VR motion.
        """
        self.model.eval()
        B, C, L = x_base.shape
        x_base = x_base.to(self.device)
        mask = mask.to(self.device)
        
        # Inject partial forward noise at t = noise_step
        t_refine = torch.full((B,), noise_step, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_base)
        x_t = self.q_sample(x_0=x_base, t=t_refine, noise=noise)
        
        for step in reversed(range(noise_step)):
            t = torch.full((B,), step, device=self.device, dtype=torch.long)
            
            x_t_in = x_t.detach().requires_grad_(True)
            eps_pred = self.model(x_t_in, t)
            x_0_hat = self.predict_x0_from_xt(x_t_in, t, eps_pred)
            
            obs_loss = F.mse_loss(mask * x_0_hat, mask * x_base, reduction="sum")
            if guidance_scale > 0.0 and step > 0:
                grad = torch.autograd.grad(obs_loss, x_t_in)[0]
                sqrt_one_minus_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
                eps_guided = eps_pred + guidance_scale * sqrt_one_minus_bar * grad
            else:
                eps_guided = eps_pred
                
            with torch.no_grad():
                x_prev = self.p_sample(x_t, t, eps_guided.detach())
                if step > 0:
                    q_obs = self.q_sample(x_base, t - 1)
                    x_t = mask * q_obs + (1.0 - mask) * x_prev
                else:
                    x_t = mask * x_base + (1.0 - mask) * x_prev
                    
        return x_t
