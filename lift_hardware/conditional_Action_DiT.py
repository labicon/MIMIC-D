# -*- coding: utf-8 -*-
"""
Taken from https://github.com/ZibinDong/AlignDiff-ICLR2024/blob/main/utils/dit_utils.py

Prediction of the states only

conditional DiT based on the style of trajectories:
"""

import os
import math
import time
import torch
import einops
import pdb
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from typing import Optional
import matplotlib.pyplot as plt


#%% Diffusion Transformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ContinuousCondEmbedder(nn.Module):
    """Modified from DiscreteCondEmbedder to embed the initial state,
    a continuous variable instead of a 1-hot vector
    The embedding transforms the discrete 1-hot into a continuous vector, don't need that here.
    Just a regular affine layer to make the initial state of the right dimension."""
    
    def __init__(self, attr_dim: int, hidden_size: int, lin_scale: int):
        super().__init__()
        self.attr_dim = attr_dim
        self.lin_scale = lin_scale
        self.embedding = nn.Linear(attr_dim, int(attr_dim*lin_scale)) # 1 layer affine to transform initial state into embedding vector
        self.attn = nn.MultiheadAttention(128, num_heads=2, batch_first=True)
        self.linear = nn.Linear(128 * attr_dim, hidden_size)
    
    def forward(self, attr: torch.Tensor, mask: torch.Tensor = None):
        '''
        attr: (batch_size, attr_dim)
        mask: (batch_size, attr_dim) 0 or 1, 0 means ignoring
        '''
        emb = self.embedding(attr).reshape((-1, self.attr_dim, self.lin_scale)) # (b, attr_dim, 128)
        if mask is not None: emb *= mask.unsqueeze(-1) # (b, attr_dim, 128)
        emb, _ = self.attn(emb, emb, emb) # (b, attr_dim, 128)
        return self.linear(einops.rearrange(emb, 'b c d -> b (c d)')) # (b, hidden_size)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, dim), nn.Mish(), nn.Linear(dim, dim))
    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x,x,x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Finallayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

    
class DiT1d(nn.Module):
    def __init__(self, x_dim: int, attr_dim: int, d_model: int = 384, 
                 n_heads: int = 6, depth: int = 12, dropout: float = 0.1, lin_scale: int = 256):
        super().__init__()
        self.attr_dim = attr_dim # dimension of the attributes
        self.x_dim, self.d_model, self.n_heads, self.depth = x_dim, d_model, n_heads, depth
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_emb = TimeEmbedding(d_model)
        self.attr_proj = ContinuousCondEmbedder(attr_dim, d_model, lin_scale)
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.pos_emb_cache = None
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = Finallayer1d(d_model, x_dim)
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                attr: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        '''
        Input:  x: (batch, horizon, x_dim)     t:  (batch, 1)
             attr: (batch, attr_dim)         mask: (batch, attr_dim)
        
        Output: y: (batch, horizon, x_dim)
        '''
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.x_proj(x) + self.pos_emb_cache[None,]
        t = self.t_emb(t)
        if attr is not None:
            t += self.attr_proj(attr, mask)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)
        return x
    
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    



#%% Diffusion ODE with multiple models

class Conditional_ODE():
    def __init__(self, env, attr_dim: list, sigma_data: list, sigma_min: float = 0.001, sigma_max: float = 50,
                 rho: float = 7, p_mean: float = -1.2, p_std: float = 1.2, 
                 d_model: int = 384, n_heads: int = 6, depth: int = 12,
                 device: str = "cpu", N: int = 5, lr: float = 2e-4, lin_scale = 128,
                 n_models: int = 2):
        """
        Predicts the sequence of actions to apply conditioned on the initial state.
        Diffusion is trained according to EDM: "Elucidating the Design Space of Diffusion-Based Generative Models"
        This version supports training any number (n_models) of diffusion transformers simultaneously.
        
        Parameters:
         - env: environment object that must have attributes `name`, `state_size`, and `action_size`.
         - sigma_data: list of sigma_data values, one per transformer. (Length must equal n_models.)
         - attr_dim: should equal env.state_size * 2.
        """
        self.task = env.name
        self.specs = f"{d_model}_{n_heads}_{depth}"
        self.filename = "Cond_ODE_" + self.task + "_specs_" + self.specs
        
        self.state_size = env.state_size
        self.action_size = env.action_size
        if attr_dim is None:
            attr_dim = [env.state_size * 2]*n_models
        # assert attr_dim[0] == self.state_size * 2, "Attribute dimension must equal 2*state_size"
        
        # Expect sigma_data to be a list with one sigma per model.
        assert isinstance(sigma_data, list), "sigma_data must be a list"
        assert len(sigma_data) == n_models, "Length of sigma_data must equal n_models"
        self.sigma_data_list = sigma_data
        
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.rho, self.p_mean, self.p_std = rho, p_mean, p_std
        self.device = device
        
        # Create n_models diffusion transformers and their EMA copies.
        self.n_models = n_models
        self.F_list = nn.ModuleList()
        self.F_ema_list = []
        for i in range(n_models):
            model = DiT1d(self.action_size, attr_dim=attr_dim[i], d_model=d_model,
                           n_heads=n_heads, depth=depth, dropout=0.1, lin_scale=lin_scale).to(device)
            model.train()
            self.F_list.append(model)
            self.F_ema_list.append(deepcopy(model).requires_grad_(False).eval())
        
        # Create a single optimizer for all transformer parameters.
        all_params = []
        for model in self.F_list:
            all_params += list(model.parameters())
        self.optim = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
        
        self.set_N(N)  # number of noise scales
        
        total_params = sum(p.numel() for p in all_params)
        print(f'Initialized {self.n_models} Diffusion Transformer(s) with total parameters: {total_params}')
        
    def ema_update(self, decay=0.999):
        """Update the EMA copy for each transformer."""
        for i in range(self.n_models):
            for p, p_ema in zip(self.F_list[i].parameters(), self.F_ema_list[i].parameters()):
                p_ema.data = decay * p_ema.data + (1 - decay) * p.data

    def set_N(self, N):
        self.N = N
        self.sigma_s = (self.sigma_max**(1/self.rho) +
                        torch.arange(N, device=self.device) / (N-1) *
                        (self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho)))**self.rho
        self.t_s = self.sigma_s
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        if self.t_s is not None:
            self.coeff1 = (self.dot_sigma_s / self.sigma_s + self.dot_scale_s / self.scale_s)
            self.coeff2 = self.dot_sigma_s / self.sigma_s * self.scale_s
            
    # The following helper functions now use the per-model sigma_data.
    def c_skip(self, sigma, model_index=0):
        sigma_data = self.sigma_data_list[model_index]
        return sigma_data**2 / (sigma_data**2 + sigma**2)
    
    def c_out(self, sigma, model_index=0):
        sigma_data = self.sigma_data_list[model_index]
        return sigma * sigma_data / ((sigma_data**2 + sigma**2)**0.5)
    
    def c_in(self, sigma, model_index=0):
        sigma_data = self.sigma_data_list[model_index]
        return 1 / ((sigma_data**2 + sigma**2)**0.5)
    
    def c_noise(self, sigma, model_index=0):
        return 0.25 * sigma.log()
    
    def loss_weighting(self, sigma, model_index=0):
        sigma_data = self.sigma_data_list[model_index]
        return (sigma_data**2 + sigma**2) / ((sigma * sigma_data)**2)
    
    def sample_noise_distribution(self, N):
        log_sigma = torch.randn((N, 1, 1), device=self.device) * self.p_std + self.p_mean
        return log_sigma.exp()
    
    def D(self, x, sigma, condition=None, mask=None, use_ema=False, model_index=0):
        """
        Denoising function using the specified transformer.
        """
        c_skip = self.c_skip(sigma, model_index=model_index)
        c_out  = self.c_out(sigma, model_index=model_index)
        c_in   = self.c_in(sigma, model_index=model_index)
        c_noise = self.c_noise(sigma, model_index=model_index)
        F = self.F_ema_list[model_index] if use_ema else self.F_list[model_index]
        return c_skip * x + c_out * F(c_in * x, c_noise.squeeze(-1), condition, mask)
    
    def train(self,
              x_normalized_list: list,
              attributes_list: list,
              n_gradient_steps: int,
              batch_size: int = 32,
              extra: str = "",
              time_limit=None,
              endpoint_loss: bool = False):
        """
        Trains the diffusion transformers on multiple datasets.
        
        x_normalized_list: list of training data tensors, one per transformer.
            Each tensor should have shape (n_trajs, horizon, action_size).
        attributes_list: list of attribute tensors, one per transformer.
            Each tensor should have shape (n_trajs, attr_dim) where attr_dim = state_size * 2.
        n_gradient_steps: number of gradient steps.
        batch_size: batch size per transformer.
        time_limit: training time limit in seconds (optional).
        """
        print(f'Begins training of {self.n_models} Diffusion Transformer(s): {self.filename + extra}')
        if time_limit is not None:
            t0 = time.time()
            print(f"Training limited to {time_limit:.0f}s")
            
        assert len(x_normalized_list) == self.n_models and len(attributes_list) == self.n_models, \
            "Length of training data lists must equal n_models"
        
        N_trajs_list = [x.shape[0] for x in x_normalized_list]
        loss_avg = 0.0
        
        pbar = tqdm(range(n_gradient_steps))
        for step in range(n_gradient_steps):
            loss_total = 0.0
            for i in range(self.n_models):
                idx = np.random.randint(0, N_trajs_list[i], batch_size)
                x = x_normalized_list[i][idx].clone()   # shape: (batch_size, horizon, action_size)
                attr = attributes_list[i][idx].clone()    # shape: (batch_size, attr_dim)
                
                sigma = self.sample_noise_distribution(x.shape[0])
                eps = torch.randn_like(x) * sigma
                loss_mask = torch.ones_like(x)
                mask = (torch.rand(*attr.shape, device=self.device) > 0.2).int()
                
                pred = self.D(x + eps, sigma, condition=attr, mask=mask, model_index=i)
                loss = (loss_mask * self.loss_weighting(sigma, model_index=i) * (pred - x) ** 2).mean()
                
                if endpoint_loss:
                    pred_start = pred[:, 0, :self.state_size]
                    cond_start = attr[:, :self.state_size]
                    endpoint_loss = ((pred_start - cond_start) ** 2).mean()
                    loss = loss + 2.0 * endpoint_loss
                else:
                    loss_total += loss
            
            self.optim.zero_grad()
            loss_total.backward()
            all_params = []
            for model in self.F_list:
                all_params += list(model.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 10.0)
            self.optim.step()
            self.ema_update()
            
            loss_avg += loss_total.item()
            if (step + 1) % 10 == 0:
                pbar.set_description(f'step: {step+1} loss: {loss_avg/10:.4f} grad_norm: {grad_norm:.4f}')
                pbar.update(10)
                loss_avg = 0.0
                self.save(extra)
                if time_limit is not None and time.time() - t0 > time_limit:
                    print(f"Time limit reached at {time.time() - t0:.0f}s")
                    break
        print('\nTraining completed!')

    def train_mask(self,
               x_normalized_list: list,
               attributes_list: list,
               n_gradient_steps: int,
               batch_size: int = 32,
               extra: str = "",
               time_limit=None,
               endpoint_loss: bool = False,
               pad_masks: list = None,
               wait_masks: list = None):
        """
        A training loop that ignores padded and “wait” frames when computing loss.

        pad_masks & wait_masks: each a list of length self.n_models, where
        pad_masks[i]  is an np.ndarray of shape (N_i, H), dtype=bool
        wait_masks[i] is an np.ndarray of shape (N_i, H), dtype=bool
        """
        assert pad_masks is not None and wait_masks is not None, "Must pass pad_masks & wait_masks"
        assert len(pad_masks) == self.n_models and len(wait_masks) == self.n_models

        print(f'Begins training of {self.n_models} models: {self.filename + extra}')
        if time_limit is not None:
            t0 = time.time()
            print(f"Training limited to {time_limit:.0f}s")

        N_list = [x.shape[0] for x in x_normalized_list]
        pbar = tqdm(range(n_gradient_steps))
        for step in range(n_gradient_steps):
            loss_total = 0.0

            for i in range(self.n_models):
                # 1) sample a minibatch of data + masks
                idx = np.random.randint(0, N_list[i], batch_size)
                x_batch    = x_normalized_list[i][idx]           # (B, H, D)
                attr_batch = attributes_list[i][idx]             # (B, attr_dim)

                pad_mask_i  = pad_masks[i][idx]                  # (B, H), bool
                wait_mask_i = wait_masks[i][idx]                 # (B, H), bool

                # 2) convert to torch
                x    = torch.tensor(x_batch,    device=self.device)
                attr = torch.tensor(attr_batch, device=self.device)
                # combined learn‐mask: only real & moving frames
                learn_mask = torch.tensor(
                    pad_mask_i & wait_mask_i,
                    device=self.device,
                    dtype=torch.float32
                ).unsqueeze(-1)  # (B, H, 1)

                # 3) sample noise & model output
                sigma = self.sample_noise_distribution(x.shape[0])
                eps   = torch.randn_like(x) * sigma
                cond_mask = (torch.rand(*attr.shape, device=self.device) > 0.2).int()
                pred = self.D(
                    x + eps,
                    sigma,
                    condition=attr,
                    mask=cond_mask,
                    model_index=i
                )

                # 4) compute masked diff‐MSE
                # loss_weighting returns shape (B, 1, 1), broadcastable to x
                w = self.loss_weighting(sigma, model_index=i)
                se = (pred - x)**2 * w
                masked_se = se * learn_mask
                loss_i = masked_se.sum() / learn_mask.sum()

                # 5) optional endpoint‐start anchoring
                if endpoint_loss:
                    pred_s  = pred[:, 0, :self.state_size]
                    cond_s  = attr[:, :self.state_size]
                    end_l   = ((pred_s - cond_s)**2).mean()
                    loss_i += 2.0 * end_l

                loss_total += loss_i

            # 6) backward & step
            self.optim.zero_grad()
            loss_total.backward()
            # gradient clipping
            all_params = []
            for model in self.F_list:
                all_params += list(model.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 10.0)
            self.optim.step()
            self.ema_update()

            # 7) logging & checkpointing
            if (step + 1) % 10 == 0:
                avg = (loss_total.item() / self.n_models)
                pbar.set_description(f'step {step+1} | loss/model: {avg:.4f} | grad: {grad_norm:.4f}')
                pbar.update(10)
                self.save(extra)
                if time_limit is not None and (time.time() - t0) > time_limit:
                    print("Time limit reached")
                    break

        print("Training completed!")

        
    @torch.no_grad()
    def sample(self, attr, traj_len, n_samples: int, w: float = 1.5, N: int = None, model_index: int = 0):
        """
        Samples a trajectory using the EMA copy of the specified transformer.
        
        attr: attribute tensor of shape (n_samples, attr_dim)
        traj_len: trajectory length.
        model_index: which transformer to use.
        """
        if N is not None and N != self.N:
            self.set_N(N)
        
        x = torch.randn((n_samples, traj_len, self.action_size), device=self.device) * self.sigma_s[0] * self.scale_s[0]
        # x[:, 0, :self.state_size] = attr[:, :self.state_size]
        # x[:, -1, :self.state_size] = attr[:, self.state_size:]
        original_attr = attr.clone()
        
        attr_mask = torch.ones_like(attr)
        attr_cat = attr.repeat(2, 1)
        attr_mask_cat = attr_mask.repeat(2, 1)
        attr_mask_cat[n_samples:] = 0
        
        for i in range(self.N):
            with torch.no_grad():
                D_out = self.D(x.repeat(2, 1, 1) / self.scale_s[i],
                               torch.ones((2 * n_samples, 1, 1), device=self.device) * self.sigma_s[i],
                               condition=attr_cat,
                               mask=attr_mask_cat,
                               use_ema=True,
                               model_index=model_index)
                D_out = w * D_out[:n_samples] + (1 - w) * D_out[n_samples:]
            delta = self.coeff1[i] * x - self.coeff2[i] * D_out
            dt = self.t_s[i] - self.t_s[i+1] if i != self.N - 1 else self.t_s[i]
            x = x - delta * dt
            # x[:, 0, :self.state_size] = original_attr[:, :self.state_size]
            # x[:, -1, :self.state_size] = original_attr[:, self.state_size:]
        return x
    
    def save(self, extra: str = ""):
        """Saves the state dictionaries for all transformers and their EMA copies."""
        state = {}
        for i in range(self.n_models):
            state[f"model_{i}"] = self.F_list[i].state_dict()
            state[f"model_ema_{i}"] = self.F_ema_list[i].state_dict()
        torch.save(state, "trained_models/" + self.filename + extra + ".pt")
        
    def load(self, extra: str = ""):
        """Loads state dictionaries for all transformers and their EMA copies."""
        name = "trained_models/" + self.filename + extra + ".pt"
        if os.path.isfile(name):
            print("Loading " + name)
            checkpoint = torch.load(name, map_location=self.device)
            for i in range(self.n_models):
                self.F_list[i].load_state_dict(checkpoint[f"model_{i}"])
                self.F_ema_list[i].load_state_dict(checkpoint[f"model_ema_{i}"])
            return True
        else:
            print("File " + name + " doesn't exist. Not loading anything.")
            return False


#%%


class Conditional_Planner():
    def __init__(self, env, ode: Conditional_ODE, normalizer):
        """Planner enables trjaectory prediction """
        self.env = env
        self.ode = ode
        self.normalizer = normalizer
        self.device = ode.device
        self.state_size = env.state_size
        self.action_size = env.action_size
      
        
    @torch.no_grad()
    def traj(self, s0, traj_len, N:int=None):
        """Returns n_samples action sequences of length traj_len starting from
        UNnormalized state s0."""
        
        if type(s0) == np.ndarray:
            nor_s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)[None,]
        else: # s0 = Tensor
            nor_s0 = s0.clone()
            s0 = s0.squeeze().numpy()
        nor_s0 = self.normalizer.normalize(nor_s0)
            
        assert nor_s0.shape == (1, self.state_size), "Only works for a single state"
        
        action_pred = self.ode.sample(attr=nor_s0, traj_len=traj_len-1, n_samples=1, N=N)
        traj_pred, traj_reward = self._open_loop(s0, action_pred.numpy())
        return traj_pred, action_pred, traj_reward   
          
    
    @torch.no_grad()
    def best_traj(self, s0, traj_len, n_samples_per_s0=1, N:int=None):
        """Returns 1 trajectory of length traj_len starting from each
        UNnormalized states s0.
        For each s0  n_samples_per_s0 are generated, the one with the longest survival is chosen"""
        
        if s0.shape == (self.state_size,):
            s0.reshape((1, self.state_size))
        assert s0.shape[1] == self.state_size
        
        if type(s0) == np.ndarray:
            nor_s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)
        else: # s0 = Tensor
            nor_s0 = s0.clone()
            s0 = s0.numpy()
        
        N_s0 = nor_s0.shape[0] # 
        nor_s0 = self.normalizer.normalize(nor_s0)
        nor_s0 = nor_s0.repeat_interleave(n_samples_per_s0, dim=0)
        n_samples = nor_s0.shape[0] # total number of samples = N_s0 * n_samples_per_s0
        
        # Sample all the sequences of actions at once: faster
        Actions_pred = self.ode.sample(attr=nor_s0, traj_len=traj_len-1, n_samples=n_samples, N=N)
        
        Best_Trajs = np.zeros((N_s0, traj_len, self.state_size))
        Best_rewards = np.zeros(N_s0)
        Best_Actions = np.zeros((N_s0, traj_len-1, self.action_size))
        
        for s_id in range(N_s0): # index of the s0
            highest_reward = -np.inf # look for the trajectory with highest reward for each s0
            i = s_id*n_samples_per_s0
            Sampled_Trajs = []
            
            for sample_id in range(n_samples_per_s0):    
                traj, reward = self._open_loop(s0[s_id], Actions_pred[i + sample_id].numpy())
                Sampled_Trajs.append(traj)
                if reward > highest_reward:
                    highest_reward = reward
                    id_highest_reward = sample_id
            
            t = Sampled_Trajs[id_highest_reward].shape[0]
            Best_Trajs[s_id, :t] = Sampled_Trajs[id_highest_reward].copy()
            Best_rewards[s_id] = highest_reward
            Best_Actions[s_id] = Actions_pred[i + id_highest_reward]
        
        return Best_Trajs, Best_Actions, Best_rewards


    def _open_loop(self, s0, Actions):
        """Applies the sequence of actions in open-loop on the initial state s0"""
        assert s0.shape[0] == self.state_size
        assert Actions.shape[1] == self.action_size
        
        N_steps = Actions.shape[0]
        Traj = np.zeros((N_steps+1, self.state_size))
        Traj[0] = self.env.reset_to(s0)
        traj_reward = 0.
        
        for t in range(N_steps):
            Traj[t+1], reward, done, _, _ = self.env.step(Actions[t])
            traj_reward += reward
            if done: break
        
        return Traj[:t+2], traj_reward




