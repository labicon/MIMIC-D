# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:42:51 2024

@author: Jean-Baptiste Bouvier
"""

import torch
import random
import numpy as np


#%% Generic utils

def norm(x, dim=None, axis=None):
    """Calculates the norm of a vector x, either torch.Tensor or numpy.ndarray"""
    if type(x) == np.ndarray:
        if axis is not None:
            return np.linalg.norm(x, axis=axis)
        return np.linalg.norm(x)
    elif type(x) == torch.Tensor:
        if dim is not None:
            return torch.linalg.vector_norm(x, dim=dim)
        return torch.linalg.vector_norm(x).item()
    else:
        raise Exception(f"norm only works for torch.Tensor and numpy.ndarray, not {type(x)}")


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
class Normalizer():
    def __init__(self, x: torch.Tensor):
        """Normalizing states for the Diffusion, same mean std for all timesteps
        But different mean for each state"""
        if len(x.shape) == 2: # single trajectory
            self.mean, self.std = x.mean(dim=0), x.std(dim=0)
        elif len(x.shape) == 3: # dataset of trajectories
            self.mean, self.std = x.mean(dim=(0,1)), x.std(dim=(0,1))
        print(f"Normalization vector has shape {self.mean.shape}")
        
    def normalize(self, x: torch.Tensor):
        """Normalize a trajectory starting"""
        return (x - self.mean) / self.std
    
    def unnormalize(self, x: torch.Tensor):
        """Unnormalize a whole trajectory"""
        return x * self.std + self.mean  
    
