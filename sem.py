# sem.py

import torch


def small_energy_masking(energy: torch.Tensor) -> torch.Tensor:
    device = energy.device
    energy = energy.to(device)

    peak_energy = torch.quantile(energy, 0.95, dim=-1, keepdim=True)

    eta_th = -20
    
    e_th = peak_energy * (10 ** (eta_th / 10))
    
    mask = (energy >= e_th).float()
    
    masked_energy = energy * mask
    scaling_factor = torch.sum(energy) / (torch.sum(masked_energy) + 1e-9)
    masked_energy *= scaling_factor
    
    return masked_energy

