# Utilities for diffusion.
from typing import List, Optional, Union

import d4rl
import gin
import gym
import numpy as np
import torch
from ipdb import set_trace as st
# GIN-required Imports.
from synther.diffusion.denoiser_network_cond import ResidualMLPDenoiser
from synther.diffusion.elucidated_diffusion import ElucidatedDiffusion
from synther.diffusion.norm import normalizer_factory
from torch import nn

# Convert diffusion samples back to (s, a, r, s') format.
@gin.configurable
def split_diffusion_samples(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modelled_terminals: bool = False,
        terminal_threshold: Optional[float] = None,
):
    # Compute dimensions from env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Split samples into (s, a, r, s') format
    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    next_obs = samples[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, next_obs, terminals
    else:
        return obs, actions, rewards, next_obs


@gin.configurable
def construct_diffusion_model(
        inputs: torch.Tensor,
        normalizer_type: str,
        denoising_network: nn.Module,
        activation: str = "relu",
        disable_terminal_norm: bool = False,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = None,
        cfg_dropout: float = 0.0,
) -> ElucidatedDiffusion:
    event_dim = inputs.shape[1]
    model = denoising_network(d_in=event_dim, activation=activation, 
                              cond_dim=cond_dim, cfg_dropout=cfg_dropout)

    if disable_terminal_norm:
        terminal_dim = event_dim - 1
        if terminal_dim not in skip_dims:
            skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")
    
    # import ipdb; ipdb.set_trace()

    normalizer = normalizer_factory(normalizer_type, inputs, skip_dims=skip_dims)
    if cond_dim is not None:
        cond_inputs = torch.zeros((128, cond_dim)).float()
        cond_normalizer = normalizer_factory(normalizer_type, cond_inputs, skip_dims=[])
    else:
        cond_normalizer = None

    diffusion_model = ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        cond_normalizer=cond_normalizer,
        event_shape=[event_dim],
    )

    # Counter number of parameters in diffusion model
    num_params = sum(p.numel() for p in diffusion_model.net.parameters() if p.requires_grad)
    print(f"Number of parameters in diffusion model: {num_params}")

    return diffusion_model

