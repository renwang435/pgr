# Denoiser networks for diffusion.

import math

import gin
import torch
import torch.nn as nn
import torch.optim
from einops import rearrange
from torch.nn import functional as F
from ipdb import set_trace as st

from torch.distributions import Bernoulli


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(
            self,
            dim: int,
            is_random: bool = False,
    ):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# Residual MLP of the form x_{L+1} = MLP(LN(x_L)) + x_L
class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            self.ln = nn.LayerNorm(dim_in)
        else:
            self.ln = torch.nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))


class ResidualMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            cond_dim: int,
            width: int,
            depth: int,
            output_dim: int,
            activation: str = "relu",
            layer_norm: bool = False,
    ):
        super().__init__()

        assert cond_dim is not None, "Residual MLP constructor requires cond_dim"
        self.x_proj = nn.Linear(input_dim, width)
        self.cond_proj = nn.Linear(cond_dim, width)

        self.network = nn.ModuleList(
            [ResidualBlock(width * 2, width * 2, activation, layer_norm) for _ in range(depth)]
        )

        self.activation = getattr(F, activation)
        self.final_linear = nn.Linear(2 * width, output_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.x_proj(x)
        cond = self.cond_proj(cond)
        x = torch.cat((x, cond), dim=-1)
        for layer in self.network:
            x = layer(x)
        return self.final_linear(self.activation(x))


@gin.configurable
class ResidualMLPDenoiser(nn.Module):
    def __init__(
            self,
            d_in: int,
            dim_t: int = 128,
            mlp_width: int = 1024,
            num_layers: int = 6,
            learned_sinusoidal_cond: bool = False,
            random_fourier_features: bool = True,
            learned_sinusoidal_dim: int = 16,
            activation: str = "relu",
            layer_norm: bool = True,
            cond_dim: int = None,
            cfg_dropout: float = 0.0,
    ):
        super().__init__()
        self.residual_mlp = ResidualMLP(
            input_dim=d_in,
            cond_dim=dim_t * 2,
            width=mlp_width,
            depth=num_layers,
            output_dim=d_in,
            activation=activation,
            layer_norm=layer_norm,
        )
        assert cond_dim is not None, "Conditional denoiser constructor requires cond_dim"

        # Conditional dropout
        self.cond_dropout = Bernoulli(probs=1 - cfg_dropout)

        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim_t * 2),
            nn.Mish(),
            nn.Linear(dim_t * 2, dim_t),
        )

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t * 4),
            nn.Mish(),
            nn.Linear(dim_t * 4, dim_t)
        )

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            cond=None,
    ) -> torch.Tensor:
        
        t = self.time_mlp(timesteps)
        
        if cond is not None:
            c = self.cond_mlp(cond)
            
            # Do conditional dropout during training
            if self.training:
                mask = self.cond_dropout.sample(sample_shape=(c.shape[0], 1)).to(c.device)
                c = c * mask
        else:
            c = torch.zeros_like(t).to(t.device)

        t = torch.cat((c, t), dim=-1)

        return self.residual_mlp(x, t)


if __name__ == "__main__":
    pass
