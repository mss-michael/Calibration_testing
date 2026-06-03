from torch import nn
import torch
import numpy as np
from typing import Tuple


class UnitNormDecoder(nn.Linear):
    def forward(self, x):
        with torch.no_grad():
            W = self.weight
            W.div(W.norm(dim=0, keepdim=True).clamp_min(1e-8))
        return super().forward(x)

class SparseAutoencoder(nn.Module):
    """Two‑layer sparse auto‑encoder for 512-D latent vectors with an over‑complete
    hidden "concept" layer. The decoder weights constitute the learned
    dictionary atoms.
    """

    def __init__(
        self,
        input_dim: int = 512,
        code_dim: int = 2048,
        activation: nn.Module = nn.ReLU(),
        tied_weights: bool = False,
        use_circuits_implementation = False,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, code_dim, bias=True)
        self.decoder = UnitNormDecoder(code_dim, input_dim, bias=True)
        self.activation = activation
        self.use_circuits_implementation = use_circuits_implementation
        if tied_weights:
            # tie decoder weights to encoder weights (transpose)
            self.decoder.weight = nn.Parameter(self.encoder.weight.t())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_circuits_implementation:
            x_line = x - self.decoder.bias
            z = self.activation(self.encoder(x_line))
            x_hat = self.decoder(z)
        else:
            z = self.activation(self.encoder(x))
            x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.activation(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(z)

    def concept_vector(self, concept_idx: int) -> torch.Tensor:
        return self.decoder.weight[:, concept_idx].detach().clone()
