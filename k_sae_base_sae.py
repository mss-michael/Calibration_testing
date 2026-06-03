from torch import nn
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

class UnitNormDecoder(nn.Linear):
    @torch.no_grad()
    def renorm_(self) -> None:
        self.weight.div_(self.weight.norm(dim=0, keepdim=True).clamp_min_(1e-8))

    def forward(self, x):
        return super().forward(x)


class TiedDecoder(nn.Module):
    def __init__(self, encoder: nn.Linear, output_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.bias = nn.Parameter(torch.zeros(output_dim))

    @property
    def weight(self) -> torch.Tensor:
        return self.encoder.weight.t()

    @torch.no_grad()
    def renorm_(self) -> None:
        self.encoder.weight.div_(self.encoder.weight.norm(dim=1, keepdim=True).clamp_min_(1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
class SparseAutoencoder(nn.Module):
    """Two‑layer sparse auto‑encoder for 512‑D latent vectors with an over‑complete
    hidden "concept" layer. The decoder weights constitute the learned
    dictionary atoms.
    """
    def __init__(
        self,
        input_dim: int = 512,
        code_dim: int = 2048,
        activation: nn.Module = nn.ReLU(),
        tied_weights: bool = False,
        use_circuits_implementation: bool = False,
        sae_type = "sae"
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, code_dim, bias=True)
        self.tied_weights = tied_weights
        self.decoder = TiedDecoder(self.encoder, input_dim) if tied_weights else UnitNormDecoder(code_dim, input_dim, bias=True)
        self.activation = activation
        self.use_circuits_implementation = use_circuits_implementation
        self.sae_type = sae_type

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
            if self.use_circuits_implementation:
                x = x - self.decoder.bias
            return self.activation(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(z)

    @torch.no_grad()
    def renorm_decoder_(self) -> None:
        if hasattr(self.decoder, "renorm_"):
            self.decoder.renorm_()

    def concept_vector(self, concept_idx: int) -> torch.Tensor:
        return self.decoder.weight[:, concept_idx].detach().clone()


def _topk_mask_per_row(z: torch.Tensor, k: int) -> torch.Tensor:
    """
    z: [B, D]
    Returns mask: [B, D] with exactly k True per row (unless k>=D).
    """
    B, D = z.shape
    kk = min(k, D)
    if kk <= 0:
        return torch.zeros_like(z, dtype=torch.bool)
    idx = torch.topk(z, kk, dim=1, largest=True, sorted=False).indices  # [B, kk]
    mask = torch.zeros_like(z, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def _batchtopk_mask_flat(z: torch.Tensor, k_per_sample: int, by_abs: bool = False) -> torch.Tensor:
    """
    Canonical BatchTopK:
      - z: [B, D]
      - keep top (B * k_per_sample) activations across the ENTIRE batch (flattened)
      - returns mask: [B, D] with exactly B*k True (unless truncated by size)

    by_abs=False matches "by value" selection as described in the BatchTopK SAE paper.
    by_abs=True selects by magnitude, useful if you allow signed codes.
    """
    if z.dim() != 2:
        raise ValueError(f"Expected z to have shape [B, D], got {tuple(z.shape)}")

    B, D = z.shape
    total = B * D
    K = int(k_per_sample) * B
    K = max(0, min(K, total))

    mask_flat = torch.zeros(total, device=z.device, dtype=torch.bool)
    if K == 0:
        return mask_flat.view(B, D)

    scores = z.abs() if by_abs else z
    flat_scores = scores.reshape(-1)

    # Exact-K selection via indices (avoids threshold tie issues)
    idx = torch.topk(flat_scores, K, largest=True, sorted=False).indices
    mask_flat[idx] = True
    return mask_flat.view(B, D)


class TopKSparseAutoencoder(SparseAutoencoder):
    """
    Per-sample TopK gating: exactly k non-zeros in z per example (post-activation).
    """
    def __init__(self, *args, k: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = k
        self.sae_type = "top_k_sae"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_circuits_implementation:
            x_line = x - self.decoder.bias
            z = self.activation(self.encoder(x_line))
        else:
            z = self.activation(self.encoder(x))

        mask = _topk_mask_per_row(z, self.k)
        z_sparse = z * mask.to(z.dtype)

        x_hat = self.decoder(z_sparse)
        return x_hat, z_sparse

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.use_circuits_implementation:
                x = x - self.decoder.bias
            z = self.activation(self.encoder(x))
            mask = _topk_mask_per_row(z, self.k)
            return z * mask.to(z.dtype)


class BatchTopKSparseAutoencoder(SparseAutoencoder):
    """
    Canonical BatchTopK SAE:
    keep top (B*k) activations across the flattened batch => average k active per sample.
    """
    def __init__(self, *args, k: int = 32, by_abs: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = k
        self.by_abs = by_abs
        self.sae_type = "batch_top_k_sae"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_circuits_implementation:
            x_line = x - self.decoder.bias
            z = self.activation(self.encoder(x_line))
        else:
            z = self.activation(self.encoder(x))

        mask = _batchtopk_mask_flat(z, self.k, by_abs=self.by_abs)
        z_sparse = z * mask.to(z.dtype)

        x_hat = self.decoder(z_sparse)
        return x_hat, z_sparse

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.use_circuits_implementation:
                x = x - self.decoder.bias
            z = self.activation(self.encoder(x))
            mask = _batchtopk_mask_flat(z, self.k, by_abs=self.by_abs)
            z_sparse = z * mask.to(z.dtype)
            return z_sparse
  