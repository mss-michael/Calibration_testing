from torch import nn
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')
from typing import Any
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import deque
import warnings
import math
from k_sae_base_sae import SparseAutoencoder, TopKSparseAutoencoder, BatchTopKSparseAutoencoder
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


# --- anti-duplicate penalty on decoder atoms (columns) ---
def decoder_cosine_penalty(W: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    # W: [input_dim, D] with atoms as columns
    Wn = W / (W.norm(dim=0, keepdim=True) + 1e-8)
    cos = Wn.T @ Wn                      # [D, D]
    off_diag = cos - torch.diag(torch.diag(cos))
    off_diag_abs = off_diag.abs()
    if threshold > 0.0:
        off_diag_abs = torch.clamp(off_diag_abs - threshold, min=0.0)
    return off_diag_abs.mean()         # encourage incoherence (small off-diagonals)


@torch.no_grad()
def duplicate_metrics(W: torch.Tensor, cos_thresh: float = 0.9):
    Wn = W / (W.norm(dim=0, keepdim=True) + 1e-8)
    cos = Wn.T @ Wn
    D = cos.shape[0]
    off = cos - torch.diag(torch.diag(cos))
    upper = torch.triu(off, diagonal=1)
    dup_pairs = (upper > cos_thresh).sum().item()
    total_pairs = D * (D - 1) // 2
    dup_ratio = dup_pairs / max(1, total_pairs)
    return dup_pairs, dup_ratio, off.abs().mean().item()


def _get_decoder_weight(sae) -> Optional[torch.Tensor]:
    candidates = [
        getattr(sae, "decoder", None),
        getattr(sae, "decode", None),
        getattr(sae, "proj_out", None),
        getattr(sae, "W_dec", None),
        getattr(sae, "W", None),
    ]
    for mod in candidates:
        if mod is None:
            continue
        if isinstance(mod, torch.Tensor):
            return mod
        if hasattr(mod, "weight"):
            return mod.weight
    return None


def _get_encoder_weight(sae) -> Optional[torch.Tensor]:
    enc = getattr(sae, "encoder", None)
    if enc is None:
        return None
    if isinstance(enc, torch.Tensor):
        return enc
    if hasattr(enc, "weight"):
        return enc.weight
    return None


def _get_encoder_bias(sae) -> Optional[torch.Tensor]:
    enc = getattr(sae, "encoder", None)
    if enc is None:
        return None
    if hasattr(enc, "bias"):
        return enc.bias
    return None


class ActivationFrequencyWindow:
    """Track per-feature activation frequency over a fixed window of steps."""

    def __init__(self, code_dim: int, window_steps: int) -> None:
        self.window_steps = max(1, int(window_steps))
        self._counts: deque[torch.Tensor] = deque()
        self._totals: deque[int] = deque()
        self._sum_counts = torch.zeros(code_dim, dtype=torch.float32)
        self._sum_total = 0

    def update(self, z: torch.Tensor, *, act_threshold: float = 0.0) -> None:
        with torch.no_grad():
            act = (z > act_threshold).float()
            counts = act.sum(dim=0).detach().cpu()
            total = int(z.shape[0])
        if len(self._counts) >= self.window_steps:
            old_counts = self._counts.popleft()
            old_total = self._totals.popleft()
            self._sum_counts -= old_counts
            self._sum_total -= old_total
        self._counts.append(counts)
        self._totals.append(total)
        self._sum_counts += counts
        self._sum_total += total

    def rates(self) -> Optional[torch.Tensor]:
        if self._sum_total <= 0:
            return None
        return self._sum_counts / float(self._sum_total)

    def ready(self) -> bool:
        return len(self._counts) >= self.window_steps


def _zero_optimizer_state_for_param(
    optimiser: optim.Optimizer,
    param: torch.nn.Parameter,
    *,
    row_idx: Optional[int] = None,
    col_idx: Optional[int] = None,
) -> None:
    state = optimiser.state.get(param)
    if not state:
        return
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if v.shape == param.shape:
            if row_idx is not None and col_idx is None:
                v[row_idx, ...] = 0
            elif col_idx is not None and row_idx is None:
                v[..., col_idx] = 0
            elif row_idx is not None and col_idx is not None:
                v[row_idx, col_idx] = 0
        elif v.ndim == 1 and row_idx is not None and v.shape[0] == param.shape[0]:
            v[row_idx] = 0


def _quantile_1d(x: torch.Tensor, q: float) -> torch.Tensor:
    q = float(q)
    if hasattr(torch, "quantile"):
        return torch.quantile(x, q)
    # Fallback for older torch versions
    k = max(1, int(math.ceil(q * x.numel())))
    return x.kthvalue(k).values


@torch.no_grad()
def resample_dead_features(
    sae,
    batch: torch.Tensor,
    dead_indices:Union[List[int], torch.Tensor],
    *,
    alpha: float = 1.0,
    target_activation_rate: float = 0.05,
    optimiser: Optional[optim.Optimizer] = None,
    residual_norm: str = "l2",
    per_feature_residual: bool = True,
) -> Dict[str, Any]:
    """
    Reinitialize dead features using the residual of the worst-reconstructed sample.

    Procedure (per dead unit j):
      - compute residual R = X - X_hat on a batch X
      - choose i = argmax ||R_i||
      - set decoder column j to normalized R_i
      - set encoder row j to alpha * decoder_col^T
      - set encoder bias j so activation rate on X is ~target_activation_rate
      - zero optimiser state for touched params

    If per_feature_residual is True, residuals are recomputed per feature.
    """
    if isinstance(dead_indices, torch.Tensor):
        dead_list = dead_indices.flatten().tolist()
    else:
        dead_list = list(dead_indices)
    if not dead_list:
        return {"resampled": 0}

    encW = _get_encoder_weight(sae)
    encb = _get_encoder_bias(sae)
    decW = _get_decoder_weight(sae)
    if encW is None or encb is None or decW is None:
        raise ValueError("SAE must expose encoder/decoder weights and encoder bias.")

    device = next(sae.parameters()).device
    x = batch.to(device)

    # clamp target activation rate away from extremes
    p = float(target_activation_rate)
    p = min(max(p, 1e-6), 1.0 - 1e-6)

    use_circuits = getattr(sae, "use_circuits_implementation", False)
    if use_circuits and hasattr(sae, "decoder") and hasattr(sae.decoder, "bias") and sae.decoder.bias is not None:
        x_line = x - sae.decoder.bias
    else:
        x_line = x

    def _compute_atom() -> Tuple[torch.Tensor, int]:
        recon, _ = sae(x)
        residual = x - recon
        if residual_norm == "l1":
            scores = residual.abs().sum(dim=1)
        else:
            scores = residual.pow(2).sum(dim=1)
        idx = int(torch.argmax(scores).item())
        r = residual[idx]
        r_norm = r.norm().clamp_min(1e-8)
        return (r / r_norm).detach(), idx

    sample_index = None
    atom = None
    for j in dead_list:
        if per_feature_residual or atom is None:
            atom, sample_index = _compute_atom()
        # decoder column
        decW[:, j] = atom
        # encoder row
        encW[j, :] = alpha * atom

        # bias to hit target activation rate p on this batch
        preact = x_line @ encW[j, :].t()
        q = _quantile_1d(preact, 1.0 - p)
        encb[j] = -q

        if optimiser is not None:
            _zero_optimizer_state_for_param(optimiser, encW, row_idx=j)
            _zero_optimizer_state_for_param(optimiser, encb, row_idx=j)
            _zero_optimizer_state_for_param(optimiser, decW, col_idx=j)

    return {"resampled": len(dead_list), "sample_index": sample_index}


def train_sae(
    sae,
    latent_tensor: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,

    # Sparsity / activation regularization:
    # For hard TopK / BatchTopK you typically keep l1_lambda=0.0.
    l1_lambda: float = 0.0,
    z_l2_lambda: float = 0.0,   # useful for TopK/BatchTopK to control activation blow-up

    # Anti-duplicate penalty controls:
    w_cos: float = 0.0,         # set >0 to enable (e.g., 1e-3)
    cos_threshold: float = 0.0, # soft margin inside cosine penalty (e.g., 0.2)

    # Metrics thresholds:
    dead_thresh: float = 1e-3,  # unit considered "dead" if active fraction < dead_thresh per epoch
    dup_cos_thresh: float = 0.9,

    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose_every: int = 10,

    # Dead feature resampling (disabled if resample_every <= 0)
    resample_every: int = 0,
    resample_window: int = 200,
    resample_dead_threshold: Optional[float] = None,
    resample_max_features: int = 0,
    resample_batch_size: Optional[int] = None,
    resample_alpha: float = 1.0,
    resample_target_rate: float = 0.05,
    resample_act_threshold: float = 0.0,
    resample_use_current_batch: bool = False,
    resample_residual_norm: str = "l2",
    resample_verbose: bool = False,
) -> None:
    """
    Optimise reconstruction (+ optional L1, optional z-L2, optional decoder incoherence).

    Notes:
    - For hard TopK/BatchTopK SAEs: keep l1_lambda = 0.0, optionally use z_l2_lambda > 0.
    - For plain (soft) SAEs: you may set l1_lambda > 0 to induce sparsity.
    - Dead feature resampling can be enabled with resample_every > 0.
    """
    sae.to(device)
    dataset = TensorDataset(latent_tensor)
    pin_memory = device.startswith("cuda") and not latent_tensor.is_cuda
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_memory)

    criterion = nn.MSELoss(reduction="mean")
    optimiser = optim.Adam(sae.parameters(), lr=lr)

    # locate decoder weights (for cosine penalty & duplicate metrics)
    decW = _get_decoder_weight(sae)
    if w_cos > 0.0 and decW is None:
        print("[WARN] w_cos > 0 but decoder weights not found; disabling cosine penalty.")
        w_cos = 0.0

    # dead-feature resampling setup
    resample_enabled = resample_every is not None and int(resample_every) > 0
    resample_dead_threshold = dead_thresh if resample_dead_threshold is None else float(resample_dead_threshold)
    resample_window = max(1, int(resample_window))
    act_window: Optional[ActivationFrequencyWindow] = None
    resample_loader = None
    resample_iter = None
    if resample_enabled:
        encW = _get_encoder_weight(sae)
        if encW is None:
            print("[WARN] resample enabled but encoder weights not found; disabling resampling.")
            resample_enabled = False
        else:
            act_window = ActivationFrequencyWindow(encW.shape[0], resample_window)
            if not resample_use_current_batch:
                resample_bsz = batch_size if resample_batch_size is None else int(resample_batch_size)
                resample_loader = DataLoader(dataset, batch_size=resample_bsz, shuffle=True, drop_last=False, pin_memory=pin_memory)
                resample_iter = iter(resample_loader)

    N = len(loader.dataset)

    global_step = 0
    for epoch in range(epochs):
        sae.train()

        running_total = 0.0
        running_recon = 0.0
        running_l1 = 0.0
        running_zl2 = 0.0
        running_cos = 0.0

        active_counts = None
        total_seen = 0

        # Track mean L0 (avg number of active codes per sample)
        l0_sum = 0.0

        for (batch,) in loader:
            batch = batch.to(device, non_blocking=pin_memory)
            optimiser.zero_grad(set_to_none=True)

            recon, z = sae(batch)  # expect z already sparse if TopK/BatchTopK SAE

            # Reconstruction
            recon_loss = criterion(recon, batch)
            loss = recon_loss

            # Optional L1 sparsity (typically OFF for hard TopK/BatchTopK)
            if l1_lambda > 0.0:
                l1_term = z.abs().mean()
                loss = loss + l1_lambda * l1_term
            else:
                l1_term = None

            # Optional activation magnitude control (often helpful for hard gating)
            if z_l2_lambda > 0.0:
                zl2_term = (z.pow(2)).mean()
                loss = loss + z_l2_lambda * zl2_term
            else:
                zl2_term = None

            # Optional decoder cosine penalty (anti-duplicate)
            if w_cos > 0.0 and decW is not None:
                cos_pen = decoder_cosine_penalty(decW, threshold=cos_threshold)
                loss = loss + w_cos * cos_pen
            else:
                cos_pen = None

            loss.backward()
            optimiser.step()
            if hasattr(sae, "renorm_decoder_"):
                sae.renorm_decoder_()

            global_step += 1
            if resample_enabled and act_window is not None:
                act_window.update(z, act_threshold=resample_act_threshold)
                if (global_step % resample_every == 0) and act_window.ready():
                    rates = act_window.rates()
                    if rates is not None:
                        dead_idx = torch.nonzero(rates < resample_dead_threshold, as_tuple=False).flatten()
                        if dead_idx.numel() > 0:
                            if resample_max_features and dead_idx.numel() > resample_max_features:
                                perm = torch.randperm(dead_idx.numel())[:resample_max_features]
                                dead_idx = dead_idx[perm]
                            if resample_use_current_batch:
                                info = resample_dead_features(
                                    sae,
                                    batch,
                                    dead_idx,
                                    alpha=resample_alpha,
                                    target_activation_rate=resample_target_rate,
                                    optimiser=optimiser,
                                    residual_norm=resample_residual_norm,
                                )
                                if resample_verbose:
                                    print(f"[Resample] step {global_step} | resampled {info['resampled']}")
                            else:
                                try:
                                    (rbatch,) = next(resample_iter)
                                except StopIteration:
                                    resample_iter = iter(resample_loader)
                                    (rbatch,) = next(resample_iter)
                                info = resample_dead_features(
                                    sae,
                                    rbatch,
                                    dead_idx,
                                    alpha=resample_alpha,
                                    target_activation_rate=resample_target_rate,
                                    optimiser=optimiser,
                                    residual_norm=resample_residual_norm,
                                    per_feature_residual=False,
                                )
                                if resample_verbose:
                                    print(f"[Resample] step {global_step} | resampled {info.get('resampled', 0)}")

            bsz = batch.size(0)

            # Track epoch sums for averages
            running_total += loss.item() * bsz
            running_recon += recon_loss.item() * bsz

            if l1_term is not None:
                running_l1 += (l1_lambda * l1_term).item() * bsz

            if zl2_term is not None:
                running_zl2 += (z_l2_lambda * zl2_term).item() * bsz

            if cos_pen is not None:
                running_cos += (w_cos * cos_pen).item() * bsz

            # Dead% + L0 stats
            with torch.no_grad():
                act = (z > 0).float()  # [B, D] assumes ReLU-ish codes; OK for hard-gated nonneg codes
                batch_counts = act.sum(dim=0)  # [D]
                active_counts = batch_counts if active_counts is None else (active_counts + batch_counts)

                total_seen += z.shape[0]
                l0_sum += act.sum(dim=1).float().sum().item()

        # Epoch-end averages
        avg_total = running_total / N
        avg_recon = running_recon / N
        avg_l1 = running_l1 / N
        avg_zl2 = running_zl2 / N
        avg_cos = running_cos / N
        avg_l0 = l0_sum / max(1.0, float(N))

        dead_pct = 0.0
        if active_counts is not None and total_seen > 0:
            per_unit_rate = (active_counts / float(total_seen)).to(torch.float32)  # [D]
            dead_pct = (per_unit_rate < dead_thresh).float().mean().item()

        dup_pairs = dup_ratio = mean_abs_off = 0.0
        if decW is not None:
            with torch.no_grad():
                W_now = _get_decoder_weight(sae)
                if W_now is not None:
                    dup_pairs, dup_ratio, mean_abs_off = duplicate_metrics(W_now, cos_thresh=dup_cos_thresh)

        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0 or epoch == epochs - 1):
            msg = (
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"total {avg_total:.10f}  recon {avg_recon:.10f}"
            )
            if l1_lambda > 0.0:
                msg += f"  l1 {avg_l1:.10f}"
            if z_l2_lambda > 0.0:
                msg += f"  zL2 {avg_zl2:.10f}"
            if w_cos > 0.0:
                msg += f"  cos_pen {avg_cos:.6f}"
            msg += f"  dead% {100*dead_pct:5.2f}  L0 {avg_l0:.3f}"
            if decW is not None:
                msg += f"  dup_pairs {dup_pairs}  dup_ratio {dup_ratio:.4f}  |offdiag| {mean_abs_off:.4f}"
            print(msg)

def save_checkpoint(
    sae: SparseAutoencoder,
    path: Path,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model + meta to a file."""
    meta = dict(meta or {})
    # Persist model type/config so load_checkpoint can restore gated variants.
    meta.setdefault("sae_type", getattr(sae, "sae_type", "sae"))
    if hasattr(sae, "k"):
        meta.setdefault("k", getattr(sae, "k"))
    if hasattr(sae, "by_abs"):
        meta.setdefault("by_abs", getattr(sae, "by_abs"))
    meta.setdefault("tied_weights", getattr(sae, "tied_weights", False))
    meta.setdefault("use_circuits_implementation", getattr(sae, "use_circuits_implementation", False))
    ckpt = {
        "state_dict": sae.state_dict(),
        "input_dim": sae.encoder.in_features,
        "code_dim": sae.encoder.out_features,
        "meta": meta,
    }
    torch.save(ckpt, Path(path))
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: Path, device: str = "cpu") -> SparseAutoencoder:
    """Load a checkpoint and return a ready-to-use model."""
    ckpt = torch.load(Path(path), map_location=device)
    meta = ckpt.get("meta") or {}
    sae_type = meta.get("sae_type") or ckpt.get("sae_type") or "sae"
    input_dim = ckpt["input_dim"]
    code_dim = ckpt["code_dim"]
    if sae_type in {"batch_top_k_sae", "batch_top_k", "BatchTopKSparseAutoencoder"}:
        model = BatchTopKSparseAutoencoder(
            input_dim=input_dim,
            code_dim=code_dim,
            k=meta.get("k", 32),
            by_abs=meta.get("by_abs", False),
            tied_weights=meta.get("tied_weights", False),
            use_circuits_implementation=meta.get("use_circuits_implementation", False),
        )
    elif sae_type in {"top_k_sae", "top_k", "TopKSparseAutoencoder"}:
        model = TopKSparseAutoencoder(
            input_dim=input_dim,
            code_dim=code_dim,
            k=meta.get("k", 32),
            tied_weights=meta.get("tied_weights", False),
            use_circuits_implementation=meta.get("use_circuits_implementation", False),
        )
    else:
        model = SparseAutoencoder(
            input_dim=input_dim,
            code_dim=code_dim,
            tied_weights=meta.get("tied_weights", False),
            use_circuits_implementation=meta.get("use_circuits_implementation", False),
        )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def _shannon_entropy(weights: torch.Tensor) -> float:
    """Return Shannon entropy (base‑*e*) of a non‑negative weight vector."""
    tot = weights.sum()
    if tot == 0:
        return float("nan")
    probs = weights / tot
    # add epsilon to avoid log(0)
    entropy = -(probs * (probs + 1e-12).log()).sum().item()
    return entropy / math.log(2)  # convert to bits

def activation_summary_stats(
    acts : torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.0,
    top_k: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[int, Dict[str, float]]:
    
    labels = labels.cpu()
    # Determine if labels are numeric (for std‑dev)
    numeric_labels: Optional[torch.Tensor] = None
    if labels.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float64):
        numeric_labels = labels.float()

    N, C = acts.shape
    stats: Dict[int, Dict[str, float]] = {}

    for cid in range(C):
        z = acts[:, cid]

        # Active sample mask --------------------------------------------------
        if top_k is not None:
            # Guard against top_k > N
            k = min(top_k, N)
            idx = torch.topk(z, k, largest=True).indices
            active_mask = torch.zeros_like(z, dtype=torch.bool)
            active_mask[idx] = True
        else:
            active_mask = z > threshold
        num_active = int(active_mask.sum())
        sparsity = num_active / N

        if num_active == 0:
            stats[cid] = {
                "sparsity": sparsity,
                "mean_activation": 0.0,
                "label_entropy": float("nan"),
                "label_std": float("nan"),
            }
            continue

        pos_acts = z[active_mask]
        mean_act = float(pos_acts.mean())

        # Label entropy -------------------------------------------------------
        lbl_subset = labels[active_mask]
        uniq_lbl, inverse = lbl_subset.unique(return_inverse=True)
        sums = torch.zeros_like(uniq_lbl, dtype=torch.float)
        sums = sums.scatter_add(0, inverse, pos_acts)
        entropy = _shannon_entropy(sums)

        # Label standard deviation -------------------------------------------
        if numeric_labels is not None:
            std = float(numeric_labels[active_mask].std())
        else:
            std = float("nan")

        stats[cid] = {
            "sparsity": sparsity,
            "mean_activation": mean_act,
            "label_entropy": entropy,
            "label_std": std,
        }
    return stats


def concept_summary_stats(
    sae: "SparseAutoencoder",
    latent_tensor: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.0,
    top_k: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[int, Dict[str, float]]:
    """Compute Lim‑et‑al metrics for **every** concept.

    Definitions (identical to the paper):
        • *Sparsity*  = fraction of *all* samples whose activation > 0. High is common or uninterpretable.
        • *Mean activation*  = average of **positive** activation values. High is meaningful concept.
        • *Label entropy*  = Shannon entropy over labels where each label is
          weighted by the *sum* of its positive activations (Eq.(2)).
        • *Label std‑dev*  = standard deviation of **numeric** labels among
          the activated samples (NaN if labels are not numeric).

    Parameters
    ----------
    threshold : float, optional
        Minimum activation value for a sample to be considered "active" if
        *top_k* is *None*.
    top_k : int, optional
        If provided, ignore *threshold* and use the *k* highest‑activation
        samples per concept instead.
    """
    sae.eval()
    acts = sae.encode(latent_tensor.to(device)).cpu()  # (N, C)
    labels = labels.cpu()

    # Determine if labels are numeric (for std‑dev)
    numeric_labels: Optional[torch.Tensor] = None
    if labels.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float64):
        numeric_labels = labels.float()

    N, C = acts.shape
    stats: Dict[int, Dict[str, float]] = {}

    for cid in range(C):
        z = acts[:, cid]

        # Active sample mask --------------------------------------------------
        if top_k is not None:
            # Guard against top_k > N
            k = min(top_k, N)
            idx = torch.topk(z, k, largest=True).indices
            active_mask = torch.zeros_like(z, dtype=torch.bool)
            active_mask[idx] = True
        else:
            active_mask = z > threshold
        num_active = int(active_mask.sum())
        sparsity = num_active / N

        if num_active == 0:
            stats[cid] = {
                "sparsity": sparsity,
                "mean_activation": 0.0,
                "label_entropy": float("nan"),
                "label_std": float("nan"),
            }
            continue

        pos_acts = z[active_mask]
        mean_act = float(pos_acts.mean())

        # Label entropy -------------------------------------------------------
        lbl_subset = labels[active_mask]
        uniq_lbl, inverse = lbl_subset.unique(return_inverse=True)
        sums = torch.zeros_like(uniq_lbl, dtype=torch.float)
        sums = sums.scatter_add(0, inverse, pos_acts)
        entropy = _shannon_entropy(sums)

        # Label standard deviation -------------------------------------------
        if numeric_labels is not None:
            std = float(numeric_labels[active_mask].std())
        else:
            std = float("nan")

        stats[cid] = {
            "sparsity": sparsity,
            "mean_activation": mean_act,
            "label_entropy": entropy,
            "label_std": std,
        }
    return stats


def rank_concepts(
    stats: Dict[int, Dict[str, float]],
    *,
    key: str = "label_entropy",
    top_n: int = 10,
    ascending: bool = False,
    return_scores: bool = False,
) -> List[Any]:
    """Return *top_n* concept IDs sorted by a chosen metric.

    Parameters
    ----------
    stats : dict
        Output from :func:`concept_summary_stats`.
    key : str, default "label_entropy"
        Which metric to sort on.
    ascending : bool, default False
        If *True*, smallest values rank highest.
    return_scores : bool, default False
        If *True*, return list of *(concept_id, score)* pairs; otherwise just
        the concept IDs.
    """
    if not stats:
        warnings.warn("Empty stats dictionary; returning empty list.")
        return []
    if key not in next(iter(stats.values())):
        raise KeyError(f"Metric '{key}' not found in stats dictionary")

# filter out NaNs before sorting
    valid_pairs = [
        (cid, m[key]) for cid, m in stats.items() if not math.isnan(m[key])
    ]
    if not valid_pairs:
        warnings.warn(f"All values are NaN for metric '{key}'.")
        return []

    ranked_pairs = sorted(valid_pairs, key=lambda kv: kv[1], reverse=not ascending)
    ranked_pairs = ranked_pairs[:top_n]
    return ranked_pairs if return_scores else [cid for cid, _ in ranked_pairs]

def query_concepts(
    stats: Dict[int, Dict[str, float]],
    *,
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]],
    sort_key: Optional[str] = None,
    ascending: bool = True,
    return_scores: bool = False,
) -> List[Any]:
    """Filter concepts using **multiple metric ranges**.

    Parameters
    ----------
    stats : dict
        Output dictionary from :func:`concept_summary_stats`.
    bounds : dict
        Mapping ``metric_name -> (lower, upper)`` where *lower* or *upper* can
        be *None* to leave that side unbounded.  Example::

            bounds = {
                "sparsity": (None, 0.05),          # sparsity ≤ 5%
                "label_entropy": (None, 1.0),      # entropy ≤ 1 bit
                "mean_activation": (1e-3, None),   # mean_act ≥ 1e-3
            }
    sort_key : str, optional
        Metric to sort by.  Defaults to the first key in ``bounds``.
    ascending : bool, default True
        Sort order (ignored if *sort_key* is *None*).
    return_scores : bool, default False
        If *True* return list of ``(concept_id, metric_dict)``; otherwise just
        the concept IDs.
    """
    if not bounds:
        raise ValueError("bounds dict cannot be empty.")
    for k in bounds:
        if k not in next(iter(stats.values())):
            raise KeyError(f"Metric '{k}' not found in stats dictionary.")

    def in_range(val: float, lo: Optional[float], hi: Optional[float]) -> bool:
        if math.isnan(val):
            return False
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    selected: List[Tuple[int, Dict[str, float]]] = []
    for cid, m in stats.items():
        if all(in_range(m[k], *bounds[k]) for k in bounds):
            selected.append((cid, m))

    if sort_key is None:
        sort_key = next(iter(bounds))  # first metric key

    selected.sort(key=lambda kv: kv[1][sort_key], reverse=not ascending)
    return (
        [(cid, mdict) for cid, mdict in selected]
        if return_scores
        else [cid for cid, _ in selected]
    )


def plot_metrics_figure(
    stats: Dict[int, Dict[str, float]],
    *,
    figsize: Tuple[int, int] = (10, 7),
    cmap: str = "plasma",
    show: bool = True,
    save_path: Optional[Path] = None,
    shapes = None
) -> None:
    """Replicate Lim‑et‑al style figure.

    Scatter of log10 Activated Frequency (x) vs log10 Mean Activation (y),
    colour‑coded by Label Entropy, with marginal histograms.
    """
    # Assemble data -----------------------------------------------------
    cids = []
    freq = []
    mean_act = []
    ent = []
    for cid, m in stats.items():
        if any(math.isnan(m[k]) for k in ("sparsity", "mean_activation", "label_entropy")):
            continue
        # Skip zero activations to avoid -inf
        if m["sparsity"] <= 0 or m["mean_activation"] <= 0:
            continue
        cids.append(cid)
        freq.append(m["sparsity"])
        mean_act.append(m["mean_activation"])
        ent.append(m["label_entropy"])

    if not cids:
        warnings.warn("No finite data to plot.")
        return

    freq = np.log10(np.array(freq))
    mean_act = np.log10(np.array(mean_act))
    ent = np.array(ent)

    # Figure layout -----------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1.2], height_ratios=[1.2, 5], hspace=0.05, wspace=0.05)
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    ax_scatter = fig.add_subplot(gs[1, 0])

    # Main scatter ------------------------------------------------------
    if shapes == None:
        shapes = ["."]*len(stats)
    #sc = ax_scatter.scatter(freq, mean_act, c=ent, cmap=cmap, s=10, alpha=0.8,marker=shapes)
    for lab in np.unique(shapes):
        idx = (np.array(shapes)[cids] == lab)
        sc = ax_scatter.scatter(freq[idx], mean_act[idx], c=ent[idx], cmap=cmap, marker=lab, s=20, alpha=0.8)

    ax_scatter.set_xlabel("Log Activated Frequency")
    ax_scatter.set_ylabel("Log Mean Activation Value")
    ax_scatter.grid(True, linestyle="--", alpha=0.3)

    # Marginal histograms ----------------------------------------------
    ax_histx.hist(freq, bins=60, color="steelblue")
    ax_histy.hist(mean_act, bins=60, orientation="horizontal", color="steelblue")
    ax_histx.axis("off")
    ax_histy.axis("off")

    # Colorbar ----------------------------------------------------------
    cbar = fig.colorbar(sc, ax=[ax_scatter, ax_histy,ax_histx], pad=0.01, label="Label Entropy (bits)")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Metrics figure saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def most_activated_indices(
    sae: SparseAutoencoder,
    latent_tensor: torch.Tensor,
    concept_idx: int,
    *,
    k: int = 25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    bottom = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    sae.eval()
    largest = not bottom
    with torch.no_grad():
        codes = sae.encode(latent_tensor.to(device))
        activations = codes[:, concept_idx].cpu()
        k_eff = max(1, min(int(k), activations.numel()))
        topk = torch.topk(activations, k_eff, largest=largest, sorted=True)
    return topk.indices, topk.values


def _topk_similarity_matrix(
    acts_a: torch.Tensor,
    acts_b: torch.Tensor,
    *,
    k: int = 50,
    metric: str = "jaccard",
    threshold: Optional[float] = None,
    act_threshold: float = 0.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> np.ndarray:
    """
    Compare two activation matrices by overlap of per-concept top-k active samples.
    Returns a dense [Da, Db] similarity matrix.
    """
    if metric not in {"jaccard", "overlap"}:
        raise ValueError("metric must be 'jaccard' or 'overlap'.")

    a = _to_numpy(acts_a)
    b = _to_numpy(acts_b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("acts_a and acts_b must be 2D arrays [N, D].")
    if a.shape[0] != b.shape[0]:
        raise ValueError("acts_a and acts_b must have the same number of rows (N).")

    n, da = a.shape
    _, db = b.shape
    k_eff = max(1, min(int(k), n))

    def _membership(x: np.ndarray) -> np.ndarray:
        m = np.zeros((n, x.shape[1]), dtype=np.uint8)
        for j in range(x.shape[1]):
            vals = x[:, j]
            active_idx = np.flatnonzero(vals > act_threshold)
            if active_idx.size == 0:
                continue
            if active_idx.size > k_eff:
                active_vals = vals[active_idx]
                top_local = np.argpartition(active_vals, -k_eff)[-k_eff:]
                idx = active_idx[top_local]
            else:
                idx = active_idx
            m[idx, j] = 1
        return m

    mem_a = _membership(a)
    mem_b = _membership(b)

    inter = mem_a.T.astype(np.int32) @ mem_b.astype(np.int32)
    sizes_a = mem_a.sum(axis=0, dtype=np.int32)[:, None]
    sizes_b = mem_b.sum(axis=0, dtype=np.int32)[None, :]

    if metric == "jaccard":
        denom = sizes_a + sizes_b - inter
    else:
        denom = np.minimum(sizes_a, sizes_b)

    scores = np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float32), where=denom > 0)
    return scores.astype("float16")


def most_activated_images(image_paths: List[Path], indices: torch.Tensor) -> List[Path]:
    return [image_paths[i] for i in indices]


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rank_transform_cols(x: np.ndarray) -> np.ndarray:
    ranked = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        ranked[:, j] = rankdata(x[:, j], method="average")
    return ranked


def _corr_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom_a = np.sqrt((a * a).sum(axis=0, keepdims=True))
    denom_b = np.sqrt((b * b).sum(axis=0, keepdims=True))
    denom = (denom_a.T @ denom_b)
    denom = np.where(denom == 0, 1.0, denom)
    sim = (a.T @ b) / denom
    sim = np.clip(sim, -1.0, 1.0)
    return sim.astype('float16')

def _cos_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    sim = cosine_similarity(a.T,b.T)
    return sim.astype('float16')

def _get_random_similarity_matrix(
    a: np.ndarray,
    b: np.ndarray,
    method: str = "cos",
    N: int = 10,
    fixed_sae: bool = False,
    rng: Optional[np.random.Generator] = None,
):
    rng = np.random.default_rng() if rng is None else rng
    max_similarities = np.zeros(a.shape[1], dtype=np.float32)
    a_perm = a.copy()
    for _ in range(N):
        perm = rng.permutation(a_perm.shape[0])
        a_perm = a_perm[perm]
        if method == "cos":
            max_sim = _cos_similarity_matrix(a_perm, b)
        else:
            max_sim = _corr_similarity_matrix(a_perm, b)

        if fixed_sae:
            max_similarities += np.diag(max_sim)
        else:
            max_similarities += max_sim.max(0)

    return (max_similarities / max(1, N)).astype("float16")


def _hungarian_from_similarity(
    sim: np.ndarray,
    *,
    allow_unmatched: bool,
    dummy_sim: float,
    min_sim: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    da, db = sim.shape
    if allow_unmatched:
        size = max(da, db)
        sim_pad = np.full((size, size), dummy_sim, dtype=float)
        sim_pad[:da, :db] = sim
        cost = -sim_pad
        row_ind, col_ind = linear_sum_assignment(cost)
        match_ids = np.full(da, -1, dtype=int)
        match_scores = np.full(da, dummy_sim, dtype=float)
        for r, c in zip(row_ind, col_ind):
            if r < da:
                if c < db and (min_sim is None or sim_pad[r, c] >= min_sim):
                    match_ids[r] = c
                    match_scores[r] = sim_pad[r, c]
    else:
        cost = -sim
        row_ind, col_ind = linear_sum_assignment(cost)
        match_ids = np.full(da, -1, dtype=int)
        match_scores = np.full(da, np.nan, dtype=float)
        for r, c in zip(row_ind, col_ind):
            match_ids[r] = c
            match_scores[r] = sim[r, c]
        if min_sim is not None:
            low = match_scores < min_sim
            match_ids[low] = -1
    return match_ids, match_scores

#def align_concepts_max(


#):


def align_concepts_hungarian(
    sim: np.array,
    *,
    rank_transform: Optional[bool] = None,
    zscore: bool = True,
    allow_unmatched: bool = True,
    dummy_sim: float = 0.0,
    min_sim: Optional[float] = 0.2,
    thresholds: Tuple[float, float] = (0.7, 0.4),
    **kwargs
) -> Dict[str, Any]:
    """
    Align concepts across two activation matrices using Hungarian matching.

    acts_a/acts_b are activation matrices with shape [N, D].
    """
    da, db = sim.shape

    match_ids, match_scores = _hungarian_from_similarity(
        sim,
        allow_unmatched=allow_unmatched,
        dummy_sim=dummy_sim,
        min_sim=min_sim,
    )

    matched_mask = match_ids >= 0
    matched_scores = match_scores[matched_mask]
    if matched_scores.size:
        median = float(np.median(matched_scores))
        q1 = float(np.percentile(matched_scores, 25))
        q3 = float(np.percentile(matched_scores, 75))
    else:
        median = q1 = q3 = float("nan")

    summary = {
        "mean_matched": float(np.mean(matched_scores)) if matched_scores.size else float("nan"),
        "median_matched": median,
        "iqr_matched": float(q3 - q1) if matched_scores.size else float("nan"),
        "matched_fraction": float(matched_mask.mean()) if match_ids.size else float("nan"),
    }
    for t in thresholds:
        summary[f"frac_ge_{t}"] = float(np.mean(matched_scores >= t)) if matched_scores.size else float("nan")

    out = {
        "match_ids": match_ids,
        "match_scores": match_scores,
        "summary": summary,
    }
    return out


def get_saes_similarities(
    sae_a: SparseAutoencoder,
    latent_tensor_a: torch.Tensor,
    acts_b: torch.Tensor,
    fixed_sae: bool = False,
    metric = "jaccard",
    method = "spearman",
    random_state: Optional[int] = None,
    rank_transform: Optional[bool] = None,
    zscore: bool = False,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs: Any,
) -> Dict[str, Any]:
    sae_a.eval()
    with torch.no_grad():
        acts_a = sae_a.encode(latent_tensor_a.to(device)).cpu()

    rng = np.random.default_rng(random_state)

    a = _to_numpy(acts_a)
    b = _to_numpy(acts_b)

    sim_cos = _cos_similarity_matrix(a.copy(), b.copy())
    random_cos = _get_random_similarity_matrix(a.copy(), b.copy(), method="cos", N=50, fixed_sae=fixed_sae, rng=rng)

    sim_topk = _topk_similarity_matrix(a, b, metric=metric, **kwargs)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("acts_a and acts_b must be 2D arrays [N, D].")
    if a.shape[0] != b.shape[0]:
        raise ValueError("acts_a and acts_b must have the same number of rows (N).")

    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'.")

    if rank_transform is None:
        rank_transform = (method == "spearman")

    if rank_transform:
        a_use = _rank_transform_cols(a.copy())
        b_use = _rank_transform_cols(b.copy())
    else:
        a_use = a.copy()
        b_use = b.copy()

    if zscore:
        a_use = (a_use - a_use.mean(axis=0, keepdims=True)) / (a_use.std(axis=0, keepdims=True) + 1e-12)
        b_use = (b_use - b_use.mean(axis=0, keepdims=True)) / (b_use.std(axis=0, keepdims=True) + 1e-12)

    sim_corr = _corr_similarity_matrix(a_use.copy(), b_use.copy())
    random_corr = _get_random_similarity_matrix(a_use.copy(), b_use.copy(), method="cor", N=50, fixed_sae=fixed_sae, rng=rng)
    #hungarian_res = align_concepts_hungarian(sim_corr, **kwargs)
    return {"cos_sim" : sim_cos, "cor_sim" : sim_corr, "topk_sim" : sim_topk,
     "cos_random":random_cos, "cor_random" : random_corr}


def plot_dictionary_atoms(
    sae: SparseAutoencoder,
    *,
    num_atoms: int = 16,
    atoms_per_row: int = 4,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    atom_indexes_bool = False,
    atom_indexes = None
) -> None:
    """Visualise selected dictionary atoms (decoder columns) as 1‑D bar plots."""
    
    decW = _get_decoder_weight(sae)
    if decW is None:
        raise ValueError("Could not find decoder weight matrix for this SAE.")
    num_atoms = min(num_atoms, int(decW.shape[1]))
    cols = atoms_per_row
    rows = math.ceil(num_atoms / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for i in range(num_atoms):
        r, c = divmod(i, cols)
        if atom_indexes_bool:
            i = atom_indexes[i]

        atom = sae.concept_vector(i).cpu().numpy()
        axes[r][c].bar(range(len(atom)), atom)
        axes[r][c].set_title(f"Atom {i}")
        axes[r][c].set_xticks([])
    for ax in axes.flat[num_atoms:]:
        ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Dictionary figure saved to {save_path}")
    else:
        plt.show()


def plot_top_k_images(
    image_paths: List[Path],
    data,
    *,
    cols: int = 5,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    top_act = None,
    stats = None,
    neuron_nr = None
) -> None:
    """Display the images corresponding to the given indices in a grid."""
    paths = []
    for name in image_paths:
        p = int(name.split("/")[2].split("_")[0])
        paths.append(p)
    rows = math.ceil(len(paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for idx, p in enumerate(paths):
        r, c = divmod(idx, cols)
        try:
            img = data[p]
            axes[r][c].imshow(img)
            axes[r][c].set_title( str(np.round(top_act[idx],5)), fontsize=8)
            axes[r][c].axis("off")
        except Exception as e:
            axes[r][c].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center")
            axes[r][c].axis("off")
    # hide any empty axes
    for ax in axes.flat[len(paths):]:
        ax.axis("off")

    plt.suptitle(f'Sparse Neuron {neuron_nr}, sparsity: {np.round(stats["sparsity"],2)}, mean_act: {np.round(stats["mean_activation"],2)}, entropy: {np.round(stats["label_entropy"],2)}')
    plt.tight_layout()


    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Image grid saved to {save_path}")
    else:
        plt.show()


