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
from typing import List, Tuple, Optional, Dict, Any
import math
from src.base_sae import SparseAutoencoder


# --- anti-duplicate penalty on decoder atoms (columns) ---
def decoder_cosine_penalty(W: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    # W: [input_dim, D] with atoms as columns
    Wn = W / (W.norm(dim=0, keepdim=True) + 1e-8)
    cos = Wn.T @ Wn                      # [D, D]
    off_diag = cos - torch.diag(torch.diag(cos))
    if threshold > 0.0:
        off_diag = torch.clamp(off_diag - threshold, min=0.0)
    return off_diag.abs().mean()         # encourage incoherence (small off-diagonals)

# --- metrics helpers ---
@torch.no_grad()
def duplicate_metrics(W: torch.Tensor, cos_thresh: float = 0.9):
    Wn = W / (W.norm(dim=0, keepdim=True) + 1e-8)
    cos = Wn.T @ Wn
    D = cos.shape[0]
    off = cos - torch.diag(torch.diag(cos))
    # count unique pairs i<j
    upper = torch.triu(off, diagonal=1)
    dup_pairs = (upper > cos_thresh).sum().item()
    total_pairs = D * (D - 1) // 2
    dup_ratio = dup_pairs / max(1, total_pairs)
    return dup_pairs, dup_ratio, off.abs().mean().item()

def _get_decoder_weight(sae) -> torch.Tensor | None:
    # Try common names / structures
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

def train_sae(
    sae,
    latent_tensor: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    l1_lambda: float = 1e-3,
    # --- new: anti-duplicate penalty controls ---
    w_cos: float = 0.0,           # set >0 to enable (e.g., 1e-3)
    cos_threshold: float = 0.0,   # soft margin inside cosine penalty (e.g., 0.2)
    # --- metrics thresholds ---
    dead_thresh: float = 1e-3,    # unit considered "dead" if active fraction < dead_thresh per epoch
    dup_cos_thresh: float = 0.9,  # report pairs with cosine > this
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose_every: int = 10, 
) -> None:
    """Optimise reconstruction + L1 sparsity (+ optional decoder incoherence)."""
    sae.to(device)
    dataset = TensorDataset(latent_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.MSELoss(reduction="mean")
    optimiser = optim.Adam(sae.parameters(), lr=lr)

    # locate decoder weights (for cosine penalty & duplicate metrics)
    decW = _get_decoder_weight(sae)
    if w_cos > 0.0 and decW is None:
        print("[WARN] w_cos > 0 but decoder weights not found; disabling cosine penalty.")
        w_cos = 0.0

    N = len(loader.dataset)

    for epoch in range(epochs):
        sae.train()
        running_loss = 0.0
        running_l1 = 0.0
        running_cos = 0.0

        # for dead%: accumulate activation counts
        active_counts = None
        total_seen = 0

        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon, z = sae(batch)

            # core losses
            recon_loss = criterion(recon, batch)
            l1_term = z.abs().mean()

            loss = recon_loss + l1_lambda * l1_term 

            # optional decoder cosine penalty (anti-duplicate)
            if w_cos > 0.0 and decW is not None:
                cos_pen = decoder_cosine_penalty(decW, threshold=cos_threshold)
                loss = loss + w_cos * cos_pen
                running_cos += (w_cos * cos_pen).item() * batch.size(0)

            loss.backward()
            optimiser.step()

            # track sums for epoch averages
            running_loss += loss.item() * batch.size(0)
            running_l1 += (l1_lambda * l1_term).item() * batch.size(0)

            # dead% stats
            with torch.no_grad():
                act = (z > 0).float()  # [B, D]
                batch_counts = act.sum(dim=0)  # [D]
                if active_counts is None:
                    active_counts = batch_counts
                else:
                    active_counts += batch_counts
                total_seen += z.shape[0]

        # epoch-end metrics
        avg_loss = running_loss / N
        avg_l1 = running_l1 / N
        avg_cos = (running_cos / N) if w_cos > 0.0 else 0.0

        # dead units (fraction of samples with z_i>0 < dead_thresh)
        dead_pct = 0.0
        if active_counts is not None and total_seen > 0:
            per_unit_rate = (active_counts / float(total_seen)).to(torch.float32)  # [D]
            dead_pct = (per_unit_rate < dead_thresh).float().mean().item()

        # duplicate metrics from decoder
        dup_pairs = dup_ratio = mean_abs_off = 0.0
        if decW is not None:
            with torch.no_grad():
                W_now = _get_decoder_weight(sae)  # refresh handle in case it changed
                if W_now is not None:
                    dup_pairs, dup_ratio, mean_abs_off = duplicate_metrics(W_now, cos_thresh=dup_cos_thresh)

        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0 or epoch == epochs - 1):
            msg = (
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"loss {avg_loss:.6f}  l1 {avg_l1:.6f}"
            )
            if w_cos > 0.0:
                msg += f"  cos_pen {avg_cos:.6f}"
            msg += f"  dead% {100*dead_pct:5.2f}"
            if decW is not None:
                msg += f"  dup_pairs {dup_pairs}  dup_ratio {dup_ratio:.4f}  |offdiag| {mean_abs_off:.4f}"
            print(msg)

def save_checkpoint(
    sae: SparseAutoencoder,
    path: Path,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model + meta to a file."""
    ckpt = {
        "state_dict": sae.state_dict(),
        "input_dim": sae.encoder.in_features,
        "code_dim": sae.encoder.out_features,
        "meta": meta or {},
    }
    torch.save(ckpt, Path(path))
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: Path, device: str = "cpu") -> SparseAutoencoder:
    """Load a checkpoint and return a ready‑to‑use model."""
    ckpt = torch.load(Path(path), map_location=device)
    model = SparseAutoencoder(
        input_dim=ckpt["input_dim"],
        code_dim=ckpt["code_dim"],
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
        warnings.warn("All values are NaN for metric '{key}'.")
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
        topk = torch.topk(activations, k, largest=largest, sorted=True)
    return topk.indices, topk.values


def most_activated_images(image_paths: List[Path], indices: torch.Tensor) -> List[Path]:
    return [image_paths[i] for i in indices]



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
    
    num_atoms = min(num_atoms, sae.decoder.out_features)
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


