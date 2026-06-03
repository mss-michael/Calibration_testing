
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Sequence

def denormalize(normalized, basis):
    return basis.min() + (basis.max() - basis.min())*((normalized + 1)/2)

def normalize(unnormalized, basis):
    return (2 * (unnormalized - basis.min()) / (basis.max() - basis.min()) - 1)

def normalize_feat_wise(unnormalized, basis):
    return (2 * (unnormalized - basis.min(0)) / (basis.max(0) - basis.min(0)) - 1)

def denormalize_feat_wise(normalized, basis):
    return basis.min(0) + (basis.max(0) - basis.min(0))*((normalized + 1)/2)

def denormalize_std(normalized, basis):
    return normalized * basis.std(0) + basis.mean(0)

def normalize_std(unnormalized, basis):
    return (unnormalized - basis.mean(0))/basis.std(0)

def feature_normalize(X, axis=1, eps=1e-12):
    """Normalize along axis (rows if axis=1) with numerical stability. From mahalanobis ++"""
    X = np.asarray(X)
    norms = np.linalg.norm(X, ord=2, axis=axis, keepdims=True)
    return X / np.clip(norms, eps, None)

def show_images_grid(
    data,
    filenames: Sequence[str],
    max_cols: int = 10,
    titles: Optional[Sequence[str]] = None,
    figsize_per_cell: float = 2.0,
    turn_off_axes: bool = True,
    save_path: Optional[str] = None,
):
    """
    Display images in a grid with up to `max_cols` columns.

    Args:
        root: Root directory for images (can be "" if `filenames` are absolute).
        filenames: List of image paths (relative to `root`, or absolute paths).
        max_cols: Maximum number of columns in the grid.
        titles: Optional list of titles per image (same length as filenames).
        figsize_per_cell: Size multiplier for each cell in inches.
        turn_off_axes: If True, hides axes.
        save_path: If provided, saves the figure to this path.

    Returns:
        (fig, axes) from matplotlib for further customization.
    """
    
    paths = []
    infos = []
    for name in filenames:
        p = int(name.split("/")[2].split("_")[0])
        infos.append(name.split("/")[2])
        paths.append(p)

    # Load images (RGB), skipping missing ones with a warning
    imgs, kept_paths, kept_titles = [], [], []
    for i, p in enumerate(paths):
        try:
            img = data[p]
            imgs.append(img)
            kept_paths.append(infos[i])
            if titles is not None and i < len(titles):
                kept_titles.append(infos[i])
            else:
                kept_titles.append(None)
        except Exception as e:
            print(f"[warn] Could not load {p}: {e}")

    n = len(imgs)
    if n == 0:
        print("No images to display.")
        return None, None

    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig_w = max(1, cols * figsize_per_cell)
    fig_h = max(1, rows * figsize_per_cell)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes, dtype=object).reshape(rows, cols)  # handles 1D cases

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < n:
                ax.imshow(imgs[idx])
                if turn_off_axes:
                    ax.axis("off")
                if kept_titles[idx]:
                    ax.set_title(str(kept_titles[idx]), fontsize=9)
            else:
                ax.axis("off")
            idx += 1

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()

    return fig, axes
