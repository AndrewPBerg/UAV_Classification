import os
from typing import Dict
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def save_fisher_heatmap(
    fisher: Dict[str, torch.Tensor],
    output_dir: str,
    title: str = "Fisher Information Heatmap",
    filename: str = "fim_heatmap.png",
) -> str:
    """Save a heatmap visualising per-parameter Fisher scores.

    For each parameter we compute the mean of its diagonal Fisher entries and
    plot these as a single-column heatmap (parameter names on y-axis).

    Returns the filepath of the saved PNG.
    """

    if not fisher:
        raise ValueError("Fisher dictionary is empty – nothing to visualise.")

    # Compute scalar score per parameter (mean of diagonal elements)
    names = []
    values = []
    for name, tensor in fisher.items():
        names.append(name)
        values.append(tensor.mean().item())

    # Convert to 2-D array for seaborn heatmap (shape: len(names) x 1)
    import numpy as np
    data = np.array(values).reshape(-1, 1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    plt.figure(figsize=(4, max(4, len(names) * 0.25)))
    ax = sns.heatmap(
        data,
        yticklabels=names,
        xticklabels=["Fisher"],
        cmap="viridis",
        cbar=True,
        annot=True,
        fmt=".2e",
        annot_kws={"size": 8},
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()
    return filepath 