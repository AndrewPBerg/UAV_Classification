import os
from typing import Dict, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def save_fisher_heatmap(
    fisher: Dict[str, torch.Tensor],
    output_dir: str,
    title: str = "Fisher Information Heatmap",
    filename: str = "fim_heatmap.png",
    peft_method: Optional[str] = None,
    epoch: Optional[int] = None,
) -> str:
    """Save a heatmap visualising per-parameter Fisher scores.

    For each parameter we compute the mean of its diagonal Fisher entries and
    plot these as a single-column heatmap (parameter names on y-axis).

    Args:
        fisher: Dictionary of parameter names to Fisher information tensors
        output_dir: Directory to save the heatmap
        title: Base title for the heatmap
        filename: Name of the output file
        peft_method: Current PEFT method being used (if any)
        epoch: Current epoch number (if applicable)

    Returns:
        The filepath of the saved PNG.
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

    # Build enhanced title with PEFT and epoch information
    enhanced_title = title
    title_parts = []
    
    if epoch is not None:
        title_parts.append(f"Epoch {epoch}")
    
    if peft_method is not None:
        title_parts.append(f"PEFT: {peft_method}")
    
    if title_parts:
        enhanced_title = f"{title}\n({' | '.join(title_parts)})"

    # Calculate dynamic figure size based on content
    max_name_length = max(len(name) for name in names) if names else 10
    fig_width = max(12, max_name_length * 0.15)  # Wider figure for long parameter names
    fig_height = max(6, len(names) * 0.4)  # More height per parameter
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Set font sizes
    plt.rcParams.update({'font.size': 10})
    
    ax = sns.heatmap(
        data,
        yticklabels=names,
        xticklabels=["Fisher Information"],
        cmap="viridis",
        cbar=True,
        annot=True,
        fmt=".2e",
        annot_kws={"size": 12, "weight": "bold"},  # Larger, bold annotation text
        cbar_kws={"shrink": 0.8}  # Slightly smaller colorbar
    )
    
    # Improve text visibility
    ax.set_title(enhanced_title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Parameter Names", fontsize=12, fontweight="bold")
    ax.set_xlabel("Fisher Information Score", fontsize=12, fontweight="bold")
    
    # Rotate y-axis labels for better readability
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=11)
    
    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.3)  # More space for parameter names
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset font parameters to avoid affecting other plots
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
    
    return filepath 