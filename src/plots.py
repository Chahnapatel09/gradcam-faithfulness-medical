import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
STRATEGY_COLORS = {"targeted": "#d62728", "random": "#7f7f7f"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def unnormalize(image_tensor):
    """(1, C, H, W) normalized tensor -> (H, W, C) numpy array in [0, 1]."""
    img = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def plot_deletion_curves(curves, metric="mean_accuracy", save_path=None):
    """One subplot per CAM method: metric vs. % pixels masked, targeted vs random."""
    cam_methods = sorted(curves["cam_method"].unique())
    fig, axes = plt.subplots(1, len(cam_methods), figsize=(5 * len(cam_methods), 4), sharey=True)
    if len(cam_methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, cam_methods):
        subset = curves[curves["cam_method"] == method]
        for strategy, color in STRATEGY_COLORS.items():
            line = subset[subset["strategy"] == strategy].sort_values("percent")
            ax.plot(line["percent"], line[metric], marker="o", label=strategy, color=color)
        ax.set_title(method)
        ax.set_xlabel("% pixels masked")

    axes[0].set_ylabel(metric.replace("_", " "))
    axes[-1].legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_faithfulness_gap(gap_df, metric="accuracy_gap", save_path=None):
    """Grouped bar chart: gap (random - targeted) per CAM method at each mask percent."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=gap_df, x="percent", y=metric, hue="cam_method", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlabel("% pixels masked")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_masking_example(original, targeted_masked, random_masked, percent, save_path=None):
    """Original image vs. targeted-masked vs. random-masked, side by side.

    original / targeted_masked / random_masked: (1, C, H, W) normalized tensors.
    """
    images = [unnormalize(original), unnormalize(targeted_masked), unnormalize(random_masked)]
    titles = ["original", f"targeted masked ({percent}%)", f"random masked ({percent}%)"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_deletion_curves_by_split(curves, split_col, metric="mean_accuracy",
                                   cam_method="gradcam", save_path=None):
    """For one CAM method, facet by a split column (e.g. group=TP/TN/FP, or confidence_bin)."""
    subset = curves[curves["cam_method"] == cam_method]
    split_values = sorted(subset[split_col].unique())

    fig, axes = plt.subplots(1, len(split_values), figsize=(5 * len(split_values), 4), sharey=True)
    if len(split_values) == 1:
        axes = [axes]

    for ax, value in zip(axes, split_values):
        group_subset = subset[subset[split_col] == value]
        for strategy, color in STRATEGY_COLORS.items():
            line = group_subset[group_subset["strategy"] == strategy].sort_values("percent")
            ax.plot(line["percent"], line[metric], marker="o", label=strategy, color=color)
        ax.set_title(f"{split_col}={value}")
        ax.set_xlabel("% pixels masked")

    axes[0].set_ylabel(metric.replace("_", " "))
    axes[-1].legend()
    fig.suptitle(cam_method)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
