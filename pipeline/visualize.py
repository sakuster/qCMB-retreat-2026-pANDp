"""
visualize.py — Figures and heatmaps for interpreting model results.

Produces three types of output:
  1. UMAP scatter plot  — 2D projection of all sample embeddings, coloured by
                          class. Outlier clusters suggest undiscovered subtypes.
  2. GradCAM heatmaps  — Overlay on each brain slice showing which spatial
                          regions drove the model's classification decision.
  3. Confusion matrix  — How often each class was correctly identified.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; safe for HPC nodes without displays
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import umap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ---------------------------------------------------------------------------
# UMAP embedding visualisation
# ---------------------------------------------------------------------------

def plot_umap(embeddings, true_labels, label_names, ood_scores, ood_threshold_std, output_path):
    """
    Project 512-dim embeddings to 2D and plot coloured by class.
    Samples flagged as potential new subtypes are marked with a red X.

    Clusters that sit away from all labelled groups in this plot are the
    primary visual signal for undiscovered prion subtypes.
    """
    print("  Computing UMAP projection...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
    coords: np.ndarray = np.array(reducer.fit_transform(embeddings))

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("tab10", n_colors=len(label_names))

    for i, name in enumerate(label_names):
        mask = true_labels == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[palette[i]], label=name, alpha=0.75, s=70, edgecolors="white", linewidths=0.3
        )

    # Overlay OOD-flagged samples
    ood_mean = ood_scores.mean()
    ood_std  = ood_scores.std()
    ood_mask = ood_scores > (ood_mean + ood_threshold_std * ood_std)
    if ood_mask.any():
        ax.scatter(
            coords[ood_mask, 0], coords[ood_mask, 1],
            c="red", marker="X", s=180, zorder=5, linewidths=0.5,
            edgecolors="darkred", label="Potential new subtype"
        )

    ax.set_title(
        "Embedding Space (UMAP)\n"
        "Red X = potential undiscovered prion subtype  |  "
        "Separate clusters may indicate new classes",
        fontsize=11
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# GradCAM heatmaps
# ---------------------------------------------------------------------------

def save_gradcam(model, dataset, output_dir: str, device, max_images: int = 30):
    """
    For each image, generate a GradCAM heatmap showing which brain regions
    the model focused on when making its classification decision.

    Warm colours (red/yellow) indicate high-attention regions — in prion
    diagnosis, these should correspond to known anatomical sites of PrP
    accumulation for each subtype.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Target the final convolutional block of the EfficientNet backbone.
    # This layer has the richest spatial feature maps before global pooling.
    target_layer = [model.backbone[-1]]
    cam_extractor = GradCAM(model=model, target_layers=target_layer)

    n = min(max_images, len(dataset))
    print(f"  Generating GradCAM for {n} images...")

    for idx in range(n):
        img_tensor, label, filepath = dataset[idx]
        input_batch = img_tensor.unsqueeze(0).to(device)

        grayscale_cam = cam_extractor(input_tensor=input_batch)[0]

        # Reconstruct the original (unnormalised) image for the overlay
        original = np.array(
            Image.open(filepath).convert("RGB").resize(
                (img_tensor.shape[2], img_tensor.shape[1])
            )
        ).astype(np.float32) / 255.0

        overlay = show_cam_on_image(original, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("GradCAM — model attention")
        axes[1].axis("off")

        fig.suptitle(os.path.basename(filepath), fontsize=9)
        fig.tight_layout()

        stem    = os.path.splitext(os.path.basename(filepath))[0]
        outfile = os.path.join(output_dir, f"{idx:04d}_{stem}.png")
        fig.savefig(outfile, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {n} GradCAM image(s) to: {output_dir}")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(true_labels, pred_labels, class_names, output_path):
    """
    Heatmap of true vs. predicted classes. The diagonal should be dark if
    the model is working correctly.
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 2), max(5, len(class_names) * 1.8)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (validation set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
