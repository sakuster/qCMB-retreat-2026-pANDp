"""
evaluate.py — Model evaluation and novel-subtype detection.

After training, this module:
  1. Generates predictions for every validation image
  2. Extracts 512-dim embedding vectors for every image
  3. Computes an OOD (out-of-distribution) score per image using cosine distance
     to the nearest known class centroid in embedding space
  4. Flags images whose embeddings are statistically distant from all known
     classes — these are candidates for previously undiscovered prion subtypes
  5. Saves a full predictions table and a filtered OOD table to results/

Why cosine distance for OOD:
  Neural network embeddings are high-dimensional and their magnitude is not
  meaningful — direction is. Cosine distance (1 - cosine similarity) measures
  angle between vectors, making it robust to scale differences across images.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from .model import PrionClassifier
from .quantify import quantify_dataset


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model: PrionClassifier, loader, device) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """
    Pass every image through the model and collect embeddings, true labels,
    and file paths. Returns three parallel arrays/lists.
    """
    model.eval()
    embeddings_list, labels_list, paths_list, preds_list = [], [], [], []

    for images, labels, paths in tqdm(loader, desc="  Extracting embeddings"):
        images = images.to(device)
        # Single forward pass returns both logits and embeddings
        logits, embs = model.forward_with_embeddings(images)

        embeddings_list.append(embs.cpu().numpy())
        labels_list.extend(labels.numpy())
        paths_list.extend(paths)
        preds_list.extend(logits.argmax(1).cpu().numpy())

    return (
        np.vstack(embeddings_list),
        np.array(labels_list),
        paths_list,
        np.array(preds_list),
    )


# ---------------------------------------------------------------------------
# OOD scoring
# ---------------------------------------------------------------------------

def compute_ood_scores(embeddings: np.ndarray, pred_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    For each image, compute the cosine distance to its nearest class centroid.

    A score near 0 means the image looks like a known class.
    A high score means the image is unlike anything the model was trained on —
    a potential indicator of an undiscovered prion subtype.
    """
    # Compute the mean embedding vector (centroid) for each known class
    centroids = []
    for c in range(num_classes):
        class_embeddings = embeddings[pred_labels == c]
        if len(class_embeddings) == 0:
            centroids.append(np.zeros(embeddings.shape[1]))
        else:
            centroid = class_embeddings.mean(axis=0)
            # Normalise to unit length for cosine comparison
            centroids.append(centroid / (np.linalg.norm(centroid) + 1e-8))

    centroids = np.array(centroids)

    # Normalise each embedding to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)

    # Cosine similarity: dot product of unit vectors. Range: [-1, 1]
    similarities = normed @ centroids.T  # shape: (N, num_classes)

    # Distance = 1 - max similarity to any known class centroid
    ood_scores = 1.0 - similarities.max(axis=1)
    return ood_scores


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate(config: dict, model: PrionClassifier, val_loader, label_encoder: LabelEncoder, checkpoint_path: str):
    """
    Load the best checkpoint, run evaluation, flag OOD samples, and save results.
    Returns (results_df, embeddings, true_labels, ood_scores) for use in visualisation.
    """
    tr_cfg  = config["training"]
    out_cfg = config["output"]
    out_dir = out_cfg["results_dir"]
    os.makedirs(out_dir, exist_ok=True)

    ood_threshold = out_cfg.get("ood_threshold", 3.0)
    class_names   = list(label_encoder.classes_)
    num_classes   = len(class_names)

    device = torch.device(
        "cuda" if tr_cfg.get("use_gpu", True) and torch.cuda.is_available() else "cpu"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)

    embeddings, true_labels, filepaths, pred_labels = extract_embeddings(model, val_loader, device)

    # Standard classification metrics
    print("\n  Classification Report (validation set):")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    # OOD detection — centroids built from true labels so the reference clusters
    # are not contaminated by the model's own mispredictions
    ood_scores = compute_ood_scores(embeddings, true_labels, num_classes)
    ood_mean   = ood_scores.mean()
    ood_std    = ood_scores.std()
    ood_flags  = ood_scores > (ood_mean + ood_threshold * ood_std)

    n_flagged = ood_flags.sum()
    if n_flagged > 0:
        print(f"\n  *** {n_flagged} image(s) flagged as potential new prion subtypes ***")
        print(f"  These samples scored >{ood_threshold:.1f} standard deviations above the mean OOD score.")
        print(f"  Inspect 'ood_flagged_samples.csv' and 'umap.png' for details.\n")
    else:
        print(f"\n  No samples flagged as potential new subtypes at threshold={ood_threshold:.1f}σ.\n")

    # Extract brain region from filepath (second path component: Condition/Region/4x/img.tiff)
    def _region(fp: str) -> str:
        parts = Path(fp).parts
        return parts[-3] if len(parts) >= 3 else ""

    # Build and save full results table
    results_df = pd.DataFrame({
        "filepath":              filepaths,
        "brain_region":          [_region(fp) for fp in filepaths],
        "true_label":            label_encoder.inverse_transform(true_labels),
        "predicted_label":       label_encoder.inverse_transform(pred_labels),
        "correct":               (true_labels == pred_labels),
        "ood_score":             ood_scores,
        "potential_new_subtype": ood_flags,
    })
    results_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # Flagged samples get their own file for easy manual review
    flagged_df = results_df[results_df["potential_new_subtype"]]
    flagged_df.to_csv(os.path.join(out_dir, "ood_flagged_samples.csv"), index=False)

    print(f"  Saved: predictions.csv ({len(results_df)} rows)")
    print(f"  Saved: ood_flagged_samples.csv ({len(flagged_df)} rows)")

    # --- Infection quantification ---
    # Only run if enabled in config (true by default).
    # Works on the original full-resolution images — not the resized tensors —
    # so color deconvolution has full chromatic information to work with.
    if out_cfg.get("quantify_infection", True):
        print("\n  Step 4b: Quantifying infection severity (DAB staining)...")
        dab_threshold = out_cfg.get("dab_threshold", 0.15)
        save_maps     = out_cfg.get("save_infection_maps", False)
        maps_dir      = os.path.join(out_dir, "infection_maps")
        maps_max      = out_cfg.get("infection_maps_max_images", 20)

        quant_filepaths = filepaths[:maps_max] if save_maps else filepaths
        quant_results   = quantify_dataset(
            filepaths,
            dab_threshold=dab_threshold,
            save_maps=save_maps,
            maps_dir=maps_dir,
        )

        quant_df = pd.DataFrame(quant_results).rename(columns={"filepath": "filepath"})
        results_df = results_df.merge(
            quant_df[["filepath", "infection_score", "mean_dab_intensity",
                      "clean_tissue_area", "artifact_area"]],
            on="filepath", how="left",
        )
        results_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
        print(f"  Infection scores added to predictions.csv")
        if save_maps:
            print(f"  Infection maps saved to: {maps_dir}/")

    return results_df, embeddings, true_labels, pred_labels, ood_scores
