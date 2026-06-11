"""
quantify.py — Artifact-aware DAB staining quantification.

Measures the degree of prion protein (PrP) deposition in each brain slice by
quantifying the fraction of clean tissue that stains DAB-positive (brown).

The key challenge is that IHC slides contain non-biological artifacts — tears,
folds, and debris — that can mimic or obscure real staining. This module:
  1. Builds a tissue mask to exclude glass/background
  2. Identifies and excludes artifact regions within the tissue
  3. Applies H-DAB color deconvolution to isolate the DAB channel
  4. Quantifies DAB positivity only within artifact-free tissue

Color deconvolution method:
  Ruifrok & Johnston (2001). "Quantification of histochemical staining by
  color deconvolution." Analytical and Quantitative Cytology and Histology.
"""

import numpy as np
import scipy.ndimage as ndi
from pathlib import Path
from PIL import Image
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# H-DAB stain separation
# ---------------------------------------------------------------------------

# Standard optical density vectors for H-DAB staining.
# Each row: [R, G, B] contribution of one stain in OD space.
# These values are the widely-used Ruifrok & Johnston reference vectors.
_H_VECTOR   = np.array([0.65, 0.70, 0.29])   # Hematoxylin (blue-purple)
_DAB_VECTOR = np.array([0.27, 0.57, 0.78])   # DAB (brown)

def _build_deconvolution_matrix() -> np.ndarray:
    """Construct the 3×3 stain separation matrix with an orthogonal residual vector."""
    h   = _H_VECTOR   / np.linalg.norm(_H_VECTOR)
    dab = _DAB_VECTOR / np.linalg.norm(_DAB_VECTOR)
    residual = np.cross(h, dab)
    residual /= np.linalg.norm(residual)
    return np.linalg.inv(np.array([h, dab, residual]).T)

_DECONV_MATRIX = _build_deconvolution_matrix()


def _rgb_to_dab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to a 2D DAB optical density map.
    Higher values = more brown DAB staining = more PrP deposition.
    """
    # Optical density: OD = -log10(I/255). Clamp to avoid log(0).
    od = -np.log10((rgb.astype(float) + 1) / 256)
    # Separate stain channels; DAB is index 1
    stains = od @ _DECONV_MATRIX
    return stains[:, :, 1].clip(min=0)


# ---------------------------------------------------------------------------
# Tissue and artifact detection
# ---------------------------------------------------------------------------

def _tissue_mask(rgb: np.ndarray, bg_threshold: int = 220) -> np.ndarray:
    """
    Binary mask of tissue vs. glass background.

    Background (empty glass) appears near-white in all channels.
    Morphological closing fills small gaps, then small isolated fragments
    (dust, debris on the slide) are removed by minimum size filtering.
    """
    is_background = np.all(rgb > bg_threshold, axis=2)
    tissue = ~is_background

    # Close small gaps within the tissue body
    tissue = ndi.binary_closing(tissue, structure=np.ones((15, 15)))

    # Remove small disconnected fragments — keep only large tissue regions
    _label_out: tuple[np.ndarray, int] = ndi.label(tissue)  # type: ignore[assignment]
    labeled, n = _label_out
    if n == 0:
        return tissue
    sizes = ndi.sum(tissue, labeled, range(1, n + 1))
    tissue = np.isin(labeled, np.where(np.array(sizes) > 5000)[0] + 1)
    return tissue.astype(bool)


def _artifact_mask(rgb: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    """
    Identify artifact pixels within the tissue mask and return a boolean
    mask where True = artifact (exclude from quantification).

    Two artifact types are detected:

    Tears — large bright voids within the tissue boundary.
      Tears appear as near-white regions surrounded by tissue. They differ
      from true background in that they are enclosed within the tissue
      outline rather than at its edge. We flag bright regions > 2000 pixels
      (smaller bright areas are unstained cells, not tears).

    Folds — regions of abnormally high staining density.
      When tissue folds, a double layer of section creates artificially
      intense DAB signal. We flag pixels in the top 0.5% of DAB optical
      density within tissue, as these rarely represent real biology.
    """
    luminance = rgb.mean(axis=2)

    # --- Tears: large bright regions embedded within tissue ---
    bright_in_tissue = tissue & (luminance > 215)
    _label_out2: tuple[np.ndarray, int] = ndi.label(bright_in_tissue)  # type: ignore[assignment]
    labeled, n = _label_out2
    if n > 0:
        sizes = ndi.sum(bright_in_tissue, labeled, range(1, n + 1))
        tears = np.isin(labeled, np.where(np.array(sizes) > 2000)[0] + 1)
    else:
        tears = np.zeros_like(tissue, dtype=bool)

    # --- Folds: extreme DAB density outliers ---
    dab = _rgb_to_dab(rgb)
    tissue_dab = dab[tissue]
    if len(tissue_dab) > 0:
        fold_threshold = np.percentile(tissue_dab, 99.5)
        folds = tissue & (dab > fold_threshold)
    else:
        folds = np.zeros_like(tissue, dtype=bool)

    return tears | folds


# ---------------------------------------------------------------------------
# Public quantification function
# ---------------------------------------------------------------------------

def quantify_infection(image_path: str, dab_threshold: float = 0.15) -> dict:
    """
    Quantify PrP/DAB staining in a single brain slice image, excluding
    glass background and artifact regions (tears, folds, debris).

    Parameters
    ----------
    image_path    : path to the IHC image (TIFF, PNG, or JPG)
    dab_threshold : optical density cutoff for calling a pixel DAB-positive.
                    0.15 is a standard starting value for H-DAB slides;
                    adjust in config.yaml if your staining protocol differs.

    Returns
    -------
    dict with four keys:
      infection_score      — DAB-positive fraction of clean tissue (0.0–1.0)
                             Primary measure of PrP deposition severity.
      mean_dab_intensity   — Mean DAB optical density over clean tissue.
                             Captures graded staining intensity, not just
                             positive/negative.
      clean_tissue_area    — Fraction of total image area that is clean,
                             analyzable tissue (excludes background + artifacts).
      artifact_area        — Fraction of total image area flagged as artifact.
                             High values (>0.15) indicate a poor-quality slide.
    """
    rgb = np.array(Image.open(image_path).convert("RGB"))

    tissue   = _tissue_mask(rgb)
    artifact = _artifact_mask(rgb, tissue)
    clean    = tissue & ~artifact

    total_px  = rgb.shape[0] * rgb.shape[1]
    clean_px  = int(clean.sum())

    if clean_px == 0:
        return {
            "infection_score":    float("nan"),
            "mean_dab_intensity": float("nan"),
            "clean_tissue_area":  0.0,
            "artifact_area":      float(tissue.sum()) / total_px,
        }

    dab          = _rgb_to_dab(rgb)
    dab_in_clean = dab[clean]

    return {
        "infection_score":    float((dab_in_clean > dab_threshold).mean()),
        "mean_dab_intensity": float(dab_in_clean.mean()),
        "clean_tissue_area":  float(clean_px / total_px),
        "artifact_area":      float(artifact.sum() / total_px),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_infection_map(image_path: str, output_path: str, dab_threshold: float = 0.15):
    """
    Save a four-panel diagnostic image for one brain slice:
      Panel 1 — Original IHC image
      Panel 2 — Clean tissue mask (green) with artifacts highlighted (red)
      Panel 3 — DAB optical density heatmap (within clean tissue only)
      Panel 4 — DAB-positive pixels above threshold overlaid on original

    This allows visual verification that the quantification is working
    correctly and that artifacts are being properly excluded.
    """
    rgb = np.array(Image.open(image_path).convert("RGB"))

    tissue   = _tissue_mask(rgb)
    artifact = _artifact_mask(rgb, tissue)
    clean    = tissue & ~artifact
    dab      = _rgb_to_dab(rgb)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: original
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Panel 2: tissue mask with artifacts
    overlay = rgb.copy()
    overlay[clean]    = (overlay[clean] * 0.5 + np.array([0, 180, 0]) * 0.5).clip(0, 255).astype(np.uint8)
    overlay[artifact] = (overlay[artifact] * 0.5 + np.array([220, 0, 0]) * 0.5).clip(0, 255).astype(np.uint8)
    axes[1].imshow(overlay)
    axes[1].set_title("Tissue mask\n(green=clean, red=artifact)")
    axes[1].axis("off")

    # Panel 3: DAB heatmap within clean tissue
    dab_display = np.zeros_like(dab)
    dab_display[clean] = dab[clean]
    im = axes[2].imshow(dab_display, cmap="hot", vmin=0, vmax=0.5)
    axes[2].set_title("DAB intensity\n(clean tissue only)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: DAB-positive overlay
    positive = clean & (dab > dab_threshold)
    pos_overlay = rgb.copy()
    pos_overlay[positive] = [220, 60, 0]
    axes[3].imshow(pos_overlay)
    score = float((dab[clean] > dab_threshold).mean()) if clean.sum() > 0 else float("nan")
    axes[3].set_title(f"DAB-positive pixels\ninfection score = {score:.3f}")
    axes[3].axis("off")

    fig.suptitle(os.path.basename(image_path), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def quantify_dataset(filepaths: list, dab_threshold: float = 0.15,
                     save_maps: bool = False, maps_dir: str = "results/infection_maps") -> list:
    """
    Run quantify_infection on every image in the dataset.
    Optionally saves a diagnostic four-panel image for each slide.
    Returns a list of dicts (one per image) in the same order as filepaths.
    """
    from tqdm import tqdm

    if save_maps:
        os.makedirs(maps_dir, exist_ok=True)

    results = []
    for fp in tqdm(filepaths, desc="  Quantifying infection"):
        metrics = quantify_infection(fp, dab_threshold=dab_threshold)
        metrics["filepath"] = fp
        results.append(metrics)

        if save_maps:
            stem    = os.path.splitext(os.path.basename(fp))[0]
            out     = os.path.join(maps_dir, f"{stem}_infection.png")
            save_infection_map(fp, out, dab_threshold=dab_threshold)

    return results
