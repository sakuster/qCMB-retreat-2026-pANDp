#!/usr/bin/env python3
"""
Prion Brain Slice Classification Pipeline
==========================================
Classifies prion subtypes from IHC brain slice images and flags samples that
may represent previously undiscovered prion types.

QUICKSTART
----------
1. Edit config.yaml to point to your dataset folder and set label_source.
2. Run:  python run.py
3. Results are saved to the folder specified in config.yaml (default: results/)

USAGE
-----
  python run.py                            Use config.yaml in this folder
  python run.py --config my_config.yaml    Use a different config file
  python run.py --predict path/to/images/  Classify new images (no training)

REQUIREMENTS
------------
  pip install -r requirements.txt
"""

import argparse
import os
import sys
import yaml
import torch


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"\nERROR: Config file not found at '{path}'")
        print("Make sure config.yaml is in the same folder as run.py,")
        print("or pass a path with --config path/to/config.yaml\n")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Prediction-only mode (no training needed)
# ---------------------------------------------------------------------------

def predict_new_images(config: dict, image_dir: str, checkpoint_path: str):
    """
    Run a trained model on a folder of new images and save predictions.
    Used when you have already trained the model and want to classify new slides.
    """
    from pipeline.data import PrionDataset, find_images
    from pipeline.model import build_model
    import pandas as pd
    import numpy as np

    # Load class names from the training checkpoint's companion file
    label_path = checkpoint_path.replace("best_model.pt", "label_classes.txt")
    if not os.path.exists(label_path):
        print(f"ERROR: Cannot find label list at '{label_path}'.")
        print("Run the full pipeline first to generate a trained model.")
        sys.exit(1)

    with open(label_path) as f:
        class_names = [line.strip() for line in f if line.strip()]

    tr_cfg = config["training"]
    device = torch.device("cuda" if tr_cfg.get("use_gpu", True) and torch.cuda.is_available() else "cpu")
    model  = build_model(len(class_names)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    extensions = config["dataset"].get("extensions", [".tiff", ".tif", ".png", ".jpg"])
    image_pairs = find_images(image_dir, extensions)
    filepaths   = [p for p, _ in image_pairs]

    labels_placeholder = [0] * len(filepaths)
    dataset = PrionDataset(filepaths, labels_placeholder, tr_cfg.get("image_size", 1024), augment=False)

    records = []
    with torch.no_grad():
        for _item in dataset:
            img_tensor: torch.Tensor = _item[0]  # type: ignore[assignment]
            filepath: str = _item[2]
            logits     = model(img_tensor.unsqueeze(0).to(device))
            probs      = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_idx   = int(probs.argmax())
            pred_label = class_names[pred_idx]
            confidence = float(probs[pred_idx])
            records.append({
                "filepath":        filepath,
                "predicted_label": pred_label,
                "confidence":      f"{confidence:.3f}",
                **{f"prob_{name}": f"{probs[i]:.3f}" for i, name in enumerate(class_names)},
            })

    out_dir = config["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "new_image_predictions.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prion brain slice classification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config",  default="config.yaml",  help="Path to config YAML file")
    parser.add_argument("--predict", default=None,           help="Path to new images (skips training)")
    args = parser.parse_args()

    config = load_config(args.config)

    out_dir = config["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "best_model.pt")

    # ---- Predict-only mode ----
    if args.predict:
        print(f"\nPredict mode: classifying images in '{args.predict}'")
        predict_new_images(config, args.predict, checkpoint_path)
        return

    # ---- Full pipeline ----
    from pipeline.data      import build_dataloaders, PrionDataset
    from pipeline.model     import build_model
    from pipeline.train     import train
    from pipeline.evaluate  import evaluate
    from pipeline.visualize import plot_umap, plot_confusion_matrix, save_gradcam

    header = lambda n, title: print(f"\n{'='*60}\nSTEP {n}: {title}\n{'='*60}")

    # STEP 1 — Load data
    header(1, "Loading dataset")
    train_loader, val_loader, label_encoder = build_dataloaders(config)
    _classes    = label_encoder.classes_ if label_encoder.classes_ is not None else []
    class_names: list[str] = [str(c) for c in _classes]
    num_classes = len(class_names)

    # Save class names alongside the model so predict mode works later
    with open(os.path.join(out_dir, "label_classes.txt"), "w") as f:
        f.write("\n".join(class_names))

    # STEP 2 — Build model
    header(2, "Building model")
    model = build_model(num_classes)

    # STEP 3 — Train
    header(3, "Training")
    checkpoint_path = train(config, model, train_loader, val_loader)

    # STEP 4 — Evaluate + OOD detection
    header(4, "Evaluation and novel-subtype detection")
    device = torch.device(
        "cuda" if config["training"].get("use_gpu", True) and torch.cuda.is_available() else "cpu"
    )

    results_df, embeddings, true_labels, pred_labels, ood_scores = evaluate(
        config, model, val_loader, label_encoder, checkpoint_path
    )

    # STEP 5 — Visualisations
    header(5, "Saving visualisations")

    if config["output"].get("save_umap", True):
        plot_umap(
            embeddings, true_labels, label_encoder.classes_, ood_scores,
            config["output"].get("ood_threshold", 3.0),
            os.path.join(out_dir, "umap.png"),
        )

    plot_confusion_matrix(
        results_df["true_label"], results_df["predicted_label"],
        label_encoder.classes_,
        os.path.join(out_dir, "confusion_matrix.png"),
    )

    if config["output"].get("save_gradcam", True):
        # Rebuild validation dataset without augmentation for GradCAM overlays
        tr_cfg  = config["training"]
        val_ds  = PrionDataset(
            results_df["filepath"].tolist(),
            list(label_encoder.transform(results_df["true_label"])),  # type: ignore[arg-type]
            tr_cfg.get("image_size", 1024),
            augment=False,
        )
        save_gradcam(
            model.to(device), val_ds,
            os.path.join(out_dir, "gradcam"),
            device,
            max_images=config["output"].get("gradcam_max_images", 30),
        )

    # Summary
    infection_maps_note = (
        "  infection_maps/          Per-slide DAB diagnostic panels\n"
        if config["output"].get("save_infection_maps", False) else ""
    )
    print(f"""
{'='*60}
PIPELINE COMPLETE
{'='*60}
All outputs saved to: {out_dir}

  best_model.pt            Trained model weights
  label_classes.txt        Class names (required for --predict mode)
  predictions.csv          Per-image predictions, OOD scores, and infection scores
  ood_flagged_samples.csv  Images flagged as potential new subtypes
  umap.png                 Embedding space — look for unexpected clusters
  confusion_matrix.png     Per-class accuracy
  gradcam/                 Brain region attention heatmaps
{infection_maps_note}
Infection score columns in predictions.csv:
  infection_score      Fraction of clean tissue that is DAB-positive (0–1)
  mean_dab_intensity   Mean DAB optical density over artifact-free tissue
  clean_tissue_area    Fraction of image area that is analyzable tissue
  artifact_area        Fraction flagged as artifact (tears, folds, debris)
""")


if __name__ == "__main__":
    main()
