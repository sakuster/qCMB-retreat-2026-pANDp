"""
data.py — Dataset discovery, label loading, and DataLoader construction.

Supports four ways of organizing your labeled images:
  1. "folders"        — subfolders named after each class  (qCMB default)
  2. "region_folders" — label extracted from filename, region from folder name
  3. "csv"            — a CSV file mapping filenames to labels
  4. "filename"       — a regex pattern extracts the label from each filename
"""

import os
import re
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def find_images(root: str, extensions: list) -> list:
    """Return (absolute_path, path_relative_to_root) for every image under root."""
    root_path = Path(root)
    ext_set = {e.lower() for e in extensions}
    results = []
    for path in sorted(root_path.rglob("*")):
        if path.suffix.lower() in ext_set:
            results.append((str(path), str(path.relative_to(root_path))))
    if not results:
        raise FileNotFoundError(
            f"No images found in '{root}' with extensions {extensions}.\n"
            "Check that 'dataset.root' and 'dataset.extensions' are correct in config.yaml."
        )
    return results


# ---------------------------------------------------------------------------
# Label loading — one function per label_source option
# ---------------------------------------------------------------------------

def _labels_from_folders(root: str, extensions: list, path_filter: str = "") -> pd.DataFrame:
    """
    The top-level subfolder of root is used as the class label.
    Supports arbitrarily nested structures — label is always the first folder
    component, not the immediate parent.

    Example (flat):   data/control/img001.tiff         → label = "control"
    Example (nested): data/GtElk/Hippocampus/4x/img.tiff → label = "GtElk"
                      brain_region                         = "Hippocampus"

    path_filter: if set, only include images whose relative path contains this
    string (e.g. "4x" to skip higher-magnification subdirectories).
    """
    records = []
    for filepath, relpath in find_images(root, extensions):
        # Apply path_filter first — skips root-level files and non-data directories
        if path_filter and path_filter not in relpath:
            continue
        parts = Path(relpath).parts
        if not parts or len(parts) < 2:
            raise ValueError(
                f"Image '{filepath}' is directly in the root folder with no class subfolder.\n"
                "With label_source='folders', images must be inside subfolders named after their class.\n"
                "Expected: data/ClassName/image.tiff  or  data/ClassName/Region/4x/image.tiff"
            )
        label        = parts[0]
        raw_region   = parts[1] if len(parts) > 2 else ""
        brain_region = re.sub(r'_4[xX]$', '', raw_region).lower()  # "Cerebellum_4x" → "cerebellum"
        records.append({"filepath": filepath, "label": label, "brain_region": brain_region})
    return pd.DataFrame(records)


def _labels_from_csv(root: str, csv_path: str, image_col: str, label_col: str) -> pd.DataFrame:
    """
    Read a CSV where one column has filenames and another has class labels.
    Image paths are resolved relative to root.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: '{csv_path}'\n"
            "Check 'dataset.csv_path' in config.yaml."
        )
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in [image_col, label_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing expected column(s): {missing_cols}\n"
            f"Available columns: {list(df.columns)}\n"
            "Update 'csv_image_column' and 'csv_label_column' in config.yaml."
        )
    df["filepath"] = df[image_col].apply(lambda f: str(Path(root) / f))
    df["label"] = df[label_col].astype(str)
    return pd.DataFrame(df[["filepath", "label"]])


def _labels_from_region_folders(root: str, extensions: list, path_filter: str = "") -> pd.DataFrame:
    """
    For datasets organized as [label_]region_4x/ folders with files named
    {label}_{region}_4x_{number}.tif.

    Handles two folder naming conventions:
      cerebellum_4x/WT_control_cerebellum_4x_01.tif → label="WT_control", region="cerebellum"
      GtDeer_treatment_cerebellum_4x/GtDeer_treatment_cerebellum_4x_01.tif
                                                     → label="GtDeer_treatment", region="cerebellum"

    The region is always the LAST underscore-separated component of the folder
    name before the _4x suffix (e.g. "cerebellum" from both "cerebellum_4x" and
    "GtDeer_treatment_cerebellum_4x").

    The label is extracted from the filename stem using "_{region}_4x_" as a
    delimiter — this uniquely marks where the label ends. Files whose names do
    not contain this marker are silently skipped (prevents picking up
    non-data images from unrelated subdirectories such as venv/).
    """
    records = []
    for filepath, relpath in find_images(root, extensions):
        if path_filter and path_filter not in relpath:
            continue
        parts = Path(relpath).parts
        if len(parts) < 2:
            continue
        folder_name = parts[0]
        # Region = last component before the _4x suffix
        folder_prefix = re.sub(r'_4[xX]$', '', folder_name).lower()
        region = folder_prefix.split('_')[-1]
        # Label = stem up to _{region}_4x_ which uniquely marks the boundary
        stem   = Path(parts[-1]).stem
        marker = f'_{region}_4x_'
        idx    = stem.find(marker)
        if idx == -1:
            continue  # not a data file (e.g. venv icons, unrelated images)
        label  = stem[:idx]
        records.append({"filepath": filepath, "label": label, "brain_region": region})
    return pd.DataFrame(records)


def _labels_from_filename(root: str, pattern: str, extensions: list) -> pd.DataFrame:
    """
    Extract the class label from each filename using a regex pattern.
    The pattern must contain a named group called 'label'.
    Example pattern: "(?P<label>[a-zA-Z_]+)_\\d+"
    matches filenames like "control_001.tiff" or "subtype_a_007.tiff"
    """
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        raise ValueError(
            f"Invalid regex pattern '{pattern}': {e}\n"
            "Update 'dataset.filename_pattern' in config.yaml.\n"
            "Test your pattern at https://regex101.com"
        )
    if "label" not in compiled.groupindex:
        raise ValueError(
            f"Pattern '{pattern}' has no named group called 'label'.\n"
            "The pattern must contain (?P<label>...) to identify the class name.\n"
            "Example: (?P<label>[a-zA-Z_]+)_\\d+"
        )

    records = []
    for filepath, relpath in find_images(root, extensions):
        stem = Path(relpath).stem  # filename without extension
        match = compiled.search(stem)
        if not match:
            raise ValueError(
                f"Filename '{stem}' did not match pattern '{pattern}'.\n"
                "Update 'dataset.filename_pattern' in config.yaml, "
                "or switch to label_source='folders' or 'csv'."
            )
        records.append({"filepath": filepath, "label": match.group("label")})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PrionDataset(Dataset):
    """
    Loads brain slice TIFF images and prepares them for the model.

    Augmentation (flips, rotation, colour jitter) is applied during training
    to help the model generalise across differences in staining intensity and
    slide orientation between experiments.
    """

    # These values normalise pixel intensities to the range the ImageNet-pretrained
    # backbone expects. Do not change them.
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, filepaths: list, labels: list, image_size: int, augment: bool = False):
        self.filepaths = filepaths
        self.labels = labels

        aug_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            # Mild colour jitter accounts for staining intensity variation across slides
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        ] if augment else []

        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size))]
            + aug_transforms
            + [transforms.ToTensor(), transforms.Normalize(self._MEAN, self._STD)]
        )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx], self.filepaths[idx]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_dataloaders(config: dict):
    """
    Read the config, discover all images, encode labels, split into
    train/validation sets, and return DataLoaders + the fitted LabelEncoder.

    The LabelEncoder maps between string class names and integer indices.
    Call label_encoder.classes_ to see the list of class names.
    """
    ds  = config["dataset"]
    tr  = config["training"]
    root       = ds["root"]
    extensions = ds.get("extensions", [".tiff", ".tif", ".png", ".jpg"])
    source     = ds["label_source"]

    print(f"  Root : {os.path.abspath(root)}")
    print(f"  Mode : label_source='{source}'")

    if source == "folders":
        df = _labels_from_folders(root, extensions, path_filter=ds.get("path_filter", ""))
    elif source == "region_folders":
        df = _labels_from_region_folders(root, extensions, path_filter=ds.get("path_filter", ""))
    elif source == "csv":
        df = _labels_from_csv(root, ds["csv_path"], ds["csv_image_column"], ds["csv_label_column"])
    elif source == "filename":
        df = _labels_from_filename(root, ds["filename_pattern"], extensions)
    else:
        raise ValueError(
            f"Unknown label_source: '{source}'.\n"
            "Valid options are: 'folders', 'region_folders', 'csv', 'filename'."
        )

    print(f"  Found {len(df)} images across {df['label'].nunique()} class(es): "
          f"{sorted(df['label'].unique())}")

    # Encode string labels ("control", "subtype_a", ...) to integers (0, 1, ...)
    encoder = LabelEncoder()
    df["label_idx"] = list(encoder.fit_transform(df["label"]))  # type: ignore[arg-type]

    val_split  = tr.get("validation_split", 0.2)
    image_size = tr.get("image_size", 1024)
    batch_size = tr.get("batch_size", 16)
    n_workers  = tr.get("num_workers", 4)

    # Stratified split keeps class proportions equal in train and validation sets
    min_class_count = df["label_idx"].value_counts().min()
    stratify = df["label_idx"] if min_class_count >= 2 else None
    if stratify is None:
        print("  Warning: some classes have only 1 sample — skipping stratified split.")

    split = train_test_split(df, test_size=val_split, stratify=stratify, random_state=42)
    train_df, val_df = pd.DataFrame(split[0]), pd.DataFrame(split[1])
    print(f"  Split : {len(train_df)} train / {len(val_df)} validation")

    train_ds = PrionDataset(
        train_df["filepath"].tolist(), train_df["label_idx"].tolist(), image_size, augment=True
    )
    val_ds = PrionDataset(
        val_df["filepath"].tolist(), val_df["label_idx"].tolist(), image_size, augment=False
    )

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=batch_size, num_workers=n_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, batch_size=batch_size, num_workers=n_workers, pin_memory=True
    )

    return train_loader, val_loader, encoder
