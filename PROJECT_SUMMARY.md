# qCMB 2026 — PrP Deposition Imaging Challenge
## Project Summary: Automated Prion Subtype Classification from IHC Brain Slices

---

## 1. Scientific Framing

### 1a. Research Question

Can we use a machine learning pipeline to automatically distinguish prion infection status and subtype (amino acid change GtE vs GtQ) from 4× immunohistochemistry (IHC) brain slice images and identify which anatomical brain regions drive those classifications?

### 1b. Specific Analytical Approach

We fine-tune **EfficientNet-B4** — a convolutional neural network pretrained on ImageNet — on 691 coronal 4× IHC images of mouse brain hemispheres across three conditions:

| Class   | Description                                      | Images |
|---------|--------------------------------------------------|--------|
| Control | Wild-type, non-prion-susceptible mice            | 143    |
| GtE     | Gene-targeted "Elk-inized" mice, prion-infected  | 163    |
| GtQ     | Gene-targeted "Deer-inized" mice, prion-infected | 385    |

Four brain regions are represented per condition: **Cerebellum, Hippocampus, Midbrain, Septum**.

The pipeline has five stages:

1. **Data loading** — Automatically discovers images under a configurable root folder, extracts class labels from the top-level subfolder name, and records the brain region from the second folder level.
2. **Training** — Fine-tunes EfficientNet-B4 with class-weighted cross-entropy loss (to correct for the 1:1.1:2.7 class imbalance) and saves the best checkpoint by validation accuracy.
3. **Evaluation** — Generates per-class precision, recall, and F1 scores; produces a confusion matrix.
4. **Novel-subtype detection (OOD)** — Extracts a 512-dimensional embedding vector per image, computes cosine distance from each sample to the nearest known class centroid, and flags samples exceeding 3 standard deviations above the mean as potential undiscovered prion phenotypes.
5. **Spatial visualization** — Applies **GradCAM** to generate pixel-level attention heatmaps, and **UMAP** to project all embeddings into 2D for cluster inspection.

### 1c. Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Per-class F1, Precision, Recall | Classification accuracy for each condition |
| Confusion matrix | Error patterns between classes |
| OOD cosine distance | Embedding divergence from known class centroids |
| GradCAM attention coverage | Whether highlighted regions match known PrP anatomy |
| UMAP cluster separation | Embedding space quality and discovery potential |

---

## 2. Scope and Complexity

### 2a. Ambition

This pipeline goes beyond binary prion detection. It simultaneously:
- Performs **3-class subtype classification** (Control / GtE / GtQ)
- Provides **open-set recognition** — the ability to flag images that do not belong to any trained class, without requiring labels for novel categories
- Produces **spatially interpretable outputs** (GradCAM heatmaps tied to anatomical regions)
- Captures **brain region metadata** automatically from folder structure, enabling region-stratified analysis of results

The OOD detection component directly addresses the challenge's stated interest in biological discovery — the pipeline does not assume the known classes are exhaustive.

### 2b. Pipeline Complexity

The pipeline is intentionally kept at moderate complexity. Five focused Python modules handle one stage each (`data`, `model`, `train`, `evaluate`, `visualize`). A single YAML configuration file is the only user interface. All hyperparameters, dataset paths, and output options are exposed in plain English comments.

On a single GPU (as available on Alpine HPC), the full pipeline — data loading through visualization — runs in under 4 hours for the full 691-image dataset.

---

## 3. Communication Plan

### 3a. Presentation Outline

**Introduction (1 min)**
- Prion disease background: difference between GtE and GtQ (biorender schematic)
- Challenge: automate what neuropathologists currently do qualitatively
- Question
- 691 4× IHC coronal sections across 3 conditions and 4 brain regions
- Class imbalance, folder structure, image characteristics

**Methods (3 min)**
- EfficientNet-B4 architecture and rationale (transfer learning on limited data)
- OOD detection mechanism and biological motivation
- GradCAM + UMAP as interpretability tools

**Results (4 min)**
- Classification report and confusion matrix
- GradCAM heatmaps: do they light up the expected anatomical regions?
- UMAP: are the three conditions well-separated? Are there outlier clusters?
- Any samples flagged as potential new subtypes?

**Discussion (2 min)**
- Biological interpretation: which regions distinguish GtElk from GtDeer?
- Limitations and next steps (higher magnification, segmentation)

---

## 4. Scientific Output

### 4a. What Is Being Quantified

- **Predicted class** and **per-class probability** for every image
- **OOD score** (cosine distance to nearest class centroid in 512-dim embedding space) — a continuous measure of how atypical each image is
- **GradCAM attention maps** — pixel-level scores indicating which image regions influenced the classification decision
- **Brain region** — extracted from folder structure and included in all result tables
- **Infection severity** — four continuous metrics derived from H-DAB color deconvolution of each original image:
  - `infection_score`: fraction of clean tissue pixels that are DAB-positive (0.0–1.0); primary measure of PrP deposition density
  - `mean_dab_intensity`: mean DAB optical density across artifact-free tissue; captures graded staining without binarizing
  - `clean_tissue_area`: fraction of the image occupied by analyzable tissue (background and artifacts excluded)
  - `artifact_area`: fraction of the image flagged as artifact (tears, folds, debris); values >0.15 indicate a poor-quality slide that warrants manual review

### 4b. Interpretability of Results

Results are designed to be interpretable by biologists without ML training:

- `predictions.csv` — one row per image with true label, predicted label, brain region, OOD score, new-subtype flag, and all four infection severity metrics
- `ood_flagged_samples.csv` — filtered view of only the anomalous images for expert review
- `gradcam/` — side-by-side original and heatmap overlay for every analyzed image; warm colors indicate high model attention
- `umap.png` — scatter plot of all samples in embedding space, colored by class; outlier samples marked with a red X
- `infection_maps/` (optional) — four-panel diagnostic images per slide showing original, tissue mask with artifacts highlighted in red, DAB optical density heatmap, and DAB-positive pixel overlay with printed infection score; enables visual verification that artifacts are correctly excluded
- `confusion_matrix.png` — visual summary of classification accuracy per class

### 4c. Biological Insights Obtainable

- **Regional vulnerability**: GradCAM maps reveal which brain structures (e.g., Hippocampus vs. Cerebellum) are most diagnostic for each prion subtype — directly testable against known PrP deposition patterns in the prion literature
- **Subtype separation**: UMAP cluster geometry shows how similar or distinct GtElk and GtDeer are in feature space, and whether Control samples ever overlap with infected ones
- **Novel phenotype discovery**: OOD-flagged samples may represent animals with atypical deposition patterns, incomplete infection, or genuinely new prion conformations
- **Regional stratification**: Because brain region is stored per-result, performance can be broken down by region to reveal which areas are most or least classifiable
- **Infection severity gradient**: `infection_score` and `mean_dab_intensity` provide a continuous measure of PrP burden independent of the discrete classification label — enabling severity ranking within a class, correlation with behavioral phenotypes, and detection of partially infected animals that lie between Control and infected groups
- **Slide quality control**: `artifact_area` automatically identifies degraded slides (tears, folds) that should be excluded from downstream analysis or re-sectioned, replacing subjective manual inspection

### 4d. Usability

- A researcher with no Python experience can run the full pipeline by editing one YAML file and running one command: `python run.py`
- Supports three dataset naming conventions (folder-based, CSV-based, filename-pattern) via a single config toggle
- Generates a ready-to-review output folder with human-readable CSV files and publication-quality figures
- A `--predict` mode allows classifying new images with an already-trained model without retraining
- HPC submission requires only editing one line (`YOUR_ACCOUNT_HERE`) in `run_job.sh` and running `sbatch run_job.sh`

### 4e. Generalizability

- The pipeline is **data-agnostic**: point it at any folder of labeled IHC images by changing two lines in `config.yaml`
- It handles **arbitrary class counts** — the number of output classes is determined automatically from the data, not hardcoded
- **OOD detection scales** with the number of known classes; adding new labeled data for a newly discovered subtype requires only retraining, not code changes
- The folder structure parser handles **arbitrary nesting depth** — the label is always the top-level subfolder, regardless of how many subdirectories lie between it and the images
- Compatible with any institution's HPC system running SLURM

---

## 5. Technical Quality

### 5a. Code Clarity, Organization, and Documentation

The codebase is organized into five single-purpose modules:

| File | Responsibility |
|------|---------------|
| `pipeline/data.py` | Dataset discovery, label loading, augmentation, DataLoaders |
| `pipeline/model.py` | EfficientNet-B4 architecture with embedding extraction |
| `pipeline/train.py` | Training loop, class weighting, checkpointing |
| `pipeline/evaluate.py` | Metrics, OOD scoring, infection quantification, results CSV |
| `pipeline/quantify.py` | H-DAB color deconvolution, artifact detection, infection scoring |
| `pipeline/visualize.py` | GradCAM, UMAP, confusion matrix, infection maps |
| `run.py` | Orchestration and command-line interface |
| `config.yaml` | All user-facing parameters with plain-English comments |

Comments explain *why* decisions were made (e.g., why cosine distance is used for OOD in high-dimensional space, why class weighting is applied) rather than restating what the code does. Every public function has a docstring.

### 5b. Robustness

- **Class imbalance**: handled automatically via inverse-frequency class weighting (no manual tuning required)
- **Small datasets**: stratified train/validation splits preserve class proportions; fallback to random split if any class has only one sample
- **Bad input data**: informative error messages with specific guidance for all three label loading modes
- **GPU availability**: gracefully falls back to CPU with a warning if CUDA is not detected
- **Checkpoint safety**: only the best validation-accuracy checkpoint is saved; training cannot silently use a stale model
- **Artifact-aware quantification**: tissue tears and folds are detected and excluded before any staining measurement is made; a high `artifact_area` score (>0.15) is surfaced in the results CSV so degraded slides are visible without manual inspection

### 5c. Use and Evaluation of AI Tools

| Tool | Role | Why chosen |
|------|------|------------|
| EfficientNet-B4 | Feature extraction and classification | Strong accuracy/parameter ratio; ImageNet pretraining transfers well to IHC textures with limited data |
| GradCAM | Spatial attribution | Requires no additional training; produces interpretable heatmaps aligned with anatomical regions |
| UMAP | Embedding visualization | Preserves local and global structure better than t-SNE; scales to hundreds of samples in seconds |
| Cosine OOD scoring | Novel subtype detection | Rotation-invariant; appropriate for high-dimensional embedding spaces where Euclidean distance is unreliable |
|Claude| AI assistance engine | For assistance in generating the first version of this summary outline using the rubric provided in the qCMB Retreat Sharepoint |

Model outputs are validated against biological priors: GradCAM attention maps should highlight regions of known PrP accumulation for each subtype. Divergence from expected anatomy flags potential model failure modes.

### 5d. Degree of Automation

Running `python run.py` fully automates:
- Image discovery across nested folder structures
- Label encoding and class counting
- Stratified train/validation splitting
- Class-weight computation
- Model training with learning-rate scheduling
- Best-checkpoint selection and saving
- Embedding extraction and OOD scoring
- H-DAB color deconvolution and artifact-aware infection scoring per slide
- CSV generation for all results
- GradCAM and UMAP figure generation

Zero manual steps between raw data folder and final figures.

### 5e. Speed and Computational Efficiency

| Stage | Estimated time (single A100 GPU, 691 images) |
|-------|----------------------------------------------|
| Data loading and preprocessing | < 5 minutes |
| Training (50 epochs, batch size 16) | ~2–3 hours |
| Embedding extraction + OOD scoring | ~5 minutes |
| GradCAM (60 images) | ~10 minutes |
| UMAP + figures | ~5 minutes |
| **Total** | **~3–4 hours** |

Feature extraction across slides is embarrassingly parallel and can be distributed across multiple GPU nodes on Alpine using SLURM job arrays for larger future datasets. EfficientNet-B4's compact architecture (19M parameters) keeps memory usage well within a single GPU's VRAM budget even at 1024×1024 input resolution.

---

## Bonus Criteria

### a. Creativity

The combination of supervised classification with **unsupervised open-set detection** in a single forward pass is unusual in clinical imaging pipelines. Rather than training a separate anomaly detector, the same embedding space used for classification is repurposed for OOD scoring — making novel subtype discovery a zero-cost byproduct of training. The brain region is automatically extracted from the folder hierarchy and propagated through to every result, enabling biological stratification without any manual annotation.

**Artifact-aware infection quantification** adds a second independent analysis axis: rather than relying solely on the model's class probabilities, each slide is independently scored for PrP burden using classical color deconvolution (Ruifrok & Johnston, 2001). Critically, the algorithm distinguishes tissue tears and fold artifacts from real staining through morphological analysis — large bright voids within the tissue boundary are flagged as tears, and extreme DAB density outliers are flagged as folds — and also removal of scale bars that only appear in some images to avoid detection by scale bar presence before any measurement is taken. This produces a physically grounded severity score that can validate, challenge, or refine the model's categorical predictions.

### b. Multi-Scale Analysis (Cellular → Regional)

The pipeline operates at the **regional scale** (4× magnification captures whole hemisphere architecture) and uses GradCAM to zoom into **sub-regional attention** — identifying which structures within the hemisphere are most informative. The dataset also contains higher-magnification images (not used in this pipeline due to time constraints) that could enable cellular-scale analysis of deposit morphology in future work. The modular architecture makes adding a second magnification stage straightforward.

### c. User-Friendly Interface

All user interaction is through `config.yaml` — a plain-text file with comments written for biologists rather than engineers. No Python knowledge is required to change the dataset, adjust training parameters, or modify output settings. A dedicated `--predict` mode allows any lab to classify new slides using a shared trained model. Error messages are written in plain English with specific remediation steps.

### d. Cross-Task Solution (Classification + Spatial Mapping)

The pipeline delivers both outputs in a single training run:
- **Classification**: predicted labels, per-class probabilities, F1/precision/recall, confusion matrix
- **Spatial mapping**: GradCAM heatmaps per image showing which brain regions were most attended, UMAP showing global embedding structure, and `brain_region` stratification in the results CSV

These outputs together directly address the challenge's dual objectives of detection accuracy and spatial characterization of PrP deposition.

- **Infection quantification**: a third output layer independent of the classifier — `infection_score` and `mean_dab_intensity` provide continuous severity measures that rank animals within a class, correlate with external phenotypic data, and surface partially infected or borderline cases that a binary classifier may handle inconsistently
