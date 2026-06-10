"""
model.py — Neural network definition.

Architecture: EfficientNet-B4 backbone (pretrained on ImageNet) with a custom
classification head that outputs both class predictions and a compact embedding
vector used for out-of-distribution (OOD) detection.

Why EfficientNet-B4:
  - Strong accuracy-to-parameter ratio; works well on small medical datasets
  - ImageNet pretraining gives useful low-level texture features even for IHC images
  - Small enough to train in hours on a single GPU
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from typing import cast


# Size of the learned feature vector for each image.
# This compact representation is used for UMAP visualisation and OOD detection.
EMBEDDING_DIM = 512


class PrionClassifier(nn.Module):
    """
    EfficientNet-B4 with a two-stage custom head:
      backbone → 512-dim embedding → class logits

    The embedding layer is the key interface for OOD detection: samples whose
    embeddings fall far from all known class centroids are flagged as potential
    undiscovered prion subtypes.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Load backbone with ImageNet weights. Using pretrained weights is
        # critical when data is limited — we get useful filters for free.
        base       = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = cast(nn.Linear, base.classifier[1]).in_features  # 1792 for EfficientNet-B4

        self.backbone  = base.features
        self.pool      = base.avgpool

        # Dropout prevents overfitting on small datasets
        self.embed = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, EMBEDDING_DIM),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(EMBEDDING_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass — returns raw class scores (logits)."""
        return self.classifier(self._embed(x))

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-dim embedding without computing class scores."""
        return self._embed(x)

    def forward_with_embeddings(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, embeddings) in a single forward pass."""
        emb = self._embed(x)
        return self.classifier(emb), emb

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.backbone(x))
        return self.embed(features.flatten(1))


def build_model(num_classes: int) -> PrionClassifier:
    """Factory function — builds a fresh model for the given number of classes."""
    model = PrionClassifier(num_classes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EfficientNet-B4 | {trainable / 1e6:.1f}M trainable parameters | {num_classes} output classes")
    return model
