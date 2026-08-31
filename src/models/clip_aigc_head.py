"""Model: a FROZEN open_clip vision backbone + a small trainable MLP head.

Only the head has learnable parameters — the CLIP encoder's weights never
receive gradients, so training is cheap enough to iterate on a CPU-only
laptop. `AIGCClipDetector.forward` extracts image features under
`torch.no_grad()` (equivalent to freezing + detaching, and it also saves
memory since no activation graph is kept for the backbone) and then runs the
head on top of those features.

Label convention (fixed across this project):
    0 = real
    1 = AI-generated ("fake")
"""
from __future__ import annotations

import torch
import torch.nn as nn
import open_clip


class MlpHead(nn.Module):
    """Small 2-layer MLP: clip_embedding -> hidden -> 1 logit."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)  # (B,) raw logits


class AIGCClipDetector(nn.Module):
    """Frozen CLIP visual encoder + trainable MlpHead.

    `clip_preprocess` (CLIP's own, resolution/normalization-correct
    preprocessing transform for this backbone) is exposed so the caller can
    build datasets with it directly instead of hand-rolling resize/normalize
    constants that might not match the chosen backbone.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
            force_quick_gelu=(clip_pretrained == "openai"),
        )
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess  # deterministic resize/crop/normalize for this backbone

        # Freeze every backbone parameter — only the head trains.
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        # Ask the model itself for its output width instead of hardcoding a
        # per-architecture constant (ViT-L-14 is 768, but this keeps the
        # class correct if you swap CLIP_MODEL_NAME later).
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.clip_model.visual.image_size[0], self.clip_model.visual.image_size[0]) \
                if hasattr(self.clip_model.visual, "image_size") else torch.zeros(1, 3, 224, 224)
            embed_dim = self.clip_model.encode_image(dummy).shape[-1]

        self.embed_dim = embed_dim
        self.head = MlpHead(embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) already run through self.clip_preprocess."""
        with torch.no_grad():
            features = self.clip_model.encode_image(images).float()
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B,) — use BCEWithLogitsLoss, or sigmoid for prob."""
        features = self.encode_image(images)
        return self.head(features)

    def trainable_parameters(self):
        return self.head.parameters()

    def param_counts(self) -> dict:
        backbone_params = sum(p.numel() for p in self.clip_model.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return {"backbone": backbone_params, "head": head_params, "total": backbone_params + head_params}
