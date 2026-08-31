"""Shared vision backbone — the single encoder reused by BOTH the AI-classifier
and the similarity/clustering system, keeping total parameter count low and
avoiding a duplicated encoder.

Swap the backbone by changing `model.backbone` / `model.pretrained` in
configs/config.yaml — no other code needs to change, since everything downstream
just consumes `embedding_dim`-sized feature vectors.
"""
import torch
import torch.nn as nn
import open_clip


class SharedBackbone(nn.Module):
    """Wraps an open_clip vision encoder. Frozen by default (config.model.freeze_backbone)
    so training only updates the small classifier head — cheap and fast, and
    avoids overfitting the (relatively small) hackathon dataset.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_cfg["backbone"],
            pretrained=model_cfg["pretrained"],
            force_quick_gelu=(model_cfg["pretrained"] == "openai"),
        )
        self.encoder = model.visual
        self.preprocess = preprocess  # exposed for reference; dataset.py does its own equivalent normalization
        self.embedding_dim = model_cfg["embedding_dim"]

        unfreeze_last_n = model_cfg.get("unfreeze_last_n_blocks", 0)
        if model_cfg.get("freeze_backbone", True):
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif unfreeze_last_n > 0:
            # Partial fine-tune: freeze everything, then re-enable gradients
            # only for the last `unfreeze_last_n` transformer blocks plus the
            # final layer norm / output projection sitting directly
            # downstream of them. Early layers (patch embed, class/positional
            # embeddings, ln_pre) stay frozen — they're the least
            # task-specific and the priciest to destabilize with a small
            # fine-tuning dataset.
            for p in self.encoder.parameters():
                p.requires_grad = False
            for block in self.encoder.transformer.resblocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.encoder.ln_post.parameters():
                p.requires_grad = True
            if self.encoder.proj is not None:
                self.encoder.proj.requires_grad = True
        # else: freeze_backbone false and unfreeze_last_n_blocks unset/0 -> full fine-tune, every param trains

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) float tensor, normalized to roughly [-1, 1].
        Returns: (B, embedding_dim) embedding — used for BOTH classification
        (via classifier.py) and similarity (via similarity/embeddings.py).
        """
        return self.encoder(images)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)


def print_param_budget(config: dict, classifier_params: int = 0):
    """Quick sanity check that the combined pipeline stays under the 2B limit."""
    backbone = SharedBackbone(config)
    total = backbone.param_count() + classifier_params
    print(f"Backbone params:   {backbone.param_count():,}")
    print(f"Classifier params: {classifier_params:,}")
    print(f"Total:             {total:,}  (limit: 2,000,000,000)")
    print(f"Under budget: {total < 2_000_000_000}")
    return total


if __name__ == "__main__":
    from src.utils import load_config

    cfg = load_config("configs/config.yaml")
    print_param_budget(cfg)
