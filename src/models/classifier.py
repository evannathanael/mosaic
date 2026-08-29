"""Small classifier head: shared-backbone embedding -> AI-generated probability.

This is the ONLY part that needs training when freeze_backbone=True, which is
what keeps hackathon training fast and cheap.
"""
import torch
import torch.nn as nn

from src.models.backbone import SharedBackbone


class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, 1) — apply sigmoid for probability, or use
        with BCEWithLogitsLoss directly during training.
        """
        return self.net(embedding).squeeze(-1)


class AIGCDetector(nn.Module):
    """Full model: shared backbone + classifier head, with an optional
    temperature parameter for calibrated confidence (see src/eval/calibration.py).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.backbone = SharedBackbone(config)
        self.classifier = ClassifierHead(
            embedding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["classifier_hidden_dim"],
            dropout=config["model"]["dropout"],
        )
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)  # set by calibration.py

    def forward(self, images: torch.Tensor, return_embedding: bool = False):
        embedding = self.backbone(images)
        logits = self.classifier(embedding)
        calibrated_logits = logits / self.temperature
        if return_embedding:
            return calibrated_logits, embedding
        return calibrated_logits

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(images)
            return torch.sigmoid(logits)

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "temperature": self.temperature.item(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, config: dict, device: str = "cpu"):
        model = cls(config)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.temperature.data = torch.tensor([checkpoint.get("temperature", 1.0)])
        model.to(device)
        model.eval()
        return model
