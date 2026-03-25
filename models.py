import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import BertModel

EMBED_DIM = 256


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove final classification layer — keep everything up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: (batch, 512, 1, 1)

        self.projection = nn.Sequential(
            nn.Linear(512, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images).squeeze(-1).squeeze(-1)  # (batch, 512)
        projected = self.projection(features)                      # (batch, 256)
        return F.normalize(projected, dim=-1)                      # L2-normalized


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.projection = nn.Sequential(
            nn.Linear(768, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token — (batch, 768)
        projected = self.projection(cls_embedding)            # (batch, 256)
        return F.normalize(projected, dim=-1)                 # L2-normalized
