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


class CLIPLoss(nn.Module):
    def __init__(self, initial_temperature: float = 0.07):
        super().__init__()
        # Learnable log-temperature: optimized during training alongside the encoders.
        # We store log(temperature) so that exp() always yields a positive scale factor.
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        # Both inputs are L2-normalized, so matmul == cosine similarity
        # logits shape: (N, N)
        temperature = self.log_temperature.exp()
        logits = (image_embeddings @ text_embeddings.T) / temperature

        # Targets: the diagonal — pair i matches pair i
        N = logits.size(0)
        targets = torch.arange(N, device=logits.device)

        # Symmetric cross-entropy: image→text + text→image
        loss_i2t = F.cross_entropy(logits, targets)       # rows
        loss_t2i = F.cross_entropy(logits.T, targets)     # columns
        return (loss_i2t + loss_t2i) / 2
