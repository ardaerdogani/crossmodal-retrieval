import argparse
import os

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import BertTokenizer

# ── Config (must match train.ipynb) ───────────────────────────────────────────
CHECKPOINT = "checkpoints/best_caption_model.pt"
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
FFN_DIM = 1024
DROPOUT = 0.1
MAX_TOKEN_LENGTH = 32
VOCAB_SIZE = 30522
PAD_ID = 0
BOS_ID = 101  # [CLS]
EOS_ID = 102  # [SEP]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Model (mirrors train.ipynb exactly) ───────────────────────────────────────
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Conv2d(512, D_MODEL, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(images)
        feat = self.projection(feat)
        return feat.flatten(2).transpose(1, 2)  # (B, 49, D_MODEL)


class CaptionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(MAX_TOKEN_LENGTH, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.output = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.output.weight = self.token_embedding.weight  # weight tying

    def forward(self, input_ids: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        padding_mask = (input_ids == PAD_ID)
        out = self.decoder(
            tgt=x, memory=image_features,
            tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask,
        )
        return self.output(self.norm(out))


class CaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = CaptionDecoder()

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(input_ids, self.encoder(images))


# ── Load model ─────────────────────────────────────────────────────────────────
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found at '{CHECKPOINT}'. "
        "Run train.ipynb to completion first."
    )

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

model = CaptioningModel().to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded checkpoint — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f} | device: {device}")


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def caption(pil_image: Image.Image) -> str:
    image = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    image_features = model.encoder(image)

    tokens = [BOS_ID]
    for _ in range(MAX_TOKEN_LENGTH - 1):
        input_ids = torch.tensor([tokens], device=device)
        logits = model.decoder(input_ids, image_features)
        next_id = logits[0, -1].argmax().item()
        if next_id == EOS_ID:
            break
        tokens.append(next_id)

    return tokenizer.decode(tokens[1:], skip_special_tokens=True)


# ── Gradio UI ──────────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=caption,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Generated caption"),
    title="Image Captioning — Cross-Modal Generation",
    description=(
        "Upload any image and the model will generate a natural-language caption. "
        "Architecture: ResNet18 encoder (49 spatial tokens) → Transformer decoder, "
        f"trained on Flickr8k. Best val loss: {ckpt['val_loss']:.4f} (epoch {ckpt['epoch']})."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo.launch(share=args.share, server_port=args.port)
