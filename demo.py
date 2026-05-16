import argparse
import base64
import io
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import BertTokenizer, CLIPModel, CLIPProcessor

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT       = "checkpoints/best_caption_model.pt"
EMBEDDINGS_CACHE = "dataset/clip_embeddings.pt"
IMAGES_DIR       = "dataset/Images"

D_MODEL          = 256
N_HEADS          = 4
N_LAYERS         = 4
FFN_DIM          = 1024
DROPOUT          = 0.1
MAX_TOKEN_LENGTH = 32
VOCAB_SIZE       = 30522
PAD_ID           = 0
BOS_ID           = 101   # [CLS]
EOS_ID           = 102   # [SEP]
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]


# ── Captioning model (mirrors train.ipynb) ────────────────────────────────────
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
        self.token_embedding    = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(MAX_TOKEN_LENGTH, D_MODEL)
        self.dropout            = nn.Dropout(DROPOUT)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N_LAYERS)
        self.norm    = nn.LayerNorm(D_MODEL)
        self.output  = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        B, T      = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x         = self.token_embedding(input_ids) + self.position_embedding(positions)
        x         = self.dropout(x)
        causal_mask  = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)
        padding_mask = (input_ids == PAD_ID)
        out = self.decoder(tgt=x, memory=image_features,
                           tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask)
        return self.output(self.norm(out))


class CaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = CaptionDecoder()

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(input_ids, self.encoder(images))


# ── Device ────────────────────────────────────────────────────────────────────
device = (
    "cuda"  if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

# ── Load captioning model ─────────────────────────────────────────────────────
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found at '{CHECKPOINT}'. Run train.ipynb first."
    )

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

cap_model = CaptioningModel().to(device)
ckpt      = torch.load(CHECKPOINT, map_location=device, weights_only=True)
cap_model.load_state_dict(ckpt["model"])
cap_model.eval()
print(f"Captioning model loaded — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

# ── Load CLIP ─────────────────────────────────────────────────────────────────
print("Loading CLIP ViT-B/32...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("CLIP ready.")

# ── Pick image source: full dataset → examples/ fallback ──────────────────────
_examples_dir = Path("examples")
if os.path.isdir(IMAGES_DIR):
    _img_source  = Path(IMAGES_DIR)
    _cache_file  = EMBEDDINGS_CACHE
elif _examples_dir.is_dir() and any(_examples_dir.glob("*.[jp][pn]g")):
    _img_source  = _examples_dir
    _cache_file  = "examples/clip_embeddings.pt"
    print("Full dataset not found — retrieval will use examples/ folder.")
else:
    _img_source  = None
    _cache_file  = None
    print("No image source found — text retrieval disabled.")

# ── Pre-compute / load CLIP image embeddings ──────────────────────────────────
all_image_paths: list[str] = []
all_image_embeddings: torch.Tensor | None = None

if _img_source is not None:
    if _cache_file and os.path.exists(_cache_file):
        cache                = torch.load(_cache_file, map_location="cpu", weights_only=True)
        all_image_paths      = cache["paths"]
        all_image_embeddings = cache["embeddings"].to(device)
        print(f"Loaded {len(all_image_paths):,} CLIP embeddings from cache.")
    else:
        print(f"Computing CLIP embeddings for {_img_source} (cached after this)...")
        _paths = sorted(_img_source.glob("*.jpg")) + sorted(_img_source.glob("*.png"))
        all_image_paths = [str(p) for p in _paths]
        _batch, _accum  = 128, []
        with torch.no_grad():
            for i in range(0, len(all_image_paths), _batch):
                batch_imgs = [Image.open(p).convert("RGB") for p in all_image_paths[i : i + _batch]]
                inputs     = clip_processor(images=batch_imgs, return_tensors="pt")
                inputs     = {k: v.to(device) for k, v in inputs.items()}
                vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
                embs       = clip_model.visual_projection(vision_out.pooler_output)
                embs       = F.normalize(embs, dim=-1)
                _accum.append(embs.cpu())
                if i % (10 * _batch) == 0:
                    print(f"  {i}/{len(all_image_paths)} done…")
        all_image_embeddings = torch.cat(_accum, dim=0)
        if _cache_file:
            torch.save({"paths": all_image_paths, "embeddings": all_image_embeddings}, _cache_file)
        all_image_embeddings = all_image_embeddings.to(device)
        print(f"Embedded {len(all_image_paths):,} images.")


# ── Inference helpers ─────────────────────────────────────────────────────────
@torch.no_grad()
def generate_caption(pil_image: Image.Image) -> str:
    if pil_image is None:
        return ""
    image         = img_transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    image_features = cap_model.encoder(image)
    tokens = [BOS_ID]
    for _ in range(MAX_TOKEN_LENGTH - 1):
        input_ids = torch.tensor([tokens], device=device)
        logits    = cap_model.decoder(input_ids, image_features)
        next_id   = logits[0, -1].argmax().item()
        if next_id == EOS_ID:
            break
        tokens.append(next_id)
    return tokenizer.decode(tokens[1:], skip_special_tokens=True)


@torch.no_grad()
def text_to_image(query: str) -> list[tuple[Image.Image, float, str]]:
    """Return [(PIL.Image, similarity_score, caption), ...] for the top-3 CLIP matches."""
    if not query.strip() or all_image_embeddings is None:
        return []

    text_inputs = clip_processor(text=[query], return_tensors="pt",
                                 truncation=True, padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_out    = clip_model.text_model(input_ids=text_inputs["input_ids"],
                                        attention_mask=text_inputs["attention_mask"])
    text_emb    = clip_model.text_projection(text_out.pooler_output)
    text_emb    = F.normalize(text_emb, dim=-1)

    sims        = (text_emb @ all_image_embeddings.T).squeeze(0)
    top_k       = min(3, len(all_image_paths))
    top_indices = sims.topk(top_k).indices.cpu().tolist()

    results: list[tuple[Image.Image, float, str]] = []
    for idx in top_indices:
        img = Image.open(all_image_paths[idx]).convert("RGB")
        results.append((img, float(sims[idx].item()), generate_caption(img)))
    return results


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


def _pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/caption", methods=["POST"])
def caption_route():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400
    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400
    return jsonify({"caption": generate_caption(img)})


@app.route("/retrieve", methods=["POST"])
def retrieve_route():
    data  = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is empty"}), 400
    if all_image_embeddings is None:
        return jsonify({"error": "no image index available on this server"}), 503

    results = text_to_image(query)
    return jsonify([
        {"image": _pil_to_data_uri(img), "score": round(score, 4), "caption": cap}
        for img, score, cap in results
    ])


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
