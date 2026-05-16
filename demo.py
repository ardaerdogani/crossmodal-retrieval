import argparse
import glob
import math
import os
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# ── Pre-compute / load CLIP image embeddings ──────────────────────────────────
all_image_paths: list[str] = []
all_image_embeddings: torch.Tensor | None = None

if os.path.isdir(IMAGES_DIR):
    if os.path.exists(EMBEDDINGS_CACHE):
        cache                = torch.load(EMBEDDINGS_CACHE, map_location="cpu", weights_only=True)
        all_image_paths      = cache["paths"]
        all_image_embeddings = cache["embeddings"].to(device)
        print(f"Loaded {len(all_image_paths):,} CLIP embeddings from cache.")
    else:
        print(f"Computing CLIP embeddings for {IMAGES_DIR} (first run only — cached after this)...")
        _paths = sorted(Path(IMAGES_DIR).glob("*.jpg")) + sorted(Path(IMAGES_DIR).glob("*.png"))
        all_image_paths = [str(p) for p in _paths]
        _batch, _accum  = 128, []
        with torch.no_grad():
            for i in range(0, len(all_image_paths), _batch):
                batch_imgs = [Image.open(p).convert("RGB") for p in all_image_paths[i : i + _batch]]
                inputs     = clip_processor(images=batch_imgs, return_tensors="pt")
                inputs     = {k: v.to(device) for k, v in inputs.items()}
                embs       = clip_model.get_image_features(**inputs)
                embs       = F.normalize(embs, dim=-1)
                _accum.append(embs.cpu())
                if i % (10 * _batch) == 0:
                    print(f"  {i}/{len(all_image_paths)} done…")
        all_image_embeddings = torch.cat(_accum, dim=0)
        torch.save({"paths": all_image_paths, "embeddings": all_image_embeddings}, EMBEDDINGS_CACHE)
        all_image_embeddings = all_image_embeddings.to(device)
        print(f"Embedded {len(all_image_paths):,} images → saved to {EMBEDDINGS_CACHE}")
else:
    print(f"Warning: {IMAGES_DIR} not found — text retrieval tab will be disabled.")


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
def text_to_image(query: str):
    """Return (img1, label1, img2, label2, img3, label3) for the top-3 matches."""
    empty = [None, "", None, "", None, ""]
    if not query.strip() or all_image_embeddings is None:
        return empty

    # Encode query with CLIP
    text_inputs = clip_processor(text=[query], return_tensors="pt",
                                 truncation=True, padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_emb    = clip_model.get_text_features(**text_inputs)
    text_emb    = F.normalize(text_emb, dim=-1)

    # Cosine similarity → top 3
    sims        = (text_emb @ all_image_embeddings.T).squeeze(0)
    top_k       = min(3, len(all_image_paths))
    top_indices = sims.topk(top_k).indices.cpu().tolist()

    outputs = []
    for idx in top_indices:
        img  = Image.open(all_image_paths[idx]).convert("RGB")
        sim  = sims[idx].item()
        cap  = generate_caption(img)
        outputs.append(img)
        outputs.append(f"Similarity: {sim:.3f}\n\"{cap}\"")

    while len(outputs) < 6:
        outputs += [None, ""]

    return outputs


# ── Pre-generate gallery captions ─────────────────────────────────────────────
_example_paths = sorted(
    glob.glob("examples/*.jpg") + glob.glob("examples/*.jpeg") + glob.glob("examples/*.png")
)[:6]

gallery_items: list[tuple] = []
if _example_paths:
    print("Pre-generating gallery captions…")
    for path in _example_paths:
        img = Image.open(path).convert("RGB")
        cap = generate_caption(img)
        gallery_items.append((img, cap))
        print(f"  {os.path.basename(path)}: {cap}")


# ── Gradio UI ─────────────────────────────────────────────────────────────────
_val_loss = ckpt["val_loss"]
_ppl      = math.exp(_val_loss)
_epoch    = ckpt["epoch"]

with gr.Blocks(theme=gr.themes.Soft(), title="Cross-Modal Demo") as demo:

    gr.Markdown(f"""
# Cross-Modal Generation — Image Captioning & Text-to-Image Retrieval
ResNet18 encoder → Transformer decoder · Trained on Flickr8k &nbsp;|&nbsp;
**Val Loss: {_val_loss:.4f}** · **PPL: {_ppl:.2f}** · Best epoch: {_epoch}/20
""")

    with gr.Tabs():

        # ── Tab 1: Text → Image Retrieval ─────────────────────────────────────
        with gr.Tab("🔍 Text → Image Retrieval"):
            gr.Markdown("""
Type a description below. **CLIP** finds the closest matching images from the Flickr8k dataset
using cosine similarity in a shared text-image embedding space.
Your **custom captioning model** then reads each image and generates its own description —
showing how the two modalities connect.
""")
            with gr.Row():
                query_box  = gr.Textbox(
                    label="Text Query",
                    placeholder="e.g.  a dog playing in the water",
                    scale=4,
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    ["a dog running on the beach"],
                    ["two children playing in the snow"],
                    ["a man riding a horse"],
                    ["a woman with a red dress"],
                    ["a group of people at a party"],
                    ["a cat sitting on a chair"],
                ],
                inputs=query_box,
                label="Try one of these:",
            )

            gr.Markdown("### Top 3 Matches")
            with gr.Row():
                result_imgs = [gr.Image(show_label=False, height=260) for _ in range(3)]
            with gr.Row():
                result_caps = [
                    gr.Textbox(show_label=False, lines=3, show_copy_button=True)
                    for _ in range(3)
                ]

            search_btn.click(
                fn=text_to_image,
                inputs=query_box,
                outputs=[
                    result_imgs[0], result_caps[0],
                    result_imgs[1], result_caps[1],
                    result_imgs[2], result_caps[2],
                ],
            )
            query_box.submit(
                fn=text_to_image,
                inputs=query_box,
                outputs=[
                    result_imgs[0], result_caps[0],
                    result_imgs[1], result_caps[1],
                    result_imgs[2], result_caps[2],
                ],
            )

        # ── Tab 2: Image → Caption (Live Demo) ────────────────────────────────
        with gr.Tab("📸 Image → Caption"):
            gr.Markdown(
                "Upload any image — the model generates a natural-language caption "
                "one token at a time using greedy decoding."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    img_input  = gr.Image(type="pil", label="Input Image", height=340)
                    cap_btn    = gr.Button("Generate Caption", variant="primary", size="lg")
                with gr.Column(scale=1):
                    caption_out = gr.Textbox(
                        label="Generated Caption",
                        lines=5,
                        placeholder="Upload an image to generate a caption…",
                        show_copy_button=True,
                    )

            if _example_paths:
                gr.Examples(
                    examples=_example_paths,
                    inputs=img_input,
                    label="Example images — click to load:",
                    examples_per_page=6,
                )

            cap_btn.click(fn=generate_caption, inputs=img_input, outputs=caption_out)
            img_input.change(fn=generate_caption, inputs=img_input, outputs=caption_out)

        # ── Tab 3: Gallery ─────────────────────────────────────────────────────
        with gr.Tab("🖼️ Gallery"):
            if gallery_items:
                gr.Markdown(
                    "All captions generated automatically at server startup — "
                    "no human input, no cherry-picking."
                )
                cols = min(3, len(gallery_items))
                for row_start in range(0, len(gallery_items), cols):
                    row = gallery_items[row_start : row_start + cols]
                    with gr.Row():
                        for img, cap in row:
                            with gr.Column():
                                gr.Image(value=img, show_label=False, height=220)
                                gr.Textbox(value=f'"{cap}"', show_label=False, lines=2)
            else:
                gr.Markdown(
                    "Add `.jpg` / `.png` images to an `examples/` folder next to `demo.py` "
                    "and restart the server to populate this gallery."
                )

        # ── Tab 4: Model & Results ─────────────────────────────────────────────
        with gr.Tab("📊 Model & Results"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
### Architecture

| Component | Details |
|---|---|
| Image Encoder | ResNet18 (ImageNet pretrained), truncated → 7×7×512 → 1×1 conv → **49 image tokens** |
| Tokenizer | BERT `bert-base-uncased` (30,522 vocab) |
| Decoder | 4-layer Transformer decoder, d=256, 4 heads, FFN=1024, `norm_first=True` |
| Weight tying | Output projection shares weights with input embedding |
| Retrieval | CLIP ViT-B/32 — joint image-text embedding, cosine similarity |
| Training | Teacher forcing · AdamW · lr=3e-4 · batch=64 · 20 epochs |
| Inference | Greedy decoding — argmax at each step, stops at `[SEP]` |
""")
                with gr.Column():
                    gr.Markdown(f"""
### Training Results

| Epoch | Val Loss | Val PPL |
|:---:|:---:|:---:|
| 1 | 5.1747 | 176.74 |
| 5 | 3.3075 | 27.32 |
| 8 | 3.0613 | 21.35 |
| 10 | 3.0123 | 20.34 |
| **{_epoch}** | **{_val_loss:.4f}** | **{_ppl:.2f}** ← best |

Trained on **Flickr8k** — 8,091 images × 5 captions ≈ 40k pairs
Hardware: NVIDIA H100 NVL (~53 s / epoch)
Parameters: **23,343,936** trainable
""")


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(share=args.share, server_port=args.port)
