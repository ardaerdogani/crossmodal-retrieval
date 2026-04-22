# Crossmodal Generation — Image Captioning

An image captioning model built from scratch with PyTorch. A ResNet18 encoder produces a grid of spatial image features; a transformer decoder cross-attends to those features and autoregressively generates a caption, one token at a time.

## Overview

The task is **cross-modal generation**: conditioned on an image (non-text modality), the model produces text (a natural-language caption). During training, the decoder is shown ground-truth tokens shifted right (*teacher forcing*) and predicts the next token at every position. At inference, it generates one token at a time (greedy decoding), stopping on `[SEP]` or at max length.

```
Image (224×224)                Caption tokens (shifted right)
      │                                       │
[ResNet18 → 7×7×512]              [Token + position embeddings]
      │                                       │
[1×1 conv → d_model]                          │
      │                                       │
  49 image "tokens" ────cross-attention────> [Transformer Decoder × N]
                                              │
                                       [Linear → vocab logits]
                                              │
                                       next-token prediction
```

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,091 images with 5 human-written captions each (~40k image-caption pairs).

The dataset is downloaded automatically via `kagglehub` when the notebook is first run. A [Kaggle account and API token](https://www.kaggle.com/settings/account) are required.

## Architecture

| Component | Details |
|---|---|
| Image encoder | ResNet18 (pretrained on ImageNet), truncated before global pooling → 7×7×512 feature map → 1×1 conv projects each cell to 256-d → flattened to 49 "image tokens" |
| Tokenizer | BERT `bert-base-uncased` WordPiece (~30k vocab). `[CLS]` is reused as BOS, `[SEP]` as EOS, `[PAD]` (id 0) is ignored by loss and attention |
| Decoder | 4-layer transformer decoder (`nn.TransformerDecoder`), `norm_first=True`, causal self-attention + cross-attention to the 49 image tokens, learned token + position embeddings |
| Output head | Linear → vocab logits, **weight-tied** to the input token embedding |
| Loss | Cross-entropy over predicted-vs-next-token, `ignore_index=pad_id` |

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Batch size | 64 |
| Epochs | 20 |
| Max token length | 32 |
| `d_model` | 256 |
| Attention heads | 4 |
| Decoder layers | 4 |
| FFN dim | 1024 |
| Dropout | 0.1 |
| Gradient clip (max-norm) | 1.0 |
| Train / val split | 90 / 10 (seed 42) |

### Results

| Epoch | Val Loss | Val PPL | Checkpoint |
|:---:|:---:|:---:|:---:|
| 1  | 5.1747 | 176.74 | saved |
| 2  | 4.1907 | 66.07  | saved |
| 3  | 3.7755 | 43.62  | saved |
| 4  | 3.4815 | 32.51  | saved |
| 5  | 3.3075 | 27.32  | saved |
| 6  | 3.2293 | 25.26  | saved |
| 7  | 3.1301 | 22.88  | saved |
| 8  | 3.0613 | 21.35  | saved |
| 9  | 3.0237 | 20.57  | saved |
| 10 | 3.0123 | 20.34  | saved |
| 11 | 2.9936 | 19.96  | saved |
| **12** | **2.9827** | **19.74** | **saved ← best** |
| 13 | 2.9895 | 19.88  | |
| 14 | 3.0294 | 20.69  | |
| 15 | 3.0186 | 20.46  | |
| 16 | 3.0274 | 20.64  | |
| 17 | 3.1100 | 22.42  | |
| 18 | 3.1065 | 22.34  | |
| 19 | 3.1386 | 23.07  | |
| 20 | 3.1152 | 22.54  | |

Best validation loss **2.9827** (PPL 19.74) at epoch 12. Perplexity — `exp(loss)` — is the more interpretable metric here: on average the model is choosing the correct next word from ~20 plausible candidates out of a 30k vocabulary. Mild overfitting appears from epoch 13 onward as train loss continues to drop while val loss creeps up. Each epoch takes ~53s on an NVIDIA H100.

## Visualization

### Generated Captions vs. Ground Truth

Six unique images from the validation set, with the model-generated caption next to a human-written reference.

![Captioning Demo](assets/captioning_demo.png)

## Demo

After training, run the Gradio web app to caption any image interactively:

```bash
python demo.py          # local only  →  http://localhost:7860
python demo.py --share  # public link →  https://<random>.gradio.live
python demo.py --port 8080  # custom port
```

Upload any image and the model generates a caption in real time. No dataset required — only the checkpoint.

## Project Structure

```
crossmodal-retrieval/
├── train.ipynb              # Self-contained notebook: data, model, training, eval
├── demo.py                  # Gradio web demo (loads checkpoint, captions any image)
├── requirements.txt         # Python dependencies
├── assets/
│   └── captioning_demo.png  # Generated caption demo (committed)
└── checkpoints/             # Saved model weights (gitignored)
    └── best_caption_model.pt
```

All code lives in `train.ipynb`. Cell structure:

| Cells | Section |
|---|---|
| 1 | Install dependencies |
| 3 | Download Flickr8k dataset |
| 4–6 | Imports, config, device setup (GPU pinning via `CUDA_VISIBLE_DEVICES`) |
| 8–9 | Data pipeline (`Flickr8kDataset`, train/val split) |
| 11 | Image encoder (ResNet18 → 49 image tokens) |
| 13 | Caption decoder (transformer with causal self-attn + cross-attn) |
| 15 | Model and optimizer instantiation |
| 17 | Training functions (teacher-forced loss, validation) |
| 19 | Training loop with best-checkpoint saving |
| 21 | Greedy caption generation |
| 23 | Demo grid (generated vs. ground-truth captions) |

## Key Concepts

- **Teacher forcing** — at training time the decoder is fed the ground-truth previous tokens rather than its own predictions, so every position can be trained in parallel.
- **Causal mask** — upper-triangular mask that blocks each token from attending to future positions, enforcing left-to-right autoregressive behavior.
- **Cross-attention** — lets each decoder position query the 49 image tokens; this is how the image conditions the generated text.
- **Weight tying** — the output projection reuses the input embedding matrix. Halves the parameter count of the head and usually improves generalization.
- **Greedy decoding** — at each step, pick the token with the highest probability and append it. Simple and deterministic; beam search would improve fluency at extra compute cost.

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (tested on NVIDIA H100 NVL)

```
torch
torchvision
transformers
kagglehub
matplotlib
```

## Setup

**Local**
```bash
git clone https://github.com/ardaerdogani/crossmodal-retrieval
cd crossmodal-retrieval
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook train.ipynb   # train the model
python demo.py --share         # run the demo
```

**JupyterHub**

Clone the repo inside JupyterHub, then open `train.ipynb` and run all cells. The first two cells install dependencies and download the dataset automatically.

A Kaggle token must be configured before the dataset cell runs:
```bash
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If running on a shared GPU host, edit the `CUDA_VISIBLE_DEVICES` line in the device-setup cell to pick a free GPU before the kernel first touches CUDA.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017
- Xu et al., [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), 2015
- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet), 2015
- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805), 2018
