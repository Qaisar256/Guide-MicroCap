# TinyVideoCap: TinyML Video Captioning on Cortex-M7

TinyVideoCap is a TinyML-compatible framework for **real-time video captioning** on resource-constrained devices. It combines:
- **CLIP-based frame embeddings**
- **Temporal pooling** (mean or lightweight attention pooling)
- **Guide–MicroCap knowledge distillation** (CE + KL + attention alignment)
- **Post-training INT8 quantization**
- **TensorFlow Lite Micro (TFLM)** deployment for Cortex-M7 (STM32H7)

## Highlights
- **INT8**: ~125 ms latency, ~932 KB flash, ~318 KB RAM, ~129 µJ per inference (Cortex-M7)
- Retains high captioning quality with minimal CIDEr drop after quantization

---

## Repository Structure (example)
TinyVideoCap/
configs/
data/
models/
scripts/
tflm/
examples/
README.md

---

## 1) Quick Start (Desktop Inference)

### 1.1 Install
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
1.2 Download/Prepare Models

Place the following in models/ (names can be changed, but keep consistent):

microcap_fp32.pt (PyTorch) or microcap_fp32.tflite

microcap_int8.tflite (recommended for fast inference)

If you provide prebuilt weights, add links here (HuggingFace/Drive/etc.).
python examples/caption_video.py \
  --video path/to/video.mp4 \
  --model models/microcap_int8.tflite \
  --pooling attention \
  --num_frames 5
Output:

Prints the generated caption

Optionally saves intermediate frame embeddings (useful for MCU deployment)
2) Tutorial: End-to-End Pipeline

TinyVideoCap follows this pipeline:

Sample frames from the input clip (default: 5 frames, ~1 FPS)

Extract CLIP embeddings per frame (grid-level or pooled embeddings)

Temporal pooling to aggregate frame embeddings into a compact clip representation

Decode caption using MicroCap (student) decoder

(Optional) Quantize to INT8 and deploy on Cortex-M7 using TFLM
3) Frame Sampling + CLIP Feature Extraction
3.1 Extract Frames
python scripts/extract_frames.py \
  --video path/to/video.mp4 \
  --out_dir data/frames/sample_001 \
  --num_frames 5 \
  --size 224
3.2 Extract CLIP Embeddings
python scripts/extract_clip_embeddings.py \
  --frames_dir data/frames/sample_001 \
  --out_file data/embeddings/sample_001.npy \
  --clip_model ViT-B/32
This produces an embedding tensor of shape like:

[T, D] for per-frame embeddings, or

[T, G, D] for grid embeddings (if enabled)
4) Temporal Pooling

We support:

Mean pooling (fastest, no parameters)

Attention pooling (slightly higher compute, better CIDEr)

Example (attention pooling)
python scripts/pool_embeddings.py \
  --in_file data/embeddings/sample_001.npy \
  --out_file data/pooled/sample_001.npy \
  --pooling attention
python scripts/pool_embeddings.py \
  --in_file data/embeddings/sample_001.npy \
  --out_file data/pooled/sample_001.npy \
  --pooling attention
5) Training + Distillation (Guide → MicroCap)
5.1 Train Teacher (optional)

If you already use LightCap as teacher, you can skip this step.

