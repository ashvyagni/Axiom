# Axiom — FlagshipNet for Semiconductor Image Restoration

SEMICON India Hackathon 2026 | KLA Track (Track 2) | Team GitWarriors — VIT Vellore

Members: Ashwin Dandotiya, Hiral Jadani, Dhun Bamboli, Vairaj Malhotra

---

## Overview

Axiom implements **FlagshipNet**, a two-stage deep learning pipeline for joint denoising and 2x super-resolution of degraded grayscale semiconductor (SEM/TEM) images.

- **Input**: 128x128 degraded grayscale `.npy` image
- **Output**: 256x256 restored `.npy` image

### Architecture

```
LR Input (128x128)
    |
[Stage 1: FlagshipDenoise]
  - 3-level U-Net encoder with NAFBlocks
  - FFT Frequency Branch (captures periodic crystal structures)
  - Strip Attention (horizontal/vertical directional features)
  - Noise Estimator (confidence gating)
  - 2-pass iterative refinement
    |
Clean LR (128x128)
    |
[Stage 2: FlagshipSR]
  - 4-level U-Net encoder with NAFBlocks
  - FFT Frequency Branch + Strip Attention
  - PixelShuffle 2x learned upsampling
  - Residual connection (bilinear skip)
    |
HR Output (256x256)
```

**Parameters**: ~9.9M

---

## Training

> **Note:** Trained weights are not yet available. You must train the model before running inference.

### Option 1: Google Colab (Recommended)

1. Upload `model2_oc/` and `data/` to your Google Drive
2. Open Google Colab → Runtime → Change runtime type → **T4 GPU**
3. Copy cells from `COLAB_TRAINING.py` into a new Colab notebook
4. Run all cells — training takes ~2-4 hours on T4 GPU

### Option 2: Local Training

```bash
pip install torch torchvision pillow matplotlib numpy scikit-image
python scripts/train_flagship.py --epochs 200 --batch_size 16 --lr 2e-4
```

Best weights are saved to `weights/best_model.pt` when validation SSIM improves.

---

## Inference

```bash
# Single image
python scripts/run_flagship.py <input-dir> <output-dir>

# Example
python scripts/run_flagship.py data/val/degraded output/
```

Output folder contains:
- `*.npy` — restored 256x256 images
- `*_comparison.png` — side-by-side [LR | Restored | GT] (when GT available)

### Test-Time Augmentation (TTA)

Add `--tta` for 8-augmentation averaging (~8x slower, +0.01-0.02 SSIM):
```bash
python scripts/run_flagship.py <input-dir> <output-dir> --tta
```

---

## Dataset

Download: https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6

Place as `data/` with structure:
```
data/
  train/gt/       # 256x256 ground truth .npy
  train/degraded/ # 128x128 degraded .npy
  val/gt/
  val/degraded/
```

---

## 9-Term Composite Loss

| Term | Weight | Purpose |
|------|--------|---------|
| Charbonnier | 1.0 | Smooth L1 — better gradients at edges |
| SSIM | 0.3 | Structural similarity |
| Multi-scale SSIM | 0.1 | SSIM at 3 scales |
| FFT Magnitude | 0.25 | Frequency-domain consistency |
| FFT Phase | 0.1 | Phase preservation |
| Wavelet (Haar) | 0.3 | Multi-resolution decomposition |
| Edge (Sobel) | 0.2 | Gradient magnitude matching |
| Range | 0.05 | Penalize out-of-[0,1] outputs |
| Noise Consistency | 0.1 | Consistent denoising confidence |

Curriculum scheduling: early epochs focus on pixel fidelity, later epochs on structural quality.

---

## Key Innovations

1. **NAFNet blocks** — SimpleGate activation (parameter-free), LayerNorm2d, depthwise separable convolutions
2. **FFT Frequency Branch** — captures periodic semiconductor structures (crystal lattices, scan patterns)
3. **Strip Attention** — horizontal/vertical pooling for directional features (scan lines, layer boundaries)
4. **Noise-aware confidence gating** — adaptive blending based on estimated noise level
5. **Multi-pass iterative refinement** — progressive cleanup over 2 passes
6. **Curriculum loss scheduling** — automatic weight adjustment from pixel-fidelity to structural-quality
7. **Self-refinement loop** — iterative denoise-SR for highest quality

---

## Benchmarking

```bash
python scripts/benchmark.py --data_dir data --weights weights/best_model.pt
```

---

## ONNX Export

```bash
python scripts/export_onnx.py --weights weights/best_model.pt --output weights/flagshipnet.onnx
```

---

## Requirements

```
torch>=2.0
numpy
Pillow
scikit-image
matplotlib
pyyaml
```

```bash
pip install -r requirements.txt
```

---

## License

Internal use for SEMICON India Hackathon 2026.
