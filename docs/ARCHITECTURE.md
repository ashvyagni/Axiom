# Architecture

## Overview

Model 2 ("Grand Finale") implements two candidate architectures for semiconductor image restoration (joint denoise + 2x super-resolution):

1. **RestorationNet** (Candidate 2): Single-stage NAFNet-block U-Net with PixelShuffle upsampling
2. **TwoStageNet** (Candidate 1): Two-stage decoupled restoration (denoise → SR) with shared encoder

Both replace the plain conv blocks of Model 1 with NAFNet-style building blocks (SimpleGate, LayerNorm-in-conv) and use PixelShuffle for clean 2x upscaling without checkerboard artifacts.

## Candidate 2: RestorationNet

```
Input (1, H, W)
    ↓
Intro Conv → base_ch features
    ↓
[NAFBlock x num_blocks] → E1
    ↓ Downsample
[NAFBlock x num_blocks] → E2
    ↓ Downsample
[NAFBlock x num_blocks] → E3 + FrequencyBranch(E3)
    ↓ Upsample + skip
[NAFBlock x num_blocks] → D1
    ↓ Upsample + skip
[NAFBlock x num_blocks] → D2
    ↓ PixelShuffle 2x
Final Conv → sigmoid → Output (1, 2H, 2W)
    + bilinear upsample residual
```

## Candidate 1: TwoStageNet

```
Input (1, H, W)
    ↓ SharedEncoder
    ├→ E1, E2, E3
    ↓
DenoiseHead(E1, E2, E3, input) → Clean LR (1, H, W)
    ↓ (re-encoded)
SRHead(E1', E2', E3', clean_lr) → Clean HR (1, 2H, 2W)
```

Training uses noise-aware strategy: SR head sees both denoised output AND ground-truth clean LR.

## NAFBlock

- LayerNorm2d → 1x1 Conv (expand) → 3x3 DepthwiseConv → SimpleGate → 1x1 Conv (contract) + residual
- LayerNorm2d → 1x1 Conv (FFN expand) → SimpleGate → 1x1 Conv (FFN contract) + residual
- No softmax attention → fast inference
- Learnable scaling parameters (beta, gamma)

## FrequencyBranch

- FFT → magnitude + phase → 1x1 conv on concatenated repr → modified FFT → IFFT
- Captures periodic/periodic structure in semiconductor patterns
- Adds frequency-domain features back to spatial features

## Key Design Decisions

1. **NAFNet blocks** over plain conv: consistently outperform on denoising at comparable params
2. **PixelShuffle** over transposed conv: no checkerboard artifacts
3. **Shared encoder** in TwoStageNet: keeps param count down
4. **Frequency branch**: exploits periodic semiconductor structure
5. **Global residual**: bilinear upsample skip for stable gradients
