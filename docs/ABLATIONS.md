# Ablations

Track individual contributions of each design choice.

## Loss Ablation

For each experiment, remove one loss term and measure delta vs baseline.

| Experiment | w_l1 | w_ssim | w_perc | w_freq | w_edge | w_range | SSIM | PSNR | LPIPS | Notes |
|------------|------|--------|--------|--------|--------|---------|------|------|-------|-------|
| baseline | 1.0 | 0.5 | 0.1 | 0.3 | 0.2 | 0.1 | - | - | - | Full loss |
| no_ssim | 1.0 | 0.0 | 0.1 | 0.3 | 0.2 | 0.1 | - | - | - | |
| no_perceptual | 1.0 | 0.5 | 0.0 | 0.3 | 0.2 | 0.1 | - | - | - | |
| no_frequency | 1.0 | 0.5 | 0.1 | 0.0 | 0.2 | 0.1 | - | - | - | |
| no_edge | 1.0 | 0.5 | 0.1 | 0.3 | 0.0 | 0.1 | - | - | - | |
| no_range | 1.0 | 0.5 | 0.1 | 0.3 | 0.2 | 0.0 | - | - | - | |

## Architecture Ablation

| Architecture | Params | SSIM | PSNR | Latency | Notes |
|-------------|--------|------|------|---------|-------|
| Model 1 (baseline) | ~13.5M | 0.6996 | - | ~150ms | Plain conv U-Net |
| RestorationNet | - | - | - | - | NAFNet blocks + PixelShuffle |
| TwoStageNet | - | - | - | - | Decoupled denoise+SR |

## Data Augmentation Ablation

| Augmentation | SSIM | PSNR | Notes |
|-------------|------|------|-------|
| None (original only) | - | - | |
| + Synthetic speckle | - | - | |
| + Patch crops | - | - | |
| + All synthetic | - | - | Full pipeline |

## Frequency Branch Ablation

| Use Freq Branch | SSIM | PSNR | Delta | Notes |
|----------------|------|------|-------|-------|
| No | - | - | baseline | |
| Yes | - | - | - | |
