# Experiments Log

Track every experiment with config, metrics, and interpretation.

## Baseline

| Metric | Model 1 (Round 1) |
|--------|-------------------|
| SSIM | 0.6996 |
| Params | ~13.5M |
| Inference | ~150ms CPU |

## Candidate Experiments

### Experiment 001: RestorationNet + Full Composite Loss

- **Config**: `configs/candidate2_nafnet_unet.yaml`
- **Architecture**: RestorationNet (NAFNet blocks, PixelShuffle, freq branch)
- **Loss**: L1(1.0) + SSIM(0.5) + Perceptual(0.1) + Frequency(0.3) + Edge(0.2) + Range(0.1)
- **Data**: All training data + synthetic augmentation
- **Status**: PENDING
- **Results**:
  - Val SSIM: -
  - Val PSNR: -
  - Val LPIPS: -
  - Latency: -
  - Params: -

### Experiment 002: TwoStageNet + Full Composite Loss

- **Config**: `configs/candidate1_two_stage.yaml`
- **Architecture**: TwoStageNet (shared encoder, denoise + SR heads)
- **Loss**: L1(1.0) + SSIM(0.5) + Perceptual(0.1) + Frequency(0.3) + Edge(0.2) + Range(0.1)
- **Data**: All training data + synthetic augmentation
- **Status**: PENDING
- **Results**:
  - Val SSIM: -
  - Val PSNR: -
  - Val LPIPS: -
  - Latency: -
  - Params: -

## OOD Evaluation

Pending: Run best model on held-out OOD split.

## Final Selected Model

Pending: Planner selects winner based on combined score (SSIM * 0.4 + PSNR/50 * 0.3 - LPIPS * 0.3 - latency_penalty).
