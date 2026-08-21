import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafnet_blocks import NAFBlock, FrequencyBranch, PixelShuffleUpscaler


class SharedEncoder(nn.Module):
    """Shared feature encoder for both denoise and SR stages."""
    def __init__(self, in_ch=1, base_ch=48):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1)

        self.blocks1 = nn.Sequential(*[NAFBlock(base_ch) for _ in range(2)])
        self.blocks2 = nn.Sequential(*[NAFBlock(base_ch * 2) for _ in range(2)])
        self.blocks3 = nn.Sequential(*[NAFBlock(base_ch * 4) for _ in range(2)])

    def forward(self, x):
        inp = self.intro(x)
        e1 = self.blocks1(inp)
        e2 = self.blocks2(self.down1(e1))
        e3 = self.blocks3(self.down2(e2))
        return e1, e2, e3


class DenoiseHead(nn.Module):
    """Stage 1: Denoise - removes speckle from LR input, outputs clean LR.

    Channel flow (base_ch=48):
        up1: 192 -> Conv2d(192,192) + PixelShuffle -> 48 @ 2x spatial
        But we want 96... Let me use the same pattern as RestorationNet.

        Actually simpler: just use bilinear upsample + conv.
        e3 (192, H/4, W/4) -> upsample to (192, H/2, W/2) -> Conv to 96 -> cat with e2 (96)
        Then upsample to (96, H, W) -> Conv to 48 -> cat with e1 (48) -> Conv to output
    """
    def __init__(self, base_ch=48):
        super().__init__()
        self.freq = FrequencyBranch(base_ch * 4)

        # Decoder upsamples using bilinear + conv
        self.up1_conv = nn.Conv2d(base_ch * 4 + base_ch * 2, base_ch * 2, 3, 1, 1)
        self.up2_conv = nn.Conv2d(base_ch * 2 + base_ch, base_ch, 3, 1, 1)

        self.blocks1 = nn.Sequential(*[NAFBlock(base_ch * 2) for _ in range(2)])
        self.blocks2 = nn.Sequential(*[NAFBlock(base_ch) for _ in range(2)])

        self.out_conv = nn.Conv2d(base_ch, 1, 3, 1, 1)

    def forward(self, e1, e2, e3, lr_input):
        e3 = e3 + self.freq(e3)

        # Upsample e3 to match e2 spatial dims, then concat
        e3_up = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.blocks1(self.up1_conv(torch.cat([e3_up, e2], dim=1)))

        # Upsample d1 to match e1 spatial dims, then concat
        d1_up = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.blocks2(self.up2_conv(torch.cat([d1_up, e1], dim=1)))

        denoised = self.out_conv(d2)
        denoised = torch.sigmoid(denoised)

        return torch.clamp(denoised + lr_input, 0.0, 1.0)


class SRHead(nn.Module):
    """Stage 2: Super-resolution - upsamples clean LR to HR."""
    def __init__(self, base_ch=48):
        super().__init__()
        self.freq = FrequencyBranch(base_ch * 4)

        self.up1_conv = nn.Conv2d(base_ch * 4 + base_ch * 2, base_ch * 2, 3, 1, 1)
        self.up2_conv = nn.Conv2d(base_ch * 2 + base_ch, base_ch, 3, 1, 1)

        self.blocks1 = nn.Sequential(*[NAFBlock(base_ch * 2) for _ in range(2)])
        self.blocks2 = nn.Sequential(*[NAFBlock(base_ch) for _ in range(2)])

        # Final 2x learned upscale
        self.final_up = PixelShuffleUpscaler(base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, 1, 1)

    def forward(self, e1, e2, e3, hr_base):
        e3 = e3 + self.freq(e3)

        e3_up = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.blocks1(self.up1_conv(torch.cat([e3_up, e2], dim=1)))

        d1_up = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.blocks2(self.up2_conv(torch.cat([d1_up, e1], dim=1)))

        out = self.final_up(d2)
        out = self.out_conv(out)
        out = torch.sigmoid(out)

        return torch.clamp(out + hr_base, 0.0, 1.0)


class TwoStageNet(nn.Module):
    """Candidate 1: Two-stage decoupled restoration with shared encoder.

    Stage 1 (Denoise): noisy-LR -> clean-LR
    Stage 2 (SR): clean-LR -> clean-HR

    Both share a lightweight feature encoder. Each has its own decoder head.
    During training, the SR head also sees ground-truth clean LR (noise-aware training).
    At inference, SR head takes denoised output from stage 1.
    """
    def __init__(self, in_ch=1, base_ch=48):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, base_ch)
        self.denoise_head = DenoiseHead(base_ch)
        self.sr_head = SRHead(base_ch)

    def forward(self, x, gt_clean_lr=None):
        e1, e2, e3 = self.encoder(x)

        # Stage 1: Denoise
        denoised = self.denoise_head(e1, e2, e3, x)

        # Stage 2: SR - use GT clean LR during training if available
        sr_input = denoised if gt_clean_lr is None else gt_clean_lr

        # Re-encode the SR input
        sr_e1, sr_e2, sr_e3 = self.encoder(sr_input)
        hr_base = F.interpolate(sr_input, size=(sr_input.shape[2]*2, sr_input.shape[3]*2),
                                mode='bilinear', align_corners=False)

        restored = self.sr_head(sr_e1, sr_e2, sr_e3, hr_base)

        return denoised, restored
