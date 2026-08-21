import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafnet_blocks import NAFBlock, FrequencyBranch, PixelShuffleUpscaler
from .two_stage import TwoStageNet  # noqa: F401


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.conv(x)


class RestorationNet(nn.Module):
    """Candidate 2: NAFNet-block U-Net with PixelShuffle upsampling.

    Replaces plain conv blocks with NAFBlock (SimpleGate, no expensive softmax attention).
    Uses PixelShuffle for clean 2x upscaling. Lightweight frequency-aware branch in bottleneck.
    Single-stage joint denoise+SR.

    Channel flow (base_ch=48):
        intro:       1 -> 48
        encoder_blocks_1: 48 -> 48       (e1, 64x64)
        down1:       48 -> 96
        encoder_blocks_2: 96 -> 96       (e2, 32x32)
        down2:       96 -> 192
        encoder_blocks_3: 192 -> 192     (e3, 16x16)
        freq_branch: 192 -> 192
        up1:         192 -> 96           (32x32)
        cat:         96 + 96 = 192
        decoder_blocks_1: 192 -> 192    (d1)
        up2:         192 -> 48           (64x64)
        cat:         48 + 48 = 96
        decoder_blocks_2: 96 -> 96      (d2)
        final_up:    96 -> 96            (128x128)
        out_conv:    96 -> 1
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=48, num_blocks=2, use_freq_branch=True):
        super().__init__()
        self.use_freq_branch = use_freq_branch

        # Initial convolution
        self.intro = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        # Encoder
        self.down1 = Downsample(base_ch, base_ch * 2)
        self.down2 = Downsample(base_ch * 2, base_ch * 4)

        # NAFBlocks at each level
        self.encoder_blocks_1 = nn.Sequential(*[NAFBlock(base_ch) for _ in range(num_blocks)])
        self.encoder_blocks_2 = nn.Sequential(*[NAFBlock(base_ch * 2) for _ in range(num_blocks)])
        self.encoder_blocks_3 = nn.Sequential(*[NAFBlock(base_ch * 4) for _ in range(num_blocks)])

        # Frequency branch at bottleneck
        if use_freq_branch:
            self.freq_branch = FrequencyBranch(base_ch * 4)

        # Decoder — upsample channels must match the concat inputs
        self.up1 = Upsample(base_ch * 4, base_ch * 2)       # 192 -> 96
        self.up2 = Upsample(base_ch * 2 + base_ch * 2, base_ch)  # 96+96=192 -> 48

        self.decoder_blocks_1 = nn.Sequential(*[NAFBlock(base_ch * 2 + base_ch * 2) for _ in range(num_blocks)])  # cat(96,96)=192
        self.decoder_blocks_2 = nn.Sequential(*[NAFBlock(base_ch + base_ch) for _ in range(num_blocks)])          # cat(48,48)=96

        # Final: 2x learned upscale + output
        self.final_upscale = PixelShuffleUpscaler(base_ch * 2)
        self.out_conv = nn.Conv2d(base_ch * 2, out_ch, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        target_h, target_w = h * 2, w * 2
        base_upsample = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)

        inp = self.intro(x)

        # Encode
        e1 = self.encoder_blocks_1(inp)           # (B, 48, H, W)
        e2 = self.encoder_blocks_2(self.down1(e1))  # (B, 96, H/2, W/2)
        e3 = self.encoder_blocks_3(self.down2(e2))  # (B, 192, H/4, W/4)

        # Frequency branch at bottleneck
        if self.use_freq_branch:
            e3 = e3 + self.freq_branch(e3)

        # Decode with skip connections
        # up1(e3): (B, 96, H/2, W/2), cat with e2: (B, 192, H/2, W/2)
        d1 = self.decoder_blocks_1(torch.cat([self.up1(e3), e2], dim=1))
        # up2(d1): (B, 48, H, W), cat with e1: (B, 96, H, W)
        d2 = self.decoder_blocks_2(torch.cat([self.up2(d1), e1], dim=1))

        # Final upscale to target resolution
        out = self.final_upscale(d2)
        out = self.out_conv(out)
        out = torch.sigmoid(out)

        # Global residual
        return torch.clamp(out + base_upsample, 0.0, 1.0)
