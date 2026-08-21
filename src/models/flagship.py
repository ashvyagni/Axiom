import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafnet_blocks import NAFBlock, FrequencyBranch, PixelShuffleUpscaler

__all__ = ["FlagshipNet", "build_flagship", "count_params"]


# ============================================================
# ATTENTION MODULES
# ============================================================

class StripAttention(nn.Module):
    """Multi-scale strip-shaped attention for directional features."""
    def __init__(self, ch):
        super().__init__()
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.v_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.proj = nn.Conv2d(ch * 2, ch, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        B, C, H, W = x.shape
        h_feat = self.h_pool(x).expand(-1, -1, H, W)
        v_feat = self.v_pool(x).expand(-1, -1, H, W)
        attn = self.sig(self.proj(torch.cat([h_feat, v_feat], dim=1)))
        return x * attn


class ECA(nn.Module):
    """Efficient Channel Attention — 1D conv, negligible cost."""
    def __init__(self, ch, k=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k//2, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        y = self.avg(x).flatten(1).unsqueeze(1)
        y = self.sig(self.conv(y)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * y


class SpatialNoiseEstimator(nn.Module):
    """Per-pixel noise confidence map (not just global scalar).

    Outputs a spatial map [0,1] indicating noise level at each pixel.
    This lets the denoiser apply different denoising strength per region.
    """
    def __init__(self, ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, ch // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(ch // 4 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(ch, ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, target_size=None):
        B, C, H, W = x.shape
        global_conf = self.head(x).view(B, 1, 1, 1)
        spatial_map = self.spatial_proj(x)
        raw = global_conf * spatial_map + (1 - global_conf) * spatial_map.mean(dim=(2,3), keepdim=True)
        if target_size is not None:
            raw = F.interpolate(raw, size=target_size, mode='bilinear', align_corners=False)
        return raw


# ============================================================
# ENCODER / DECODER
# ============================================================

class Encoder(nn.Module):
    def __init__(self, in_ch, base_ch, depth, blocks_per_level, dropout=0.0):
        super().__init__()
        self.depth = depth
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.intro = nn.Conv2d(in_ch, chs[0], 3, 1, 1)
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(nn.Sequential(*[NAFBlock(chs[i], drop_out_rate=dropout) for _ in range(blocks_per_level)]))
            if i < depth - 1:
                self.downs.append(nn.Conv2d(chs[i], chs[i+1], 3, 2, 1))
        self.chs = chs

    def forward(self, x):
        skips = []
        feat = self.intro(x)
        for i in range(self.depth):
            feat = self.blocks[i](feat)
            if i < self.depth - 1:
                skips.append(feat)
                feat = self.downs[i](feat)
        return feat, skips


class Decoder(nn.Module):
    def __init__(self, chs, blocks_per_level, dropout=0.0):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            in_c = chs[-1 - i]
            out_c = chs[-2 - i]
            self.ups.append(nn.Sequential(
                nn.Conv2d(in_c, out_c * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ))
            self.blocks.append(nn.Sequential(
                nn.Conv2d(out_c * 2, out_c, 1),
                *[NAFBlock(out_c, drop_out_rate=dropout) for _ in range(blocks_per_level)],
            ))
        self.out_ch = chs[0]

    def forward(self, feat, skips):
        for i in range(len(self.ups)):
            feat = self.ups[i](feat)
            skip = skips[-(i+1)]
            feat = torch.cat([feat, skip], dim=1)
            feat = self.blocks[i](feat)
        return feat


# ============================================================
# DENOISE STAGE
# ============================================================

class FlagshipDenoise(nn.Module):
    """Multi-pass denoising with proper NAFBlocks, spatial noise estimation, frequency branch."""
    def __init__(self, base_ch=48, depth=3, blocks=2, n_passes=2, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(1, base_ch, depth, blocks, dropout)
        chs = self.encoder.chs
        self.freq = FrequencyBranch(chs[-1])
        self.freq_proj = nn.Conv2d(chs[-1], chs[-1], 1)
        self.bottleneck = nn.Sequential(
            *[NAFBlock(chs[-1], drop_out_rate=dropout) for _ in range(blocks)],
            StripAttention(chs[-1]),
            ECA(chs[-1]),
        )
        self.decoder = Decoder(chs, blocks, dropout)
        self.noise_est = SpatialNoiseEstimator(chs[-1])
        self.n_passes = n_passes
        self.final_conv = nn.Conv2d(chs[0], 1, 3, 1, 1)

    def _pass(self, x):
        feat, skips = self.encoder(x)
        freq_f = self.freq_proj(self.freq(feat))
        feat = feat + freq_f
        feat = self.bottleneck(feat)
        out = self.decoder(feat, skips)
        return self.final_conv(out), feat

    def forward(self, lr):
        current = lr
        intermediate_outputs = []
        for p in range(self.n_passes):
            denoised, bottleneck_feat = self._pass(current)
            noise_conf = self.noise_est(bottleneck_feat, target_size=(lr.shape[2], lr.shape[3]))
            if p == 0:
                current = lr + denoised * (1 - noise_conf)
            else:
                current = current + denoised * 0.5
            intermediate_outputs.append(current)
        return torch.clamp(current, 0, 1), intermediate_outputs


# ============================================================
# SR STAGE
# ============================================================

class FlagshipSR(nn.Module):
    """2x super-resolution with proper NAFBlocks, frequency branch, strip attention."""
    def __init__(self, base_ch=48, depth=4, blocks=2, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(1, base_ch, depth, blocks, dropout)
        chs = self.encoder.chs
        self.freq = FrequencyBranch(chs[-1])
        self.freq_proj = nn.Conv2d(chs[-1], chs[-1], 1)
        self.bottleneck = nn.Sequential(
            *[NAFBlock(chs[-1], drop_out_rate=dropout) for _ in range(blocks)],
            StripAttention(chs[-1]),
            ECA(chs[-1]),
        )
        self.decoder = Decoder(chs, blocks, dropout)
        self.final_up = nn.Sequential(
            nn.Conv2d(chs[0], chs[0] * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(chs[0], 1, 3, 1, 1),
        )

    def forward(self, clean_lr):
        feat, skips = self.encoder(clean_lr)
        freq_f = self.freq_proj(self.freq(feat))
        feat = feat + freq_f
        feat = self.bottleneck(feat)
        out = self.decoder(feat, skips)
        return self.final_up(out)


# ============================================================
# FLAGSHIP NET: TWO-STAGE WITH SELF-REFINEMENT
# ============================================================

class FlagshipNet(nn.Module):
    """The nuclear option for semiconductor image restoration.

    Stage 1: Multi-pass denoising with spatial noise estimation
    Stage 2: 2x super-resolution with frequency branch
    Inference: Single-pass (fast) or self-refined (best quality)
    """
    def __init__(self, denoise_base=48, sr_base=48, denoise_depth=3, sr_depth=4,
                 denoise_blocks=2, sr_blocks=2, denoise_passes=2, dropout=0.0):
        super().__init__()
        self.denoise = FlagshipDenoise(denoise_base, denoise_depth, denoise_blocks, denoise_passes, dropout)
        self.sr = FlagshipSR(sr_base, sr_depth, sr_blocks, dropout)

    def forward(self, lr):
        clean_lr, _ = self.denoise(lr)
        hr = self.sr(clean_lr)
        hr_base = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        return torch.clamp(hr + hr_base, 0, 1)

    def forward_two_stage(self, lr):
        clean_lr, denoise_intermediates = self.denoise(lr)
        hr = self.sr(clean_lr)
        hr_base = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        return clean_lr, torch.clamp(hr + hr_base, 0, 1), denoise_intermediates

    def forward_refined(self, lr, n_refine=1):
        clean_lr, _ = self.denoise(lr)
        hr = self.sr(clean_lr)
        hr_base = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        hr = hr + hr_base
        for _ in range(n_refine):
            lr_re = F.interpolate(hr, size=lr.shape[-2:], mode='bilinear', align_corners=False)
            clean_re, _ = self.denoise(lr_re)
            hr_re = self.sr(clean_re)
            hr_base_re = F.interpolate(lr_re, size=hr.shape[-2:], mode='bilinear', align_corners=False)
            hr = hr + 0.3 * (hr_re + hr_base_re - hr)
        return clean_lr, torch.clamp(hr, 0, 1)


def build_flagship(cfg=None):
    cfg = cfg or {}
    return FlagshipNet(
        denoise_base=cfg.get('denoise_base', 48),
        sr_base=cfg.get('sr_base', 48),
        denoise_depth=cfg.get('denoise_depth', 3),
        sr_depth=cfg.get('sr_depth', 4),
        denoise_blocks=cfg.get('denoise_blocks', 2),
        sr_blocks=cfg.get('sr_blocks', 2),
        denoise_passes=cfg.get('denoise_passes', 2),
        dropout=cfg.get('dropout', 0.0),
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
