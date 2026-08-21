import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FlagshipNet", "build_flagship", "count_params"]

# ============================================================
# BUILDING BLOCKS
# ============================================================

class LayerNorm2d(nn.Module):
    def __init__(self, ch, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(ch))
        self.b = nn.Parameter(torch.zeros(ch))
        self.eps = eps
    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = F.layer_norm(x, (x.shape[-1],), self.w, self.b, self.eps)
        return x.permute(0,3,1,2)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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


class NAFBlock(nn.Module):
    """NAFNet block + ECA. gamma init 0 → identity start."""
    def __init__(self, ch):
        super().__init__()
        dw = ch * 2
        self.conv1 = nn.Conv2d(ch, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw)
        self.norm = LayerNorm2d(dw)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(ch, ch, 1)
        self.eca = ECA(ch)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        inp = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.norm(y)
        y = self.sg(y)
        y = self.conv3(y)
        y = self.eca(y)
        return inp + y * self.gamma


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


# ============================================================
# FREQUENCY PROCESSING
# ============================================================

class FrequencyBranch(nn.Module):
    """FFT magnitude → spatial features. Exploits periodic semiconductor structure."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
        )
    def forward(self, x):
        fft = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(fft)
        rec = torch.fft.irfft2(torch.complex(mag, torch.zeros_like(mag)), s=x.shape[-2:], norm='ortho')
        rec = rec / (rec.std(dim=(2,3), keepdim=True) + 1e-8)
        return self.conv(torch.cat([rec, x], dim=1))


# ============================================================
# ENCODER / DECODER
# ============================================================

class Encoder(nn.Module):
    def __init__(self, in_ch, base_ch, depth, blocks_per_level):
        super().__init__()
        self.depth = depth
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.intro = nn.Conv2d(in_ch, chs[0], 3, 1, 1)
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(nn.Sequential(*[NAFBlock(chs[i]) for _ in range(blocks_per_level)]))
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
    def __init__(self, chs, blocks_per_level):
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
                *[NAFBlock(out_c) for _ in range(blocks_per_level)],
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
# NOISE ESTIMATION
# ============================================================

class NoiseEstimator(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(ch * 16, ch),
            nn.ReLU(inplace=True),
            nn.Linear(ch, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.head(x).view(x.shape[0], 1, 1, 1)


# ============================================================
# DENOISE STAGE
# ============================================================

class FlagshipDenoise(nn.Module):
    """Multi-pass denoising with noise estimation + frequency branch."""
    def __init__(self, base_ch=48, depth=3, blocks=2, n_passes=2):
        super().__init__()
        self.encoder = Encoder(1, base_ch, depth, blocks)
        chs = self.encoder.chs
        self.freq = FrequencyBranch(chs[-1], chs[-1] // 2)
        self.freq_proj = nn.Conv2d(chs[-1] // 2, chs[-1], 1)
        self.bottleneck = nn.Sequential(
            *[NAFBlock(chs[-1]) for _ in range(blocks)],
            StripAttention(chs[-1]),
        )
        self.decoder = Decoder(chs, blocks)
        self.noise_est = NoiseEstimator(chs[-1])
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
        for p in range(self.n_passes):
            denoised, bottleneck_feat = self._pass(current)
            noise_conf = self.noise_est(bottleneck_feat)
            if p == 0:
                current = lr + denoised * (1 - noise_conf)
            else:
                current = current + denoised * 0.5
        return torch.clamp(current, 0, 1)


# ============================================================
# SR STAGE
# ============================================================

class FlagshipSR(nn.Module):
    """2x super-resolution with frequency + strip attention."""
    def __init__(self, base_ch=48, depth=4, blocks=2):
        super().__init__()
        self.encoder = Encoder(1, base_ch, depth, blocks)
        chs = self.encoder.chs
        self.freq = FrequencyBranch(chs[-1], chs[-1] // 2)
        self.freq_proj = nn.Conv2d(chs[-1] // 2, chs[-1], 1)
        self.bottleneck = nn.Sequential(
            *[NAFBlock(chs[-1]) for _ in range(blocks)],
            StripAttention(chs[-1]),
        )
        self.decoder = Decoder(chs, blocks)
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

    Stage 1: Multi-pass denoising with noise estimation
    Stage 2: 2x super-resolution with frequency branch
    Inference: Single-pass (fast) or self-refined (best quality)
    """
    def __init__(self, denoise_base=48, sr_base=48, denoise_depth=3, sr_depth=4,
                 denoise_blocks=2, sr_blocks=2, denoise_passes=2):
        super().__init__()
        self.denoise = FlagshipDenoise(denoise_base, denoise_depth, denoise_blocks, denoise_passes)
        self.sr = FlagshipSR(sr_base, sr_depth, sr_blocks)

    def forward(self, lr):
        clean_lr = self.denoise(lr)
        hr = self.sr(clean_lr)
        hr_base = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        return torch.clamp(hr + hr_base, 0, 1)

    def forward_two_stage(self, lr):
        clean_lr = self.denoise(lr)
        hr = self.sr(clean_lr)
        hr_base = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        return clean_lr, torch.clamp(hr + hr_base, 0, 1)

    def forward_refined(self, lr, n_refine=1):
        clean_lr = self.denoise(lr)
        hr = self.sr(clean_lr)
        hr_base = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        hr = hr + hr_base
        for _ in range(n_refine):
            lr_re = F.interpolate(hr, size=lr.shape[-2:], mode='bilinear', align_corners=False)
            clean_re = self.denoise(lr_re)
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
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
