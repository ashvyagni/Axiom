import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FlagshipLoss"]


def _fspecial_gauss(size, channel, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32)
    coords = coords - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] @ g[None, :]
    kernel = kernel / kernel.sum()
    return kernel[None, None].repeat(channel, 1, 1, 1)


class DifferentiableSSIM(nn.Module):
    def __init__(self, data_range=1.0, window_size=11):
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
    def forward(self, pred, target):
        b, c, h, w = pred.shape
        ws = self.window_size
        if min(h, w) < ws:
            f = (ws // min(h, w)) + 1
            pred = F.interpolate(pred, scale_factor=f, mode="bilinear", align_corners=False)
            target = F.interpolate(target, scale_factor=f, mode="bilinear", align_corners=False)
        pad = ws // 2
        window = _fspecial_gauss(ws, c, 1.5).to(pred.device, dtype=pred.dtype)
        mu1 = F.conv2d(pred, window, padding=pad, groups=c)
        mu2 = F.conv2d(target, window, padding=pad, groups=c)
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        sigma11 = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu1 * mu1
        sigma22 = F.conv2d(target * target, window, padding=pad, groups=c) - mu2 * mu2
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=c) - mu1 * mu2
        cs_map = (2.0 * sigma12 + c2) / (sigma11 + sigma22 + c2)
        ssim_map = ((2.0 * mu1 * mu2 + c1) / (mu1 * mu1 + mu2 * mu2 + c1)) * cs_map
        return ssim_map.mean()


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1.0,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sy = torch.tensor([[-1.0,-2.0,-1.0],[0,0,0],[1.0,2.0,1.0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)
    def _grad(self, x):
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(x, self.sx, padding=1)
        gy = F.conv2d(x, self.sy, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)
    def forward(self, pred, target):
        return F.l1_loss(self._grad(pred), self._grad(target))


class CharbonnierLoss(nn.Module):
    """Smooth L1 approximation — better gradient behavior at edges than plain L1."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class FFTMagnitudeLoss(nn.Module):
    """L1 on FFT magnitude spectrum — enforces frequency-domain consistency."""
    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred, norm='ortho')
        tf = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(pf.abs(), tf.abs())


class FFTPhaseLoss(nn.Module):
    """L1 on FFT phase — phase carries structural information."""
    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred, norm='ortho')
        tf = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(pf.angle(), tf.angle())


class WaveletLoss(nn.Module):
    """Haar wavelet subband L1 loss — preserves scale+orientation of high-freq content."""
    def __init__(self):
        super().__init__()
        self.register_buffer('ll', torch.tensor([[1,1],[1,1]], dtype=torch.float32).view(1,1,2,2) / 2.0)
        self.register_buffer('lh', torch.tensor([[1,-1],[1,-1]], dtype=torch.float32).view(1,1,2,2) / 2.0)
        self.register_buffer('hl', torch.tensor([[1,1],[-1,-1]], dtype=torch.float32).view(1,1,2,2) / 2.0)
        self.register_buffer('hh', torch.tensor([[1,-1],[-1,1]], dtype=torch.float32).view(1,1,2,2) / 2.0)
    def _haar(self, x):
        """Decompose into LL, LH, HL, HH subbands."""
        return (F.conv2d(x, self.ll, stride=2),
                F.conv2d(x, self.lh, stride=2),
                F.conv2d(x, self.hl, stride=2),
                F.conv2d(x, self.hh, stride=2))
    def forward(self, pred, target):
        p_ll, p_lh, p_hl, p_hh = self._haar(pred)
        t_ll, t_lh, t_hl, t_hh = self._haar(target)
        # Weight high-freq subbands more (edges, fine structure)
        loss = (0.1 * F.l1_loss(p_ll, t_ll) +
                0.3 * F.l1_loss(p_lh, t_lh) +
                0.3 * F.l1_loss(p_hl, t_hl) +
                0.8 * F.l1_loss(p_hh, t_hh))
        return loss / 1.5  # normalize


class RangePenalty(nn.Module):
    """Penalize outputs outside [0,1]."""
    def forward(self, pred):
        excess = F.relu(pred - 1.0) + F.relu(-pred)
        return (excess ** 2).mean()


class MultiScaleSSIM(nn.Module):
    """Multi-scale SSIM — structural similarity at multiple resolutions."""
    def __init__(self, levels=3, data_range=1.0):
        super().__init__()
        self.ssim = DifferentiableSSIM(data_range)
        self.levels = levels
    def forward(self, pred, target):
        total = 1.0 - self.ssim(pred, target)
        p, t = pred, target
        cnt = 1
        for _ in range(self.levels):
            p = F.avg_pool2d(p, 2)
            t = F.avg_pool2d(t, 2)
            if p.shape[-1] < 11 or p.shape[-2] < 11:
                break
            total = total + (1.0 - self.ssim(p, t))
            cnt += 1
        return total / cnt


class NoiseConsistencyLoss(nn.Module):
    """Penalizes inconsistent denoising across the image.
    If the denoise stage outputs have high local variance in uniform regions,
    this loss encourages spatially smooth denoising."""
    def forward(self, denoised, lr):
        # Denoised should have lower high-freq energy than input
        diff_lr = lr - F.avg_pool2d(lr, 3, 1, 1)
        diff_dn = denoised - F.avg_pool2d(denoised, 3, 1, 1)
        return F.l1_loss(diff_dn.abs(), diff_lr.abs()) * 0.1


class FlagshipLoss(nn.Module):
    """The ultimate composite loss for semiconductor image restoration.

    L_total = w1*L_charb + w2*L_ssim + w3*L_ms_ssim
            + w4*L_fft_mag + w5*L_fft_phase + w6*L_wavelet
            + w7*L_edge + w8*L_range + w9*L_noise_consist

    Curriculum: L1/edge/fft heavy early, SSIM/wavelet heavy late.
    """
    def __init__(self, weights=None, curriculum=True, total_epochs=200):
        super().__init__()
        w = weights or {}
        self.w = {
            'charb': float(w.get('charb', 1.0)),
            'ssim': float(w.get('ssim', 0.3)),
            'ms_ssim': float(w.get('ms_ssim', 0.1)),
            'fft_mag': float(w.get('fft_mag', 0.25)),
            'fft_phase': float(w.get('fft_phase', 0.1)),
            'wavelet': float(w.get('wavelet', 0.3)),
            'edge': float(w.get('edge', 0.2)),
            'range': float(w.get('range', 0.05)),
            'noise_consist': float(w.get('noise_consist', 0.1)),
        }
        self.curriculum = curriculum
        self.total_epochs = total_epochs
        self.current_epoch = 0

        self.charb = CharbonnierLoss()
        self.ssim = DifferentiableSSIM()
        self.ms_ssim = MultiScaleSSIM()
        self.fft_mag = FFTMagnitudeLoss()
        self.fft_phase = FFTPhaseLoss()
        self.wavelet = WaveletLoss()
        self.edge = SobelEdgeLoss()
        self.range_pen = RangePenalty()
        self.noise_consist = NoiseConsistencyLoss()

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _get_weights(self):
        """Curriculum weighting: early=charb/edge heavy, late=SSIM/wavelet heavy."""
        if not self.curriculum:
            return self.w
        progress = self.current_epoch / max(self.total_epochs, 1)
        # Early: focus on fidelity. Late: focus on structure.
        ssim_mult = 0.5 + 0.5 * progress  # 0.5→1.0
        wavelet_mult = 0.5 + 0.5 * progress  # 0.5→1.0
        charb_mult = 1.2 - 0.4 * progress  # 1.2→0.8
        edge_mult = 1.0 + 0.2 * progress  # 1.0→1.2
        return {
            'charb': self.w['charb'] * charb_mult,
            'ssim': self.w['ssim'] * ssim_mult,
            'ms_ssim': self.w['ms_ssim'] * ssim_mult,
            'fft_mag': self.w['fft_mag'],
            'fft_phase': self.w['fft_phase'],
            'wavelet': self.w['wavelet'] * wavelet_mult,
            'edge': self.w['edge'] * edge_mult,
            'range': self.w['range'],
            'noise_consist': self.w['noise_consist'],
        }

    def forward(self, pred, target, denoised=None, lr_input=None):
        """Compute composite loss.

        Args:
            pred: SR output (B, 1, 2H, 2W)
            target: GT HR (B, 1, 2H, 2W)
            denoised: optional denoise stage output for noise consistency
            lr_input: optional LR input for noise consistency
        """
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        w = self._get_weights()

        losses = {}
        losses['total'] = (
            w['charb'] * self.charb(pred, target) +
            w['ssim'] * (1.0 - self.ssim(pred, target)) +
            w['ms_ssim'] * self.ms_ssim(pred, target) +
            w['fft_mag'] * self.fft_mag(pred, target) +
            w['fft_phase'] * self.fft_phase(pred, target) +
            w['wavelet'] * self.wavelet(pred, target) +
            w['edge'] * self.edge(pred, target) +
            w['range'] * self.range_pen(pred)
        )
        losses['charb'] = self.charb(pred, target).item()
        losses['ssim'] = (1.0 - self.ssim(pred, target)).item()
        losses['fft_mag'] = self.fft_mag(pred, target).item()
        losses['wavelet'] = self.wavelet(pred, target).item()
        losses['edge'] = self.edge(pred, target).item()
        losses['range'] = self.range_pen(pred).item()

        if denoised is not None and lr_input is not None:
            nc = self.noise_consist(denoised, lr_input)
            losses['total'] = losses['total'] + w['noise_consist'] * nc
            losses['noise_consist'] = nc.item()
        else:
            losses['noise_consist'] = 0.0

        return losses['total'], losses
