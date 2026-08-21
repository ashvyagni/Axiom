import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FlagshipLoss"]


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
        coords = torch.arange(ws, dtype=torch.float32, device=pred.device)
        coords = coords - (ws - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        kernel = g[:, None] @ g[None, :]
        kernel = kernel / kernel.sum()
        window = kernel[None, None].repeat(c, 1, 1, 1)
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
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def set_eps(self, eps):
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class FFTMagnitudeLoss(nn.Module):
    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred, norm='ortho')
        tf = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(pf.abs(), tf.abs())


class FFTPhaseLoss(nn.Module):
    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred, norm='ortho')
        tf = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(pf.angle(), tf.angle())


class FFTFreqBandLoss(nn.Module):
    """Freq-weighted FFT loss: high-freq corruption penalized 3x harder."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred, norm='ortho')
        tf = torch.fft.rfft2(target, norm='ortho')
        pmag = pf.abs()
        tmag = tf.abs()
        B, C, H, W_half = pmag.shape
        cy = torch.arange(H, dtype=torch.float32, device=pred.device) / H
        cx = torch.arange(W_half, dtype=torch.float32, device=pred.device) / W_half
        fy, fx = torch.meshgrid(cy, cx, indexing='ij')
        freq_dist = torch.sqrt(fy ** 2 + fx ** 2)
        weight = 1.0 + 2.0 * freq_dist
        weight = weight.unsqueeze(0).unsqueeze(0)
        return (weight * (pmag - tmag).abs()).mean()


class WaveletLoss(nn.Module):
    """Rebalanced Haar wavelet: reduced HH from 0.8 to 0.5 to preserve fine detail."""
    def __init__(self):
        super().__init__()
        self.register_buffer('ll', torch.tensor([[1,1],[1,1]], dtype=torch.float32).view(1,1,2,2) / 2.0)
        self.register_buffer('lh', torch.tensor([[1,-1],[1,-1]], dtype=torch.float32).view(1,1,2,2) / 2.0)
        self.register_buffer('hl', torch.tensor([[1,1],[-1,-1]], dtype=torch.float32).view(1,1,2,2) / 2.0)
        self.register_buffer('hh', torch.tensor([[1,-1],[-1,1]], dtype=torch.float32).view(1,1,2,2) / 2.0)

    def _haar(self, x):
        return (F.conv2d(x, self.ll, stride=2),
                F.conv2d(x, self.lh, stride=2),
                F.conv2d(x, self.hl, stride=2),
                F.conv2d(x, self.hh, stride=2))

    def forward(self, pred, target):
        p_ll, p_lh, p_hl, p_hh = self._haar(pred)
        t_ll, t_lh, t_hl, t_hh = self._haar(target)
        loss = (0.1 * F.l1_loss(p_ll, t_ll) +
                0.4 * F.l1_loss(p_lh, t_lh) +
                0.4 * F.l1_loss(p_hl, t_hl) +
                0.5 * F.l1_loss(p_hh, t_hh))
        return loss / 1.4


class RangePenalty(nn.Module):
    def forward(self, pred):
        excess = F.relu(pred - 1.0) + F.relu(-pred)
        return (excess ** 2).mean()


class MultiScaleSSIM(nn.Module):
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
    def forward(self, denoised, lr):
        diff_lr = lr - F.avg_pool2d(lr, 3, 1, 1)
        diff_dn = denoised - F.avg_pool2d(denoised, 3, 1, 1)
        return F.l1_loss(diff_dn.abs(), diff_lr.abs()) * 0.1


class LPIPSLoss(nn.Module):
    """VGG-based perceptual loss using shallow+mid layers."""
    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
            self.blocks = nn.ModuleList()
            feature_layers = [2, 9, 16, 23]
            prev = 0
            for layer_idx in feature_layers:
                self.blocks.append(vgg[prev:layer_idx+1])
                prev = layer_idx + 1
            for param in self.parameters():
                param.requires_grad = False
            self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
            self.register_buffer('std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
            self._available = True
        except Exception:
            self._available = False

    def forward(self, pred, target):
        if not self._available:
            return torch.tensor(0.0, device=pred.device)
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = torch.tensor(0.0, device=pred.device)
        pred_feat = pred
        target_feat = target
        for block in self.blocks:
            pred_feat = block(pred_feat)
            target_feat = block(target_feat)
            loss = loss + F.l1_loss(pred_feat, target_feat.detach())
        return loss


class DeepSupervisionLoss(nn.Module):
    """Weighted loss at intermediate denoise passes: earlier passes get lower weight."""
    def __init__(self, base_loss_fn):
        super().__init__()
        self.base_loss = base_loss_fn

    def forward(self, intermediates, target):
        n = len(intermediates)
        total = torch.tensor(0.0, device=target.device)
        for i, inter in enumerate(intermediates):
            w = (i + 1) / n
            upsampled = F.interpolate(inter, size=target.shape[-2:], mode='bilinear', align_corners=False)
            total = total + w * self.base_loss(upsampled, target)
        return total / n


class FlagshipLoss(nn.Module):
    """12-term composite loss with curriculum, LPIPS, freq-band, annealing eps, deep supervision.

    L = w1*L_charb + w2*L_ssim + w3*L_ms_ssim
      + w4*L_fft_mag + w5*L_fft_phase + w6*L_freq_band
      + w7*L_wavelet + w8*L_edge + w9*L_range
      + w10*L_noise_consist + w11*L_lpips
      + w12*L_deep_sup (on denoise intermediates)
    """
    def __init__(self, weights=None, curriculum=True, total_epochs=200):
        super().__init__()
        w = weights or {}
        self.w = {
            'charb': float(w.get('charb', 1.0)),
            'ssim': float(w.get('ssim', 0.4)),
            'ms_ssim': float(w.get('ms_ssim', 0.15)),
            'fft_mag': float(w.get('fft_mag', 0.2)),
            'fft_phase': float(w.get('fft_phase', 0.1)),
            'freq_band': float(w.get('freq_band', 0.15)),
            'wavelet': float(w.get('wavelet', 0.3)),
            'edge': float(w.get('edge', 0.2)),
            'range': float(w.get('range', 0.05)),
            'noise_consist': float(w.get('noise_consist', 0.1)),
            'lpips': float(w.get('lpips', 0.1)),
            'deep_sup': float(w.get('deep_sup', 0.15)),
        }
        self.curriculum = curriculum
        self.total_epochs = total_epochs
        self.current_epoch = 0

        self.charb = CharbonnierLoss(eps=1e-3)
        self.ssim = DifferentiableSSIM()
        self.ms_ssim = MultiScaleSSIM()
        self.fft_mag = FFTMagnitudeLoss()
        self.fft_phase = FFTPhaseLoss()
        self.freq_band = FFTFreqBandLoss()
        self.wavelet = WaveletLoss()
        self.edge = SobelEdgeLoss()
        self.range_pen = RangePenalty()
        self.noise_consist = NoiseConsistencyLoss()
        self.lpips_loss = LPIPSLoss()
        self.deep_sup = DeepSupervisionLoss(CharbonnierLoss(eps=1e-3))

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        progress = epoch / max(self.total_epochs, 1)
        eps = 1e-3 * (1 - progress) + 1e-5 * progress
        self.charb.set_eps(eps)
        self.deep_sup.base_loss.set_eps(eps)

    def _get_weights(self):
        if not self.curriculum:
            return self.w
        progress = self.current_epoch / max(self.total_epochs, 1)
        ssim_mult = 0.4 + 0.6 * progress
        wavelet_mult = 0.4 + 0.6 * progress
        charb_mult = 1.3 - 0.5 * progress
        edge_mult = 1.0 + 0.3 * progress
        lpips_mult = 0.3 + 0.7 * progress
        deep_sup_mult = 1.0 - 0.5 * progress
        freq_band_mult = 0.5 + 0.5 * progress
        return {
            'charb': self.w['charb'] * charb_mult,
            'ssim': self.w['ssim'] * ssim_mult,
            'ms_ssim': self.w['ms_ssim'] * ssim_mult,
            'fft_mag': self.w['fft_mag'],
            'fft_phase': self.w['fft_phase'],
            'freq_band': self.w['freq_band'] * freq_band_mult,
            'wavelet': self.w['wavelet'] * wavelet_mult,
            'edge': self.w['edge'] * edge_mult,
            'range': self.w['range'],
            'noise_consist': self.w['noise_consist'],
            'lpips': self.w['lpips'] * lpips_mult,
            'deep_sup': self.w['deep_sup'] * deep_sup_mult,
        }

    def forward(self, pred, target, denoised=None, lr_input=None, denoise_intermediates=None):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
        w = self._get_weights()

        losses = {}
        losses['total'] = (
            w['charb'] * self.charb(pred, target) +
            w['ssim'] * (1.0 - self.ssim(pred, target)) +
            w['ms_ssim'] * self.ms_ssim(pred, target) +
            w['fft_mag'] * self.fft_mag(pred, target) +
            w['fft_phase'] * self.fft_phase(pred, target) +
            w['freq_band'] * self.freq_band(pred, target) +
            w['wavelet'] * self.wavelet(pred, target) +
            w['edge'] * self.edge(pred, target) +
            w['range'] * self.range_pen(pred) +
            w['lpips'] * self.lpips_loss(pred, target)
        )
        losses['charb'] = self.charb(pred, target).item()
        losses['ssim'] = (1.0 - self.ssim(pred, target)).item()
        losses['fft_mag'] = self.fft_mag(pred, target).item()
        losses['wavelet'] = self.wavelet(pred, target).item()
        losses['edge'] = self.edge(pred, target).item()
        losses['lpips'] = self.lpips_loss(pred, target).item()

        if denoised is not None and lr_input is not None:
            nc = self.noise_consist(denoised, lr_input)
            losses['total'] = losses['total'] + w['noise_consist'] * nc
            losses['noise_consist'] = nc.item()
        else:
            losses['noise_consist'] = 0.0

        if denoise_intermediates is not None and len(denoise_intermediates) > 1:
            ds = self.deep_sup(denoise_intermediates, target)
            losses['total'] = losses['total'] + w['deep_sup'] * ds
            losses['deep_sup'] = ds.item()
        else:
            losses['deep_sup'] = 0.0

        return losses['total'], losses
