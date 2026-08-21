import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss: 1 - SSIM."""
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _create_window(self, window_size, channel):
        _1D_window = torch.tensor(
            [1.0 / (window_size * window_size)] * (window_size * window_size),
            dtype=torch.float32
        ).view(1, 1, window_size, window_size).expand(channel, 1, -1, -1)
        return _1D_window

    def forward(self, img1, img2):
        channel = img1.size(1)
        if channel != self.channel:
            self.channel = channel
            self.window = self._create_window(self.window_size, channel).to(img1.device)

        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class EdgeLoss(nn.Module):
    """Sobel gradient loss for edge preservation."""
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def _sobel(self, x):
        gx = F.conv2d(x, self.sx, padding=1)
        gy = F.conv2d(x, self.sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target):
        return F.l1_loss(self._sobel(pred), self._sobel(target))


class FrequencyLoss(nn.Module):
    """FFT magnitude domain L1 loss. Targets periodic structure recovery."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)

        mag_loss = F.l1_loss(pred_mag, target_mag)
        phase_loss = F.l1_loss(pred_phase, target_phase)

        return mag_loss + 0.5 * phase_loss


class RangeLoss(nn.Module):
    """Penalize outputs outside [0,1] range.

    Specific to speckle noise: speckle pushes pixel values beyond true range.
    This loss directly penalizes that behavior.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred):
        lower_violation = F.relu(0.0 - pred).mean()
        upper_violation = F.relu(pred - 1.0).mean()
        return lower_violation + upper_violation


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss using shallow layers (edge/texture level).

    Uses shallow VGG layers only to minimize domain mismatch with grayscale scientific images.
    """
    def __init__(self, feature_layers=None):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
            self.blocks = nn.ModuleList()
            if feature_layers is None:
                feature_layers = [2, 9, 16]  # shallow layers: edges, textures

            prev = 0
            for layer_idx in feature_layers:
                self.blocks.append(vgg[prev:layer_idx+1])
                prev = layer_idx + 1

            # Freeze VGG
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        except Exception:
            self.blocks = None

    def forward(self, pred, target):
        if self.blocks is None:
            return torch.tensor(0.0, device=pred.device)

        # Convert grayscale to 3-channel for VGG
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Normalize to VGG input range
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        pred = (pred - mean) / std
        target = (target - mean) / std

        loss = torch.tensor(0.0, device=pred.device)
        pred_feat = pred
        target_feat = target

        with torch.no_grad() if not self.training else torch.enable_grad():
            for block in self.blocks:
                pred_feat = block(pred_feat)
                target_feat = block(target_feat)
                loss = loss + F.l1_loss(pred_feat, target_feat)

        return loss
