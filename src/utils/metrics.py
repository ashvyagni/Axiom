import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_ssim(pred, target):
    """SSIM between two numpy arrays in [0,1], shape (H, W)."""
    return float(ssim(pred, target, data_range=1.0))


def compute_psnr(pred, target):
    """PSNR between two numpy arrays in [0,1], shape (H, W)."""
    return float(psnr(pred, target, data_range=1.0))


def compute_lpips(pred, target, lpips_model=None):
    """LPIPS between two tensors (1,1,H,W) in [0,1]."""
    if lpips_model is None:
        lpips_model = load_lpips(str(pred.device))
    with torch.no_grad():
        # LPIPS expects 3-channel input
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        score = lpips_model(pred, target)
    return float(score.item())


def load_lpips(device="cpu"):
    """Load LPIPS model for perceptual quality assessment."""
    model = lpips.LPIPS(net="alex")
    model = model.to(device)
    model.eval()
    return model


def validate_range(arr, name="array"):
    """Check if array is in [0,1] and report violations."""
    min_val = float(arr.min())
    max_val = float(arr.max())
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    return {
        'name': name,
        'min': min_val,
        'max': max_val,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'in_range': min_val >= 0.0 and max_val <= 1.0,
    }
