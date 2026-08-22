import random
import numpy as np


def cutmix_patches(gt, lr, patch_size=64, prob=0.5):
    """CutMix: paste a random patch from one image onto another."""
    if random.random() > prob:
        return gt, lr
    h, w = gt.shape
    if h < patch_size or w < patch_size:
        return gt, lr
    rh = random.randint(patch_size // 2, h - patch_size // 2)
    rw = random.randint(patch_size // 2, w - patch_size // 2)
    top = max(0, rh - patch_size // 2)
    left = max(0, rw - patch_size // 2)
    patch_gt = gt[top:top+patch_size, left:left+patch_size].copy()
    patch_lr = lr[top:top+patch_size, left:left+patch_size].copy()
    gt[top:top+patch_size, left:left+patch_size] = patch_gt
    lr[top:top+patch_size, left:left+patch_size] = patch_lr
    return gt, lr


def random_erasing(img, prob=0.3, sl=0.02, sh=0.15):
    """Random erasing: fill a rectangular region with noise or constant."""
    if random.random() > prob:
        return img
    h, w = img.shape
    area = h * w
    target_area = random.uniform(sl, sh) * area
    aspect_ratio = random.uniform(0.3, 3.3)
    rh = int(round(np.sqrt(target_area * aspect_ratio)))
    rw = int(round(np.sqrt(target_area / aspect_ratio)))
    if rh >= h or rw >= w:
        return img
    top = random.randint(0, h - rh)
    left = random.randint(0, w - rw)
    img[top:top+rh, left:left+rw] = np.random.uniform(0, 1, (rh, rw)).astype(np.float32)
    return img


def contrast_jitter(img, factor_range=(0.7, 1.3), prob=0.4):
    """Random contrast jitter."""
    if random.random() > prob:
        return img
    factor = random.uniform(*factor_range)
    mean = img.mean()
    img = (img - mean) * factor + mean
    return np.clip(img, 0, 1).astype(np.float32)


def gaussian_noise(img, sigma_range=(0.01, 0.08), prob=0.3):
    """Additive Gaussian noise."""
    if random.random() > prob:
        return img
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1).astype(np.float32)


def elastic_deformation(img, alpha=30, sigma=4, prob=0.2):
    """Light elastic deformation for SEM image augmentation."""
    if random.random() > prob:
        return img
    try:
        from scipy.ndimage import gaussian_filter, map_coordinates
        h, w = img.shape
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = [np.reshape(y + dy, (-1,)), np.reshape(x + dx, (-1,))]
        distorted = map_coordinates(img, indices, order=1, mode='reflect')
        return distorted.reshape(h, w).astype(np.float32)
    except ImportError:
        return img
