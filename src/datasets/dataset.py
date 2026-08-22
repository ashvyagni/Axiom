import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentations import (
    cutmix_patches, random_erasing, contrast_jitter,
    gaussian_noise, elastic_deformation
)


def add_synthetic_speckle(image, severity_range=(0.05, 0.25)):
    sigma = random.uniform(*severity_range)
    noise = np.random.normal(1.0, sigma, image.shape).astype(np.float32)
    noisy = image * noise
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def add_synthetic_gaussian_blur(image, sigma_range=(0.5, 2.0)):
    try:
        from scipy.ndimage import gaussian_filter
        sigma = random.uniform(*sigma_range)
        return gaussian_filter(image, sigma=sigma).astype(np.float32)
    except ImportError:
        return image


def add_synthetic_downsample(image, scale_range=(1.5, 3.0)):
    from scipy.ndimage import zoom
    scale = random.uniform(*scale_range)
    h, w = image.shape
    new_h, new_w = max(1, int(h / scale)), max(1, int(w / scale))
    downsampled = zoom(image, (new_h / h, new_w / w), order=1)
    upsampled = zoom(downsampled, (h / new_h, w / new_w), order=1)
    return upsampled.astype(np.float32)


def random_crop_pair(gt, lr, patch_size=256):
    h, w = gt.shape
    if h < patch_size or w < patch_size:
        return gt, lr
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    gt_crop = gt[top:top+patch_size, left:left+patch_size]
    scale = lr.shape[0] / gt.shape[0]
    lr_top = int(top * scale)
    lr_left = int(left * scale)
    lr_patch_size = int(patch_size * scale)
    lr_crop = lr[lr_top:lr_top+lr_patch_size, lr_left:lr_left+lr_patch_size]
    return gt_crop, lr_crop


class KLADataset(Dataset):
    def __init__(self, root_dir, split='train', augment=True,
                 synthetic_aug=True, patch_size=None):
        self.root = root_dir
        self.split = split
        self.augment = augment and (split == 'train')
        self.synthetic_aug = synthetic_aug and (split == 'train')
        self.patch_size = patch_size

        gt_dir = os.path.join(root_dir, split, 'gt')
        lr_dir = os.path.join(root_dir, split, 'degraded')

        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

        self.pairs = []
        for f in gt_files:
            gp = os.path.join(gt_dir, f)
            lp = os.path.join(lr_dir, f)
            if os.path.exists(lp):
                self.pairs.append((gp, lp))

    def __len__(self):
        return len(self.pairs)

    def _random_augment(self, gt_np, lr_np):
        if random.random() > 0.5:
            gt_np = np.flip(gt_np, axis=1).copy()
            lr_np = np.flip(lr_np, axis=1).copy()
        if random.random() > 0.5:
            gt_np = np.flip(gt_np, axis=0).copy()
            lr_np = np.flip(lr_np, axis=0).copy()
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            gt_np = np.rot90(gt_np, k=k, axes=(0, 1)).copy()
            lr_np = np.rot90(lr_np, k=k, axes=(0, 1)).copy()
        return gt_np, lr_np

    def _apply_synthetic_aug(self, gt_np, lr_np):
        aug_type = random.choice(['speckle', 'blur', 'combined'])
        if aug_type == 'speckle':
            lr_np = add_synthetic_speckle(lr_np)
        elif aug_type == 'blur':
            lr_np = add_synthetic_gaussian_blur(lr_np)
        elif aug_type == 'combined':
            lr_np = add_synthetic_speckle(lr_np, severity_range=(0.03, 0.15))
            lr_np = add_synthetic_gaussian_blur(lr_np, sigma_range=(0.5, 1.5))
        return gt_np, lr_np

    def _apply_advanced_aug(self, gt_np, lr_np):
        lr_np = contrast_jitter(lr_np)
        lr_np = gaussian_noise(lr_np)
        lr_np = random_erasing(lr_np)
        lr_np = elastic_deformation(lr_np)
        return gt_np, lr_np

    def __getitem__(self, idx):
        gt_path, lr_path = self.pairs[idx]

        gt_np = np.load(gt_path).astype(np.float32)
        lr_np = np.load(lr_path).astype(np.float32)

        if gt_np.max() > 1.0:
            gt_np = gt_np / 255.0

        if self.patch_size and self.augment:
            gt_np, lr_np = random_crop_pair(gt_np, lr_np, self.patch_size)

        if self.augment:
            gt_np, lr_np = self._random_augment(gt_np, lr_np)

        if self.synthetic_aug:
            gt_np, lr_np = self._apply_synthetic_aug(gt_np, lr_np)
            gt_np, lr_np = self._apply_advanced_aug(gt_np, lr_np)

        gt_np = np.clip(gt_np, 0.0, 1.0)
        lr_np = np.clip(lr_np, 0.0, 1.0)

        gt_t = torch.from_numpy(gt_np).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr_np).unsqueeze(0).float()

        return lr_t, gt_t


def get_dataloaders(root_dir, batch_size=8, num_workers=0,
                    synthetic_aug=True, patch_size=None):
    train_ds = KLADataset(root_dir, split='train', augment=True,
                          synthetic_aug=synthetic_aug, patch_size=patch_size)
    val_ds = KLADataset(root_dir, split='val', augment=False,
                        synthetic_aug=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
