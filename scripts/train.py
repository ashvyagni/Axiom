import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import structural_similarity as ski_ssim
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from datasets.dataset import get_dataloaders
from models.restoration import RestorationNet
from models.two_stage import TwoStageNet
from losses.composite import CompositeLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(model_type, base_ch=48):
    if model_type == 'twostage':
        return TwoStageNet(in_ch=1, base_ch=base_ch)
    return RestorationNet(in_ch=1, out_ch=1, base_ch=base_ch)


@torch.no_grad()
def validate(model, val_loader, device, loss_fn, model_type='restoration'):
    model.eval()
    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    for lr_batch, gt_batch in val_loader:
        lr_batch = lr_batch.to(device)
        gt_batch = gt_batch.to(device)

        if model_type == 'twostage':
            denoised, pred = model(lr_batch)
        else:
            pred = model(lr_batch)

        loss, _ = loss_fn(pred, gt_batch)

        pred_np = pred.cpu().numpy()
        gt_np = gt_batch.cpu().numpy()
        bs = pred_np.shape[0]
        for i in range(bs):
            p = pred_np[i, 0]
            g = gt_np[i, 0]
            total_ssim += ski_ssim(p, g, data_range=1.0)
            total_psnr += ski_psnr(p, g, data_range=1.0)

        total_loss += loss.item() * bs
        n += bs

    return total_loss / n, total_ssim / n, total_psnr / n


def main():
    parser = argparse.ArgumentParser(description='Model 2 Training')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--output_dir', type=str, default='../weights')
    parser.add_argument('--model_type', type=str, default='restoration',
                        choices=['restoration', 'twostage'])
    parser.add_argument('--base_ch', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--synthetic_aug', action='store_true', default=True)
    parser.add_argument('--patch_size', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    # Loss weights
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=0.5)
    parser.add_argument('--w_perceptual', type=float, default=0.1)
    parser.add_argument('--w_frequency', type=float, default=0.3)
    parser.add_argument('--w_edge', type=float, default=0.2)
    parser.add_argument('--w_range', type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        args.data_dir, args.batch_size, num_workers=0,
        synthetic_aug=args.synthetic_aug, patch_size=args.patch_size
    )
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Model
    model = get_model(args.model_type, args.base_ch).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}, Parameters: {param_count:,}")

    # Loss
    loss_weights = {
        'l1': args.w_l1, 'ssim': args.w_ssim,
        'perceptual': args.w_perceptual, 'frequency': args.w_frequency,
        'edge': args.w_edge, 'range': args.w_range,
    }
    loss_fn = CompositeLoss(weights=loss_weights).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Resume
    start_epoch = 0
    best_ssim = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt.get('epoch', 0)
            best_ssim = ckpt.get('metrics', {}).get('val_ssim', -1.0)
        print(f"Resumed from epoch {start_epoch}")

    print(f"Training {args.epochs} epochs, lr={args.lr}")
    print(f"Loss weights: {loss_weights}")
    print("-" * 70)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n = 0

        for lr_batch, gt_batch in train_loader:
            lr_batch = lr_batch.to(device)
            gt_batch = gt_batch.to(device)

            optimizer.zero_grad()

            with autocast(device.type, enabled=(device.type == 'cuda')):
                if args.model_type == 'twostage':
                    denoised, pred = model(lr_batch)
                    loss, _ = loss_fn(pred, gt_batch)
                    # Also add denoise loss
                    denoise_loss, _ = loss_fn(denoised, lr_batch, for_denoise=True)
                    loss = loss + 0.5 * denoise_loss
                else:
                    pred = model(lr_batch)
                    loss, _ = loss_fn(pred, gt_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * lr_batch.size(0)
            n += lr_batch.size(0)

        scheduler.step()
        train_loss = running_loss / max(n, 1)

        # Validate
        val_loss, val_ssim, val_psnr = validate(
            model, val_loader, device, loss_fn, args.model_type
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_ssim={val_ssim:.4f} | val_psnr={val_psnr:.2f} | "
            f"{elapsed:.1f}s"
        )

        # Save best
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'metrics': {'val_ssim': val_ssim, 'val_psnr': val_psnr, 'val_loss': val_loss},
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> New best SSIM: {val_ssim:.4f}")

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict()},
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            )

    # Save final
    torch.save(
        {'epoch': args.epochs, 'model_state': model.state_dict()},
        os.path.join(args.output_dir, 'final_model.pt')
    )
    print(f"Done. Best SSIM: {best_ssim:.4f}")


if __name__ == '__main__':
    main()
