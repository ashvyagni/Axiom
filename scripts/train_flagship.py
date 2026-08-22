import argparse
import os
import sys
import time
import json

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..'))
_SRC = os.path.join(_PROJECT_ROOT, 'src')
for _p in [_SRC, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from models.flagship import FlagshipNet, build_flagship, count_params
from losses.flagship_loss import FlagshipLoss
from data.dataset import get_dataloaders, KLADataset
from utils.metrics import compute_ssim, compute_psnr
from utils.guards import clip_output
from utils.ema import EMA


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device, n_limit=200):
    model.eval()
    ssims, psnrs = [], []
    cnt = 0
    for lr_b, gt_b in loader:
        lr_b, gt_b = lr_b.to(device), gt_b.to(device)
        clean_lr, hr, _ = model.forward_two_stage(lr_b)
        out = torch.nan_to_num(hr.clamp(0.0, 1.0))
        out_np = out.squeeze(1).cpu().numpy()
        gt_np = gt_b.clamp(0.0, 1.0).squeeze(1).cpu().numpy()
        for i in range(out_np.shape[0]):
            ssims.append(compute_ssim(out_np[i], gt_np[i]))
            psnrs.append(compute_psnr(out_np[i], gt_np[i]))
            cnt += 1
            if cnt >= n_limit:
                break
        if cnt >= n_limit:
            break
    return float(np.mean(ssims)), float(np.mean(psnrs))


def main():
    ap = argparse.ArgumentParser(description='FlagshipNet Training — Nuclear Edition')
    ap.add_argument('--data_dir', type=str, default='../data')
    ap.add_argument('--output_dir', type=str, default='../weights')
    ap.add_argument('--model_type', type=str, default='flagship',
                    choices=['flagship', 'restoration', 'twostage'])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_every', type=int, default=25)
    ap.add_argument('--synthetic_aug', action='store_true', default=True)
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--no_curriculum', action='store_true')
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--use_onecycle', action='store_true', default=True)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--dropout', type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(
        args.data_dir, args.batch_size, num_workers=0,
        synthetic_aug=args.synthetic_aug
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    if args.model_type == 'flagship':
        model = build_flagship({'dropout': args.dropout}).to(device)
    elif args.model_type == 'twostage':
        from models.two_stage import TwoStageNet
        model = TwoStageNet(in_ch=1, base_ch=48).to(device)
    else:
        from models.restoration import RestorationNet
        model = RestorationNet(in_ch=1, out_ch=1, base_ch=48).to(device)

    params = count_params(model)
    print(f"Model: {args.model_type} | Params: {params:,} ({params/1e6:.2f}M)")

    ema = EMA(model, decay=args.ema_decay)
    print(f"EMA enabled | Decay: {args.ema_decay}")

    loss_fn = FlagshipLoss(curriculum=not args.no_curriculum, total_epochs=args.epochs).to(device)

    effective_batch = args.batch_size * args.grad_accum
    base_lr = args.lr
    actual_lr = base_lr * (effective_batch / 8)
    print(f"Gradient accumulation: {args.grad_accum}x | Effective batch: {effective_batch} | LR scaled: {actual_lr:.6f}")

    optimizer = AdamW(model.parameters(), lr=actual_lr, weight_decay=args.weight_decay)

    if args.use_onecycle:
        steps_per_epoch = len(train_loader) // args.grad_accum + (1 if len(train_loader) % args.grad_accum else 0)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = steps_per_epoch * args.warmup_epochs
        scheduler = OneCycleLR(
            optimizer, max_lr=actual_lr, total_steps=total_steps,
            pct_start=warmup_steps / max(total_steps, 1),
            anneal_strategy='cos', div_factor=10, final_div_factor=100
        )
        print(f"OneCycleLR: warmup {args.warmup_epochs} epochs ({warmup_steps} steps), total {total_steps} steps")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_scaler = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    start_epoch = 0
    best_ssim = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
            start_epoch = ckpt.get('epoch', 0)
            best_ssim = ckpt.get('val_ssim', -1.0)
            if 'ema_shadow' in ckpt:
                ema.load_state_dict(ckpt['ema_shadow'])
        print(f"Resumed from epoch {start_epoch}, best_ssim={best_ssim:.4f}")

    print(f"Training {args.epochs} epochs, lr={actual_lr}")
    print("=" * 70)

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_fn.set_epoch(epoch)
        running = 0.0
        seen = 0
        optimizer.zero_grad()

        for step, (lr_b, gt_b) in enumerate(train_loader):
            lr_b, gt_b = lr_b.to(device), gt_b.to(device)

            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                if args.model_type == 'flagship':
                    denoised, hr, denoise_inters = model.forward_two_stage(lr_b)
                    total, ldict = loss_fn(hr, gt_b, denoised, lr_b, denoise_inters)
                else:
                    pred = model(lr_b)
                    total, ldict = loss_fn(pred, gt_b)

            scaled_loss = total / args.grad_accum

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running += total.item() * lr_b.size(0)
            seen += lr_b.size(0)

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                ema.update()
                optimizer.zero_grad()

                if args.use_onecycle:
                    scheduler.step()

        if not args.use_onecycle:
            scheduler.step()

        train_loss = running / max(seen, 1)

        ema.apply_shadow()
        val_ssim, val_psnr = validate(model, val_loader, device)
        ema.restore()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
              f"val_ssim={val_ssim:.4f} | val_psnr={val_psnr:.2f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "loss": train_loss, "val_ssim": val_ssim,
                        "val_psnr": val_psnr, "lr": lr_now})

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            ema.apply_shadow()
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "ema_shadow": ema.state_dict(),
                "epoch": epoch, "val_ssim": val_ssim, "val_psnr": val_psnr,
                "model_type": args.model_type, "cfg": {},
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pt'))
            ema.restore()
            print(f"  -> New best SSIM: {val_ssim:.4f} (EMA weights saved)")

        if epoch % args.save_every == 0:
            ema.apply_shadow()
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "ema_shadow": ema.state_dict()},
                       os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))
            ema.restore()

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best SSIM: {best_ssim:.4f}")


if __name__ == '__main__':
    main()
