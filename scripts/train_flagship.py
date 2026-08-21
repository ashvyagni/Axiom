import argparse
import os
import sys
import time
import json

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from models.flagship import FlagshipNet, build_flagship, count_params
from losses.flagship_loss import FlagshipLoss
from data.dataset import get_dataloaders, KLADataset
from utils.metrics import compute_ssim, compute_psnr
from utils.guards import clip_output


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device, loss_fn, n_limit=200):
    model.eval()
    ssims, psnrs = [], []
    cnt = 0
    for lr_b, gt_b in loader:
        lr_b, gt_b = lr_b.to(device), gt_b.to(device)
        _, out = model.forward_two_stage(lr_b)
        out = torch.nan_to_num(out.clamp(0.0, 1.0))
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
    ap = argparse.ArgumentParser(description='Flagship Model 2 Training')
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
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        args.data_dir, args.batch_size, num_workers=0,
        synthetic_aug=args.synthetic_aug
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Model
    if args.model_type == 'flagship':
        model = build_flagship().to(device)
    elif args.model_type == 'twostage':
        from models.two_stage import TwoStageNet
        model = TwoStageNet(in_ch=1, base_ch=48).to(device)
    else:
        from models.restoration import RestorationNet
        model = RestorationNet(in_ch=1, out_ch=1, base_ch=48).to(device)

    params = count_params(model)
    print(f"Model: {args.model_type} | Params: {params:,} ({params/1e6:.2f}M)")

    # Loss
    loss_fn = FlagshipLoss(curriculum=not args.no_curriculum, total_epochs=args.epochs).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        print(f"Resumed from epoch {start_epoch}, best_ssim={best_ssim:.4f}")

    print(f"Training {args.epochs} epochs, lr={args.lr}")
    print("-" * 70)

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_fn.set_epoch(epoch)
        running = 0.0
        seen = 0

        for lr_b, gt_b in train_loader:
            lr_b, gt_b = lr_b.to(device), gt_b.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                if args.model_type == 'flagship':
                    denoised, hr = model.forward_two_stage(lr_b)
                    total, ldict = loss_fn(hr, gt_b, denoised, lr_b)
                else:
                    pred = model(lr_b)
                    total, ldict = loss_fn(pred, gt_b)

            if use_scaler:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running += total.item() * lr_b.size(0)
            seen += lr_b.size(0)

        scheduler.step()
        train_loss = running / max(seen, 1)
        val_ssim, val_psnr = validate(model, val_loader, device, loss_fn)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
              f"val_ssim={val_ssim:.4f} | val_psnr={val_psnr:.2f} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "loss": train_loss, "val_ssim": val_ssim, "val_psnr": val_psnr})

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "val_ssim": val_ssim, "val_psnr": val_psnr,
                "model_type": args.model_type,
                "cfg": {},
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> New best SSIM: {val_ssim:.4f}")

        if epoch % args.save_every == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict()},
                       os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best SSIM: {best_ssim:.4f}")


if __name__ == '__main__':
    main()
