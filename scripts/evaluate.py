import argparse
import os
import sys
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..'))
_SRC = os.path.join(_PROJECT_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.flagship import build_flagship, count_params
from utils.metrics import compute_ssim, compute_psnr
from utils.ema import EMA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--weights', type=str, default='../weights/best_model.pt')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = build_flagship({'dropout': 0.0}).to(device)
    model.load_state_dict(ckpt['model_state'])
    if 'ema_shadow' in ckpt:
        ema = EMA(model, decay=0.999)
        ema.load_state_dict(ckpt['ema_shadow'])
        ema.apply_shadow()
        print(f"Loaded EMA from epoch {ckpt.get('epoch', '?')} (SSIM={ckpt.get('val_ssim', '?'):.4f})")
    model.eval()

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.npy')])
    print(f"Found {len(files)} samples in {args.input_dir}")

    all_ssims, all_psnrs = [], []
    for i, fname in enumerate(files):
        lr = np.load(os.path.join(args.input_dir, fname)).astype(np.float32)
        if lr.max() > 1.0:
            lr = lr / 255.0
        lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            _, hr, _ = model.forward_two_stage(lr_t)
            out = torch.nan_to_num(hr.clamp(0.0, 1.0)).squeeze().cpu().numpy()

        np.save(os.path.join(args.output_dir, fname), out)

        gt_path = None
        for split in ['val', 'train']:
            for sub in ['gt', 'GT']:
                p = os.path.join(_HERE, '..', 'data', split, sub, fname)
                if os.path.exists(p):
                    gt_path = p
                    break
            if gt_path:
                break

        if gt_path:
            gt = np.load(gt_path).astype(np.float32)
            if gt.max() > 1.0:
                gt = gt / 255.0
            all_ssims.append(compute_ssim(out, gt))
            all_psnrs.append(compute_psnr(out, gt))

        if (i + 1) % 50 == 0:
            avg_ssim = np.mean(all_ssims) if all_ssims else 0
            print(f"  [{i+1}/{len(files)}] SSIM={avg_ssim:.4f}")

    if all_ssims:
        print(f"\nFinal: SSIM={np.mean(all_ssims):.4f} +/- {np.std(all_ssims):.4f}")
        print(f"       PSNR={np.mean(all_psnrs):.2f} +/- {np.std(all_psnrs):.2f}")
    print(f"Saved {len(files)} restored .npy to {args.output_dir}")


if __name__ == '__main__':
    main()
