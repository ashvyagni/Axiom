import argparse
import os
import sys
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity as ski_ssim
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from models.restoration import RestorationNet
from models.two_stage import TwoStageNet
from utils.metrics import load_lpips, compute_lpips, validate_range
from utils.guards import nan_guard, clip_output


def load_model(weights_path, device, model_type='restoration', base_ch=48):
    if model_type == 'twostage':
        model = TwoStageNet(in_ch=1, base_ch=base_ch)
    else:
        model = RestorationNet(in_ch=1, out_ch=1, base_ch=base_ch)

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def benchmark_model(model, val_loader, device, model_type='restoration', num_samples=None):
    """Run benchmark: SSIM, PSNR, LPIPS, latency, parameter count."""
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    latencies = []

    lpips_model = load_lpips(device)

    with torch.no_grad():
        for i, (lr_batch, gt_batch) in enumerate(val_loader):
            if num_samples and i >= num_samples:
                break

            lr_batch = lr_batch.to(device)
            gt_batch = gt_batch.to(device)

            # Warmup + measure latency
            t0 = time.time()
            if model_type == 'twostage':
                _, pred = model(lr_batch)
            else:
                pred = model(lr_batch)
            elapsed_ms = (time.time() - t0) * 1000

            pred_np = pred.cpu().numpy()
            gt_np = gt_batch.cpu().numpy()

            bs = pred_np.shape[0]
            for j in range(bs):
                p = pred_np[j, 0]
                g = gt_np[j, 0]

                # Clip and guard
                p = clip_output(p)
                range_info = validate_range(p, f"sample_{i*bs+j}")

                ssim_scores.append(ski_ssim(p, g, data_range=1.0))
                psnr_scores.append(ski_psnr(p, g, data_range=1.0))

                p_t = torch.from_numpy(p).unsqueeze(0).unsqueeze(0).to(device)
                g_t = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(device)
                lpips_scores.append(compute_lpips(p_t, g_t, lpips_model))

                latencies.append(elapsed_ms / bs)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'ssim': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
        'psnr': np.mean(psnr_scores),
        'psnr_std': np.std(psnr_scores),
        'lpips': np.mean(lpips_scores),
        'lpips_std': np.std(lpips_scores),
        'latency_mean_ms': np.mean(latencies),
        'latency_p95_ms': np.percentile(latencies, 95),
        'param_count': param_count,
        'model_size_mb': param_count * 4 / (1024 * 1024),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Model 2')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='restoration',
                        choices=['restoration', 'twostage'])
    parser.add_argument('--base_ch', type=int, default=48)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Load data
    from data.dataset import KLADataset
    from torch.utils.data import DataLoader
    ds = KLADataset(args.data_dir, split=args.split, augment=False, synthetic_aug=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(ds)} samples")

    # Load model
    model = load_model(args.weights, device, args.model_type, args.base_ch)

    # Run benchmark
    results = benchmark_model(model, loader, device, args.model_type, args.num_samples)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"SSIM:          {results['ssim']:.4f} +/- {results['ssim_std']:.4f}")
    print(f"PSNR:          {results['psnr']:.2f} +/- {results['psnr_std']:.2f} dB")
    print(f"LPIPS:         {results['lpips']:.4f} +/- {results['lpips_std']:.4f}")
    print(f"Latency (ms):  {results['latency_mean_ms']:.1f} mean, {results['latency_p95_ms']:.1f} p95")
    print(f"Parameters:    {results['param_count']:,}")
    print(f"Model size:    {results['model_size_mb']:.1f} MB")
    print("=" * 60)

    # Combined score (higher is better)
    combined = results['ssim'] * 0.4 + results['psnr'] / 50.0 * 0.3 - results['lpips'] * 0.3
    latency_penalty = max(0, (results['latency_mean_ms'] - 200) / 200) * 0.1
    final_score = combined - latency_penalty
    print(f"\nCombined score: {final_score:.4f}")

    return results


if __name__ == '__main__':
    main()
