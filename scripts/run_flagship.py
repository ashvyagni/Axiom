import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', 'src'))
sys.path.insert(0, _SRC)

from models.flagship import FlagshipNet, build_flagship, count_params
from models.restoration import RestorationNet
from models.two_stage import TwoStageNet
from utils.guards import clip_output


def find_weights():
    root = os.path.abspath(os.path.join(_HERE, '..'))
    cands = [
        os.path.join(root, 'weights', 'best_model.pt'),
        os.path.join(root, '..', 'weights', 'best_model.pt'),
        os.path.join(root, '..', 'model2', 'weights', 'best_model.pt'),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        model_type = ckpt.get('model_type', 'flagship')
        state = ckpt.get('model_state', ckpt)
    else:
        model_type = 'flagship'
        state = ckpt

    if model_type == 'flagship':
        model = build_flagship()
    elif model_type == 'twostage':
        model = TwoStageNet(in_ch=1, base_ch=48)
    else:
        model = RestorationNet(in_ch=1, out_ch=1, base_ch=48)

    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, model_type


def prepare_input(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[2] == 1:
            arr = arr[..., 0]
    return arr


def pad_to_mult(x, m=16):
    _, _, h, w = x.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return x, h, w
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w


def tta_forward(model, x):
    """8-augmentation TTA: 4 rotations x 2 flips."""
    outs = []
    for flip in (False, True):
        xi = x.flip(-1) if flip else x
        for k in range(4):
            xik = torch.rot90(xi, k, (-2, -1))
            o = model(xik)
            o = torch.rot90(o, -k, (-2, -1))
            if flip:
                o = o.flip(-1)
            outs.append(o)
    return torch.stack(outs).mean(0)


def make_comparison_grid(lr, out, gt, border=2):
    """Create [LR | Output | GT] comparison grid as numpy uint8."""
    h, w = lr.shape
    h2, w2 = out.shape
    h3, w3 = gt.shape

    max_h = max(h, h2, h3)
    total_w = w + border + w2 + border + w3

    canvas = np.ones((max_h, total_w), dtype=np.uint8) * 40
    lr_u8 = (np.clip(lr, 0, 1) * 255).astype(np.uint8)
    out_u8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    gt_u8 = (np.clip(gt, 0, 1) * 255).astype(np.uint8)

    canvas[:h, :w] = lr_u8
    canvas[:h2, w+border:w+border+w2] = out_u8
    canvas[:h3, w+border+w2+border:w+border+w2+border+w3] = gt_u8

    return canvas


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input-dir> [output-dir]")
        print(f"  If output-dir not specified, shows 5 comparison images")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    wpath = find_weights()
    if wpath is None:
        print("ERROR: No weights found. Train first.")
        sys.exit(1)

    model, model_type = load_model(wpath, device)
    params = count_params(model) if hasattr(model, 'parameters') else sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type} | Params: {params:,} | Weights: {wpath}")

    # Find .npy files
    files = sorted(f for f in os.listdir(input_dir) if f.endswith('.npy'))
    if not files:
        print("No .npy files found.")
        sys.exit(0)

    # Check for GT directory
    gt_dir = None
    parent = os.path.dirname(input_dir.rstrip('/'))
    for candidate in ['gt', 'ground_truth', 'target']:
        c = os.path.join(parent, candidate)
        if os.path.isdir(c):
            gt_dir = c
            break

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process all files
    total_time = 0.0
    processed = 0
    comparisons = []

    for i, fname in enumerate(files):
        try:
            arr = prepare_input(np.load(os.path.join(input_dir, fname)))
            H, W = arr.shape
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            x_pad, orig_h, orig_w = pad_to_mult(x)

            # Inference
            t0 = time.time()
            with torch.no_grad():
                if model_type == 'flagship':
                    out = tta_forward(model, x_pad) if not output_dir else model(x_pad)
                else:
                    out = model(x_pad)
            elapsed_ms = (time.time() - t0) * 1000

            out_np = out.squeeze().cpu().numpy().astype(np.float32)
            out_np = out_np[:2*H, :2*W]
            out_np = clip_output(out_np)

            # Load GT if available
            gt_np = None
            if gt_dir and os.path.isfile(os.path.join(gt_dir, fname)):
                gt_np = np.load(os.path.join(gt_dir, fname)).astype(np.float32)

            # Save output
            if output_dir:
                np.save(os.path.join(output_dir, fname), out_np)

            # Save comparison
            if output_dir and gt_np is not None:
                from PIL import Image
                cmp = make_comparison_grid(arr, out_np, gt_np)
                cmp_path = os.path.join(output_dir, fname.replace('.npy', '_comparison.png'))
                Image.fromarray(cmp).save(cmp_path)
                comparisons.append(cmp)

            total_time += elapsed_ms
            processed += 1
            print(f"[{i+1}/{len(files)}] {fname} | {elapsed_ms:.1f}ms")

        except Exception as e:
            print(f"WARNING: {fname}: {e}")

    avg = total_time / max(processed, 1)
    print(f"\n{'='*60}")
    print(f"Processed: {processed}/{len(files)}")
    print(f"Avg latency: {avg:.1f} ms/image")
    print(f"{'='*60}")

    # Show 5 comparison images if no output dir
    if not output_dir and comparisons:
        try:
            from PIL import Image
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            n_show = min(5, len(comparisons))
            fig, axes = plt.subplots(n_show, 1, figsize=(15, 5*n_show))
            if n_show == 1:
                axes = [axes]
            for i in range(n_show):
                axes[i].imshow(comparisons[i], cmap='gray')
                axes[i].set_title(f"Sample {i+1}: LR | Restored | GT", fontsize=12)
                axes[i].axis('off')
            plt.tight_layout()
            save_path = os.path.join(_HERE, '..', 'outputs', 'flagship_comparisons.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved comparison grid: {save_path}")
        except ImportError:
            print("\nMatplotlib not available. Comparisons saved as individual PNGs.")


if __name__ == '__main__':
    main()
