import os
import sys
import time

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from models.restoration import RestorationNet
from models.two_stage import TwoStageNet
from utils.guards import nan_guard, clip_output


def load_model(weights_path, device, model_type='restoration', base_ch=48):
    """Load trained model for inference."""
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


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve model weights
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try restoration first, then twostage
    weights_path = os.path.join(script_dir, "..", "weights", "best_model.pt")
    if not os.path.isfile(weights_path):
        weights_path = os.path.join(script_dir, "..", "model2", "weights", "best_model.pt")

    if not os.path.isfile(weights_path):
        print("ERROR: Model weights not found. Expected at model2_oc/weights/best_model.pt")
        sys.exit(1)

    # Detect model type from checkpoint
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    is_twostage = False
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
        is_twostage = any('denoise_head' in k for k in state.keys())
    model_type = 'twostage' if is_twostage else 'restoration'

    model = load_model(weights_path, device, model_type)
    print(f"Loaded {model_type} model from {weights_path}")

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))

    if not files:
        print("No .npy files found in input directory.")
        sys.exit(0)

    total_time = 0.0
    processed = 0

    for fname in files:
        try:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)

            arr = np.load(in_path).astype(np.float32)

            # Normalize input if needed
            if arr.max() > 1.0:
                arr = arr / 255.0

            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

            t0 = time.time()
            with torch.no_grad():
                if model_type == 'twostage':
                    _, output = model(tensor)
                else:
                    output = model(tensor)
            elapsed_ms = (time.time() - t0) * 1000

            out_np = output.squeeze().cpu().numpy().astype(np.float32)

            # Defensive guards
            out_np = nan_guard(torch.from_numpy(out_np))[0].numpy()
            out_np = clip_output(out_np, 0.0, 1.0)

            # Save .npy
            np.save(out_path, out_np)

            total_time += elapsed_ms
            processed += 1
            print(f"{fname} {elapsed_ms:.1f}ms")

        except Exception as e:
            print(f"WARNING: failed on {fname}: {e}")
            continue

    avg = total_time / processed if processed > 0 else 0
    print(f"\nProcessed {processed}/{len(files)} images")
    print(f"Total time: {total_time:.1f} ms")
    print(f"Avg time per image: {avg:.1f} ms")


if __name__ == "__main__":
    main()
