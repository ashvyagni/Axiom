import argparse
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from models.restoration import RestorationNet
from models.two_stage import TwoStageNet


def export_to_onnx(weights_path, output_path, model_type='restoration', base_ch=48):
    """Export trained model to ONNX format."""
    device = torch.device('cpu')

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

    # Dummy input (128x128 grayscale)
    dummy_input = torch.randn(1, 1, 128, 128).to(device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        }
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported ONNX model to {output_path} ({file_size:.1f} MB)")

    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    test_input = np.random.randn(1, 1, 128, 128).astype(np.float32)
    outputs = session.run(None, {'input': test_input})
    print(f"ONNX verification: output shape = {outputs[0].shape}")
    print(f"Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description='Export Model 2 to ONNX')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--output', type=str, default='../weights/model.onnx')
    parser.add_argument('--model_type', type=str, default='restoration',
                        choices=['restoration', 'twostage'])
    parser.add_argument('--base_ch', type=int, default=48)
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output, args.model_type, args.base_ch)


if __name__ == '__main__':
    main()
