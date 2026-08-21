import os
import sys

import torch

# Add model2_oc src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


def load_model_for_inference(weights_path, model_type='restoration', device='cpu'):
    """Load a trained model for inference.

    Args:
        weights_path: path to .pt checkpoint
        model_type: 'restoration' for RestorationNet, 'twostage' for TwoStageNet
        device: device to load onto
    Returns:
        model in eval mode
    """
    if model_type == 'twostage':
        from models.two_stage import TwoStageNet
        model = TwoStageNet(in_ch=1, base_ch=48)
    else:
        from models.restoration import RestorationNet
        model = RestorationNet(in_ch=1, out_ch=1, base_ch=48)

    from utils.checkpoint import load_model_weights
    model = load_model_weights(model, weights_path, device)
    return model
