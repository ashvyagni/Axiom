import torch
import numpy as np


class TTAWrapper:
    """Test-Time Augmentation wrapper.

    Runs inference on the image plus its flips/rotations,
    averages results for free quality boost (+0.1-0.3 dB PSNR typically).
    """
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def predict(self, x, enabled=True):
        """Run TTA inference.

        Args:
            x: input tensor (1, 1, H, W)
            enabled: if False, just run single forward pass
        Returns:
            averaged output tensor
        """
        if not enabled:
            return self.model(x)

        augmented = [
            x,
            torch.flip(x, dims=[3]),          # horizontal flip
            torch.flip(x, dims=[2]),          # vertical flip
            torch.flip(x, dims=[2, 3]),       # both flips
        ]

        results = []
        for aug_x in augmented:
            out = self.model(aug_x)
            # Reverse augmentation
            if aug_x is augmented[1]:
                out = torch.flip(out, dims=[3])
            elif aug_x is augmented[2]:
                out = torch.flip(out, dims=[2])
            elif aug_x is augmented[3]:
                out = torch.flip(out, dims=[2, 3])
            results.append(out)

        return torch.stack(results).mean(dim=0)
