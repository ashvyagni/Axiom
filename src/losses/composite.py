import torch
import torch.nn as nn

from .components import (
    L1Loss, SSIMLoss, EdgeLoss, FrequencyLoss,
    RangeLoss, PerceptualLoss
)


class CompositeLoss(nn.Module):
    """Total composite loss for Model 2.

    L_total = w1*L1 + w2*L_SSIM + w3*L_perceptual + w4*L_frequency + w5*L_edge + w6*L_range

    All weights are configurable. Default values are starting points;
    tune via ablation grid search logged in docs/ABLATIONS.md.
    """
    def __init__(self, weights=None):
        super().__init__()
        default_weights = {
            'l1': 1.0,
            'ssim': 0.5,
            'perceptual': 0.1,
            'frequency': 0.3,
            'edge': 0.2,
            'range': 0.1,
        }
        self.w = weights if weights is not None else default_weights

        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()
        self.freq_loss = FrequencyLoss()
        self.range_loss = RangeLoss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target, for_denoise=False):
        """Compute composite loss.

        Args:
            pred: model output (B, 1, H, W) in [0,1]
            target: ground truth (B, 1, H, W) in [0,1]
            for_denoise: if True, skip range loss (input may not be in [0,1])
        Returns:
            total_loss: weighted sum
            loss_dict: individual loss values for logging
        """
        loss_dict = {}

        loss_dict['l1'] = self.l1_loss(pred, target)
        loss_dict['ssim'] = self.ssim_loss(pred, target)
        loss_dict['edge'] = self.edge_loss(pred, target)
        loss_dict['frequency'] = self.freq_loss(pred, target)
        loss_dict['perceptual'] = self.perceptual_loss(pred, target)

        if not for_denoise:
            loss_dict['range'] = self.range_loss(pred)
        else:
            loss_dict['range'] = torch.tensor(0.0, device=pred.device)

        total = (
            self.w['l1'] * loss_dict['l1'] +
            self.w['ssim'] * loss_dict['ssim'] +
            self.w['perceptual'] * loss_dict['perceptual'] +
            self.w['frequency'] * loss_dict['frequency'] +
            self.w['edge'] * loss_dict['edge'] +
            self.w['range'] * loss_dict['range']
        )

        return total, loss_dict

    def get_config(self):
        """Return current loss config for logging."""
        return {
            'weights': dict(self.w),
            'losses': [
                'l1', 'ssim', 'perceptual', 'frequency', 'edge', 'range'
            ]
        }
