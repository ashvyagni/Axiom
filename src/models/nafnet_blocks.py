import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet-style block with two sub-blocks, each using SimpleGate activation.

    Sub-block 1 (SPF): LayerNorm -> Conv1x1 -> DWConv3x3 -> SimpleGate -> Conv1x1
    Sub-block 2 (FFN): LayerNorm -> Conv1x1 -> SimpleGate -> Conv1x1
    """
    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = channels * dw_expand
        ffn_channel = channels * ffn_expand

        # Sub-block 1: Spatial Processing (SPF)
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, channels, 1, 1, 0)

        # Sub-block 2: Feed-Forward Network (FFN)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channel, 1, 1, 0)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, channels, 1, 1, 0)

        # Learnable scaling parameters
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else None

    def forward(self, inp):
        # Sub-block 1: SPF
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = self.conv3(x)
        x = x * self.beta + inp

        # Sub-block 2: FFN
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        if self.dropout is not None:
            y = self.dropout(y)
        y = self.conv5(y)
        y = y * self.gamma + x

        return y


class FrequencyBranch(nn.Module):
    """Small parallel branch operating in FFT domain.

    Modifies FFT coefficients and converts back to spatial domain.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        n, c, h, w = x.shape
        fft_x = torch.fft.rfft2(x, norm='ortho')

        # Decompose into real and imaginary parts
        fft_real = fft_x.real
        fft_imag = fft_x.imag

        # Apply learnable scaling (acts as a frequency-domain filter)
        # Use magnitude-aware gating
        fft_mag = torch.sqrt(fft_real**2 + fft_imag**2 + 1e-8)

        # Simple attention-like mechanism in frequency domain
        gate = torch.sigmoid(fft_mag)
        modified_real = fft_real * gate
        modified_imag = fft_imag * gate

        modified_fft = torch.complex(modified_real, modified_imag)
        spatial_feat = torch.fft.irfft2(modified_fft, s=(h, w), norm='ortho')

        return x + spatial_feat


class PixelShuffleUpscaler(nn.Module):
    """2x upscaling using PixelShuffle (no checkerboard artifacts)."""
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.up(x)
