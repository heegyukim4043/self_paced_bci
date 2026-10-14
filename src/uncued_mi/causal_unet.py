"""Small strictly-causal 1-D U-Net for sample-wise idle/command segmentation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5,
                 dilation: int = 1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=0,
                         dilation=dilation, **kwargs)
        self.left = dilation * (kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left, 0)))


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each time point (causal-safe)."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(cin, cout, 5, dilation=dilation, bias=False),
            ChannelLayerNorm(cout), nn.ELU(),
            CausalConv1d(cout, cout, 3, bias=False),
            ChannelLayerNorm(cout), nn.ELU(),
        )
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class CausalUNet1D(nn.Module):
    """Causal encoder-decoder preserving the input time grid.

    Downsampling uses causal strided convolutions. Decoder interpolation only
    repeats already-computed coarse states; it never reads a future input.
    """
    def __init__(self, channels: int, width: int = 16):
        super().__init__()
        self.enc0 = Block(channels, width)
        self.down1 = CausalConv1d(width, width * 2, 5, stride=2, bias=False)
        self.enc1 = Block(width * 2, width * 2)
        self.down2 = CausalConv1d(width * 2, width * 4, 5, stride=2, bias=False)
        self.bottleneck = Block(width * 4, width * 4, dilation=2)
        self.dec1 = Block(width * 4 + width * 2, width * 2)
        self.dec0 = Block(width * 2 + width, width)
        self.head = nn.Conv1d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, samples)")
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        b = self.bottleneck(self.down2(e1))
        u1 = F.interpolate(b, size=e1.shape[-1], mode="nearest")
        u1 = self.dec1(torch.cat((u1, e1), dim=1))
        u0 = F.interpolate(u1, size=e0.shape[-1], mode="nearest")
        return self.head(self.dec0(torch.cat((u0, e0), dim=1))).squeeze(1)
