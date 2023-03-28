import torch
import torch.nn as nn

# 3 x 256 x 256 -> 3 x 256 x 256


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down: bool, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, img_channels: int = 3, num_features: int = 64, num_residuals: int = 6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect"
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        self.last(nn.Sequential(
            nn.Conv2d(
                num_features,
                img_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            )
        ))

    def forward(self, x):
        pass
