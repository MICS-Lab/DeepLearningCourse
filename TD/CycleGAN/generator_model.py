import torch
import torch.nn as nn

# 3 x 256 x 256 -> 3 x 256 x 256


class Block(nn.Module):



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

