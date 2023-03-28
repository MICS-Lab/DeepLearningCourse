import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                padding_mode="reflect"
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: tuple[int] = (64, 128, 256, 1)):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        layers = []
        in_channels = out_channels[0]
        for num_channel in out_channels[1:]:
            layers.append(Block(
                in_channels,
                num_channel,
                stride=1 if num_channel == out_channels[-1] else 2
            ))
            in_channels = num_channel
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return torch.sigmoid(x)


def test():
    x = torch.randn((5, 3, 256, 256))
    discriminator = Discriminator(in_channels=3)
    print(discriminator)
    
    preds = discriminator(x)
    print(x.shape)
    print(preds.shape)


if __name__ == "__main__":
    test()

