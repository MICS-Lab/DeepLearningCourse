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


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.block = nn.Sequential(
            Block(in_channels = num_channels, out_channels = num_channels, down=True, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)



class Generator(nn.Module):
    def __init__(self, img_channels: int = 3, num_features: int = 64):
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
        self.down_blocks = nn.ModuleList([
            Block(num_features, num_features*2, down=True, kernel_size=3, stride=2, padding=1),
            Block(num_features*2, num_features*4, down=True, kernel_size=3, stride=2, padding=1)
        ])
        self.res_blocks = nn.ModuleList([
            ResBlock(num_features*4),
            ResBlock(num_features*4),
            ResBlock(num_features*4),
            ResBlock(num_features*4),
            ResBlock(num_features*4),
            ResBlock(num_features*4),
        ])
        self.up_blocks = nn.ModuleList([
            Block(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            Block(num_features*2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])
        self.last = nn.Sequential(
            nn.Conv2d(
                num_features,
                img_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            )
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        for layer in self.res_blocks:
            x = layer(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = self.last(x)
        return torch.tanh(x)


def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    print(x.shape)

    generator = Generator()
    print(generator(x).shape)


if __name__ == "__main__":
    test()



