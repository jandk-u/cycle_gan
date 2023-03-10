import torch
import torch.nn as nn


class BlockConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block1 = BlockConv2D(in_channels=in_channels, out_channels=64, kernel=4, stride=2)
        self.block2 = BlockConv2D(in_channels=64, out_channels=128, kernel=4, stride=2)
        self.block3 = BlockConv2D(in_channels=128, out_channels=256, kernel=4, stride=2)
        self.block4 = BlockConv2D(in_channels=256, out_channels=512, kernel=4, stride=1)
        self.final_block = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final_block(x)
        return x


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    disc = Discriminator()
    preds = disc(x)
    print(preds.shape)