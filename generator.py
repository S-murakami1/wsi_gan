import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(out_ch)
        )

    def forward(self, x):
        return self.block(x)


class ResidualGenerator(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        self.enc1 = nn.Sequential(
            ConvBlock(in_ch, base_ch),
            ConvBlock(base_ch, base_ch)
        )
        self.enc2 = nn.Sequential(
            ConvBlock(base_ch, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 2)
        )
        self.enc3 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 4)
        )
        self.enc4 = nn.Sequential(
            ConvBlock(base_ch * 4, base_ch * 8),
            ConvBlock(base_ch * 8, base_ch * 8)
        )

        self.bottom = nn.Sequential(
            ConvBlock(base_ch * 8, base_ch * 16),
            ConvBlock(base_ch * 16, base_ch * 16)
        )

        self.dec4 = nn.Sequential(
            ConvBlock(base_ch * 16 + base_ch * 8, base_ch * 8),
            ConvBlock(base_ch * 8, base_ch * 8)
        )
        self.dec3 = nn.Sequential(
            ConvBlock(base_ch * 8 + base_ch * 4, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 4)
        )
        self.dec2 = nn.Sequential(
            ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 2)
        )
        self.dec1 = nn.Sequential(
            ConvBlock(base_ch * 2 + base_ch, base_ch),
            ConvBlock(base_ch, base_ch)
        )

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, in_ch, kernel_size=3),
            nn.Tanh()
        )

    def down(self, x):
        return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

    def up(self, x, size):
        return F.interpolate(x, size=size, mode="nearest")

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        e4 = self.enc4(self.down(e3))

        b = self.bottom(self.down(e4))

        d4 = self.up(b, e4.shape[-2:])
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up(d4, e3.shape[-2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up(d3, e2.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up(d2, e1.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        residual = self.out_conv(d1)

        out = x + 2.0 * residual
        return out

if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    generator = ResidualGenerator()
    logger.info(generator(x).shape)