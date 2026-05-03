import torch
import torch.nn as nn
from loguru import logger

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        def block(in_c, out_c, norm=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_ch, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 256),
            nn.Conv2d(256, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    discriminator = PatchDiscriminator()
    logger.info(discriminator(x).shape)