import torch
import torch.nn as nn
from configurations.configuration import DCGAN_NOISE_DIM

class DCGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 1 x 1 -> 7 x 7
            nn.ConvTranspose2d(
                DCGAN_NOISE_DIM,
                256,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7 x 7 -> 14 x 14
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 14 x 14 -> 28 x 28
            nn.ConvTranspose2d(
                128,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        return self.network(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 28 x 28 -> 14 x 14
            nn.Conv2d(
                1,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # 14 x 14 -> 7 x 7
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class DCGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = DCGANGenerator()
        self.discriminator = DCGANDiscriminator()

    def generate(self, num_samples, device):
        z = torch.randn(num_samples, DCGAN_NOISE_DIM, 1, 1, device=device)
        return self.generator(z)