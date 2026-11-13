import torch
import torch.nn as nn
from configurations.configuration import (IMAGE_DIM, HIDDEN_DIM, GAN_NOISE_DIM)

class VanillaGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(GAN_NOISE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, IMAGE_DIM),
            nn.Tanh()
        )

    def forward(self, z):
        generated = self.network(z)
        return generated.view(-1, 1, 28, 28)


class VanillaDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(IMAGE_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class VanillaGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = VanillaGenerator()
        self.discriminator = VanillaDiscriminator()

    def generate(self, num_samples, device):
        z = torch.randn(
            num_samples,
            GAN_NOISE_DIM,
            device=device
        )
        return self.generator(z)