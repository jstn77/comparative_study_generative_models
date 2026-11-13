import torch.nn as nn
from configurations.configuration import (IMAGE_DIM, HIDDEN_DIM, LATENT_DIM)

class SparseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        #encoder part
        self.encoder = nn.Sequential(
            nn.Linear(IMAGE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
            nn.ReLU()
        )
        #decoder part
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, IMAGE_DIM),
            nn.Tanh()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder(z)
        return x.view(-1, 1, 28, 28)

    def forward(self, x, return_latent=False):
        z = self.encode(x)
        reconstruction = self.decode(z)
        if return_latent:
            return reconstruction, z
        return reconstruction