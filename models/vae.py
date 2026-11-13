import torch
import torch.nn as nn
from configurations.configuration import (IMAGE_DIM, HIDDEN_DIM, LATENT_DIM)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        #encoder part
        self.encoder = nn.Sequential(
            nn.Linear(IMAGE_DIM, HIDDEN_DIM),
            nn.ReLU()
        )
        #latent distribution
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        #decoder part
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, IMAGE_DIM),
            nn.Tanh()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, LATENT_DIM, device=device)
        return self.decode(z)
