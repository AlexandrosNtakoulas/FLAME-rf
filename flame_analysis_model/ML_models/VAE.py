import torch
import torch.nn as nn
from .autoencoder import AutoEncoder

class VAE(nn.Module):
    def __init__(self, num_input: int, num_latent: int, num_hidden: int):
        super().__init__()

        # Encoder
        self.in_2hid = nn.Linear(num_input, num_hidden)
        self.hid_2mu = nn.Linear(num_hidden, num_latent)
        self.hid_2logvar = nn.Linear(num_hidden, num_latent)  # changed name

        # Decoder
        self.z_2hid = nn.Linear(num_latent, num_hidden)
        self.hid_2in = nn.Linear(num_hidden, num_input)

        # Between
        self.between = nn.Linear(num_hidden, num_hidden)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.in_2hid(x))
        k = self.relu(self.between(h))
        mu = self.hid_2mu(k)
        logvar = self.hid_2logvar(k)
        return mu, logvar

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        k = self.relu(self.between(h))
        return self.hid_2in(k)  # maps to (0,1), fine for normalized inputs

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


