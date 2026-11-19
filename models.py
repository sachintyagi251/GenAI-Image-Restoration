import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    """Compact convolutional autoencoder for 64x64 RGB images."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True), # 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True), # 16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True), # 8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 16
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 32
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 64
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class VAE(nn.Module):
    """Convolutional VAE. Returns reconstruction, mu and log-variance."""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 8
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 4
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 32
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid() # 64
        )
    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    def decode(self, z):
        h = self.fc_decode(z).view(-1, 256, 4, 4)
        return self.decoder(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
