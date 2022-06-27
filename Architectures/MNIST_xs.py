import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(784, 49)

    def forward(self, x):
        # 1x28x28 = 784 -> 784
        x = torch.flatten(x, start_dim=1)
        # 784 -> 49
        x = F.leaky_relu(self.fc1(x))

        return x


class EncoderToLatents(nn.Module):
    def __init__(self, num_latents):
        super().__init__()

        self.num_latents = num_latents

        # Fully-connected layers
        self.fc_mean = nn.Linear(49, num_latents)
        self.fc_logvar = nn.Linear(49, num_latents)

    def forward(self, x):
        # 49 -> num_latents
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class LatentsToDecoder(nn.Module):
    def __init__(self, num_latents):
        super().__init__()

        self.num_latents = num_latents

        # Fully-connected layers
        self.fc = nn.Linear(num_latents, 49)

    def forward(self, x):
        # num_latents -> 49
        x = F.leaky_relu(self.fc(x))

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Fully-connected layers
        self.fc3 = nn.Linear(49, 784)

    def forward(self, x):
        # 49 -> 784
        x = self.fc3(x)

        return x


class VAE(nn.Module):
    def __init__(self, num_latents):
        super().__init__()

        self.num_latents = num_latents

        self.encoder = Encoder()
        self.enc_to_lat = EncoderToLatents(num_latents)
        self.lat_to_dec = LatentsToDecoder(num_latents)
        self.decoder = Decoder()

    def forward(self, x):
        x_enc = self.encoder(x)
        mu, logvar = self.enc_to_lat(x_enc)
        z = self.reparameterise(mu, logvar)
        z_dec = self.lat_to_dec(z)
        logits = self.decoder(z_dec)

        return {
            "x_enc": x_enc,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_dec": z_dec,
            "logits": logits
        }

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + (std * eps)

    def x_to_mu_logvar(self, x):
        x_enc = self.encoder(x)
        mu, logvar = self.enc_to_lat(x_enc)

        return x_enc, mu, logvar

    def z_to_logits(self, z):
        z_dec = self.lat_to_dec(z)
        logits = self.decoder(z_dec)

        return z_dec, logits
