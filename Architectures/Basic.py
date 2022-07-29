import torch
from torch import nn
from torch.nn import functional as F

from Architectures.VAE import VAE as VAE2


class Canvas(nn.Module):
    def __init__(self, size, channels):
        super().__init__()

        self.fc = nn.Linear(1, channels * (size ** 2), bias=False)

    def forward(self, x):
        x = self.fc(x)

        return x


class Encoder(nn.Module):
    def __init__(self, num_latents, size, channels, out_channels):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(channels * (size ** 2), num_latents * 4)
        self.fc2 = nn.Linear(num_latents * 4, num_latents * 4)

    def forward(self, x):
        # channels x size x size -> channels * (size ** 2)
        x = torch.flatten(x, start_dim=1)
        # channels * (size ** 2) -> num_latents * 4
        x = F.leaky_relu(self.fc1(x))
        # num_latents * 4 -> num_latents * 4
        x = F.leaky_relu(self.fc2(x))

        return x


class EncoderEncoderToEncoder(nn.Module):
    def __init__(self, num_latents):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(num_latents * 8, num_latents * 4)
        self.fc2 = nn.Linear(num_latents * 4, num_latents * 4)

    def forward(self, enc_1, enc_2):
        # num_latents * 8 -> num_latents * 4
        x = F.leaky_relu(self.fc1(torch.cat([enc_1, enc_2], dim=1)))
        # num_latents * 4 -> num_latents * 4
        x = F.leaky_relu(self.fc2(x))

        return x


class EncoderToLatents(nn.Module):
    def __init__(self, num_latents, num_latents_group):
        super().__init__()

        # Fully-connected layers
        self.fc_mean = nn.Linear(num_latents * 4, num_latents_group)
        self.fc_logvar = nn.Linear(num_latents * 4, num_latents_group)

    def forward(self, x):
        # num_latents * 4 -> num_latents
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class LatentsToLatents(nn.Module):
    def __init__(self, num_latents_group):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(num_latents_group * 4, num_latents_group * 2)
        self.fc2 = nn.Linear(num_latents_group * 2, num_latents_group * 2)

    def forward(self, mu_1, logvar_1, mu_2, logvar_2):
        # num_latents_group * 4 -> num_latents_group * 2
        x = F.leaky_relu(self.fc1(torch.cat([mu_1, logvar_1, mu_2, logvar_2], dim=1)))
        # num_latents_group * 2 -> num_latents_group * 2
        x = self.fc2(x)
        # num_latents_group * 2 -> num_latents_group, num_latents_group
        mu_2, logvar_2 = x.split(x.shape[1] // 2, dim=1)

        return mu_2, logvar_2


class LatentsToDecoder(nn.Module):
    def __init__(self, num_latents, num_latents_group):
        super().__init__()

        # Fully-connected layers
        self.fc = nn.Linear(num_latents_group, num_latents * 4)

    def forward(self, x):
        # num_latents -> num_latents * 4
        x = F.leaky_relu(self.fc(x))

        return x


class Decoder(nn.Module):
    def __init__(self, num_latents, size, channels, out_channels):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(num_latents * 4, num_latents * 4)
        self.fc2 = nn.Linear(num_latents * 4, channels * (size ** 2))

    def forward(self, x):
        # num_latents * 4 -> num_latents * 4
        x = F.leaky_relu(self.fc1(x))
        # num_latents * 4 -> channels * (size ** 2)
        x = self.fc2(x)

        return x


class VAE(VAE2):
    def __init__(self, num_latents, num_latents_group=None, *, size=28, channels=1, out_channels=None):
        super().__init__(
            num_latents,
            num_latents_group,
            Encoder,
            EncoderToLatents,
            LatentsToDecoder,
            Decoder,
            size=size,
            channels=channels,
            out_channels=out_channels)
