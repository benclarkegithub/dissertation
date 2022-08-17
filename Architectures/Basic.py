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


class KL(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_mu = nn.Linear(1, 1, bias=False)
        self.fc_logvar = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Encoder(nn.Module):
    def __init__(self, size, channels, hidden_size, out_channels):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(channels * (size ** 2), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # channels x size x size -> channels * (size ** 2)
        x = torch.flatten(x, start_dim=1)
        # channels * (size ** 2) -> hidden_size
        x = F.leaky_relu(self.fc1(x))
        # hidden_size -> hidden_size
        x = F.leaky_relu(self.fc2(x))

        return x


class EncoderEncoderToEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, enc_1, enc_2):
        # hidden_size * 2 -> hidden_size
        x = F.leaky_relu(self.fc1(torch.cat([enc_1, enc_2], dim=1)))
        # hidden_size -> hidden_size
        x = F.leaky_relu(self.fc2(x))

        return x


class EncoderToLatents(nn.Module):
    def __init__(self, hidden_size, num_latents_group):
        super().__init__()

        # Fully-connected layers
        self.fc_mean = nn.Linear(hidden_size, num_latents_group)
        self.fc_logvar = nn.Linear(hidden_size, num_latents_group)

    def forward(self, x):
        # hidden_size -> num_latents_group
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class EncoderLatentsToLatents(nn.Module):
    def __init__(self, hidden_size, group, num_latents_group):
        super().__init__()

        # Fully-connected layers
        encoder_plus_latents_size = hidden_size + (group * num_latents_group * 2)
        self.fc1 = nn.Linear(encoder_plus_latents_size, encoder_plus_latents_size)
        self.fc_mean = nn.Linear(encoder_plus_latents_size, num_latents_group)
        self.fc_logvar = nn.Linear(encoder_plus_latents_size, num_latents_group)

    def forward(self, x, mu, logvar):
        # hidden_size + (group * num_latents_group * 2) -> hidden_size + (group * num_latents_group * 2)
        if mu is not None:
            x = F.leaky_relu(self.fc1(torch.cat([x, mu, logvar], dim=1)))
        else:
            x = F.leaky_relu(self.fc1(x))
        # hidden_size + (group * num_latents_group) -> num_latents_group
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class LatentsToLatents(nn.Module):
    def __init__(self, num_latents_group):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(num_latents_group * 4, num_latents_group * 4)
        self.fc_mean = nn.Linear(num_latents_group * 4, num_latents_group)
        self.fc_logvar = nn.Linear(num_latents_group * 4, num_latents_group)

    def forward(self, mu_1, logvar_1, mu_2, logvar_2):
        # num_latents_group * 4 -> num_latents_group * 4
        x = F.leaky_relu(self.fc1(torch.cat([mu_1, logvar_1, mu_2, logvar_2], dim=1)))
        # num_latents_group * 4 -> num_latents_group
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class LatentsToLatentsComplicated(nn.Module):
    def __init__(self, num_latents_group):
        super().__init__()

        hidden_size = 100

        # Fully-connected layers
        self.fc1 = nn.Linear(num_latents_group * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, num_latents_group)
        self.fc_logvar = nn.Linear(hidden_size, num_latents_group)

    def forward(self, mu_1, logvar_1, mu_2, logvar_2):
        # num_latents_group * 4 -> hidden_size
        x = F.leaky_relu(self.fc1(torch.cat([mu_1, logvar_1, mu_2, logvar_2], dim=1)))
        # hidden_size -> hidden_size
        x = F.leaky_relu(self.fc2(x))
        # hidden_size -> hidden_size
        x = F.leaky_relu(self.fc3(x))
        # hidden_size -> num_latents_group
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class LatentsToDecoder(nn.Module):
    def __init__(self, hidden_size, num_latents_group):
        super().__init__()

        # Fully-connected layers
        self.fc = nn.Linear(num_latents_group, hidden_size)

    def forward(self, x):
        # num_latents -> hidden_size
        x = F.leaky_relu(self.fc(x))

        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size, size, channels, out_channels):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, channels * (size ** 2))

    def forward(self, x):
        # hidden_size -> hidden_size
        x = F.leaky_relu(self.fc1(x))
        # hidden_size -> channels * (size ** 2)
        x = self.fc2(x)

        return x


class VAE(VAE2):
    def __init__(self,
                 num_latents,
                 num_latents_group=None,
                 *,
                 size=28,
                 channels=1,
                 out_channels=None,
                 hidden_size=None):
        super().__init__(
            num_latents,
            num_latents_group,
            Encoder,
            EncoderToLatents,
            LatentsToDecoder,
            Decoder,
            size=size,
            channels=channels,
            out_channels=out_channels,
            hidden_size=hidden_size)
