import torch
from torch import nn
from torch.nn import functional as F

from Architectures.VAE import VAE as VAE2


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(784, 98)
        self.fc2 = nn.Linear(98, 49)

    def forward(self, x):
        # 1x28x28 = 784 -> 784
        x = torch.flatten(x, start_dim=1)
        # 784 -> 98
        x = F.leaky_relu(self.fc1(x))
        # 98 -> 49
        x = F.leaky_relu(self.fc2(x))

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
        self.fc2 = nn.Linear(49, 98)
        self.fc3 = nn.Linear(98, 784)

    def forward(self, x):
        # 49 -> 98
        x = F.leaky_relu(self.fc2(x))
        # 98 -> 784
        x = self.fc3(x)

        return x


class VAE(VAE2):
    def __init__(self, num_latents):
        super().__init__(num_latents, Encoder, EncoderToLatents, LatentsToDecoder, Decoder)
