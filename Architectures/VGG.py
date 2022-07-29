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

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels * 4, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(
            in_channels=out_channels * 4, out_channels=out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=out_channels * 4, out_channels=out_channels * 8, kernel_size=2, stride=2, padding=0)

        # Fully-connected layers
        self.fc1 = nn.Linear(out_channels * 8 * ((size // 2 // 2 // 2) ** 2), num_latents * 4)
        self.fc2 = nn.Linear(num_latents * 4, num_latents * 4)

    def forward(self, x):
        # channels x size x size -> (out_channels * 8) x (size // 2 // 2 // 2) x (size // 2 // 2 // 2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        # (out_channels * 8) x (size // 2 // 2 // 2) x (size // 2 // 2 // 2) -> out_channels * 8 * ((size // 2 // 2 // 2) ** 2)
        x = torch.flatten(x, start_dim=1)
        # out_channels * 8 * ((size // 2 // 2 // 2) ** 2) -> num_latents * 4
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

        self.size = size
        self.out_channels = out_channels

        # Fully-connected layers
        self.fc1 = nn.Linear(num_latents * 4, num_latents * 4)
        self.fc2 = nn.Linear(num_latents * 4, out_channels * 8 * ((size // 2 // 2 // 2) ** 2))

        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=out_channels * 8, out_channels=out_channels * 4, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=out_channels * 4, out_channels=out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=out_channels * 4, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=out_channels * 2, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(
            in_channels=out_channels * 2, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        self.deconv6 = nn.ConvTranspose2d(
            in_channels=out_channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # num_latents * 4 -> num_latents * 4
        x = F.leaky_relu(self.fc1(x))
        # num_latents * 4 -> out_channels * 8 * ((size // 2 // 2 // 2) ** 2)
        x = F.leaky_relu(self.fc2(x))
        # out_channels * 8 * ((size // 2 // 2 // 2) ** 2) -> (out_channels * 8) x (size // 2 // 2 // 2) x (size // 2 // 2 // 2)
        x = x.reshape(x.shape[0], self.out_channels * 8, self.size // 2 // 2 // 2, self.size // 2 // 2 // 2)
        # (out_channels * 8) x (size // 2 // 2 // 2) x (size // 2 // 2 // 2) -> channels x size x size
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
        x = F.leaky_relu(self.deconv3(x))
        x = F.leaky_relu(self.deconv4(x))
        x = F.leaky_relu(self.deconv5(x))
        x = self.deconv6(x)
        # channels x size x size -> channels * (size ** 2)
        x = torch.flatten(x, start_dim=1)

        return x


class VAE(VAE2):
    def __init__(self, num_latents, num_latents_group=None, *, size=28, channels=1, out_channels=8):
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
