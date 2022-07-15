import torch
from torch import nn
from torch.nn import functional as F

from Architectures.VAE import VAE as VAE2


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=0)

        # Fully-connected layers
        self.fc1 = nn.Linear(784, 196)
        self.fc2 = nn.Linear(196, 49)

    def forward(self, x):
        # 1x28x28 = 784 -> 2x28x28 = 1568
        x = F.leaky_relu(self.conv1(x))
        # 2x28x28 = 1568 -> 4x14x14 = 784
        x = F.leaky_relu(self.conv2(x))
        # 4x14x14 = 784 -> 8x14x14 = 1568
        x = F.leaky_relu(self.conv3(x))
        # 8x14x14 = 1568 -> 16x7x7 = 784
        x = F.leaky_relu(self.conv4(x))
        # 16x7x7 = 784 -> 784
        x = torch.flatten(x, start_dim=1)
        # 784 -> 196
        x = F.leaky_relu(self.fc1(x))
        # 196 -> 49
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
        self.fc2 = nn.Linear(49, 196)
        self.fc3 = nn.Linear(196, 784)

        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 49 -> 196
        x = F.leaky_relu(self.fc2(x))
        # 196 -> 784
        x = F.leaky_relu(self.fc3(x))
        # 784 -> 16x7x7
        x = torch.reshape(x, (-1, 16, 7, 7))
        # 16x7x7 = 784 -> 8x14x14 = 1568
        x = F.leaky_relu(self.deconv1(x))
        # 8x14x14 = 1568 -> 4x14x14 = 784
        x = F.leaky_relu(self.deconv2(x))
        # 4x14x14 = 784 -> 2x28x28 = 1568
        x = F.leaky_relu(self.deconv3(x))
        # 2x28x28 = 1568 -> 1x28x28 = 784
        x = self.deconv4(x)
        # 1x28x28 = 784
        x = torch.flatten(x, start_dim=1)

        return x


class VAE(VAE2):
    def __init__(self, num_latents):
        super().__init__(num_latents, Encoder, EncoderToLatents, LatentsToDecoder, Decoder)
