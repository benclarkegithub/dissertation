import torch
from torch import nn
from torch.nn import functional as F

NUM_LATENTS = 2


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Convolutional layers
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input image channels, 12 output channels, 3x3 square convolution
        self.conv2 = nn.Conv2d(6, 12, 3)
        # Fully-connected layers
        self.fc1 = nn.Linear(5 * 5 * 12, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc_mean = nn.Linear(50, NUM_LATENTS)
        self.fc_logvar = nn.Linear(50, NUM_LATENTS)

    def forward(self, x):
        # (28x28x1 -> 24x24x6)
        x = F.leaky_relu(self.conv1(x))
        # (24x24x6 -> 12x12x6)
        x = F.max_pool2d(x, 2)
        # (12x12x6 -> 10x10x12)
        x = F.leaky_relu(self.conv2(x))
        # (10x10x12 -> 5x5x12)
        x = F.max_pool2d(x, 2)
        # (5x5x12 -> 300)
        x = torch.flatten(x, 1)
        # (300 -> 100)
        x = F.leaky_relu(self.fc1(x))
        # (100 -> 50)
        x = F.leaky_relu(self.fc2(x))
        # (50 -> NUM_LATENTS)
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Fully-connected layers
        self.fc1 = nn.Linear(NUM_LATENTS, 50)
        self.fc2 = nn.Linear(50, 250)
        self.fc3 = nn.Linear(250, 28 * 28)

    def forward(self, x):
        # (NUM_LATENTS -> 50)
        x = F.leaky_relu(self.fc1(x))
        # (50 -> 250)
        x = F.leaky_relu(self.fc2(x))
        # (250 -> 784)
        probs = torch.sigmoid(self.fc3(x))

        return probs


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        probs = self.decoder(z)

        return probs, mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + (std * eps)
