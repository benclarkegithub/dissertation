# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

# Neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input image channels, 12 output channels, 3x3 square convolution
        self.conv2 = nn.Conv2d(6, 12, 3)
        # Affine operations: y = Wx + b
        self.fc1 = nn.Linear(5 * 5 * 12, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        # (28x28x1 -> 24x24x6)
        x = F.leaky_relu(self.conv1(x))
        # (24x24x6 -> 12x12x6)
        x = F.max_pool2d(x, 2)
        # (12x12x6 -> 10x10x12)
        x = F.leaky_relu(self.conv2(x))
        # (10x10x12 -> 5x5x12)
        x = F.max_pool2d(x, 2)
        # Flatten the input
        x = torch.flatten(x, 1)
        # 2 non-linear fully-connected layers
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # Linear fully connected layer
        x = self.fc3(x)

        # Return the output
        return x
