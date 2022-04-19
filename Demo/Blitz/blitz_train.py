# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn

import blitz

# Data
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()]
)

batch_size = 16

# Get the data and set up classes
train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

classes = [str(x) for x in range(10)]

# Preview some images
# import blitz_helper
#
# data_iter = iter(train_loader)
# images, labels = data_iter.next()
# blitz_helper.show_image(torchvision.utils.make_grid(images))
# print(" ".join(f"{classes[labels[i]]}" for i in range(batch_size)))

# Neural network
net = blitz.Net()

# Loss function and optimiser
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(net.parameters(), lr=0.001)

# Train
for epoch in range(10):
    running_loss = 0

    for i, data in enumerate(train_loader):
        # Get the input and labels
        inputs, labels = data
        # Zero the parameter's gradients
        optimiser.zero_grad()
        # Forward, backward, loss, step
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimiser.step()
        # Print statistics
        running_loss += loss.item()
        if (i % 2000) == 1999:
            print(f"[{epoch + 1}, {i + 1}] Loss: {running_loss:.3f}")
            running_loss = 0

print("Finished training")
torch.save(net.state_dict(), "MNIST_net.pth")
