from os.path import exists
import numpy as np
import torch
import torchvision
import torch.distributions
import torchvision.transforms as transforms
import torch.optim as optim

import VAE

EPOCHS = 30
BATCH_SIZE = 128
TRAIN_UPDATE_FREQ = 100

# Data
transform = transforms.Compose(
    [transforms.ToTensor()]
)

# Get the data and set up classes
train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

classes = [str(x) for x in range(10)]

# Preview some images
# import VAE_helper
#
# data_iter = iter(train_loader)
# images, labels = data_iter.next()
# blitz_helper.show_image(torchvision.utils.make_grid(images))
# print(" ".join(f"{classes[labels[i]]}" for i in range(batch_size)))

# Neural network
vae = VAE.VAE()

# Optimiser and loss function
optimiser = optim.Adam(vae.parameters(), lr=0.001)


def ELBO(probs, x, mu, logvar):
    # The continuous Bernoulli: fixing a pervasive error in variational autoencoders, Loaiza-Ganem G and Cunningham JP,
    # NeurIPS 2019. https://arxiv.org/abs/1907.06845.
    CB = torch.distributions.ContinuousBernoulli(probs=probs)
    CB_log_prob = CB.log_prob(x.view(-1, 28 * 28)).sum(dim=-1)

    KLD = -0.5 * (1 + logvar - (mu ** 2) - logvar.exp()).sum(dim=-1)

    return (CB_log_prob - KLD).mean()


if exists("./VAE.pth"):
    print("Loading saved model...")
    vae.load_state_dict(torch.load("./VAE.pth"))
else:
    print("No saved model exists...")

best_model_loss = np.inf
best_model_parameters = vae.state_dict()

# Train
for epoch in range(EPOCHS):
    running_loss = 0

    for i, data in enumerate(train_loader):
        # Get the input and labels
        inputs, labels = data
        # Zero the parameter's gradients
        optimiser.zero_grad()
        # Forward, backward, loss, step
        probs, mu, logvar = vae(inputs)
        # Because optimisers minimise, and we want to maximise the ELBO, we multiply it by -1
        loss = -ELBO(probs, inputs, mu, logvar)
        loss.backward()
        optimiser.step()
        # Print statistics
        running_loss += loss.item()

        if (i % TRAIN_UPDATE_FREQ) == (TRAIN_UPDATE_FREQ - 1):
            print(f"[{epoch + 1}, {i + 1}]\tAvg. ELBO: {-running_loss / TRAIN_UPDATE_FREQ}")

            if running_loss < best_model_loss:
                print("Found new best model.")
                best_model_loss = running_loss
                best_model_parameters = vae.state_dict()

            running_loss = 0

print("Finished training!")
torch.save(best_model_parameters, "VAE.pth")
