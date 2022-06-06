from os.path import exists
import numpy as np
import torch
import torchvision
import torch.distributions
import torch.optim as optim
import torchvision.transforms as transforms

import VAE
import VAE_helper

TRAIN_SET_SIZE = 54000
VAL_SET_SIZE = 6000
NUM_VAES = 3
MAX_EPOCHS = 20
BATCH_SIZE = 128
MAX_NO_IMPROVEMENT = 10

# Data
transform = transforms.Compose(
    [transforms.ToTensor()]
)

# Get the data and set up classes
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# Split the dataset into train and validation sets
train_set, val_set = torch.utils.data.random_split(dataset, [TRAIN_SET_SIZE, VAL_SET_SIZE])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE)

classes = [str(x) for x in range(10)]

# Preview some images
# data_iter = iter(train_loader)
# images, labels = data_iter.next()
# blitz_helper.show_image(torchvision.utils.make_grid(images))
# print(" ".join(f"{classes[labels[i]]}" for i in range(batch_size)))


def CB_log_prob_fn(x, *, logits=None, probs=None):
    # The continuous Bernoulli: fixing a pervasive error in variational autoencoders, Loaiza-Ganem G and Cunningham JP,
    # NeurIPS 2019. https://arxiv.org/abs/1907.06845.
    if logits is not None:
        CB = torch.distributions.ContinuousBernoulli(logits=logits)
        CB_log_prob = CB.log_prob(x).sum(dim=-1)
    elif probs is not None:
        CB = torch.distributions.ContinuousBernoulli(probs=probs)
        CB_log_prob = CB.log_prob(x).sum(dim=-1)
    else:
        RuntimeError("Either logits or probs needs to be passed to CB_log_prob_fn.")

    return CB_log_prob


def ELBO(logits, x, mu, logvar):
    CB_log_prob = CB_log_prob_fn(x, logits=logits)
    KLD = -0.5 * (1 + logvar - (mu ** 2) - logvar.exp()).sum(dim=-1)

    return (CB_log_prob - KLD).mean()


VAEs = []

# Train
for VAE_i in range(NUM_VAES):
    print(f"Training VAE {VAE_i}...")

    # Neural network
    VAEs.append(VAE.VAE())

    # Load the VAE if it exists
    VAE_path = f"./VAE_{VAE_i}.pth"
    if exists(VAE_path):
        print(f"Loading saved model {VAE_i}...")
        VAEs[VAE_i].load_state_dict(torch.load(VAE_path))
        continue
    else:
        print("No saved model exists...")

    best_val_epoch = 0
    best_val_loss = np.inf
    best_parameters = VAEs[VAE_i].state_dict()

    # Optimiser and loss function
    optimiser = optim.Adam(VAEs[VAE_i].parameters(), lr=1e-3) # 0.001

    for epoch in range(MAX_EPOCHS):
        train_loss = 0

        for i, data in enumerate(train_loader):
            # Get the inputs
            images, _ = data
            # Transform the images to appropriate input
            _, _, _, all_logits = VAE_helper.VAEs_output_logits(VAEs[:VAE_i], images)
            # Zero the parameter's gradients
            optimiser.zero_grad()
            # Forward, backward, loss, step
            logits, mu, logvar = VAEs[VAE_i](all_logits[-1])
            # Because optimisers minimise, and we want to maximise the ELBO, we multiply it by -1
            loss = -ELBO(logits, torch.sigmoid(all_logits[-1].view(-1, 28 * 28)), mu, logvar)
            loss.backward()
            optimiser.step()
            # Keep track of loss
            train_loss += loss.item()

        # Get the validation log probability
        val_loss = 0
        for i, data in enumerate(val_loader):
            # Get the inputs
            images, _ = data
            # Get the outputs of all the VAEs combined (including the one we're training)
            logits, mu, logvar, _ = VAE_helper.VAEs_output_logits(VAEs, images)
            # Calculate loss
            val_loss += -ELBO(logits, images.view(-1, 28 * 28), mu, logvar).item()

        # At the end of each epoch, print the total loss and validation log probability
        print(f"[Epoch {epoch:03}]\tTrain ELBO: {-train_loss}\tValidation ELBO: {-val_loss}")

        if val_loss < best_val_loss:
            print("Found new best model.")
            best_val_epoch = epoch
            best_val_loss = val_loss
            best_parameters = VAEs[VAE_i].state_dict()
        elif (epoch - best_val_epoch) == MAX_NO_IMPROVEMENT:
            print(f"No improvement after {MAX_NO_IMPROVEMENT} epochs.")
            print(f"Training next model...")

    print(f"Finished training VAE {VAE_i}!")
    torch.save(best_parameters, VAE_path)
