import torch
import torchvision
import torchvision.transforms as transforms

import VAE
import VAE_helper

BATCH_SIZE = 16
NUM_VAES = 3

transform = transforms.Compose(
    [transforms.ToTensor()]
)

test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

classes = [str(x) for x in range(10)]

data_iter = iter(test_loader)
images, labels = data_iter.next()

VAEs = []

for VAE_i in range(NUM_VAES):
    # Load VAE
    VAEs.append(VAE.VAE())
    VAEs[VAE_i].load_state_dict(torch.load(f"./VAE_{VAE_i}.pth"))

    _, _, _, all_logits = VAE_helper.VAEs_output_logits(VAEs[:VAE_i], images)

    # Pass the last set of logits (corresponding to the (i - 1)th VAE) through the decoder
    mu, _ = VAEs[VAE_i].encoder(all_logits[-1])
    logits = VAEs[VAE_i].decoder(mu)

    # Preview produced images
    output_images = VAE_helper.logits_to_images(logits, images.shape)
    target_and_output = torch.cat([torch.sigmoid(all_logits[-1]), output_images], dim=0)
    output_images_grid = torchvision.utils.make_grid(target_and_output)
    VAE_helper.show_image(output_images_grid, f"Target vs. output (VAE {VAE_i})")

logits, _, _, _ = VAE_helper.VAEs_output_logits(VAEs, images)
output_images = VAE_helper.logits_to_images(logits, images.shape)
output_images_grid = torchvision.utils.make_grid(output_images)
VAE_helper.show_image(output_images_grid, f"All VAEs")

print(f"True labels: ", " ".join(f"{classes[labels[i]]}" for i in range(BATCH_SIZE)))
# print(f"Mu: ", " ".join(f"{mu[i]}" for i in range(BATCH_SIZE)))
