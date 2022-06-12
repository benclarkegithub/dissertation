import torch
import torchvision
import torchvision.transforms as transforms

import VAE
import VAE_old
import VAE_helper

USE_OLD = False

BATCH_SIZE = 16

transform = transforms.Compose(
    [transforms.ToTensor()]
)

test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

classes = [str(x) for x in range(10)]

data_iter = iter(test_loader)
images, labels = data_iter.next()

# Neural network
if not USE_OLD:
    net = VAE.VAE()
    path = "./VAE.pth"
else:
    net = VAE_old.VAE()
    path = "./VAE.old.pth"

net.load_state_dict(torch.load(path))

mu, _ = net.encoder(images)
output = net.decoder(mu)

output_images = output.unsqueeze(dim=1).reshape(BATCH_SIZE, 1, 28, 28).detach()
output_images_grid = torchvision.utils.make_grid(output_images)
VAE_helper.show_image(output_images_grid)

print(f"True labels: ", " ".join(f"{classes[labels[i]]}" for i in range(BATCH_SIZE)))
print(f"Mu: ", " ".join(f"{mu[i]}" for i in range(BATCH_SIZE)))
