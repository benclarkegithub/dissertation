import matplotlib.pyplot as plt
import numpy as np
import torch


def show_image(image, title):
    image = image.numpy()
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.title(title)
    plt.show()


def VAEs_output(VAEs, images):
    logits = torch.zeros_like(images.view(-1, 28 * 28), requires_grad=False)
    mu = None
    logvar = None
    all_images = torch.clone(images).unsqueeze(dim=0) # (len(VAEs) + 1, BATCH_SIZE, 1, 28, 28)

    for VAE_i, VAE in enumerate(VAEs):
        with torch.no_grad():
            VAE_i_logits, VAE_i_mu, VAE_i_logvar = VAE(all_images[-1])
            logits += VAE_i_logits
            if VAE_i == 0:
                mu = VAE_i_mu
                logvar = VAE_i_logvar
            else:
                mu = torch.hstack([mu, VAE_i_mu])
                logvar = torch.hstack([logvar, VAE_i_logvar])
            transformed_images = transform_images(all_images[-1], VAE_i_logits)
            all_images = torch.cat([all_images, transformed_images.unsqueeze(dim=0)], dim=0)

    return logits, mu, logvar, all_images


def transform_images(images, logits):
    # SOME OF THESE VALUES COULD BE OUT OF THE RANGE (0, 1)!!!
    transformed_images = images - logits_to_images(logits, images.shape)
    transformed_images = clip_images(transformed_images)

    return transformed_images


def logits_to_images(logits, shape):
    images = torch.sigmoid(logits).reshape(shape)

    return images


def clip_images(images):
    # Hack, clip the values for training, not a perfect solution...
    images = torch.where(images < torch.tensor(1e-6), torch.tensor(1e-6), images)
    images = torch.where(images > 1 - torch.tensor(1e-6), 1 - torch.tensor(1e-6), images)

    return images
