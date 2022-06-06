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
            VAE_logits, VAE_mu, VAE_logvar = VAE(all_images[-1])
            logits += VAE_logits
            if VAE_i == 0:
                mu = VAE_mu
                logvar = VAE_logvar
            else:
                mu = torch.hstack([mu, VAE_mu])
                logvar = torch.hstack([logvar, VAE_logvar])
            transformed_images = transform_images(all_images[-1], VAE_logits)
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


def VAEs_output_logits(VAEs, images):
    LOGITS_RANGE = 10.0

    logits = torch.zeros_like(images.view(-1, 28 * 28), requires_grad=False)
    mu = None
    logvar = None

    all_images = torch.clone(images).unsqueeze(dim=0) # (1 + len(VAEs), BATCH_SIZE, 1, 28, 28)
    all_logits = clip_logits(torch.logit(all_images), LOGITS_RANGE) # (1 + len(VAEs), BATCH_SIZE, 1, 28, 28)

    for VAE_i, VAE in enumerate(VAEs):
        with torch.no_grad():
            VAE_logits, VAE_mu, VAE_logvar = VAE(all_logits[-1])
            # Clip the output logits
            VAE_logits = clip_logits(VAE_logits, LOGITS_RANGE)
            logits += VAE_logits
            if VAE_i == 0:
                mu = VAE_mu
                logvar = VAE_logvar
            else:
                mu = torch.hstack([mu, VAE_mu])
                logvar = torch.hstack([logvar, VAE_logvar])
            transformed_logits = all_logits[-1] - VAE_logits.reshape(images.shape)
            all_logits = torch.cat([all_logits, transformed_logits.unsqueeze(dim=0)], dim=0)

    return logits, mu, logvar, all_logits


def clip_logits(logits, range):
    # Hack, clip the values for training, not a perfect solution...
    logits = torch.where(logits < torch.tensor(-range), torch.tensor(-range), logits)
    logits = torch.where(logits > torch.tensor(range), torch.tensor(range), logits)

    return logits
