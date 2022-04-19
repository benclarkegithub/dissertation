import matplotlib.pyplot as plt
import numpy as np


def show_image(image):
    image = image.numpy()
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()
