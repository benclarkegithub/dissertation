from Datasets import CIFAR10
from Evaluate import Evaluate as Eval
from Methods.Standard import Standard
from Architectures.VGG import VAE


SEED = 2024
BATCH_SIZE = 128
TRAIN_SET_SIZE = 47500
VAL_SET_SIZE = 2500

cifar10 = CIFAR10(
    root="../../Data" , batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE, seed=SEED)

NUM_LATENTS = 128
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_OUT_CHANNELS = 16
LOG_PROB_FN = "N"
STD = 0.05

method = Standard(
    VAE=VAE,
    num_latents=NUM_LATENTS,
    image_size=IMAGE_SIZE,
    num_channels=NUM_CHANNELS,
    num_out_channels=NUM_OUT_CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD)

MAX_EPOCHS = 100
MAX_NO_IMPROVEMENT = 25

evaluate = Eval(method=method, experiment="CIFAR10_VGG", seed=SEED)

# Train model
evaluate.train(
    train_loader=cifar10.get_train_loader(),
    val_loader=cifar10.get_val_loader(),
    max_epochs=MAX_EPOCHS,
    max_no_improvement=MAX_NO_IMPROVEMENT)

# Test model
reconstruction_opt = { "number": 10, "size": IMAGE_SIZE, "channels": NUM_CHANNELS }
output_images_opt = { "range": 10, "number": 11, "size": IMAGE_SIZE, "channels": NUM_CHANNELS }
conceptual_compression_opt = {
    "number": 10, "size": IMAGE_SIZE, "channels": NUM_CHANNELS, "random": True, "separate": True }

evaluate.test(
    test_loader=cifar10.get_test_loader(),
    avg_var=True,
    reconstruction=True,
    reconstruction_opt=reconstruction_opt,
    output_images=False,
    output_images_opt=output_images_opt,
    conceptual_compression=False,
    conceptual_compression_opt=conceptual_compression_opt)
