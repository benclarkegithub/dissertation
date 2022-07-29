from Datasets import MNIST
from Evaluate import Evaluate as Eval
from Methods.Standard import Standard
from Architectures.Basic import VAE


SEED = 2024
BATCH_SIZE = 128
TRAIN_SET_SIZE = 57000
VAL_SET_SIZE = 3000

mnist = MNIST(
    root="../../Data" , batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE, seed=SEED)

NUM_LATENTS = 10

method = Standard(VAE=VAE, num_latents=NUM_LATENTS)

MAX_EPOCHS = 100
MAX_NO_IMPROVEMENT = 25

evaluate = Eval(method=method, experiment="MNIST_Basic_10", seed=SEED)

# Train model
evaluate.train(
    train_loader=mnist.get_train_loader(),
    val_loader=mnist.get_val_loader(),
    max_epochs=MAX_EPOCHS,
    max_no_improvement=MAX_NO_IMPROVEMENT)

# Test model
evaluate.test(test_loader=mnist.get_test_loader())
