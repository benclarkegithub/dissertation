from Datasets import MNIST
from Evaluate import Evaluate as Eval
from Methods.Standard import Standard
from Architectures.Basic import VAE


SEEDS = [2018, 2019, 2020, 2021, 2022]
BATCH_SIZE = 128
TRAIN_SET_SIZE = 57000
VAL_SET_SIZE = 3000

for trial, seed in enumerate(SEEDS):
    mnist = MNIST(
        root="../../Data" , batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE, seed=seed)

    NUM_LATENTS = 10

    method = Standard(VAE=VAE, num_latents=NUM_LATENTS)

    MAX_EPOCHS = 100
    MAX_NO_IMPROVEMENT = 10
    GET_GRAD = True

    evaluate = Eval(method=method, experiment="MNIST_10", trial=trial, seed=seed)

    # Train model
    evaluate.train(
        train_loader=mnist.get_train_loader(),
        val_loader=mnist.get_val_loader(),
        max_epochs=MAX_EPOCHS,
        max_no_improvement=MAX_NO_IMPROVEMENT,
        get_grad=GET_GRAD)

    # Test model
    evaluate.test(test_loader=mnist.get_test_loader())
