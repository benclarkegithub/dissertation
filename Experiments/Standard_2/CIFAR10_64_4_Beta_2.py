from Datasets import CIFAR10
from Evaluate import Evaluate as Eval
from Methods.Standard import Standard
from Architectures.Basic import VAE


SEEDS = [2021, 2022]
BATCH_SIZE = 128
TRAIN_SET_SIZE = 47500
VAL_SET_SIZE = 2500

for trial, seed in enumerate(SEEDS):
    cifar10 = CIFAR10(
        root="../../Data", batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE, seed=seed)

    NUM_LATENTS = 64
    SIZE = 32
    CHANNELS = 3
    LOG_PROB_FN = "N"
    BETA = 4
    STD = 0.05
    HIDDEN_SIZE = 256

    method = Standard(
        VAE=VAE,
        num_latents=NUM_LATENTS,
        size=SIZE,
        channels=CHANNELS,
        log_prob_fn=LOG_PROB_FN,
        beta=BETA,
        std=STD,
        hidden_size=HIDDEN_SIZE)

    MAX_EPOCHS = 1000
    MAX_NO_IMPROVEMENT = 20
    GET_GRAD = True

    evaluate = Eval(method=method, experiment="CIFAR10_64_4_Beta_2", trial=trial, seed=seed)

    # Train model
    evaluate.train(
        train_loader=cifar10.get_train_loader(),
        val_loader=cifar10.get_val_loader(),
        max_epochs=MAX_EPOCHS,
        max_no_improvement=MAX_NO_IMPROVEMENT,
        get_grad=GET_GRAD)

    # Test model
    reconstruction_opt = {"number": 10, "size": SIZE, "channels": CHANNELS, "z": "mu"}
    output_images_opt = {"range": 10, "number": 11, "size": SIZE, "channels": CHANNELS}
    conceptual_compression_opt = {
        "number": 10, "size": SIZE, "channels": CHANNELS, "random": True, "separate": True, "z": "mu"}

    evaluate.test(
        test_loader=cifar10.get_test_loader(),
        avg_var=True,
        reconstruction=True,
        reconstruction_opt=reconstruction_opt,
        output_images=False,
        output_images_opt=output_images_opt,
        conceptual_compression=True,
        conceptual_compression_opt=conceptual_compression_opt)
