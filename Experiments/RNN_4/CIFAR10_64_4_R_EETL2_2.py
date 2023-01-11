from Datasets import CIFAR10
from Evaluate import Evaluate as Eval
from Methods.RNN import RNN
from Architectures.Basic import Canvas, Encoder, EncoderEncoderToLatents2, LatentsToDecoder, Decoder


SEEDS = [2021]
BATCH_SIZE = 128
TRAIN_SET_SIZE = 47500
VAL_SET_SIZE = 2500

for trial, seed in enumerate(SEEDS):
    cifar10 = CIFAR10(
        root="../../Data", batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE, seed=seed)

    ARCHITECTURE = {
        "Canvas": Canvas,
        "Encoder": Encoder,
        "EncoderEncoderToLatents": EncoderEncoderToLatents2,
        "LatentsToDecoder": LatentsToDecoder,
        "Decoder": Decoder
    }

    NUM_LATENTS = 64
    NUM_LATENTS_GROUP = 4
    TYPE = "Single"
    LEARNING_RATE = 1e-4
    SIZE = 32
    CHANNELS = 3
    LOG_PROB_FN = "N"
    STD = 0.05
    HIDDEN_SIZE = 256
    CLIP = 1e5

    method = RNN(
        architecture=ARCHITECTURE,
        num_latents=NUM_LATENTS,
        num_latents_group=NUM_LATENTS_GROUP,
        reconstruction=True,
        encoders=True,
        to_latents="EncoderEncoder",
        encoder_to_latents=True,
        backprop=True,
        resample=True,
        type=TYPE,
        learning_rate=LEARNING_RATE,
        size=SIZE,
        channels=CHANNELS,
        log_prob_fn=LOG_PROB_FN,
        std=STD,
        hidden_size=HIDDEN_SIZE,
        clip=CLIP)

    MAX_EPOCHS = 500
    MAX_NO_IMPROVEMENT = 20
    GET_GRAD = True

    evaluate = Eval(method=method, experiment="CIFAR10_64_4_R_EETL2_2", trial=trial, seed=seed)

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
