from Datasets import CIFAR10
from Evaluate import Evaluate as Eval
from Methods.RNN import RNN
from Architectures.Basic import Canvas, Encoder, EncoderEncoderToEncoder, EncoderToLatents, EncoderLatentsToLatents, LatentsToLatents, LatentsToDecoder, Decoder


SEED = 2024
BATCH_SIZE = 128
TRAIN_SET_SIZE = 47500
VAL_SET_SIZE = 2500

cifar10 = CIFAR10(
    root="../../Data" , batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE, seed=SEED)

ARCHITECTURE = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderEncoderToEncoder": EncoderEncoderToEncoder,
    "EncoderToLatents": EncoderToLatents,
    "EncoderLatentsToLatents": EncoderLatentsToLatents,
    "LatentsToLatents": LatentsToLatents,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

NUM_LATENTS = 64
NUM_LATENTS_GROUP = 4
SIZE = 32
CHANNELS = 3
OUT_CHANNELS = 16
LOG_PROB_FN = "N"
STD = 0.05

method = RNN(
    architecture=ARCHITECTURE,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    encoders=False,
    encoder_encoder_to_encoder=False,
    encoder_to_latents="Many",
    backprop=True,
    resample=True,
    size=SIZE,
    channels=CHANNELS,
    out_channels=OUT_CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD)

MAX_EPOCHS = 200
MAX_NO_IMPROVEMENT = 25
GET_GRAD = True

evaluate = Eval(method=method, experiment="CIFAR10_VGG_64_4_LTL", seed=SEED)

# Train model
evaluate.train(
    train_loader=cifar10.get_train_loader(),
    val_loader=cifar10.get_val_loader(),
    max_epochs=MAX_EPOCHS,
    max_no_improvement=MAX_NO_IMPROVEMENT,
    get_grad=GET_GRAD)

# Test model
reconstruction_opt = { "number": 10, "size": SIZE, "channels": CHANNELS }
output_images_opt = { "range": 10, "number": 11, "size": SIZE, "channels": CHANNELS }
conceptual_compression_opt = {
    "number": 10, "size": SIZE, "channels": CHANNELS, "random": True, "separate": True }

evaluate.test(
    test_loader=cifar10.get_test_loader(),
    avg_var=True,
    reconstruction=True,
    reconstruction_opt=reconstruction_opt,
    output_images=False,
    output_images_opt=output_images_opt,
    conceptual_compression=True,
    conceptual_compression_opt=conceptual_compression_opt)
