from Datasets import MNIST
from Evaluate import Evaluate as Eval
from Methods.Quotient import Quotient
from Architectures.Basic import Canvas, KL, Encoder, EncoderToLatents, LatentsToDecoder, Decoder


SEED = 2030
BATCH_SIZE = 128
TRAIN_SET_SIZE = 57000
VAL_SET_SIZE = 3000

mnist = MNIST(
    root="../../Data" , batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE, seed=SEED)

ARCHITECTURE = {
    "Canvas": Canvas,
    "KL": KL,
    "Encoder": Encoder,
    "EncoderToLatents": EncoderToLatents,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

NUM_LATENTS = 2
NUM_LATENTS_GROUP = 1
LEARNING_RATE = 1e-4
CLIP = 1e6

method = Quotient(
    architecture=ARCHITECTURE,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    encoder_to_latents=True,
    resample=False,
    learnable_KL=True,
    learning_rate=LEARNING_RATE,
    clip=CLIP)

MAX_EPOCHS = 50
MAX_NO_IMPROVEMENT = 10
GET_GRAD = True

evaluate = Eval(method=method, experiment="MNIST_Basic_2_1_Both", seed=SEED)

# Train model
evaluate.train(
    train_loader=mnist.get_train_loader(),
    val_loader=mnist.get_val_loader(),
    max_epochs=MAX_EPOCHS,
    max_no_improvement=MAX_NO_IMPROVEMENT,
    get_grad=GET_GRAD)

# Test model
reconstruction_opt = { "number": 10, "size": 28, "channels": 1, "z": "Sample" }
output_images_opt = { "range": 100, "number": 11, "size": 28, "channels": 1 }
conceptual_compression_opt = { "number": 8, "size": 28, "channels": 1, "random": True, "separate": True, "z": "Sample" }

evaluate.test(
    test_loader=mnist.get_test_loader(),
    avg_var=False,
    reconstruction=True,
    reconstruction_opt=reconstruction_opt,
    output_images=True,
    output_images_opt=output_images_opt,
    conceptual_compression=True,
    conceptual_compression_opt=conceptual_compression_opt)
