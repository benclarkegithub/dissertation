from Datasets import Omniglot
from Evaluate import Evaluate as Eval
from Methods.RNN import RNN
from Architectures.Basic import Canvas, Encoder, EncoderEncoderToEncoder, EncoderToLatents, EncoderLatentsToLatents, LatentsToLatents, LatentsToDecoder, Decoder


SEED = 2030
BATCH_SIZE = 128
# 5872
TRAIN_SET_SIZE = 5285
VAL_SET_SIZE = 587

omniglot = Omniglot(
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

NUM_LATENTS = 20
NUM_LATENTS_GROUP = 4
SIZE = 105
CHANNELS = 1

method = RNN(
    architecture=ARCHITECTURE,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    encoders=True,
    encoder_encoder_to_encoder=True,
    encoder_to_latents="Many",
    backprop=True,
    resample=True,
    size=SIZE)

MAX_EPOCHS = 100
MAX_NO_IMPROVEMENT = 25
GET_GRAD = True

evaluate = Eval(method=method, experiment="OMNIGLOT_Basic_20_4_ETL", seed=SEED)

# Train model
evaluate.train(
    train_loader=omniglot.get_train_loader(),
    val_loader=omniglot.get_val_loader(),
    max_epochs=MAX_EPOCHS,
    max_no_improvement=MAX_NO_IMPROVEMENT,
    get_grad=GET_GRAD)

# Test model
reconstruction_opt = { "number": 10, "size": SIZE, "channels": CHANNELS }
output_images_opt = { "range": 10, "number": 11, "size": SIZE, "channels": CHANNELS }
conceptual_compression_opt = {
    "number": 10, "size": SIZE, "channels": CHANNELS, "random": True, "separate": True }

evaluate.test(
    test_loader=omniglot.get_test_loader(),
    avg_var=True,
    reconstruction=True,
    reconstruction_opt=reconstruction_opt,
    output_images=True,
    output_images_opt=output_images_opt,
    conceptual_compression=True,
    conceptual_compression_opt=conceptual_compression_opt)
