from Datasets import FashionMNIST
from Evaluate import Evaluate as Eval
from Methods.RNN import RNN
from Architectures.Basic import Canvas, Encoder, EncoderEncoderToEncoder, EncoderToLatents, EncoderLatentsToLatents, LatentsToLatents, LatentsToDecoder, Decoder


SEED = 2030
BATCH_SIZE = 128
TRAIN_SET_SIZE = 57000
VAL_SET_SIZE = 3000

fashion_mnist = FashionMNIST(
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

NUM_LATENTS = 10
NUM_LATENTS_GROUP = 1
LOG_PROB_FN = "N"
STD = 0.05

method = RNN(
    architecture=ARCHITECTURE,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    encoders=True,
    encoder_encoder_to_encoder=True,
    encoder_to_latents="Many",
    backprop=True,
    resample=True,
    log_prob_fn=LOG_PROB_FN,
    std=STD)

MAX_EPOCHS = 100
MAX_NO_IMPROVEMENT = 25
GET_GRAD = True

evaluate = Eval(method=method, experiment="FashionMNIST_Basic_10_1_ETL", seed=SEED)

# Train model
evaluate.train(
    train_loader=fashion_mnist.get_train_loader(),
    val_loader=fashion_mnist.get_val_loader(),
    max_epochs=MAX_EPOCHS,
    max_no_improvement=MAX_NO_IMPROVEMENT,
    get_grad=GET_GRAD)

# Test model
evaluate.test(test_loader=fashion_mnist.get_test_loader())
