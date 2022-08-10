from Datasets import CIFAR10
from Evaluate import Evaluate as Eval
from Methods.RNN import RNN
from Architectures.Basic import Canvas, Encoder, EncoderEncoderToEncoder, EncoderToLatents, EncoderLatentsToLatents, \
    LatentsToLatents, LatentsToDecoder, Decoder


# Epoch 23: (Decreased gradient clip from 1e7 to 1e6)
# ValueError: Expected parameter loc (Tensor of shape (128, 3072)) of distribution Normal(loc: torch.Size([128, 3072]), scale: torch.Size([128, 3072])) to satisfy the constraint Real(), but found invalid values:
# tensor([[0.1775, 0.1729, 0.1777,  ..., 0.1244, 0.1185, 0.1160],
#         [0.6357, 0.6290, 0.6281,  ..., 0.2961, 0.2942, 0.3067],
#         [0.8741, 0.8749, 0.8770,  ..., 0.5713, 0.5778, 0.5827],
#         ...,
#         [0.4697, 0.4769, 0.4795,  ..., 0.3561, 0.3534, 0.3497],
#         [0.9148, 0.9146, 0.9195,  ..., 0.4955, 0.5298, 0.5390],
#         [0.2690, 0.2595, 0.2587,  ..., 0.3098, 0.3005, 0.2948]],
#        grad_fn=<SigmoidBackward0>)

# Epoch 35: (Loaded best afterwards)
# ValueError: Expected parameter loc (Tensor of shape (128, 3072)) of distribution Normal(loc: torch.Size([128, 3072]), scale: torch.Size([128, 3072])) to satisfy the constraint Real(), but found invalid values:
# tensor([[0.4828, 0.4890, 0.5032,  ..., 0.3047, 0.3099, 0.3138],
#         [0.0776, 0.0766, 0.0791,  ..., 0.4116, 0.4248, 0.4363],
#         [0.1028, 0.1062, 0.1099,  ..., 0.0740, 0.0791, 0.0784],
#         ...,
#         [0.5493, 0.5233, 0.5201,  ..., 0.9144, 0.9197, 0.9198],
#         [0.9057, 0.9116, 0.9174,  ..., 0.6058, 0.6297, 0.6299],
#         [0.4775, 0.4800, 0.4898,  ..., 0.3969, 0.4006, 0.4092]],
#        grad_fn=<SigmoidBackward0>)

# Epoch 39:
# ValueError: Expected parameter loc (Tensor of shape (128, 3072)) of distribution Normal(loc: torch.Size([128, 3072]), scale: torch.Size([128, 3072])) to satisfy the constraint Real(), but found invalid values:
# tensor([[0.1062, 0.0950, 0.0890,  ..., 0.6555, 0.6638, 0.6643],
#         [0.5812, 0.5836, 0.5929,  ..., 0.3524, 0.3477, 0.3469],
#         [0.2571, 0.2443, 0.2475,  ..., 0.5021, 0.5048, 0.5254],
#         ...,
#         [0.7556, 0.7615, 0.7747,  ..., 0.2038, 0.1996, 0.1972],
#         [0.1286, 0.1353, 0.1487,  ..., 0.1389, 0.1537, 0.1540],
#         [0.8008, 0.7837, 0.7783,  ..., 0.5065, 0.5011, 0.5037]],
#        grad_fn=<SigmoidBackward0>)


SEED = 2040
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
TYPE = "Single"
SIZE = 32
CHANNELS = 3
OUT_CHANNELS = 16
LOG_PROB_FN = "N"
STD = 0.05
HIDDEN_SIZE = 256
CLIP = 1e6

method = RNN(
    architecture=ARCHITECTURE,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    reconstruction=True,
    encoders=False,
    to_latents="Encoder",
    encoder_to_latents=True,
    backprop=True,
    resample=True,
    type=TYPE,
    size=SIZE,
    channels=CHANNELS,
    out_channels=OUT_CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD,
    hidden_size=HIDDEN_SIZE,
    clip=CLIP)

MAX_EPOCHS = 100
MAX_NO_IMPROVEMENT = 10
GET_GRAD = True
LOAD_BEST = True

evaluate = Eval(method=method, experiment="CIFAR10_Basic_64_4_R_ETL_100", seed=SEED)

# Train model
evaluate.train(
    train_loader=cifar10.get_train_loader(),
    val_loader=cifar10.get_val_loader(),
    max_epochs=MAX_EPOCHS,
    max_no_improvement=MAX_NO_IMPROVEMENT,
    get_grad=GET_GRAD,
    load_best=LOAD_BEST)

# Test model
reconstruction_opt = { "number": 10, "size": SIZE, "channels": CHANNELS, "z": "mu" }
output_images_opt = { "range": 10, "number": 11, "size": SIZE, "channels": CHANNELS }
conceptual_compression_opt = {
    "number": 10, "size": SIZE, "channels": CHANNELS, "random": True, "separate": True, "z": "mu" }

evaluate.test(
    test_loader=cifar10.get_test_loader(),
    avg_var=True,
    reconstruction=True,
    reconstruction_opt=reconstruction_opt,
    output_images=False,
    output_images_opt=output_images_opt,
    conceptual_compression=True,
    conceptual_compression_opt=conceptual_compression_opt)
