from Architectures.Basic import Canvas, Encoder, EncoderEncoderToEncoder, EncoderToLatents, EncoderEncoderToLatents,\
    EncoderEncoderToLatents2, EncoderEncoderToLatents3, EncoderEncoderToLatents4, LatentsToLatents, \
    LatentsToLatentsComplicated, LatentsToDecoder, Decoder
from Datasets import CIFAR10
from Methods.RNN import RNN
from Experiments import Experiments

BATCH_SIZE = 128
TRAIN_SET_SIZE = 47500
VAL_SET_SIZE = 2500

# Doesn't need a seed!
cifar10 = CIFAR10(
        root="../../Data", batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE)

NUM_LATENTS = 64
NUM_LATENTS_GROUP = 4
TYPE = "Single"
SIZE = 32
CHANNELS = 3
LOG_PROB_FN = "N"
STD = 0.05
HIDDEN_SIZE = 256
CLIP = 5e5

ARCHITECTURE_64_4_R_EETL = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderEncoderToLatents": EncoderEncoderToLatents,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

CIFAR10_64_4_R_EETL = RNN(
    architecture=ARCHITECTURE_64_4_R_EETL,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    reconstruction=True,
    encoders=False,
    to_latents="EncoderEncoder",
    encoder_to_latents=True,
    backprop=True,
    resample=True,
    type=TYPE,
    size=SIZE,
    channels=CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD,
    hidden_size=HIDDEN_SIZE,
    clip=CLIP)

ARCHITECTURE_64_4_R_EETL2 = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderEncoderToLatents": EncoderEncoderToLatents2,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

CIFAR10_64_4_R_EETL2 = RNN(
    architecture=ARCHITECTURE_64_4_R_EETL2,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    reconstruction=True,
    encoders=False,
    to_latents="EncoderEncoder",
    encoder_to_latents=True,
    backprop=True,
    resample=True,
    type=TYPE,
    size=SIZE,
    channels=CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD,
    hidden_size=HIDDEN_SIZE,
    clip=CLIP)

ARCHITECTURE_64_4_R_EETL3 = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderEncoderToLatents": EncoderEncoderToLatents3,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

CIFAR10_64_4_R_EETL3 = RNN(
    architecture=ARCHITECTURE_64_4_R_EETL3,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    reconstruction=True,
    encoders=False,
    to_latents="EncoderEncoder",
    encoder_to_latents=True,
    backprop=True,
    resample=True,
    type=TYPE,
    size=SIZE,
    channels=CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD,
    hidden_size=HIDDEN_SIZE,
    clip=CLIP)

ARCHITECTURE_64_4_R_EETL4 = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderEncoderToLatents": EncoderEncoderToLatents4,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

CIFAR10_64_4_R_EETL4 = RNN(
    architecture=ARCHITECTURE_64_4_R_EETL4,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    reconstruction=True,
    encoders=False,
    to_latents="EncoderEncoder",
    encoder_to_latents=True,
    backprop=True,
    resample=True,
    type=TYPE,
    size=SIZE,
    channels=CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD,
    hidden_size=HIDDEN_SIZE,
    clip=CLIP)

ARCHITECTURE_CIFAR10_64_4_R_ETL = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderEncoderToEncoder": EncoderEncoderToEncoder,
    "EncoderToLatents": EncoderToLatents,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

CIFAR10_64_4_R_ETL = RNN(
    architecture=ARCHITECTURE_CIFAR10_64_4_R_ETL,
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
    log_prob_fn=LOG_PROB_FN,
    std=STD,
    hidden_size=HIDDEN_SIZE,
    clip=CLIP)

ARCHITECTURE_CIFAR10_64_4_R_LTL = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderToLatents": EncoderToLatents,
    "LatentsToLatents": LatentsToLatents,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

CIFAR10_64_4_R_LTL = RNN(
    architecture=ARCHITECTURE_CIFAR10_64_4_R_LTL,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    reconstruction=True,
    encoders=False,
    to_latents="Latents",
    encoder_to_latents=True,
    backprop=True,
    resample=True,
    type=TYPE,
    size=SIZE,
    channels=CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD,
    hidden_size=HIDDEN_SIZE,
    clip=CLIP)

ARCHITECTURE_CIFAR10_64_4_R_LTLC = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderToLatents": EncoderToLatents,
    "LatentsToLatents": LatentsToLatentsComplicated,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

CIFAR10_64_4_R_LTLC = RNN(
    architecture=ARCHITECTURE_CIFAR10_64_4_R_LTLC,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    reconstruction=True,
    encoders=False,
    to_latents="Latents",
    encoder_to_latents=True,
    backprop=True,
    resample=True,
    type=TYPE,
    size=SIZE,
    channels=CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD,
    hidden_size=HIDDEN_SIZE,
    clip=CLIP)

experiments = [
    {
        "name": "Reconstruction Encoder Encoder To Latents",
        "abbreviation": "R EETL",
        "identifier": "CIFAR10_64_4_R_EETL",
        "trials": 1,
        "method": CIFAR10_64_4_R_EETL,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Encoder Encoder To Latents 2",
        "abbreviation": "R EETL 2",
        "identifier": "CIFAR10_64_4_R_EETL2",
        "trials": 1,
        "method": CIFAR10_64_4_R_EETL2,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },{
        "name": "Reconstruction Encoder Encoder To Latents 3",
        "abbreviation": "R EETL 3",
        "identifier": "CIFAR10_64_4_R_EETL3",
        "trials": 1,
        "method": CIFAR10_64_4_R_EETL3,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },{
        "name": "Reconstruction Encoder Encoder To Latents 4",
        "abbreviation": "R EETL 4",
        "identifier": "CIFAR10_64_4_R_EETL4",
        "trials": 1,
        "method": CIFAR10_64_4_R_EETL4,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Encoder To Latents",
        "abbreviation": "R ETL",
        "identifier": "CIFAR10_64_4_R_ETL",
        "trials": 1,
        "method": CIFAR10_64_4_R_ETL,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Latents To Latents",
        "abbreviation": "R LTL",
        "identifier": "CIFAR10_64_4_R_LTL",
        "trials": 1,
        "method": CIFAR10_64_4_R_LTL,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Latents To Latents Complicated",
        "abbreviation": "R LTLC",
        "identifier": "CIFAR10_64_4_R_LTLC",
        "trials": 1,
        "method": CIFAR10_64_4_R_LTLC,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    }
]

NUM_GROUPS = 16
MAX_EPOCHS = 500
START_EPOCH = 100

experiments_2 = Experiments("CIFAR10", experiments, size=SIZE, channels=CHANNELS)
experiments_2.load_experiments()
experiments_2.test(cifar10.get_val_loader())
experiments_2.summary()
experiments_2.graphs(NUM_GROUPS, MAX_EPOCHS, start_epoch=START_EPOCH)
