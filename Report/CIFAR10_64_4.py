from Architectures.Basic import VAE, Canvas, Encoder, EncoderEncoderToEncoder, EncoderToLatents, LatentsToLatents, \
    LatentsToLatentsComplicated, LatentsToDecoder, Decoder
from Datasets import CIFAR10
from Methods.Standard import Standard
from Methods.RNN import RNN
from Experiments import Experiments

BATCH_SIZE = 128
TRAIN_SET_SIZE = 47500
VAL_SET_SIZE = 2500

# Doesn't need a seed!
cifar10 = CIFAR10(
        root="../Data" , batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE)

NUM_LATENTS = 64
SIZE = 32
CHANNELS = 3
LOG_PROB_FN = "N"
STD = 0.05
HIDDEN_SIZE = 256

CIFAR10_64_4 = Standard(
    VAE=VAE,
    num_latents=NUM_LATENTS,
    size=SIZE,
    channels=CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    std=STD,
    hidden_size=HIDDEN_SIZE)

BETA = 4

CIFAR10_64_4_Beta = Standard(
    VAE=VAE,
    num_latents=NUM_LATENTS,
    size=SIZE,
    channels=CHANNELS,
    log_prob_fn=LOG_PROB_FN,
    beta=BETA,
    std=STD,
    hidden_size=HIDDEN_SIZE)

NUM_LATENTS_GROUP = 4
TYPE = "Single"
CLIP = 5e5

ARCHITECTURE_CIFAR10_64_4_NoR_ETL = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderToLatents": EncoderToLatents,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

CIFAR10_64_4_NoR_ETL = RNN(
    architecture=ARCHITECTURE_CIFAR10_64_4_NoR_ETL,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    reconstruction=False,
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
        "name": "Standard VAE",
        "abbreviation": "VAE",
        "identifier": "CIFAR10_64",
        "trials": 2,
        "method": CIFAR10_64_4,
        "graphs": ["ELBO", "R", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Beta-VAE",
        "abbreviation": "B-VAE",
        "identifier": "CIFAR10_64_4_Beta",
        "trials": 2,
        "method": CIFAR10_64_4_Beta,
        "graphs": ["ELBO", "R", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "No Reconstruction",
        "abbreviation": "NoR",
        "identifier": "CIFAR10_64_4_NoR_ETL",
        "trials": 2,
        "method": CIFAR10_64_4_NoR_ETL,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Encoder To Latents",
        "abbreviation": "R ETL",
        "identifier": "CIFAR10_64_4_R_ETL",
        "trials": 2,
        "method": CIFAR10_64_4_R_ETL,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Latents To Latents",
        "abbreviation": "R LTL",
        "identifier": "CIFAR10_64_4_R_LTL",
        "trials": 2,
        "method": CIFAR10_64_4_R_LTL,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Latents To Latents Complicated",
        "abbreviation": "R LTLC",
        "identifier": "CIFAR10_64_4_R_LTLC",
        "trials": 2,
        "method": CIFAR10_64_4_R_LTLC,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    }
]

NUM_GROUPS = 16
MAX_EPOCHS = 200

experiments_2 = Experiments("CIFAR10", experiments, size=SIZE, channels=CHANNELS)
experiments_2.load_experiments()
experiments_2.test(cifar10.get_val_loader())
experiments_2.summary()
experiments_2.graphs(NUM_GROUPS, MAX_EPOCHS, start_epoch=5)
