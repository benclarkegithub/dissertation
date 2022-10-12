from Architectures.Basic import VAE, Canvas, Encoder, EncoderEncoderToEncoder, EncoderToLatents, LatentsToLatents, \
    LatentsToLatentsComplicated, LatentsToDecoder, Decoder
from Datasets import MNIST
from Methods.Standard import Standard
from Methods.RNN import RNN
from Experiments import Experiments

BATCH_SIZE = 128
TRAIN_SET_SIZE = 57000
VAL_SET_SIZE = 3000

# Doesn't need a seed!
mnist = MNIST(
        root="../Data" , batch_size=BATCH_SIZE, train_set_size=TRAIN_SET_SIZE, val_set_size=VAL_SET_SIZE)

NUM_LATENTS = 10

MNIST_10 = Standard(VAE=VAE, num_latents=NUM_LATENTS)

BETA = 4

MNIST_10_Beta = Standard(VAE=VAE, num_latents=NUM_LATENTS, beta=BETA)

NUM_LATENTS_GROUP = 1
TYPE = "Single"

ARCHITECTURE_MNIST_10_1_NoR_ETL = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderToLatents": EncoderToLatents,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

MNIST_10_1_NoR_ETL = RNN(
    architecture=ARCHITECTURE_MNIST_10_1_NoR_ETL,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    type=TYPE,
    reconstruction=False,
    encoders=False,
    to_latents="Encoder",
    encoder_to_latents=True,
    backprop=True,
    resample=True)

ARCHITECTURE_MNIST_10_1_R_ETL = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderEncoderToEncoder": EncoderEncoderToEncoder,
    "EncoderToLatents": EncoderToLatents,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

MNIST_10_1_R_ETL = RNN(
    architecture=ARCHITECTURE_MNIST_10_1_R_ETL,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    type=TYPE,
    reconstruction=True,
    encoders=False,
    to_latents="Encoder",
    encoder_to_latents=True,
    backprop=True,
    resample=True)

ARCHITECTURE_MNIST_10_1_R_LTL = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderToLatents": EncoderToLatents,
    "LatentsToLatents": LatentsToLatents,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

MNIST_10_1_R_LTL = RNN(
    architecture=ARCHITECTURE_MNIST_10_1_R_LTL,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    type=TYPE,
    reconstruction=True,
    encoders=False,
    to_latents="Latents",
    encoder_to_latents=True,
    backprop=True,
    resample=True)

ARCHITECTURE_MNIST_10_1_R_LTLC = {
    "Canvas": Canvas,
    "Encoder": Encoder,
    "EncoderToLatents": EncoderToLatents,
    "LatentsToLatents": LatentsToLatentsComplicated,
    "LatentsToDecoder": LatentsToDecoder,
    "Decoder": Decoder
}

MNIST_10_1_R_LTLC = RNN(
    architecture=ARCHITECTURE_MNIST_10_1_R_LTLC,
    num_latents=NUM_LATENTS,
    num_latents_group=NUM_LATENTS_GROUP,
    type=TYPE,
    reconstruction=True,
    encoders=False,
    to_latents="Latents",
    encoder_to_latents=True,
    backprop=True,
    resample=True)

experiments = [
    {
        "name": "Standard VAE",
        "abbreviation": "VAE",
        "identifier": "MNIST_10",
        "trials": 5,
        "method": MNIST_10,
        "graphs": ["ELBO", "R", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Standard B-VAE",
        "abbreviation": "B-VAE",
        "identifier": "MNIST_10_Beta",
        "trials": 5,
        "method": MNIST_10_Beta,
        "graphs": ["ELBO", "R", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "No Reconstruction",
        "abbreviation": "NoR",
        "identifier": "MNIST_10_1_NoR_ETL",
        "trials": 5,
        "method": MNIST_10_1_NoR_ETL,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Encoder To Latents",
        "abbreviation": "R ETL",
        "identifier": "MNIST_10_1_R_ETL",
        "trials": 5,
        "method": MNIST_10_1_R_ETL,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Latents To Latents",
        "abbreviation": "R LTL",
        "identifier": "MNIST_10_1_R_LTL",
        "trials": 5,
        "method": MNIST_10_1_R_LTL,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    },
    {
        "name": "Reconstruction Latents To Latents Complicated",
        "abbreviation": "R LTLC",
        "identifier": "MNIST_10_1_R_LTLC",
        "trials": 5,
        "method": MNIST_10_1_R_LTLC,
        "graphs": ["ELBO", "ELBO_Z1", "R", "R_Z1", "KLD_Z_10_16", "KLD_Z_1", "KLD_Validation"]
    }
]

NUM_GROUPS = 10
MAX_EPOCHS = 100

experiments_2 = Experiments("MNIST", experiments, size=28, channels=1)
experiments_2.load_experiments()
experiments_2.test(mnist.get_val_loader())
experiments_2.summary()
experiments_2.graphs(NUM_GROUPS, MAX_EPOCHS, start_epoch=5)
