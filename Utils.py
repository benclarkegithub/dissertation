from os import mkdir
from os.path import isdir
import random
import numpy as np
import torch


def make_dir(path):
    if not isdir(path):
        mkdir(path)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
