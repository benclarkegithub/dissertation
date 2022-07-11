from os import mkdir
from os.path import isdir
import random
import numpy as np
import torch


def make_dir(path):
    path_split = path.split("/")
    if not isdir(path_split[0]):
        mkdir(path_split[0])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
