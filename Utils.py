from os import mkdir
from os.path import isdir


def make_dir(path):
    path_split = path.split("/")
    if not isdir(path_split[0]):
        mkdir(path_split[0])
