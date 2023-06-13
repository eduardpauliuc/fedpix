import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "/Users/eduardpauliuc/PycharmProjects/federated"
VAL_DIR_ALL = ROOT_DIR + "/maps_all/val"
