import torch
from torch import nn

class PropellerNoiseGen(nn.Module):
    def __init__(self, n_harmonics=50):
