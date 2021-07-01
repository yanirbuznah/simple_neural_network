from config import *

import numpy as np

if USE_GPU:
    import cupy as np


class NeuralLayer(object):
    def __init__(self, size: int, index: int,with_bias, dropout):
        self.index = index
        self.bias = with_bias
        self.size = size
        self.dropout = dropout
        self.mask = np.random.binomial(1, 1-dropout, size=self.size) / (1-dropout)
        if with_bias:
            self.size += 1
        self.clear_feeded_values()

    def feed(self, values: np.array):
        self.feeded_values += values
        if self.dropout > 0:
            self.mask = np.random.binomial(1, 1-self.dropout, size=self.size) / (1-self.dropout)
            self.feeded_values *= self.mask

        # make sure that the bias still shut -1
        if self.bias:
            self.feeded_values[-1] = -1

    def clear_feeded_values(self):
        self.feeded_values = np.zeros(self.size)
        # update the bias neuron to -1
        if self.bias:
            self.feeded_values[-1] = -1

    def __repr__(self):
        return self.feeded_values.__repr__()
