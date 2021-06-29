import csv
import pprint
import shutil
import signal
import smtplib
import ssl
import sys
import uuid
from email.mime.text import MIMEText
from pathlib import Path
from typing import Tuple, List
from glob import glob

import numpy

import config
from config import *

import numpy as np

random.seed(SEED)
numpy.random.seed(SEED)

if USE_GPU:
    import cupy as np
    np.random.seed(SEED)

SHOULD_STOP = False



class NeuralLayer(object):
    def __init__(self, size: int, index: int,with_bias):
        self.index = index
        self.bias = with_bias
        self.size = size
        self.mask = np.random.binomial(1, 1-DROP_OUT, size=self.size) / (1-DROP_OUT)
        if with_bias:
            self.size += 1
        self.clear_feeded_values()

    def feed(self, values: np.array):
        self.feeded_values += values
        if DROP_OUT:
            self.mask = np.random.binomial(1, 1-DROP_OUT, size=self.size) / (1-DROP_OUT)
            self.feeded_values*= self.mask

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
