from enum import Enum

import numpy as np


class ActivationFunction(object):
    def __init__(self, f, d):
        self.f = f
        self.d = d


ActivationFunction.ReLU = ActivationFunction(lambda a: np.maximum(0, a), lambda a: (a > 0).astype(int))
ActivationFunction.Sigmoid = ActivationFunction(lambda a: 1/(1+np.exp(-a)), lambda a: a * (1-a))


class AdaptiveLearningRateMode(Enum):
    PREDEFINED_DICT = 1
    FORMULA = 2