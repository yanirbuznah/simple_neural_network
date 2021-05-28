import numpy as np


class ActivationFunction(object):
    def __init__(self, f, d):
        self.f = f
        self.d = d


ActivationFunction.ReLU = ActivationFunction(lambda a: np.maximum(0, a), lambda a: (a > 0).astype(int))
ActivationFunction.Sigmoid = ActivationFunction(lambda a: 1/(1+np.exp(-a)), lambda a: a * (1-a))


def softmax(x, derivative=False):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


ActivationFunction.SoftMax = ActivationFunction(lambda a: softmax(a), lambda a: softmax(a, derivative=True))
