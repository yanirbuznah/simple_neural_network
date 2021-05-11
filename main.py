import pandas as pd
import numpy as np

def sigmoid(a):
    return 1/(1+np.exp(-a))

def d_sigmoid(a):
    return a * (1-a)


class NeuralNetwork:
    def __init__(self,layers_sizes):
        self.num_layers = len(layers_sizes)
        self.sizes = layers_sizes
        self.biases = [np.random.randn(y, 1) for y in layers_sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers_sizes[:-1], layers_sizes[1:])]

    def feed_forward(self):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def train(self):
        pass

    def back_propagation(self):
        for b, l in zip(self.biases, self.num_layers):
            a = d_sigmoid(np.dot(l, a)+b)
        return a
