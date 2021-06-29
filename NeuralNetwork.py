
import sys

from typing import Tuple, List


import numpy


from config import *


import numpy as np


random.seed(SEED)
numpy.random.seed(SEED)

if USE_GPU:
    import cupy as np
    np.random.seed(SEED)

SHOULD_STOP = False


from NeuralLayer import NeuralLayer

class NeuralNetwork(object):
    def __init__(self, input_layer_size: int, hidden_layers_sizes: List[int], output_layer_size: int, activation_function:ActivationFunction, learning_rate=0.001, randrange=0.01):
        self.input_layer = NeuralLayer(input_layer_size, 0, with_bias=True)
        self.hidden_layers = [NeuralLayer(size, index + 1, with_bias=True) for index, size in enumerate(hidden_layers_sizes)]
        self.output_layer = NeuralLayer(output_layer_size, 1 + len(hidden_layers_sizes), with_bias=False)
        self.randrange = randrange

        self.weights = [np.random.uniform(-randrange, randrange, (y.size, x.size)) for x, y in zip(self.layers[1:], self.layers[:-1])]

        self.activation_function = activation_function
        self.lr = learning_rate

    @property
    def layers(self):
        return [self.input_layer] + self.hidden_layers + [self.output_layer]

    def _clear_feeded_values(self):
        for layer in self.layers:
            layer.clear_feeded_values()

    def _feed_forward(self, input_values: np.array):
        #input_values = np.append(input_values,0)
        self.input_layer.feed(input_values)
        for layer in self.hidden_layers + [self.output_layer]:
            prev_layer_index = layer.index - 1
            if SOFTMAX and layer == self.output_layer:
                f = softmax
            else:
                f = self.activation_function.f

            values = f(np.dot(self.layers[prev_layer_index].feeded_values, self.weights[prev_layer_index]))
            layer.feed(values)

    @staticmethod
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def train_set(self, data_sets: List[Tuple[np.array, np.array]], shuffle=False, mini_batch_size=1):
        if shuffle:
            numpy.random.shuffle(data_sets)

        batches = self._chunks(data_sets, mini_batch_size)

        count = 0
        for batch in batches:
            count += len(batch)
            samples_list, expected_results_list = zip(*batch)
            if count % 10 == 0:
                print('\r', end='')
                print(f"{count}/{len(data_sets)}", end='')
                sys.stdout.flush()

            self._train_mini_batch(samples_list, expected_results_list)

        print("\rFinished training")
        sys.stdout.flush()

    def validate_set(self, data_sets: List[Tuple[np.array, np.array]]):
        correct = 0
        total = 0
        certainty = 0
        for index, (sample, expected_result) in enumerate(data_sets):
            result, i = self._validate_sample(sample, expected_result)
            certainty += i
            if result:
                correct += 1
            total += 1

        average_certainty = float(certainty / total)
        print(f"Average Certainty: {average_certainty}")
        correction = float(correct / total) * 100.0
        print(f"Correct: {correction}%")
        return correction, average_certainty

    def set_weights(self, weights):
        self.weights = weights

    def _train_mini_batch(self, input_values_list: List[np.array], correct_output_list: List[np.array]):
        errors = []
        for input_values, correct_output in zip(input_values_list, correct_output_list):
            self._clear_feeded_values()
            self._feed_forward(input_values)
            current_errors = self._calculate_errors(correct_output)
            if not errors:
                errors = current_errors
            else:
                errors = [sum(l) for l in zip(errors, current_errors)]

        self._update_weights(errors)

    def classify_sample(self, input_values: np.array):
        self._clear_feeded_values()
        self._feed_forward(input_values)
        prediction = np.argmax(self.output_layer.feeded_values)
        return prediction

    def _validate_sample(self, input_values: np.array, correct_output: np.array):
        prediction = self.classify_sample(input_values)
        correct = np.argmax(correct_output)
        #print(prediction, correct, f"Certainty: {self.output_layer.feeded_values[prediction]}")
        return correct == prediction, self.output_layer.feeded_values[prediction]

    def _calculate_errors(self, correct_output: np.array):
        errors = []
        prev_layer_error = correct_output - self.output_layer.feeded_values
        errors.insert(0, prev_layer_error)
        for layer in self.layers[:-1][::-1]:
            if SOFTMAX and layer == self.output_layer:
                d = softmax_d
            else:
                d = self.activation_function.d

            prev_layer_error = errors[0]
            weighted_error = np.dot(prev_layer_error, self.weights[layer.index].T) * d(layer.feeded_values)
            errors.insert(0, weighted_error)

        return errors

    def _update_weights(self, errors: List[np.array]):
        for layer in self.layers[:-1][::-1]:
            if SOFTMAX and layer == self.output_layer:
                f = softmax
            else:
                f = self.activation_function.f
            self.weights[layer.index] = self.weights[layer.index] + self.lr * np.outer(f(layer.feeded_values), errors[layer.index + 1])

    def __str__(self):
        return f"Net[layers={','.join([str(layer.size) for layer in self.layers])}_randrange={self.randrange}]"


def softmax(x):
    return  np.exp(x)/sum(np.exp(x))


def softmax_d(x):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

