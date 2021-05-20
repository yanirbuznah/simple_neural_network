import sys
from typing import Tuple, List

import numpy.random
import pandas as pd
import numpy as np


class ActivationFunction(object):
    def __init__(self, f, d):
        self.f = f
        self.d = d


ActivationFunction.ReLU = ActivationFunction(lambda a: np.maximum(0, a), lambda a: (a > 0).astype(int))
ActivationFunction.Sigmoid = ActivationFunction(lambda a: 1/(1+np.exp(-a)), lambda a: a * (1-a))


class NeuralLayer(object):
    def __init__(self, size: int, index: int):
        self.size = size
        self.index = index
        self.feeded_values = np.zeros(size)


    def feed(self, values: np.array):
        self.feeded_values += values


    def clear_feeded_values(self):
        self.feeded_values = np.zeros(self.size)

    def __repr__(self):
        return self.feeded_values.__repr__()


class NeuralNetwork(object):
    def __init__(self, input_layer_size: int, hidden_layers_sizes: List[int], output_layer_size: int, activation_function:ActivationFunction, learning_rate=0.001, randrange=0.01):
        self.input_layer = NeuralLayer(input_layer_size, 0)
        self.hidden_layers = [NeuralLayer(size, index + 1) for index, size in enumerate(hidden_layers_sizes)]
        self.output_layer = NeuralLayer(output_layer_size, 1 + len(hidden_layers_sizes))
        self.randrange = randrange

        self.weights = [np.random.uniform(-randrange, randrange, (y.size, x.size)) for x, y in zip(self.layers[1:], self.layers[:-1])]
        self.biases = [np.random.randn(y.size) if self.input_layer.index < y.index < self.output_layer.index else None for y in self.layers]

        self.activation_function = activation_function
        self.lr = learning_rate

    @property
    def layers(self):
        return [self.input_layer] + self.hidden_layers + [self.output_layer]

    def _clear_feeded_values(self):
        for layer in self.layers:
            layer.clear_feeded_values()


    def _feed_forward(self, input_values: np.array):
        self.input_layer.feed(input_values)
        for layer in self.hidden_layers + [self.output_layer]:
            prev_layer_index = layer.index - 1
            bias = self.biases[layer.index] if self.biases[layer.index] is not None else np.zeros(layer.size)
            values = self.activation_function.f(np.dot(self.layers[prev_layer_index].feeded_values, self.weights[prev_layer_index]) + bias)
            layer.feed(values)

    def train_set(self, data_sets: List[Tuple[np.array, np.array]], shuffle=False):
        if shuffle:
            numpy.random.shuffle(data_sets)

        for index, (sample, expected_result) in enumerate(data_sets):

            if index % 10 == 0:
                print('\r', end='')
                print(f"{index}/{len(data_sets)}", end='')

            self._train_sample(sample, expected_result)
        print("\rFinish training")

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

        print(f"Average Certainty: {float(certainty / total)}")
        correction = float(correct / total) * 100.0
        print(f"Correct: {correction}%")
        return correction

    def _train_sample(self, input_values: np.array, correct_output: np.array):
        self._clear_feeded_values()
        self._feed_forward(input_values)
        errors = self._calculate_errors(correct_output)
        self._update_weights(errors)

    def _validate_sample(self, input_values: np.array, correct_output: np.array):
        self._clear_feeded_values()
        self._feed_forward(input_values)
        prediction = np.argmax(self.output_layer.feeded_values)
        correct = np.argmax(correct_output)
        print(prediction, correct, f"Certainty: {self.output_layer.feeded_values[prediction]}")
        return correct == prediction, self.output_layer.feeded_values[prediction]

    def _calculate_errors(self, correct_output: np.array):
        errors = []
        prev_layer_error = correct_output - self.output_layer.feeded_values
        errors.insert(0, prev_layer_error)
        for layer in self.layers[:-1][::-1]:
            prev_layer_error = errors[0]
            weighted_error = np.dot(prev_layer_error, self.weights[layer.index].T) * self.activation_function.d(layer.feeded_values)
            errors.insert(0, weighted_error)

        return errors

    def _update_weights(self, errors: List[np.array]):
        for layer in self.layers[:-1][::-1]:
            self.weights[layer.index] = self.weights[layer.index] + self.lr * np.outer(self.activation_function.f(layer.feeded_values), errors[layer.index + 1])


        #self.weights[-1] += np.dot(output_layer_error, self.activation_function.d(self.output_layer.feeded_values))

        #for b, l in zip(self.biases, self.num_layers):
        #    a = d_sigmoid(np.dot(l, a)+b)
        #return a

    def __str__(self):
        return f"Net[layers={','.join([str(layer.size) for layer in self.layers])}_randrange={self.randrange}]"


def csv_to_train_data(path, count=-1) -> Tuple[np.array, np.array]:
    df = pd.read_csv(path, header=None)
    output = df.loc[:, 0]
    data = df.drop(columns=0).to_numpy()
    results_indexes = output.to_numpy()
    results = np.zeros((len(results_indexes), 10))
    for i in range(len(results_indexes)):
        if not str(results_indexes[i]).isdigit():
            # This is probably a test set. Ignore expected results column
            results = []
            break

        results[i][results_indexes[i] - 1] = 1

    if count == -1:
        return data, results
    else:
        return data[:count], results[:count]

def main():
    if len(sys.argv) != 3:
        print("Not enough arguments")
        return

    train_csv = sys.argv[1]
    validate_csv = sys.argv[2]

    print(f"Reading training data from: {train_csv}")
    train_data, train_correct = csv_to_train_data(train_csv)
    validate_data, validate_correct = csv_to_train_data(validate_csv)
    #net = NeuralNetwork(len(train_data[0]), [int(np.floor((2/3)*train_data.shape[1])), int(np.floor((4/9)*train_data.shape[1]))], 10, ActivationFunction.ReLU, randrange=0.02, learning_rate=0.003)

    adaptive_learning_rate_setting = {
         0: 0.003,
        15: 0.001,
        20: 0.0005
    }

    net = NeuralNetwork(len(train_data[0]), [1000, 1000], 10, ActivationFunction.ReLU, randrange=0.04, learning_rate=adaptive_learning_rate_setting[0])

    max_correct = 0
    for i in range(30):
        #exp decay : lrate = initial_lrate * exp(-k*t)
        #net.lr = 0.003* np.exp(-0.01*i)

        if i in adaptive_learning_rate_setting:
            net.lr = adaptive_learning_rate_setting[i]

        print(f"Epoch {i}")
        print(f"Current LR: {net.lr}")
        net.train_set(list(zip(train_data, train_correct)), shuffle=True)
        correct = net.validate_set(list(zip(validate_data, validate_correct)))
        if correct > max_correct:
            max_correct = correct
            max_epoch = i
            max_weights = net.weights

    print("Done!")
    print("Saving weights...")
    for i in range(len(net.weights)):
        pd.DataFrame(net.weights[i]).to_csv(f"{correct}% adaptive_lr={adaptive_learning_rate_setting}_net={net} weights_{i}.csv")
    for i in range(len(max_weights)):
        pd.DataFrame(net.weights[i]).to_csv(f"max_epoch={max_epoch}_{max_correct}% % adaptive_lr={adaptive_learning_rate_setting}_net={net} weights_{i}.csv")


if __name__ == '__main__':
    main()
