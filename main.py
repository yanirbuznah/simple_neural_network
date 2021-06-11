import csv
import shutil
import smtplib
import ssl
import sys
import uuid
from email.mime.text import MIMEText
from pathlib import Path
from typing import Tuple, List
from glob import glob

import pandas as pd
import numpy as np
import pickle

from config import *

np.random.seed(SEED)

class TrainingStateData(object):
    def __init__(self, correct_percent, epoch, weights, biases):
        self.accuracy = correct_percent
        self.epoch = epoch
        self.weights = self.deep_copy_list_of_np_arrays(weights)
        self.biases = self.deep_copy_list_of_np_arrays(biases)

    def __str__(self):
        return f"Epoch: {self.epoch}\nAccuracy: {self.accuracy}%"

    @staticmethod
    def deep_copy_list_of_np_arrays(l: List[np.array]):
        res = []
        for arr in l:
            if arr is None:
                res.append(None)
            else:
                res.append(arr.copy())

        return res


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
            np.random.shuffle(data_sets)

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

        average_certainty = float(certainty / total)
        print(f"Average Certainty: {average_certainty}")
        correction = float(correct / total) * 100.0
        print(f"Correct: {correction}%")
        return correction, average_certainty

    def set_weights_and_biases(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def _train_sample(self, input_values: np.array, correct_output: np.array):
        self._clear_feeded_values()
        self._feed_forward(input_values)
        errors = self._calculate_errors(correct_output)
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
            prev_layer_error = errors[0]
            weighted_error = np.dot(prev_layer_error, self.weights[layer.index].T) * self.activation_function.d(layer.feeded_values)
            errors.insert(0, weighted_error)

        return errors

    def _update_weights(self, errors: List[np.array]):
        for layer in self.layers[:-1][::-1]:
            self.weights[layer.index] = self.weights[layer.index] + self.lr * np.outer(self.activation_function.f(layer.feeded_values), errors[layer.index + 1])

    def __str__(self):
        return f"Net[layers={','.join([str(layer.size) for layer in self.layers])}_randrange={self.randrange}]"


def result_classifications_to_np_layers(results_classifications: List[int]) -> np.array:
    results = np.zeros((len(results_classifications), 10))
    for i in range(len(results_classifications)):
        if not str(results_classifications[i]).isdigit():
            # This is probably a test set. Ignore expected results column
            results = []
            break

        results[i][results_classifications[i] - 1] = 1

    return results


def csv_to_data(path, count=-1) -> Tuple[np.array, np.array]:
    df = pd.read_csv(path, header=None)
    output = df.loc[:, 0]
    data = df.drop(columns=0).to_numpy()
    results_indexes = output.to_numpy()
    results = result_classifications_to_np_layers(results_indexes)

    if count == -1:
        return data, results
    else:
        return data[:count], results[:count]


def pickle_to_data(path, count=-1) -> Tuple[np.array, np.array]:
    results_indexes = []
    data = []
    pickle_files = sorted(glob(f"{path}/data_batch_*"))
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            results_indexes += data_dict[b'labels']
            data += [data_dict[b'data']/255]

    result_classifications = [i + 1 for i in results_indexes]
    results = result_classifications_to_np_layers(result_classifications)

    data = np.concatenate(data)

    if count == -1:
        return data, results
    else:
        return data[:count], results[:count]


def save_state(path: Path, prefix, state: TrainingStateData):
    if SAVED_MODEL_PICKLE_MODE:
        with open(path / f"{prefix}_epoch={state.epoch}_{state.accuracy}%.model", 'wb') as f:
            pickle.dump(state, f)
    else:
        for i, weight in enumerate(state.weights):
            pd.DataFrame(weight).to_csv(path / f"{prefix}_epoch={state.epoch}_{state.accuracy}%_weights_{i}.csv")
        for i, bias in enumerate(state.biases):
            if bias is not None:
                pd.DataFrame(bias).to_csv(path / f"{prefix}_epoch={state.epoch}_{state.accuracy}%_biases_{i}.csv")


def load_state(path: Path, net: NeuralNetwork):
    if SAVED_MODEL_PICKLE_MODE:
        pickle_model_file = glob(f"{path}/best_state*.model")
        if len(pickle_model_file) != 1:
            raise Exception("Expected only one pickle model file to be found")
        pickle_model_file = pickle_model_file[0]
        with open(pickle_model_file, 'rb') as f:
            state: TrainingStateData = pickle.load(f)
            print(f"Loaded state: {state}")
            net.set_weights_and_biases(state.weights, state.biases)
    else:
        weight_files = sorted(glob(f"{path}/best_state*weights*.csv"))
        biases_files = sorted(glob(f"{path}/best_state*biases*.csv"))
        for i, weights_file in enumerate(weight_files):
            df = pd.read_csv(weights_file)
            data = df.iloc[:, 1:].to_numpy()
            net.weights[i] = data

        for i, bias in enumerate(net.biases):
            if bias is None:
                continue

            df = pd.read_csv(biases_files[i - 1])
            data = df.iloc[:, 1:].to_numpy().flatten()
            net.biases[i] = data


def get_subset(train_data, train_correct, count):
    random_rows_idx = np.random.choice(train_data.shape[0], size=count, replace=False)
    return train_data[np.ix_(random_rows_idx)], train_correct[np.ix_(random_rows_idx)]


def apply_noise(train_data, prob):
    for i in train_data:
        indices = np.random.choice(np.arange(i.size), replace=False,
                                   size=int(i.size * prob))
        i[indices] = 0
    return train_data


def send_mail(mail, message):
    sender_email = "algobiotester@gmail.com"

    msg = MIMEText(message, _charset="UTF-8")

    port = 465  # For SSL
    password = "algobio1!"

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, mail, msg.as_string())


def separate_data(data, correct):
    #data = data.random.suffle(data)
    #correct = correct.random.suffle(correct)
    np.split(data,1000)
    np.split(correct,1000)
    return data,correct

def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        return

    train_csv = sys.argv[1]
    validate_csv = sys.argv[2]
    test_csv = sys.argv[3] if len(sys.argv) >= 4 else None

    net = NeuralNetwork(INPUT_LAYER_SIZE, HIDDEN_LAYERS_SIZES, OUTPUT_LAYER_SIZE, ACTIVATION_FUNCTION, randrange=RANDRANGE, learning_rate=LEARNING_RATE)
    csv_results = [["epoch", "LR", "train_accuracy", "train_certainty", "validate_accuracy", "validate_certainty"]]

    validate_data, validate_correct = csv_to_data(validate_csv)
#    if SEPARATE_VALIDATE:
#       validate_data_array, validate_correct_array = separate_data(validate_data,validate_correct)


    output_path = Path(str(uuid.uuid4()) if not TRAINED_NET_DIR else TRAINED_NET_DIR)

    if TRAINED_NET_DIR and Path(TRAINED_NET_DIR).exists():
        print(f"Taking best values from {TRAINED_NET_DIR}. Pickle mode = {SAVED_MODEL_PICKLE_MODE}")
        load_state(TRAINED_NET_DIR, net)

    if SHOULD_TRAIN:
        print(f"Reading training data from: {train_csv}")

        # TO TAKE FROM CSV
        train_data, train_correct = csv_to_data(train_csv)
        output_path.mkdir(exist_ok=True)
        shutil.copy2("config.py", output_path)
        # TO TAKE FROM PICKLE TODO: REMOVE THIS BEFORE SUBMITTING
        #train_data, train_correct = pickle_to_data("cifar-10-batches-py")


        print("Starting training...")

        current_validate_accuracy = 0
        overall_best_state = TrainingStateData(0, 0, net.weights, net.biases)

        for epoch in range(EPOCH_COUNT):
            if ADAPTIVE_LEARNING_RATE_MODE == AdaptiveLearningRateMode.FORMULA:
                net.lr = ADAPTIVE_LEARNING_RATE_FORMULA(epoch)
            elif ADAPTIVE_LEARNING_RATE_MODE == AdaptiveLearningRateMode.PREDEFINED_DICT:
                net.lr = ADAPTIVE_LEARNING_RATE_DICT.get(epoch, net.lr)
            else:
                raise NotImplementedError("Unknown adaptive learning rate mode")

            print(f"Epoch {epoch}")
            print(f"Current LR: {net.lr}")

            if TAKE_BEST_FROM_VALIDATE:
                print("Take best from epoch:", overall_best_state.epoch, "with accuracy", overall_best_state.accuracy,"%")
                net.weights = overall_best_state.weights
                net.biases = overall_best_state.biases

            if INPUT_LAYER_NOISE_PROB > 0:
                print(f"Applying noise of {INPUT_LAYER_NOISE_PROB * 100}% on all inputs")
                train_data = apply_noise(train_data, INPUT_LAYER_NOISE_PROB)

            if SUBSET_SIZE > 0:
                subset_train, subset_correct = get_subset(train_data, train_correct, SUBSET_SIZE)
                net.train_set(list(zip(subset_train, subset_correct)), shuffle=True)
            else:
                net.train_set(list(zip(train_data, train_correct)), shuffle=True)

            #validate_data, validate_correct = csv_to_data(validate_csv)


            print("======= Train Accuracy =======")
            current_train_accuracy, train_certainty = net.validate_set(list(zip(train_data, train_correct)))

            print("======= Validate Accuracy =======")
            current_validate_accuracy, validate_certainty = net.validate_set(list(zip(validate_data, validate_correct)))

            csv_results.append([epoch, net.lr, current_train_accuracy, train_certainty, current_validate_accuracy, validate_certainty])

            if current_validate_accuracy > overall_best_state.accuracy:
                overall_best_state = TrainingStateData(current_validate_accuracy, epoch, net.weights, net.biases)

        print("Done!")
        print("Saving results, weights and biases...")

        with open(output_path / "results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_results)

        save_state(output_path, "latest_state", TrainingStateData(current_validate_accuracy, epoch, net.weights, net.biases))
        save_state(output_path, "best_state", overall_best_state)

        #mail_content = f"Finished!\nbest state:\n {overall_best_state}\n CONFIG:\n{open('config.py', 'r').read()}"
        #send_mail("yanirbuznah@gmail.com", mail_content)
        #send_mail("ron.evenm@gmail.com", mail_content)

    if test_csv:
        print("Test csv provided. Classifying...")
        train_data, _ = csv_to_data(test_csv)
        prediction_list = []
        for i, data in enumerate(train_data):
            classification = net.classify_sample(data) + 1
            prediction_list.append(classification)

        print("Saving predicted test.csv")
        df = pd.read_csv(test_csv, header=None)
        df.drop(columns=0, inplace=True)
        df.insert(0, "", prediction_list)
        df.to_csv(output_path/"test_filled.csv")

        print(prediction_list)
        print(output_path)
        print("TODO: REMOVE ME")
        print("Testing results...")
        import result_compare
        result_compare.check_results(prediction_list)


if __name__ == '__main__':
    main()
