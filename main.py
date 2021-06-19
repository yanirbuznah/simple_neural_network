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

import pandas as pd
import numpy as np
import pickle

import config
from config import *

np.random.seed(SEED)

SHOULD_STOP = False


class EpochStateData(object):
    def __init__(self, current_validate_accuracy,current_train_accuracy, epoch, weights):
        self.validate_accuracy = current_validate_accuracy
        self.train_accuracy = current_train_accuracy
        self.epoch = epoch
        self.weights = self.deep_copy_list_of_np_arrays(weights)

    def __str__(self):
        return f"Epoch {self.epoch}\nTrain accuracy: {self.train_accuracy}% and Validate accuracy: {self.validate_accuracy}%"

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
    def __init__(self, size: int, index: int,with_bias):
        self.index = index
        self.bias = with_bias
        self.size = size
        if with_bias:
            self.size += 1
        self.feeded_values = self.clear_feeded_values()

    def feed(self, values: np.array):
        self.feeded_values += values
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

            values = self.activation_function.f(np.dot(self.layers[prev_layer_index].feeded_values, self.weights[prev_layer_index]))
            layer.feed(values)

    @staticmethod
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def train_set(self, data_sets: List[Tuple[np.array, np.array]], shuffle=False, mini_batch_size=1):
        if shuffle:
            np.random.shuffle(data_sets)

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
    df[df.shape[1]]=0
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


def save_state(path: Path, prefix, state: EpochStateData):
    with open(path / f"{prefix}epoch={state.epoch}_train{state.train_accuracy}%_validate{state.validate_accuracy}% .model", 'wb') as f:
        pickle.dump(state, f)


def load_state(path: Path, net: NeuralNetwork):
    pickle_model_file = glob(f"{path}/best_state*.model")
    if len(pickle_model_file) != 1:
        raise Exception("Expected only one pickle model file to be found")
    pickle_model_file = pickle_model_file[0]
    with open(pickle_model_file, 'rb') as f:
        state: EpochStateData = pickle.load(f)
        print(f"Loaded state: {state}")
        net.set_weights(state.weights)


def get_subset(train_data, train_correct, count):
    random_rows_idx = np.random.choice(train_data.shape[0], size=count, replace=False)
    return train_data[np.ix_(random_rows_idx)], train_correct[np.ix_(random_rows_idx)]


def apply_noise(train_data, prob):
    after_noise_data = EpochStateData.deep_copy_list_of_np_arrays(train_data)
    for i in after_noise_data:
        indices = np.random.choice(np.arange(i.size), replace=False,
                                   size=int(i.size * prob))
        i[indices] = 0
    return after_noise_data


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


def save_predictions(path, prediction_list):
    with open(path, 'w') as f:
        f.writelines([str(p) for p in prediction_list])


def interrupt_handler(sig, frame):
    answer = input("\nAre you sure you want to stop? [y/N]")
    if answer == "y":
        global SHOULD_STOP
        SHOULD_STOP = True
        print("Will stop at the end of the current epoch")


BEST_TEST_RESULT = 0
# TODO: REMOVE BEFORE SUBMITTING
def run_tests(test_data, net, epoch):
    global BEST_TEST_RESULT
    print(f"RUNNING EPOCH {epoch} MODEL ON TEST SET")
    prediction_list = []
    for i, data in enumerate(test_data):
        classification = net.classify_sample(data) + 1
        prediction_list.append(classification)

    print(prediction_list)
    print("TODO: REMOVE ME")
    import result_compare
    result = result_compare.check_results(prediction_list)
    if result > BEST_TEST_RESULT:
        BEST_TEST_RESULT = result
        print(f"NEW BEST TEST ON EPOCH {epoch} WITH RESULT {result}%")


def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        return

    train_csv = sys.argv[1]
    validate_csv = sys.argv[2]
    test_csv = sys.argv[3] if len(sys.argv) >= 4 else None
    current_train_accuracy=0
    epoch = 0

    print(" ======== Config ==========")
    pprint.pprint(list([(k, v) for (k, v) in config.__dict__.items() if k.isupper()]))
    print(" ==========================")

    net = NeuralNetwork(INPUT_LAYER_SIZE, HIDDEN_LAYERS_SIZES, OUTPUT_LAYER_SIZE, ACTIVATION_FUNCTION, randrange=RANDRANGE, learning_rate=LEARNING_RATE)
    csv_results = [["epoch", "LR", "train_accuracy", "train_certainty", "validate_accuracy", "validate_certainty"]]

#    if SEPARATE_VALIDATE:
#       validate_data_array, validate_correct_array = separate_data(validate_data,validate_correct)

    output_path = Path(str(uuid.uuid4()) if not TRAINED_NET_DIR else TRAINED_NET_DIR)

    if not TRAINED_NET_DIR:
        print(f"Will write output to {output_path}")

    if TRAINED_NET_DIR and Path(TRAINED_NET_DIR).exists():
        print(f"Taking best values from {TRAINED_NET_DIR}. Pickle mode = {SAVED_MODEL_PICKLE_MODE}")
        load_state(TRAINED_NET_DIR, net)

    if test_csv:
        print("Test csv provided")
        test_data, _ = csv_to_data(test_csv)

    if SHOULD_TRAIN:
        output_path.mkdir(exist_ok=True)
        shutil.copy2("config.py", output_path)

        validate_data, validate_correct = csv_to_data(validate_csv)

        signal.signal(signal.SIGINT, interrupt_handler)

        print(f"Reading training data from: {train_csv}")

        # TO TAKE FROM CSV
        train_data, train_correct = csv_to_data(train_csv)
        # TO TAKE FROM PICKLE TODO: REMOVE THIS BEFORE SUBMITTING
        #train_data, train_correct = pickle_to_data("cifar-10-batches-py")


        print("Starting training...")

        current_validate_accuracy = 0
        overall_best_state = EpochStateData(0, 0, 0, net.weights)

        for epoch in range(EPOCH_COUNT):
            if SHOULD_STOP:
                print("Training interrupt requested. Stopping")
                break

            if ADAPTIVE_LEARNING_RATE_MODE == AdaptiveLearningRateMode.FORMULA:
                net.lr = ADAPTIVE_LEARNING_RATE_FORMULA(epoch)
            elif ADAPTIVE_LEARNING_RATE_MODE == AdaptiveLearningRateMode.PREDEFINED_DICT:
                net.lr = ADAPTIVE_LEARNING_RATE_DICT.get(epoch, net.lr)
            else:
                raise NotImplementedError("Unknown adaptive learning rate mode")

            print(f"Epoch {epoch}")
            print(f"Current LR: {net.lr}")

            if TAKE_BEST_FROM_VALIDATE or TAKE_BEST_FROM_TRAIN:
                print("Take best from:", overall_best_state)
                net.weights = EpochStateData.deep_copy_list_of_np_arrays(overall_best_state.weights)


            if SUBSET_SIZE > 0:
                subset_train, subset_correct = get_subset(train_data, train_correct, SUBSET_SIZE)
                if INPUT_LAYER_NOISE_PROB > 0:
                    print(f"Applying noise of {INPUT_LAYER_NOISE_PROB * 100}% on all inputs")
                    subset_train = apply_noise(subset_train, INPUT_LAYER_NOISE_PROB)
                net.train_set(list(zip(subset_train, subset_correct)), shuffle=True, mini_batch_size=MINI_BATCH_SIZE)

            else:
                if INPUT_LAYER_NOISE_PROB > 0:
                    print(f"Applying noise of {INPUT_LAYER_NOISE_PROB * 100}% on all inputs")
                    after_noise_train = apply_noise(train_data, INPUT_LAYER_NOISE_PROB)
                    net.train_set(list(zip(after_noise_train, train_correct)), shuffle=True, mini_batch_size=MINI_BATCH_SIZE)
                else:
                    net.train_set(list(zip(train_data, train_correct)), shuffle=True, mini_batch_size=MINI_BATCH_SIZE)

            #validate_data, validate_correct = csv_to_data(validate_csv)


            print("======= Train Accuracy =======")
            current_train_accuracy, train_certainty = net.validate_set(list(zip(train_data, train_correct)))

            print("======= Validate Accuracy =======")
            current_validate_accuracy, validate_certainty = net.validate_set(list(zip(validate_data, validate_correct)))

            # TODO: REMOVE ME BEFORE SUBMITTING
            if test_csv:
                run_tests(test_data, net, epoch)

            csv_results.append([epoch, net.lr, current_train_accuracy, train_certainty, current_validate_accuracy, validate_certainty])

            if TAKE_BEST_FROM_TRAIN and TAKE_BEST_FROM_VALIDATE:
                if current_validate_accuracy + current_train_accuracy > overall_best_state.train_accuracy + overall_best_state.validate_accuracy:
                #if current_validate_accuracy >= overall_best_state.validate_accuracy and current_train_accuracy + 2.0 > overall_best_state.train_accuracy:
                    overall_best_state = EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights)
            elif TAKE_BEST_FROM_TRAIN:
                if current_train_accuracy > overall_best_state.train_accuracy:
                    overall_best_state = EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights)
            else:
                if current_validate_accuracy > overall_best_state.validate_accuracy:
                    overall_best_state = EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights)
            if epoch %25==0:
                save_state(output_path, f"epoch_{epoch}", EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights))
                save_state(output_path, f"best_state_until_epoch_{epoch}", overall_best_state)

        print("Done!")
        print("Saving results, weights...")

        with open(output_path / "results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_results)

        save_state(output_path, "latest_state", EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights))
        save_state(output_path, "best_state", overall_best_state)

        #mail_content = f"Finished!\nbest state:\n {overall_best_state}\n CONFIG:\n{open('config.py', 'r').read()}"
        #send_mail("yanirbuznah@gmail.com", mail_content)
        #send_mail("ron.evenm@gmail.com", mail_content)

    if test_csv:
        print("Test csv provided. Classifying...")
        test_data, _ = csv_to_data(test_csv)

        prediction_list = []
        for i, data in enumerate(test_data):
            classification = net.classify_sample(data) + 1
            prediction_list.append(classification)

        print("Saving predicted latest_test.txt")
        save_predictions("latest_test.txt", prediction_list)

        print(prediction_list)
        print(output_path)
        print("TODO: REMOVE ME")
        print("Testing results...")
        import result_compare
        result_compare.check_results(prediction_list)





        prediction_list = []
        net.set_weights(overall_best_state.weights)
        for i, data in enumerate(test_data):
            classification = net.classify_sample(data) + 1
            prediction_list.append(classification)

        print("Saving predicted best_test.txt")

        save_predictions("best_test.txt", prediction_list)

        print(prediction_list)
        print(output_path)
        print("TODO: REMOVE ME")
        print("Testing results...")
        import result_compare
        result_compare.check_results(prediction_list)


if __name__ == '__main__':
    main()
