from typing import List

from config import *

import numpy as np
import numpy

if USE_GPU:
    import cupy
    import cupy as np


def shuffle(x, y, seed=None):
    if seed:
        import random
        random.seed(seed)
        random.shuffle(x)
        random.seed(seed)
        random.shuffle(y)
    else:
        rand_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(rand_state)
        np.random.shuffle(y)
    return x, y


def result_classifications(results_classifications: List[int]) -> np.array:
    results = numpy.zeros((len(results_classifications), 10))
    for i in range(len(results_classifications)):
        if not str(results_classifications[i]).isdigit():
            # This is probably a test set. Ignore expected results column
            results = []
            break

        results[i][results_classifications[i] - 1] = 1

    return results


def separate_validation(samples, div):
    x, y = zip(*samples)
    train_len = int(len(x) * div)
    return {'train_x': x[:train_len], 'train_y': y[:train_len], 'validate_x': x[train_len:],
            'validate_y': y[train_len:]}


def load_all():
    train_x = "train_x"
    train_y = "train_y"
    test_x = "test_x"

    train_x = numpy.loadtxt(train_x) / 255
    train_y = numpy.loadtxt(train_y, dtype=numpy.int)
    train_x,train_y = shuffle(train_x,train_y)
    test_x = numpy.loadtxt(test_x) / 255

    res = separate_validation(zip(train_x, train_y), 0.9)
    train_x = res["train_x"]
    train_y = res["train_y"]
    validate_x = res["validate_x"]
    validate_y = res["validate_y"]
    train_y = result_classifications(list(train_y))
    validate_y = result_classifications(list(validate_y))

    return train_x, train_y, validate_x, validate_y, test_x
