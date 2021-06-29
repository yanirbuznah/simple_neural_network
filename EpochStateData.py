from typing import List


import numpy


from config import *


import numpy as np


random.seed(SEED)
numpy.random.seed(SEED)

if USE_GPU:
    import cupy as np
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
