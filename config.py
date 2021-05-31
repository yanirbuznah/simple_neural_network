import numpy as np

from common import ActivationFunction, AdaptiveLearningRateMode

# Neural Network Configuration
INPUT_LAYER_SIZE = 3072
HIDDEN_LAYERS_SIZES = [1000, 1000]
OUTPUT_LAYER_SIZE = 10
ACTIVATION_FUNCTION = ActivationFunction.ReLU
RANDRANGE = 0.04
LEARNING_RATE = 0.003

# Training Configuration
EPOCH_COUNT = 100
INPUT_LAYER_NOISE_PROB = 0.2
SUBSET_SIZE = -1
ADAPTIVE_LEARNING_RATE_MODE = AdaptiveLearningRateMode.FORMULA
ADAPTIVE_LEARNING_RATE_FORMULA = lambda epoch: 0.01 * np.exp(-0.01 * epoch)
ADAPTIVE_LEARNING_RATE_DICT = {
        15: 0.001,
        20: 0.0006,
        25: 0.0002,
        50: 0.0001
}

SHOULD_TRAIN = True

SAVED_MODEL_PICKLE_MODE = True  # Put False to use csv files, True to use pickle
TRAINED_NET_DIR = None  # Put None if you don't want to load a result dir

