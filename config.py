import numpy as np

from common import ActivationFunction, AdaptiveLearningRateMode

# Neural Network Configuration
SEED = 785
INPUT_LAYER_SIZE = 3072
HIDDEN_LAYERS_SIZES = [1000,1000]
OUTPUT_LAYER_SIZE = 10
ACTIVATION_FUNCTION = ActivationFunction.ReLU
RANDRANGE = 0.035
LEARNING_RATE = 0.003

# Training Configuration
EPOCH_COUNT = 100
INPUT_LAYER_NOISE_PROB = 0
SUBSET_SIZE = 8000
MINI_BATCH_SIZE = 32
ADAPTIVE_LEARNING_RATE_MODE = AdaptiveLearningRateMode.PREDEFINED_DICT
ADAPTIVE_LEARNING_RATE_FORMULA = lambda epoch: 0.01 * np.exp(-0.01 * epoch)
# ADAPTIVE_LEARNING_RATE_DICT = {
#         10: 0.009,
#         20: 0.008,
#         30: 0.007,
#         40: 0.006,
#         50: 0.005,
#         60: 0.004,
#         70: 0.003,
#         80: 0.002,
#         90: 0.001
# }
# ADAPTIVE_LEARNING_RATE_DICT = {
#         50: 0.009,
#         100: 0.008,
#         150: 0.007,
#         200: 0.006,
#         250: 0.005,
#         300: 0.004,
#         350: 0.003,
#         400: 0.002,
#         450: 0.001
# }
ADAPTIVE_LEARNING_RATE_DICT = {
        10: 0.002,
        20: 0.001,
        30: 0.0005,
        60: 0.0004,
        80: 0.0003
}
SHOULD_TRAIN = True

SAVED_MODEL_PICKLE_MODE = True  # Put False to use csv files, True to use pickle
TRAINED_NET_DIR = None  # Put None if you don't want to load a result dir

TAKE_BEST_FROM_TRAIN = False
TAKE_BEST_FROM_VALIDATE = False

SEPARATE_VALIDATE = True