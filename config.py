from common import ActivationFunction

# Neural Network Configuration
INPUT_LAYER_SIZE = 3072
HIDDEN_LAYERS_SIZES = [1000, 1000]
OUTPUT_LAYER_SIZE = 10
ACTIVATION_FUNCTION = ActivationFunction.ReLU
RANDRANGE = 0.04
LEARNING_RATE = 0.003

# Training Configuration
EPOCH_COUNT = 30
ADAPTIVE_LEARNING_RATE_SETTING = {
        15: 0.001,
        20: 0.0005
}

SHOULD_TRAIN = True

TRAINED_NET_DIR = None  # Put None if you don't want to load a result dir

