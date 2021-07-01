# simple_neural_network

A simple implimention of neural network from scratch, using numpy cupy and pandas. 
## Getting started:
### Installing
Download the zip for this repository or use git on the termianl. The terminal command is:
```
git clone https://github.com/yanirbuznah/simple_neural_network.git
```
After clonning the project, run the following command : 
```
python main.py <train.csv> <validate.csv> <test.csv>
```
(<*.csv> refers to path)<br/>
To run the network after training, use the configuration file.
### dependencies:
- [pandas](https://pandas.pydata.org/).
- [numpy](https://numpy.org/).
- [cupy](https://cupy.dev/) (only if run on gpu).
## Configuration file:
- SEED = The seed for the random functions, use random.randint(0, 100000000) to randomly seed (the network will save the seed in a separate file)
- INPUT_LAYER_SIZE = The size of the input layer
- HIDDEN_LAYERS_SIZES = A list of hidden layer sizes, for example [1000,100] will create a first layer with 1000 neurons and a second layer with 100
- OUTPUT_LAYER_SIZE = The size of the output layer (for example: if the net need to detect if a picture is cat, dog, horse, put 3)
- ACTIVATION_FUNCTION = The activation function of the lauyer : ActivationFunction.ReLU/ActivationFunction.Sigmoid
- RANDRANGE = The initial range of values of the neurons in the network
- LEARNING_RATE = The initial learnig rate of the network
- EPOCH_COUNT = The number of epochs in training mode.
- INPUT_LAYER_NOISE_PROB = The noise on the input layer, for example, if 0.2 then 20% of the values in the input layer will be 0.
- SUBSET_SIZE = The amount of data per epoch ( -1 if all the data), for example: 1000, Will randomly select 1000 values from the training data in each epoch.
- MINI_BATCH_SIZE = Number of runs (forward propagation and backpropagation) until weights are updated.
- ADAPTIVE_LEARNING_RATE_MODE = The decay parameter to the learning rate. AdaptiveLearningRateMode.PREDEFINED_DICT/AdaptiveLearningRateMode.FORMULA.
- ADAPTIVE_LEARNING_RATE_FORMULA = Decay learning rate formula, for example: lambda epoch: 0.005 * np.exp(-0.0001 * epoch)
- ADAPTIVE_LEARNING_RATE_DICT = Python dictionary with learning rate per epoch (start with "LEARNING_RATE" parameter) , for example: {20: 0.002, 40: 0.001}
- SHOULD_TRAIN = False - to check outside weights, True  - to train (you can load outside weights and train).
- SAVED_MODEL_PICKLE_MODE = False -  to use csv files, True -  to use pickle (from pickle python library).
- TRAINED_NET_DIR = None if you don't want to load a result dir or path to result dir to load weights. 

- TAKE_BEST_FROM_TRAIN = True - in each epoch, take the weights with the highest success rates on a train, False - otherwise. (False - recommended!)
- TAKE_BEST_FROM_VALIDATE = True - in each epoch, take the weights with the highest success rates on a validate, False - otherwise. (False - recommended!)
- SHOULD_SHUFFLE = True - Shuffle the train and validate (together) before the start of the training. False - otherwise.
- SOFTMAX = True - Use softmax activation function on the output layer, False - use the normal activation function.
- DROP_OUT =  A list of probabilities of turning off neurons in the hidden layer, for example [0.2,0.5] will shut down 20% of the first layer neurons and 50% of the second layer.
- USE_GPU = True - to use gpu (use cupy library, and cuda gpu, recommended in google colab), False - to use cpu only.
