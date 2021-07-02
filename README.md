# simple_neural_network

A simple implimention of neural network from scratch, using numpy cupy and pandas. 
## Getting started:
### Installing
Download the zip for this repository or use git on the termianl. The terminal command is:
```
git clone https://github.com/yanirbuznah/simple_neural_network.git
```
After cloning the project, run the following command : 
```
python main.py <train.csv> <validate.csv> <test.csv>
```
(<*.csv> refers to path)<br/>
To run the network after training, copy the `config.py` file from the saved model directory onto the `config.py` in the main directory,
and change the following parameters:
- `TRAINED_NET_DIR = <TRAINED_MODEL_DIR>`
- `SHOULD_TRAIN = False`
- Change `USE_GPU` to whether or not the local machine has CUDA support
### Dependencies:
- [pandas](https://pandas.pydata.org/).
- [numpy](https://numpy.org/).
- [cupy](https://cupy.dev/) (only if run on gpu).

## Configuration file:

- `SEED` = The seed for the random functions, use `random.randint(0, 100000000)` for a random seed (the network will save the seed in a separate file)
  

- `INPUT_LAYER_SIZE` = The size of the input layer
  

- `HIDDEN_LAYERS_SIZES` = A list of hidden layer sizes, for example [1000,100] will create a first layer with 1000 neurons and a second layer with 100
  

- `OUTPUT_LAYER_SIZE` = The size of the output layer (for example: if the net need to detect if a picture is cat, dog, horse, put 3)
  

- `ACTIVATION_FUNCTION` = The activation function the network uses. Can be one of:
  - `ActivationFunction.ReLU`
  - `ActivationFunction.Sigmoid`
    

- `RANDRANGE` = The initial range of values of the neurons in the network
  

- `LEARNING_RATE` = The initial learning rate of the network
  

- `EPOCH_COUNT` = The number of epochs in training mode.


- `INPUT_LAYER_NOISE_PROB` = The noise on the input layer, for example, if 0.2 then 20% of the values in the input layer will be randomly set to 0.
  

- `SUBSET_SIZE` = The amount of data per epoch ( -1 if all the data), for example: 1000, Will randomly select 1000 values from the training data in each epoch.
  

- `MINI_BATCH_SIZE` = Number of runs (forward propagation and backpropagation) until weights are updated.
  

- `ADAPTIVE_LEARNING_RATE_MODE` = The decay parameter to the learning rate. Can be either:
  - `AdaptiveLearningRateMode.PREDEFINED_DICT` (Configured by `ADAPTIVE_LEARNING_RATE_DICT`)
  - `AdaptiveLearningRateMode.FORMULA` (Decay function configured by `ADAPTIVE_LEARNING_RATE_FORMULA`)
    

- `ADAPTIVE_LEARNING_RATE_FORMULA` = Decay learning rate formula, for example:
<br/>`lambda epoch: 0.005 * np.exp(-0.0001 * epoch)`


- `ADAPTIVE_LEARNING_RATE_DICT` = Python dictionary with learning rate per epoch (start with `LEARNING_RATE` parameter) , for example: `{20: 0.002, 40: 0.001}`
  

- `SHOULD_TRAIN` = 
  - `True` - Enable training (you can still load existing weights and continue training on them)
  - `False` - Disable training. Should be used only when `TRAINED_NET_DIR` is configured
    

- `TRAINED_NET_DIR` = 
  - `None` if you don't want to load a result dir (generally for training mode) 
  - Path to result directory to load existing weights model. 
  

- `SAVED_MODEL_PICKLE_MODE` = 
  - `True` -  to use pickle (from pickle python library).
  - `False` -  to use csv files (legacy)

- `TAKE_BEST_FROM_TRAIN` = 
  - `True` - in each epoch, take the weights with the highest success rates on a train
  - `False` - otherwise (recommended!)
    

- `TAKE_BEST_FROM_VALIDATE` =
  - `True` - in each epoch, take the weights with the highest success rates on a validate run
  - `False` - otherwise (recommended!)
    

- `SHOULD_SHUFFLE` = 
  - `True` - Shuffle the train and validate (together) before the start of the training-
  - `False` - otherwise
    

- `SOFTMAX` = 
  - `True` - Use softmax activation function on the output layer
  - `False` - use the normal activation function.
    

- `DROP_OUT` =  A list of probabilities of turning off neurons in the hidden layer. For example `[0.2,0.5]` will shut down 20% of the first hidden layer neurons and 50% of the second hidden layer.
  

- `USE_GPU` = 
  - `True` - to use gpu (use cupy library, and cuda gpu, recommended in google colab)
  - `False` - to use cpu only.


## Saved model format
A saved model is a folder containing the following files:
  - `results.csv` - The training process output (accuracy and certainty observed in each epoch). Can be used to build a training process chart
  - `seed` - The seed that was used during training 
  - `config.py` - Configuration file used for the training, just for backup purposes
  - `*.model` - Trained model state object in pickle format. When running, make sure there is only one model file in the directory
