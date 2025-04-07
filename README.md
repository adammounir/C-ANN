# C-ANN - Neural Network from scratch for MNIST Digit Recognition

This project implements a neural network for recognizing handwritten digits from the MNIST dataset. It includes forward and backward propagation, model training, and evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Model Saving and Loading](#model-saving-and-loading)
- [Usage](#usage)

## Introduction
The project uses a neural network to perform digit recognition on the MNIST dataset. It is built with:
- Forward propagation using ReLU activation functions.
- Backpropagation for training the network with gradient descent.
- Model saving and loading for reusability.

The network consists of:
- **Input Layer**: 784 neurons (28x28 MNIST image size).
- **Hidden Layers**: Two layers with 64 neurons each.
- **Output Layer**: 10 neurons (digits 0-9).

## Files and Structure

### Core Files
- **`neural_network.h` and `neural_network.c`**: Define the structure of the neural network, including layer sizes, weights, biases, and the core functions for training (forward and backward propagation), as well as functions for saving and loading the model.
  
- **`mnist.h` and `mnist.c`**: Handle loading and preprocessing the MNIST dataset. They define how the dataset is read, converted into usable formats, and provide functions to display images.

- **`main.c`**: The entry point of the program that handles user input to either train a new model or test with a pre-trained model. It manages the overall training loop and prints progress.

- **`front_propagation.c`**: Contains the logic for forward propagation, computing the activations of each layer.

- **`back_propagation.c`**: Implements backpropagation, calculating gradients and updating weights to minimize the loss function.

## Installation
To run the project, clone this repository to your local machine and compile the C files.

### Prerequisites
- C Compiler (e.g., GCC)
- The MNIST dataset (expected at the specified paths)

### Compilation
Running the Program
To train the model, execute the following command:
```bash
make clean
make
./main
```

Then, choose option 1 to train the model. The program will display a progress bar for each epoch of training.

## Training the Model
During training, the network uses the MNIST training images and labels. The weights and biases are initialized randomly, and the model is trained using gradient descent.

## Model Evaluation
After training, the model is evaluated on the test dataset, and the accuracy is displayed based on predictions made on the test images.

## Model Saving and Loading
The trained model can be saved to a file for later use. This allows you to load the model without retraining it.


### Load Pre-Trained Model:
```bash
./main
```
Then, choose option 2 to test the model.

## License
This project is open-source and available under the MIT License.


