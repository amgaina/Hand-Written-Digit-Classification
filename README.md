# Hand-Written-Digit-Classification

### This project involves the creation of a neural network to classify handwritten digits from the famous MNIST dataset. The model was developed using TensorFlow and Keras and achieved an accuracy of 98% on the test data.

## Introduction
Handwritten digit classification is a classic machine-learning problem. The goal is to correctly identify digits (0-9) from a dataset of handwritten images. This project demonstrates the creation and training of a neural network to achieve this task.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy

## Installation
To install the required libraries, you can use the following command:
```sh
pip install tensorflow keras matplotlib numpy
```
## Dataset
The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits. Each image is 28x28 pixels.

## Model Architecture
The neural network model consists of the following layers:

Input layer: Flatten layer to convert 28x28 images to a 784-dimensional vector
Hidden layer 1: Dense layer with 110 neurons and ReLU activation
Output layer: Dense layer with 10 neurons (one for each digit) and sigmoid activation

## Training
The model is trained using the following steps:

1. Load and preprocess the MNIST dataset.
2. Define the neural network architecture.
3. Compile the model with loss function, optimizer, and evaluation metric.
4. Train the model on the training data.
5. Validate the model on the validation data.

## Evaluation
The model is evaluated on the test dataset to measure its performance. The evaluation metric used is accuracy.
