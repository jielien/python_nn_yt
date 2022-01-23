'''
    author: jielien
    date: 23.01.2022
    description: Softmax and ReLU in simple nn
'''
# numpy initialization
import numpy as np
np.random.seed(0)
# custom datasets library
from custom_datasets import spiral_data

class Layer:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights matrix and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        # calculate output matrix
        self.output = np.dot(inputs, self.weights) + self.biases

# activation function - rectified linear unit
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# activation function - softmax
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # exponentiation and overflow prevention
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True) # normalization


# inputs batch
X, y = spiral_data(100, 3)

# create layers and activation functions
layer1 = Layer(2, 3)
activation1 = Activation_ReLU()
layer2 = Layer(3, 3)
activation2 = Activation_Softmax()

# forward passes
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

# print first 5 outputs
print(activation2.output[:5])
