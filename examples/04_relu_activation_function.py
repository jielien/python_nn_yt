'''
    author: jielien
    date: 23.01.2022
    description: ReLU activation function example
'''
# numpy initialization
import numpy as np
np.random.seed(0)

# custom datasets library
from custom_datasets import spiral_data

# inputs batch
X, y = spiral_data(100, 3)

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

# create layers
layer1 = Layer(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(activation1.output)
