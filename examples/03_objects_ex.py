'''
    author: jielien
    date: 23.01.2022
    description: Layer as object
'''
import numpy as np

np.random.seed(0)

# inputs batch
X = [[ 1.0, 2.0,  3.0,  2.5],
     [ 2.0, 5.0, -1.0,  2.0],
     [-1.5, 2.7,  3.3, -0.8]]

class Layer:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights matrix and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        # calculate output matrix
        self.output = np.dot(inputs, self.weights) + self.biases

# create layers
layer1 = Layer(4, 5)
layer2 = Layer(5, 2)

# running "nn"
layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)