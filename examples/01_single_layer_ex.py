'''
    author: jielien
    date: 19.01.2022
    description: Single layer with 3 neurons
'''
import numpy as np

inputs = [1.2, 5.1, 2.1] # input for each neuron
weights =  [[ 0.12, 0.87, -0.72], # each neuron has weight for each input
            [ 0.2, -0.21,  0.67],
            [-0.1,  1.0,   0.91]]
biases = [2, 3, 0.5] # bias for each neuron

outputs = np.dot(weights, inputs) + biases # weights[0, 0]*inputs[0] + weights[0, 1]*inputs[1] + weights[0, 1]*inputs[1] + bias[0] ...

print(outputs)
