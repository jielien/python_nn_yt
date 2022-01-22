'''
    Single layer with 3 neurons example
'''

import numpy as np

inputs = [[1.2,  5.1,  2.1],
          [0.2, -3.7, -0.8],
          [4.2,  3.1,  4.7]] # batch of inputs for each neuron
weights =  [[ 0.12, 0.87, -0.72],
            [ 0.2, -0.21,  0.67],
            [-0.1,  1.0,   0.91]] # each neuron has weight for each input
biases = [2, 3, 0.5] # bias for each neuron

# weights[0, 0]*inputs[0] + weights[0, 1]*inputs[1] + weights[0, 1]*inputs[1] + bias[0] ...
outputs = np.dot(inputs, np.array(weights).T) + biases
 
print(outputs)
