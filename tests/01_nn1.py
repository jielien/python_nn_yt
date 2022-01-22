import os
import numpy as np

class Layer:
    def __init__(self, input_count, neuron_count):
        self.in_c = input_count
        self.nn_c = neuron_count
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.populate_arrays()
    
    def populate_arrays(self):
        # create neurons
        for n in range(0, self.nn_c):
            self.biases.append(np.random.uniform(-1, 1)) # create bias for each neuron
            self.weights.append([]) # create input weights array for each neuron
            self.outputs.append(0.0) # create output for each neuron
            # populate weights arrays
            for i in range(0, self.in_c):
                self.weights[n].append(np.random.uniform(-1, 1))
    
    def run(self):
        # weights[0, 0]*inputs[0] + weights[0, 1]*inputs[1] + weights[0, 2]*inputs[2] + bias[0] ...
        self.outputs = np.dot(self.weights, self.inputs) + self.biases
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.output = []
    
    def add_layer(self, layer):
        if isinstance(layer, Layer):
            self.layers.append(layer)
        else:
            print("ERROR >> Invalid layer type!")
    
    def run(self, inputs):
        if len(self.layers) < 2:
            print("ERROR >> Insuficient layers count!")
            return
        self.layers[0].inputs = inputs
        for layer in range(0, len(self.layers)):
            self.layers[layer].run()
            if layer < len(self.layers) - 1:
                self.layers[layer + 1].inputs = self.layers[layer].outputs
        self.output = self.layers[-1].outputs

# create NN
print("INFO >> Creating NN.....")
nn = NeuralNetwork()
nn.add_layer(Layer(784, 784))
nn.add_layer(Layer(784, 1024))
nn.add_layer(Layer(1024, 1024))
nn.add_layer(Layer(1024, 10))
print("INFO >> Finished creating NN!")

'''
# open train files
dir = os.path.dirname(__file__)
labels_filename = os.path.join(dir, 'datasets','mnist','train-labels.idx1-ubyte')
labels_file = open(labels_filename, "rb")
labels_header = labels_file.read(8)
#print(['{:02x}'.format(b) for b in header])
labels_data = labels_file.read()
#print(['{:02x}'.format(b) for b in data])
labels_file.close()

images_filename = os.path.join(dir, 'datasets','mnist','train-images.idx3-ubyte')
images_file = open(images_filename, "rb")
header = images_file.read(16)
print(['{:02x}'.format(b) for b in header])
images_data = images_file.read()
#print(['{:02x}'.format(b) for b in data])
images_file.close()

# TODO: learning
'''

inputs = []
for x in range(0, 784):
    inputs.append(np.random.uniform(-1, 1))
print("INFO >> Running NN.....")
nn.run(inputs)
print(nn.output)
