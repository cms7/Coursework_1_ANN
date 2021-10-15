# coursework 1

import pandas as pd
import numpy as np
from random import random, seed

# function to intitialise a network taking adjustable numbers of inputs, hidden and outputs
# funtion will generate random weights for each input value, a bias is also genererated and will be the last element in the array
# e.g. 3 inputs: weights = [0.1 , 0.3 , 0.2, 0.6] where element [-1] is the bias
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    w1 = [{'weights': [random() for i in range(n_inputs)]} for i in range(n_hidden)]
    w2 = [{'weights': [random() for i in range(n_hidden)]} for i in range(n_outputs)]
    network.append(w1)
    network.append(w2)
    return network

# hyperparameters
learning_rate = 0.1
num_epochs = 200
activation_func = "sigmoid"

# activation functions which takes an imput and produces a number between 0-1
def sigmoid(inpt):
    return 1/1+np.exp(-inpt)

def tanh(inpt):
    return np.tanh(inpt)

def relu(inpt):
    return max(inpt, 0)

def sum_of_weights(weights, inputs):
	sum = 0 
	for i in range(len(weights)-1):
		sum += weights[i]*inputs[i]+weights[-1]
	return sum

def forward_prop(network,input):
    for layer in network:
        temp = []
        for neuron in layer:
            sum = sum_of_weights(neuron['weights'],input)
            if(activation_func == "sigmoid"):
                neuron['output'] = sigmoid(sum)
            elif(activation_func == "relu"):
                neuron['output'] = relu(sum)
            elif(activation_func == "tanh"):
                neuron['output'] = tanh(sum)
            else:
                print("Please check hyperparameter activation_func")
            temp.append(neuron['output'])
        output = temp
    return output

network = initialize_network(2,1,2)
row = [1, 0, None]
output = forward_prop(network, row)
print(output)


# calculates the derivative of the neuron output
def neuron_derivitive(neuron_output):
    return neuron_output*(1-neuron_output)



# code for interpreting the data
data = pd.read_csv("DATASET.csv")
n_inputs = (len(list(data))-1)*(len(data))
