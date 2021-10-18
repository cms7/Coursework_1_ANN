# coursework 1

import pandas as pd
import numpy as np
from random import random, seed

# function to intitialise a network taking adjustable numbers of inputs, hidden and outputs
# funtion will generate random weights for each input value, a bias is also genererated and will be the last element in the array
# e.g. 3 inputs: weights = [0.1 , 0.3 , 0.2, 0.6] where element [-1] is the bias
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    w1 = [{'weights': [random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    w2 = [{'weights': [random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(w1)
    network.append(w2)
    return network

    
# hyperparameters - the hyperparameters are set by the user when the terminal prompts input 
print("Please enter a learning rate between 0.01 and 1")
learning_rate = input()
print("Please enter the number of epochs - up to 20000")
num_epochs = input()
print("Please choose an activation function of the following - sigmoid , tanh , relu ")
activation_func = input()

# activation functions which takes an imput and produces a number between 0-1
def sigmoid(inpt):
    return 1/1+np.exp(-inpt)

def tanh(inpt):
    return np.tanh(inpt)

def relu(inpt):
    return max(inpt, 0)

#function which will calculate the weighted sum of weights and inputs + a bias 
def weighted_sum(weights, inputs):
	sum = weights[-1]
	for i in range(len(weights)-1):
		sum += weights[i]*inputs[i]
	return sum

def forward_prop(network,input):
    for layer in network:
        temp = []
        for neuron in layer:
            sum = weighted_sum(neuron['weights'],input)
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

seed(1)
network = initialize_network(2,1,1)
row = [1, 0, None]
output = forward_prop(network, row)
print(output)


# calculates the derivatives of the neuron output for all activation functions
def sigmoid_derivitive(neuron_output):
    return sigmoid(neuron_output*(1-sigmoid(neuron_output)))

def tanh_derivitive(neuron_output):
    return 1 - np.tanh(neuron_output)**2

def relu_derivitive(neuron_output):
    if(neuron_output<=0):
        return 0
    else: 
        return 1

# code for interpreting the data
data = pd.read_csv("DATASET.csv")
n_inputs = (len(list(data))-1)*(len(data))
