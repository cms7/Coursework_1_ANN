from os import error
from random import random,seed
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# hyperparameters - the hyperparameters are set by the user when the terminal prompts input 
print("Please enter a learning rate between 0.01 and 1")
learning_rate = float(input())
print("Please enter the number of epochs - up to 20000")
num_epochs = int(input())
print("Please choose an activation function of the following - sigmoid , tanh , relu ")
activation_func = input()

# function to intitialise a network taking adjustable numbers of inputs, hidden and outputs
# funtion will generate random weights for each input value, a bias is also genererated and will be the last element in the array
# e.g. 3 inputs: weights = [0.1 , 0.3 , 0.2, 0.6] where element [-1] is the bias
def initialize_network(n_inputs, n_hidden, n_hidden1, n_outputs):
    network = list()
    w1 = [{'weights': [random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    w2 = [{'weights': [random() for i in range(n_hidden+1)]} for i in range(n_hidden1)]
    w3 = [{'weights': [random() for i in range(n_hidden1+1)]} for i in range(n_outputs)]
    network.append(w1)
    network.append(w2)
    network.append(w3)
    return network

# activation functions which takes an input and produces a number between 0-1
def sigmoid(inpt):
    return 1/(1+np.exp(-inpt))

def tanh(inpt):
    return np.tanh(inpt)

def relu(inpt):
    return np.maximum(0,inpt)

# calculates the derivatives of the neuron output for all activation functions
def sigmoid_derivative(neuron_output):
    return sigmoid(neuron_output)*(1-sigmoid(neuron_output))

def tanh_derivative(neuron_output):
    return 1 - tanh(neuron_output)**2

def relu_derivative(neuron_output):
    if(neuron_output<0):
        return 0
    else: 
        return 1

# Calculate neuron activation for an input
def sum_weights(weights, inputs):
    sum = weights[-1]
    for i in range(len(weights)-1):
        sum += weights[i] * float(inputs[i])
    return sum
 
# Forward propagate input to a network output 
def forward_prop(network,row):
    inputs = row
    for layer in network:
        temp = []
        for neuron in layer:
            sum = sum_weights(neuron['weights'],inputs)
            if(activation_func == "sigmoid"):
                neuron['output'] = sigmoid(sum)
            elif(activation_func == "relu"):
                neuron['output'] = relu(sum)
            elif(activation_func == "tanh"):
                neuron['output'] = tanh(sum)
            else:
                print("Please check hyperparameter activation_func")
            temp.append(neuron['output'])
        inputs = temp
    return inputs

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def train_network(network, train, n_outputs):
    error_l = []
    for epoch in range(num_epochs):
        sum_error = 0
        
        for row in train:
            outputs = forward_prop(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[-1] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
        error_l.append(sum_error)
    epoch_l = []
    for k in range(0,num_epochs):
        epoch_l += [k]
    print(error_l)
    print(epoch_l)
    x = np.array(epoch_l)
    y = np.array(error_l)
    plt.plot(x, y)
    plt.show()



    


        
    
    

    
seed(1)
dataset = pd.read_csv('DATASET.csv')
data = np.array(dataset,float)
n_inputs = len(dataset.columns) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2,2, n_outputs)
train_network(network, data, n_outputs)

