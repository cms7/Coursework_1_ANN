from os import error
from random import random,seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

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
            if(activation_func == "sigmoid"):
                neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
            elif(activation_func == "relu"):
                neuron['delta'] = errors[j] * relu_derivative(neuron['output'])
            elif(activation_func == "tanh"):
                neuron['delta'] = errors[j] * tanh_derivative(neuron['output'])
            else:
                print("Please check hyperparameter activation_func")
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
    #storing epochs values so they can be used for the following graphs
    epoch_l = []
    for k in range(0,num_epochs):
        epoch_l += [k]

    #coding of the graphs produced
    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)
    l = ax.fill_between(epoch_l, error_l)
    ax.legend(['Learning Rate = 0.01'])
    l.set_facecolors([[.5,.5,.8,.3]])
    l.set_edgecolors([[0, 0, .5, .3]])
    l.set_linewidths([3])
    #Set labels
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Summed error')
    ax.set_title('ReLU (rectified linear unit)')
    ax.grid('on')

    #Tweak labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_style('italic')
    xlab.set_size(10)
    ylab.set_style('italic')
    ylab.set_size(10)
    #tweaking plot title
    ttl = ax.title
    ttl.set_weight('bold')
    plt.show()

seed(1)
dataset = pd.read_csv('DATASET.csv')
data = np.array(dataset,float)
n_inputs = len(dataset.columns) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2,2, n_outputs)
train_network(network, data, n_outputs)

