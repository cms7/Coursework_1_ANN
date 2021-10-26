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
# number of hidden layers is configuarable through the funtion. 
def initialise_network(n_inputs, n_hidden, n_hidden1, n_outputs):
    net = list()
    w1 = [{'weights': [random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    w2 = [{'weights': [random() for i in range(n_hidden+1)]} for i in range(n_hidden1)]
    w3 = [{'weights': [random() for i in range(n_hidden1+1)]} for i in range(n_outputs)]
    net.append(w1)
    net.append(w2)
    net.append(w3)
    return net

# activation functions which takes an input and produces a number between 0-1
def sigmoid(inpt):
    return 1/(1+np.exp(-inpt))

def tanh(inpt):
    return np.tanh(inpt)

def relu(inpt):
    return np.maximum(0,inpt)

# calculates the derivatives of the neuron output for all activation functions
def sigmoid_derivative(node_output):
    return sigmoid(node_output)*(1-sigmoid(node_output))

def tanh_derivative(node_output):
    return 1 - tanh(node_output)**2

def relu_derivative(node_output):
    if(node_output<0):
        return 0
    else: 
        return 1

# This will calculate the neurons activation value
def sum_weights(weights, inputs):
    # weights[-1] will act as the bias, this takes the last element in the array of weights, which previously was identified.
    sum = weights[-1]
    for i in range(len(weights)-1):
        sum += weights[i] * float(inputs[i])
    return sum
 
# Function to forward propagte the network using loops to iterate over each node
def forward_prop(net,row):
    inputs = row[:-1]
    for layer in net:
        temp = []
        for node in layer:
            #use the weighted sum of inputs and outputs and feed the result into an activation function defined by the used 
            #using conditional statements
            sum = sum_weights(node['weights'],inputs)
            if(activation_func == "sigmoid"):
                node['output'] = sigmoid(sum)
            elif(activation_func == "relu"):
                node['output'] = relu(sum)
            elif(activation_func == "tanh"):
                node['output'] = tanh(sum)
            else:
                #this is base case to make sure that there is a valid input for the activation function, else exit program
                print("Please enter a valid hyperparameter for activation function of the following: sigmoid , tanh, relu")
                exit()
            temp.append(node['output'])
        inputs = temp
    return inputs

#error function to calculate the sum of squared errors
def squared_error(actual_output,output):
    sum_err = sum([(actual_output[i]-output[i])**2 for i in range(len(actual_output))])
    return sum_err
    

# Function traverses the network backawards calculating gradient of weights.
def backward_propagate_error(net, actual_output):
    for i in reversed(range(len(net))):
        errors_list = list() # errors are stored here
        layer = net[i]
        if i != len(net)-1:
            for j in range(len(layer)):
                node_error = 0.0
                for node in net[i + 1]:
                    node_error += (node['weights'][j] * node['diff'])
                errors_list.append(node_error)
        else:
            for j in range(len(layer)):
                node = layer[j]
                errors_list.append(actual_output[j] - node['output'])
        for j in range(len(layer)):
            node = layer[j]
            #conditional statements which set the correct derivatave
            if(activation_func == "sigmoid"):
                node['diff'] =  sigmoid_derivative(node['output']) * errors_list[j] 
            elif(activation_func == "relu"):
                node['diff'] = relu_derivative(node['output']) * errors_list[j] 
            elif(activation_func == "tanh"):
                node['diff'] = tanh_derivative(node['output']) * errors_list[j] 
            else:
                print("Please check hyperparameter activation_func")

# Function to tune the weights of each nodes connections
def update_weights(net, row):
    for i in range(len(net)):
        inputs = row[:-1]
        if i != 0:
            inputs = [node['output'] for node in net[i - 1]]
        for node in net[i]:
            for j in range(len(inputs)):
            #update each respective weight 
                node['weights'][j] += learning_rate * inputs[j] * node['diff'] 
            #updates the bias 
            node['weights'][-1] += learning_rate * node['diff']


# Network is trained over a fixed number of epochs which is set by the user
def train_net(net, data):
    error_l = []
    for epoch in range(num_epochs):
        sum_error = 0
        for row in data:
            #expected data is either 1 or 0 for binary classification problem
            actual_output = [0,1]
            #use previous functions developed for training
            forward_prop_output = forward_prop(net, row)
            sum_error += squared_error(actual_output,forward_prop_output)
            backward_propagate_error(net, actual_output)
            update_weights(net, row)
        #prints in terminal for user to follow
        print('Epoch number = %d, Learning Rate = %.3f, Total Error = %.3f' % (epoch, learning_rate, sum_error))
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

#seed(1) ensures each test will use the same generated weights
seed(1)
dataset = pd.read_csv('DATASET.csv')
data = np.array(dataset,float)

#sets the required inputs and outputs for dataset
n_inputs = len(dataset.columns) - 1
n_outputs = 2

#create and test network
net = initialise_network(n_inputs, 2,2, n_outputs)
train_net(net, data)

