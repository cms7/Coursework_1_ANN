#coursework 1

import pandas as pd
import numpy as np
from random import random
from random import seed

class NeuralNetwork:

    #function to intitialise a network taking adjustable numbers of inputs, hidden and outputs 
    def create_network(n_inputs,n_hidden,n_outputs):
        network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(hidden_layer)
        network.append(output_layer)
        return network

    #seed(1) ?? is this necessary 
    network = create_network(2,1,2)
    for layer in network:
        print(layer)

    
    #activation functions which takes an imput and produces a number between 0-1
    def sigmoid(inpt):
        return 1/1+np.exp(-inpt)

    def sigmoid_derivitive(inpt):
        return sigmoid(inpt)*(1-sigmoid(inpt))

    def tanh(inpt):
        return np.tanh(inpt)

    def relu(inpt):
        return max(inpt,0)

    #hyperparameters
    learning_rate = 0.1;

    #code for interpreting the data
    data = pd.read_csv("DATASET.csv")
    n_inputs = (len(list(data))-1)*(len(data))
    n_outputs = 2   


