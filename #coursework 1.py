#coursework 1

import pandas as pd
import numpy as np

class NeuralNetwork:


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

    def gradient_descent():
        


