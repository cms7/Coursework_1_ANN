import numpy as np

    def create_network(n_inputs,n_hidden,n_output):

        weights_ih = np.random.rand(n_inputs,n_hidden)
        weights_ho = np.random.rand(n_hidden,n_output)

        bias_ih = np.random.rand(n_hidden,1)
        bias_ho = np.random.rand(n_output,1)
    
        parameters = {"W1": weights_ih, }
        return parameters

    def sigmoid(input):
        return 1/(1+np.exp(-input))

