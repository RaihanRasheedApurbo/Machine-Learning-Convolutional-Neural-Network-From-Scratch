import logging
import random
import math
from turtle import forward
import numpy as np

# logger initialization
formatter = logging.Formatter(
    "\n*********Line no:%(lineno)d*********\n%(message)s\n***************************"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)



class ReLU:

    def __init__(self, debug=logging.ERROR) -> None:
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(debug)
        self.logger.addHandler(stream_handler)

        
    def forward(self, input_matrix):
        sgn_input = (input_matrix > 0).astype(int)
        self.logger.info(sgn_input)
        self.sgn_input = sgn_input
        return input_matrix * sgn_input
    
    def backward(self, input_matrix):
        self.logger.info(self.sgn_input)
        return self.sgn_input * input_matrix

class Flatten:

    def __init__(self, debug=logging.ERROR) -> None:
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(debug)
        self.logger.addHandler(stream_handler)
        
    def forward(self, input_matrix):
        self.shape = input_matrix.shape
        return np.ndarray.flatten(input_matrix)

    def backward(self, input_matrix):
        return np.reshape(input_matrix, self.shape)
    
def soft_max(input_matrix):
    exp = np.exp(input_matrix)
    sum = np.sum(exp)
    normalize = exp/sum
    logger.info(normalize)
    logger.info(np.sum(normalize))
    return normalize

class FullConnectedLayer:

    def __init__(self, input_size, output_size, debug=logging.ERROR, weight_matrix=None, bias_matrix=None):
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(debug)
        self.logger.addHandler(stream_handler)

        self.flatten = Flatten() # creating a flatten object if input is not properly flatten
        
        if weight_matrix.any():
            self.weight_matrix = weight_matrix
        else:
            self.weight_matrix = np.random.randn(output_size,input_size)
        
        if bias_matrix.any():
            self.bias_matrix = bias_matrix
        else:
            self.bias_matrix = np.random.randn(output_size,1) # 1 for making column matrix

    
    def forward(self, input_matrix):
        flatten_matrix = self.flatten.forward(input_matrix)
        m = flatten_matrix.shape
        logger.info(m)
        x = np.reshape(flatten_matrix, (m[0],1)) # 1 for making column matrix instead of 1D array
        

        if x.shape[0] != self.weight_matrix.shape[1]:
            raise 'Full Connected layer error shape mismatch in forward propagation'
        
        logger.info(self.weight_matrix)
        logger.info(x)
        wx = np.matmul(self.weight_matrix, x)
        self.input_matrix = x # stroring for backward prop.

        logger.info(wx)
        logger.info(self.bias_matrix)
        
        y = wx + self.bias_matrix
        
        logger.info(y)
        return  y

    # def forward(self, input_matrix):

    #     self.input_matrix = self.flatten.forward(input_matrix) # incase input is not flatten 
    #     self.logger.info(input_matrix.shape)
    #     self.logger.info(self.weight_matrix.shape)
        
    #     if input_matrix.shape[0] != self.weight_matrix.shape[1]:
    #         raise 'Full Connected layer error shape mismatch in forward propagation'
        
    #     logger.info(self.weight_matrix)
    #     logger.info(input_matrix)
        
    #     wx = np.matmul(self.weight_matrix, input_matrix)
        
    #     logger.info(wx)
    #     logger.info(self.bias_matrix)
        
    #     y = wx + self.bias_matrix
        
    #     logger.info(y)
    #     return  y

    def backward(self, output_gradiant, learning_rate=1):
        bias_gradient = output_gradiant
        weight_gradient = np.matmul(output_gradiant, self.input_matrix.T)
        input_gradient = np.matmul(self.weight_matrix.T, output_gradiant)


        logger.info(bias_gradient)
        logger.info(weight_gradient)
        logger.info(input_gradient)

        logger.info('prev weights')
        logger.info(self.weight_matrix)

        logger.info('prev bias')
        logger.info(self.bias_matrix)

        self.bias_matrix -= learning_rate * bias_gradient
        self.weight_matrix -= learning_rate * weight_gradient

        logger.info('bias')
        logger.info(self.bias_matrix)

        logger.info('weights')
        logger.info(self.weight_matrix)

        return self.flatten.backward(input_gradient) # incase input wasn't flattend


    # def backward(self, input_matrix, learning_rate=1):
        
    #     # converting 1D to 2D
    #     output_gradiant = input_matrix[np.newaxis].T  
    #     input_transpose = self.input_matrix[np.newaxis]

    #     bias_gradient = input_matrix
    #     weight_gradient = np.matmul(output_gradiant, input_transpose)
    #     input_gradient = np.matmul(self.weight_matrix.T, output_gradiant)
        
    #     logger.info(bias_gradient)
    #     logger.info(weight_gradient)
    #     logger.info(input_gradient)

    #     logger.info('prev weights')
    #     logger.info(self.weight_matrix)

    #     logger.info('prev bias')
    #     logger.info(self.bias_matrix)

    #     self.bias_matrix -= learning_rate * bias_gradient
    #     self.weight_matrix -= learning_rate * weight_gradient

    #     logger.info('bias')
    #     logger.info(self.bias_matrix)

    #     logger.info('weights')
    #     logger.info(self.weight_matrix)

    #     return self.flatten.backward(input_gradient) # incase input wasn't flattend
        
    


if __name__ == '__main__':
    

#     weight = np.array([[ 1.4401747,   0.72498046, -0.05727674],
#  [-1.15246919, -0.39990891,  0.44136903],
#  [ 1.14171484, -1.41891945,  0.73059128],
#  [ 0.60664542, -0.08249916, -1.05893566]])
#     bias = np.array([[-0.64243089], [0.51146315], [-0.17120088], [1.10775354]])

#     f = FullConnectedLayer(3,4,debug=logging.INFO, weight_matrix=weight, bias_matrix=bias)
#     output = np.random.randn(2,2)
#     output = np.array([ [0.83351854], [-0.55429203],  [0.0702855] ])
#     logger.info(output)
#     output = f.forward(output)
#     logger.info(output)
#     output = f.backward(output)
#     logger.info(output)
#     r = ReLU()
#     output = r.forward(output)
#     logger.info(output)
#     output = r.backward(output)
#     logger.info(output)
#     output = soft_max(output)
#     logger.info(output)








    