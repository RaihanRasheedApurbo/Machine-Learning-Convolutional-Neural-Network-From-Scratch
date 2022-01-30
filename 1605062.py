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
        sgn_input = (input_matrix > 0)
        self.logger.info(sgn_input)
        self.sgn_input = sgn_input
        return input_matrix * sgn_input
    
    def backward(self, input_matrix):
        self.logger.info(self.sgn_input)
        return self.sgn_input * input_matrix

class Flatten:
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

    

if __name__ == '__main__':
    a = np.random.randn(2,2)
    f = Flatten()
    logger.info(a)
    logger.info(f.forward(a))
    logger.info(f.backward(f.forward(a)))




    