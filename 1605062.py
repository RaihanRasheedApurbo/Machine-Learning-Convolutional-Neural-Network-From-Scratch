import logging
import random
import math
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

print(np.zeros(10))

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



    

    

r = ReLU(debug=logging.INFO)
a = np.random.randn(2,2)
logger.info(a)
output = r.forward(a)
logger.info(output)
output = r.backward(a)
logger.info(output)




    