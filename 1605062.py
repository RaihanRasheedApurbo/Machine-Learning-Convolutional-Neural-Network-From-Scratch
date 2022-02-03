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

class Pooling:
    def __init__(self, height, width, stride, debug=logging.ERROR) -> None:
        self.height = height
        self.width = width
        self.stride = stride

        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(debug)
        self.logger.addHandler(stream_handler)
    
    def forward(self, input_matrix):
        self.logger.info(input_matrix.shape)
        self.forward_input = input_matrix
        s = self.stride
        w = self.width
        h = self.height
        f,h1,w1,d = input_matrix.shape
        h2,w2 = int((h1-h)/s) + 1, int((w1-w)/s) + 1
        output_matrix = np.zeros((f,h2,w2,d))
        for i in range(f):
            for j in range(h2):
                prev_row_start = j*s
                prev_row_finish = prev_row_start + h
                for k in range(w2):
                    prev_col_start = k*s
                    prev_col_finish = prev_col_start + w
                    for l in range(d):
                        output_matrix[i][j][k][l] = np.max(input_matrix[i,prev_row_start:prev_row_finish,prev_col_start:prev_col_finish,l])

        return output_matrix

    def backward(self, input_matrix):
        forward_input = self.forward_input
        f,h1,w1,d = forward_input.shape
        output_matrix = np.zeros((f,h1,w1,d))
        f,h2,w2,d = input_matrix.shape
        h,w,s = self.height, self.width, self.stride

        for i in range(f):
            for j in range(h2):
                prev_row_start = j*s
                prev_row_finish = prev_row_start + h
                for k in range(w2):
                    prev_col_start = k*s
                    prev_col_finish = prev_col_start + w
                    for l in range(d):
                        window = forward_input[i,prev_row_start:prev_row_finish,prev_col_start:prev_col_finish,l]
                        max_v = np.max(window)
                        mask = (window == max_v)

                        output_matrix[i,prev_row_start:prev_row_finish,prev_col_start:prev_col_finish,l] += mask * input_matrix[i,j,k,l]

        return output_matrix

class Convolution:
    
    def __init__(self, height, width, stride, total_filters, color_channels, pad, debug=logging.ERROR) -> None:
        self.height = height
        self.width = width
        self.stride = stride
        self.total_filters = total_filters
        self.color_channels = color_channels
        self.pad = pad

        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(debug)
        self.logger.addHandler(stream_handler)

        self.weight_matrix = np.random.randn(height, width, color_channels, total_filters)
        self.bias_matrix = np.random.randn(total_filters)

        self.bias_matrix = np.zeros(2)

        self.weight_matrix = np.arange(54).reshape((3,3,3,2))



    def forward(self, input_matrix):
        self.forward_input = input_matrix
        s,p = self.stride, self.pad

        input_matrix_with_pad = np.pad(input_matrix, ((0,0),(p,p),(p,p),(0,0)))

        s1,h1,w1,ch1 = input_matrix.shape
        h2,w2,ch2,tf2 = self.weight_matrix.shape
        
    
        h3,w3 = int((h1+2*p-h2)/s)+1,int((w1+2*p-w2)/s)+1
        output_matrix = np.zeros((s1,h3,w3,tf2))
        

        for i in range(s1):
            for j in range(h3):
                row_start = j*s
                row_end = row_start + h2
                for k in range(w3):
                    col_start = k*s
                    col_end = col_start + w2
                    for l in range(tf2):
                        arr_slice = input_matrix_with_pad[i,row_start:row_end,col_start:col_end,:]
                        weight = self.weight_matrix[:,:,:,l]
                        bias = self.bias_matrix[l]
                        y = arr_slice * weight
                        y_sum = np.sum(y)

                        output_matrix[i,j,k,l] = y_sum + bias

        return output_matrix

    def backward(self, input_matrix, learning_rate=1):
        forward_input = self.forward_input
        weight_matrix = self.weight_matrix
        bias_matrix = self.bias_matrix
        s,p = self.stride, self.pad

        
        
        forward_input_derivative = np.zeros(forward_input.shape)
        weight_matrix_derivative = np.zeros(weight_matrix.shape)
        bias_matrix_derivative = np.zeros(bias_matrix.shape)

        forward_input_with_pad = np.pad(forward_input, ((0,0),(p,p),(p,p),(0,0)))
        forward_input_derivative_with_pad = np.pad(forward_input_derivative, ((0,0),(p,p),(p,p),(0,0)))

        (s1,h1,w1,c1) = input_matrix.shape

        h2,w2,c2,t2 = weight_matrix.shape
        
        for i in range(s1):
            for j in range(h1):
                row_start = j*s
                row_end = row_start + h2
                for k in range(w1):
                    col_start = k*s
                    col_end = col_start + w2
                    for l in range(c1):
                        bias_matrix_derivative[l] += input_matrix[i,j,k,l]
                        forward_input_derivative_with_pad[i,row_start:row_end,col_start:col_end,:] += weight_matrix[:,:,:,l] * input_matrix[i,j,k,l]

                        slice_of_forward_input_pad = forward_input_with_pad[i,row_start:row_end,col_start:col_end,:]
                        weight_matrix_derivative[:,:,:,l] += slice_of_forward_input_pad * input_matrix[i,j,k,l]
            forward_input_derivative[i,:,:,:] = forward_input_derivative_with_pad[i,p:-p,p:-p:,:]

        self.bias_matrix -= learning_rate * bias_matrix_derivative
        self.weight_matrix -= learning_rate * weight_matrix_derivative

        return forward_input_derivative 

        

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

    # np.random.seed(1)
    
    # output = np.random.randn(1,5,5,3)
    # logger.info(output)
    # p = Pooling(2,2,1)
    # logger.info(output)
    # output = p.forward(output)
    # logger.info(output.shape)
    # logger.info(output)
    # output = np.random.randn(1,4,4,3)
    # logger.info((output))
    # output = p.backward(output)
    
    # logger.info(output)

#             np.random.seed(1)
# A_prev = np.random.randn(10,5,7,4)
# W = np.random.randn(3,3,4,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 1,
#                "stride": 2}

# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("Z's mean =\n", np.mean(Z))
# print("Z[3,2,1] =\n", Z[3,2,1])
# print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])

    # np.random.seed(1)
    # output = np.random.randn(10,4,4,3)
    # c = Convolution(2,2,2,8,3,2)
    # output = c.forward(output)
    # # logger.info((output))
    # # logger.info(output.shape)
    # # logger.info(output[3,2,1])
    # # logger.info()
    # a,b,c = c.backward(output)
    # logger.info(np.mean(a))
    # logger.info(np.mean(b))
    # logger.info(np.mean(c))


    # We'll run conv_forward to initialize the 'Z' and 'cache_conv",
# which we'll use to test the conv_backward function
# np.random.seed(1)
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 2,
#                "stride": 2}
# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

# # Test conv_backward
# dA, dW, db = conv_backward(Z, cache_conv)
# print("dA_mean =", np.mean(dA))
# print("dW_mean =", np.mean(dW))
# print("db_mean =", np.mean(db))
    # input_arr = np.arange(75).reshape(1,5,5,3)
    # c = Convolution(3,3,2,2,3,1)
    # output = c.forward(input_arr)
    # logger.info(output)
    # d_output = np.arange(18).reshape((1,3, 3, 2))
    # output = c.backward(d_output)
    # logger.info(output)



    





