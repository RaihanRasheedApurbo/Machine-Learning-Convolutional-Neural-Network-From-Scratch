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
        
        # self.logger.info(bias_matrix.shape)
        # self.logger.info(bias_matrix_derivative.shape)
        # self.logger.info(weight_matrix.shape)
        # self.logger.info(weight_matrix_derivative.shape)

        # self.bias_matrix -= learning_rate * bias_matrix_derivative
        # self.weight_matrix = self.weight_matrix - learning_rate * weight_matrix_derivative # -= operator throws error # details https://techoverflow.net/2019/05/22/how-to-fix-numpy-typeerror-cannot-cast-ufunc-subtract-output-from-dtypefloat64-to-dtypeint64-with-casting-rule-same_kind/

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
        sample_size = self.shape[0]
        inner_data = self.shape[1]*self.shape[2]*self.shape[3]
        return np.reshape(input_matrix,(sample_size,inner_data))
       

    def backward(self, input_matrix):
        return np.reshape(input_matrix, self.shape)
    
def soft_max(input_matrix):
    sample_size, prediction = input_matrix.shape
    arr = []
    for i in range(sample_size):

        exp = np.exp(input_matrix[i])
        sum = np.sum(exp)
        normalize = exp/sum
        arr.append(normalize)
    
    return np.array(arr)

class FullConnectedLayer:
    def __init__(self, output_size, debug=logging.ERROR, weight_matrix=None, bias_matrix=None):
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(debug)
        self.logger.addHandler(stream_handler)

        self.flatten = Flatten() # creating a flatten object if input is not properly flatten
        self.output_size = output_size



        if weight_matrix:
            self.weight_matrix = weight_matrix
        
        
        if bias_matrix:
            self.bias_matrix = bias_matrix
       
            

    
    def forward(self, input_matrix):



        flatten_matrix = self.flatten.forward(input_matrix)
        
        sample_size,data_size = flatten_matrix.shape

        if not hasattr(self, 'weight_matrix'):
            self.weight_matrix = np.random.randn(self.output_size, data_size)
        if not hasattr(self, 'bias_matrix'):
            self.bias_matrix = np.random.randn(self.output_size,1) # 1 for making column matrix
        
        x = flatten_matrix.T   # making inputs column vectors
        self.input_matrix = x # stroring for backward prop.

        if x.shape[0] != self.weight_matrix.shape[1]:
            raise 'Full Connected layer error shape mismatch in forward propagation'
        
        self.logger.info(self.weight_matrix)
        self.logger.info(x)
        wx = np.matmul(self.weight_matrix, x)
        

        self.logger.info(wx)
        self.logger.info(self.bias_matrix)
        
        y = wx + self.bias_matrix
        
        logger.info(y)
        return  y


    def backward(self, output_gradiant, learning_rate=1):
        bias_gradient = np.average(output_gradiant,axis=1) # row matrix
        bias_gradient = np.reshape(bias_gradient,(bias_gradient.shape[0],1)) # making column matrix again
        weight_gradient = np.matmul(output_gradiant, self.input_matrix.T)
        input_gradient = np.matmul(self.weight_matrix.T, output_gradiant)
        input_gradient = input_gradient.T  # converting column matrix to row matrix

        self.logger.info(bias_gradient)
        self.logger.info(weight_gradient)
        self.logger.info(input_gradient)

        self.logger.info('prev weights')
        self.logger.info(self.weight_matrix)

        self.logger.info('prev bias')
        self.logger.info(self.bias_matrix)

        self.bias_matrix -= learning_rate * bias_gradient
        self.weight_matrix -= learning_rate * weight_gradient

        self.logger.info('bias')
        self.logger.info(self.bias_matrix)

        self.logger.info('weights')
        self.logger.info(self.weight_matrix)

        return self.flatten.backward(input_gradient) # incase input wasn't flattend
    
        
    

if __name__ == '__main__':
    

    ##### fc and relu test #####

    np.random.seed(1)
    f = FullConnectedLayer(4)
    output = np.random.randn(2,2,2,3)
    logger.info(output)
    output = f.forward(output)
    logger.info(output)
    output = f.backward(output)
    logger.info(output)
    # r = ReLU()
    # output = r.forward(output)
    # logger.info(output)
    # output = r.backward(output)
    # logger.info(output)




    ##### soft max test #####
    # output = soft_max(output)
    # logger.info(output)

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


   

    ### convolution forward test #####
    # np.random.seed(1)
    # output = np.random.randn(10,5,7,4)
    # c = Convolution(3,3,2,8,4,1,debug=logging.INFO)
    # output = c.forward(output)
    # logger.info(np.mean(output))  # expected 0.6923608807576933


    #### convolution backward test #####


    # np.random.seed(1)
    # output = np.random.randn(10,4,4,3)
    # c = Convolution(2,2,2,8,3,2,debug=logging.INFO)
    # output = c.forward(output)
    
    # a = c.backward(output)
    # logger.info(np.mean(a))   # expected 1.4524377775388075
    # # logger.info(np.mean(b))
    # # logger.info(np.mean(c))


   



    ###### mnist data import test ######
    # from keras.datasets import mnist
    # from matplotlib import pyplot
    
    # #loading
    # (train_X, train_y), (test_X, test_y) = mnist.load_data()
    
    # #shape of dataset
    # print('X_train: ' + str(train_X.shape))
    # print('Y_train: ' + str(train_y.shape))
    # print('X_test:  '  + str(test_X.shape))
    # print('Y_test:  '  + str(test_y.shape))
    
    # #plotting
    # from matplotlib import pyplot
    # for i in range(9):  
    #     pyplot.subplot(330 + 1 + i)
    #     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    # pyplot.show()


    ##### flatten test #####
    # np.random.seed(1)
    # output = np.random.randn(2,28,28,3)
    # a = output
    # logger.info(output.shape)
    # f = Flatten()
    # output = f.forward(output)
    # logger.info(output.shape)
    # output = f.backward(output)
    # logger.info(output.shape)
    # logger.info(np.sum(a==output)==2*28*28*3)
    


    





