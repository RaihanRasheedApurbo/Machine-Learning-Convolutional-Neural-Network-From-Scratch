import logging
import random
import math
from tkinter import N
import numpy as np

from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# logger initialization
formatter = logging.Formatter(
    "\n*********Line no:%(lineno)d*********\n%(message)s\n***************************"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__file__)
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

    def backward(self, input_matrix, learning_rate=0.001):
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
    
    def __init__(self, height, width, stride, total_filters, pad, debug=logging.ERROR) -> None:
        self.height = height
        self.width = width
        self.stride = stride
        self.total_filters = total_filters
        self.pad = pad

        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(debug)
        self.logger.addHandler(stream_handler)

        

        



    def forward(self, input_matrix):
        self.forward_input = input_matrix
        s,p = self.stride, self.pad

        s1,h1,w1,ch1 = input_matrix.shape
        input_matrix_with_pad = np.pad(input_matrix, ((0,0),(p,p),(p,p),(0,0)))
        
        
        if not hasattr(self, 'weight_matrix'):
            self.weight_matrix = np.random.randn(self.height, self.width, ch1, self.total_filters)
        if not hasattr(self, 'bias_matrix'):
            self.bias_matrix = np.random.randn(self.total_filters)

        

        
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

    def backward(self, input_matrix, learning_rate=0.001):
        forward_input = self.forward_input
        weight_matrix = self.weight_matrix
        bias_matrix = self.bias_matrix
        s,p = self.stride, self.pad

        
        
        forward_input_derivative = np.zeros(forward_input.shape)
        weight_matrix_derivative = np.zeros(weight_matrix.shape)
        bias_matrix_derivative = np.zeros(bias_matrix.shape)

        forward_input_with_pad = np.pad(forward_input, ((0,0),(p,p),(p,p),(0,0)))
        forward_input_derivative_with_pad = np.pad(forward_input_derivative, ((0,0),(p,p),(p,p),(0,0)))
        _,row,col,_ =  forward_input_derivative_with_pad.shape # used for unfold padding later in loop


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
            forward_input_derivative[i,:,:,:] = forward_input_derivative_with_pad[i,p:row-p,p:col-p,:]
        
        # self.logger.info(bias_matrix.shape)
        # self.logger.info(bias_matrix_derivative.shape)
        # self.logger.info(weight_matrix.shape)
        # self.logger.info(weight_matrix_derivative.shape)

        self.bias_matrix -= learning_rate * bias_matrix_derivative
        self.weight_matrix = self.weight_matrix - learning_rate * weight_matrix_derivative # -= operator throws error # details https://techoverflow.net/2019/05/22/how-to-fix-numpy-typeerror-cannot-cast-ufunc-subtract-output-from-dtypefloat64-to-dtypeint64-with-casting-rule-same_kind/

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
    
    def backward(self, input_matrix, learning_rate=0.001):
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
       

    def backward(self, input_matrix, learning_rate=0.001):
        return np.reshape(input_matrix, self.shape)

class SoftMax:
    
    def __init__(self, debug=logging.ERROR) -> None:
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(debug)
        self.logger.addHandler(stream_handler)

    def forward(self, input_matrix):
        maxs = np.max(input_matrix, axis=0)
        maxs = np.reshape(maxs,(1,maxs.shape[0]))
        input_matrix = input_matrix - maxs
        exp = np.exp(input_matrix)
        sums = np.sum(exp,axis=0)
        sums = np.reshape(sums,(1,sums.shape[0]))
        return exp/sums

    def backward(self, input_matrix, learning_rate=0.001):
        return input_matrix
        

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
        
        self.logger.info(self.weight_matrix)
        self.logger.info(self.bias_matrix)

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
        
        self.logger.info(y)
        return  y


    def backward(self, output_gradiant, learning_rate=0.001):
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
    
    logger.setLevel(logging.WARNING)
    np.random.seed(1) # for reproducible training
    
    file = open('arch.txt','r')
    layers = []
    debug = logging.ERROR
    
    

    for line in file.readlines():
        words = line.split()
        logger.info(words)
        if words[0]=='Conv':
            height = int(words[2])
            width = height
            total_filters = int(words[1])
            stride = int(words[3])
            padding = int(words[4])
            c = Convolution(height,width,stride,total_filters,padding,debug)
            layers.append(c)
        elif words[0] == 'ReLU':
            layers.append(ReLU(debug))
        elif words[0] == 'Pool':
            height = int(words[1])
            width = height
            stride = int(words[2])
            p = Pooling(height, width, stride, debug)
            layers.append(p)
        elif words[0] == 'FC':
            output_size = int(words[1])
            fc = FullConnectedLayer(output_size, debug)
            layers.append(fc)
        elif words[0] == 'FL':
            layers.append(Flatten(debug))
        elif words[0] == 'Softmax':
            layers.append(SoftMax(debug))
        else:
            logger.error('invalid input')

    

     ###### mnist data import test ######
    from keras.datasets import mnist
    from keras.datasets import cifar10
    from matplotlib import pyplot
    from tensorflow.keras.utils import to_categorical # for one hot encoding
    
    model = 2
    color_channel = 3

    # #loading
    if model == 1:
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
    else:
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    max_pixel = 255
    train_x, test_x = train_x/max_pixel, test_x/max_pixel

    # logger.info(test_x[0])
    # logger.info(test_y[0])
    validation_x, test_x = np.split(test_x, 2)
    validation_y, test_y = np.split(test_y, 2)

     #shape of dataset
    logger.info('X_train: ' + str(train_x.shape))
    logger.info('Y_train: ' + str(train_y.shape))
    logger.info('X_test:  '  + str(test_x.shape))
    logger.info('Y_test:  '  + str(test_y.shape))
    logger.info('validation_x:  '  + str(validation_x.shape))
    logger.info('validation_y:  '  + str(validation_y.shape))
    
    sub_sample_ratio = 100
    test_data_count = len(train_x)
    batch_size = 32
    iteration_per_epoch = int(test_data_count/batch_size)
    iteration_per_epoch = int(iteration_per_epoch/sub_sample_ratio)
    total_epoch = 5
    learning_rate = 0.1

    output_class = 10 # total probable posibility of output of an input

    for i in range(total_epoch):
        for j in range(25):
            
            random_indices = np.random.choice(test_data_count, batch_size)
            output = train_x[random_indices] # renaming it output for using in loop
            if color_channel == 1:
                a,b,c = output.shape
                output = np.reshape(output, (a,b,c,1)) # adding another dimension
            test_this_iteration_y = train_y[random_indices]
            y_encode = to_categorical(test_this_iteration_y, num_classes= output_class, dtype="int").T
            layer_size = len(layers)

            for k in range(layer_size):
                logger.info(output.shape)
                output = layers[k].forward(output)
            

            output = output-y_encode # error dericative dE/dY

            for k in range(-1,-(layer_size+1),-1):
                logger.info(output.shape)
                output = layers[k].backward(output,learning_rate)
            logger.info(output.shape)
            # print(i,j)
            if j%10==0:
                print(i,j)

        
        
        validation_batch_size = 1000
        validation_data_count = len(validation_x)
        random_indices = np.random.choice(validation_data_count, validation_batch_size)
        output = validation_x[random_indices] # renaming it output for using in loop
        if color_channel == 1:
            a,b,c = output.shape
            output = np.reshape(output, (a,b,c,1)) # adding another dimension
        validation_samples_y = validation_y[random_indices]
        # y_encode = to_categorical(validation_samples_y, num_classes= output_class, dtype="int").T
        layer_size = len(layers)

        for k in range(layer_size):
            logger.info(output.shape)
            output = layers[k].forward(output)
        
        probabilities = output
        output = np.argmax(output, axis=0)  # predicted labels
        # print(len(output), len(validation_samples_y))

        
        f1 = f1_score(validation_samples_y, output, average='macro')
        print(f1)
        
        a1 = accuracy_score(validation_samples_y, output)
        # print(np.sum(validation_samples_y==output))
        print(a1)

        
        # print(validation_samples_y.shape, probabilities.T.shape)
        e1 = log_loss(validation_samples_y, probabilities.T)
        print(e1)

    
    logger.warning("completed")
    








    
    # #plotting
    # from matplotlib import pyplot
    # for i in range(9):  
    #     pyplot.subplot(330 + 1 + i)
    #     pyplot.imshow(train_x[i], cmap=pyplot.get_cmap('gray'))
    #     logger.info(train_y[i])
    # pyplot.show()


    ##### fc and relu test #####

    # np.random.seed(1)
    # f = FullConnectedLayer(4, debug=logging.INFO)
    # output = np.random.randn(2,2,2,3)
    # logger.info(output)
    # output = f.forward(output)
    # logger.info(output)
    # output = f.backward(output)
    # logger.info(output)
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
    # c = Convolution(3,3,2,8,1,debug=logging.INFO)
    # output = c.forward(output)
    # logger.info(np.mean(output))  # expected 0.6923608807576933


    #### convolution backward test #####


    # np.random.seed(1)
    # output = np.random.randn(10,4,4,3)
    # c = Convolution(2,2,2,8,2,debug=logging.INFO)
    # output = c.forward(output)
    
    # a = c.backward(output)
    # logger.info(np.mean(a))   # expected 1.4524377775388075
    # # logger.info(np.mean(b))
    # # logger.info(np.mean(c))

    #### polling layer test ######

    # np.random.seed(1)
    # output = np.random.randn(2,5,5,3)
    # p = Pooling(3,3,1)
    # output = p.forward(output)
    # logger.info(output.shape)
    # logger.info(output)  # first row [[[[ 1.74481176  0.90159072  1.65980218] # last row [ 1.62765075  1.12141771  0.79280687]]]]
                                                                               
    ##### another polling test #####
    # np.random.seed(1)
    # output = np.random.randn(2,5,5,3)
    # p = Pooling(3,3,2)
    # output = p.forward(output)
    # logger.info(output.shape)
    # logger.info(output)  # first row [[[[1.74481176 0.90159072 1.65980218] # last row [ 1.62765075  1.12141771  0.79280687]]]]

   
    #### polling backward test #####

    # np.random.seed(1)
    # output = np.random.randn(5,5,3,2)
    # p = Pooling(2,2,1)
    # output = p.forward(output)
    # d1 = np.random.randn(5,4,2,2)

    # output = p.backward(d1)
    # logger.info(output[1,1])


   


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
    


    ##### soft max test #####
    # np.random.seed(1)
    # output = np.random.randn(3,2)
    # logger.info(output)
    # s = SoftMax()
    # output = s.forward(output)
    # logger.info(output)
    #### another soft max test #####
    # a = np.arange(10).reshape((2,5))
    # b = SoftMax().forward(a)
    # logger.info(b)






