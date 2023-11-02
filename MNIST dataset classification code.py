"""
Modified Multilayer Perceptron: A New Architecture
This code is based on Charanjeet Singh code "Digit_recognition_numpy"
https://github.com/charan89/Digit_recognition_numpy.git
which we modify it for our new architecture. Please visit link above for more 
information about Singh code.
"""
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import sklearn
import sklearn.datasets
import scipy
from PIL import Image
from scipy import ndimage


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)

X_train = np.array(X_train)/255
X_test = np.array(X_test)/255

y_train = np.array(y_train)
y_test = np.array(y_test)


def one_hot(j):
    # input is the target dataset of shape (m,) where m is the number of data points
    # returns a 2 dimensional array of shape (10, m) where each target value is converted to a one hot encoding
    # Look at the next block of code for a better understanding of one hot encoding
    n = j.shape[0]
    new_array = np.zeros((10, n))
    index = 0
    for res in j:
        new_array[int(res)][index] = 1.0
        index = index + 1
    return new_array

def data_wrapper():
    
    training_inputs = np.array(X_train[:]).T
    training_results = np.array(y_train[:])
    train_set_y = one_hot(training_results)
    
    
    test_inputs = np.array(X_test[:]).T
    test_results = np.array(y_test[:])
    test_set_y = one_hot(test_results)
    
    return (training_inputs, train_set_y, test_inputs, test_set_y)

train_set_x, train_set_y, test_set_x, test_set_y = data_wrapper()


y = pd.DataFrame(train_set_y)


def sigmoid(Z):
    
    # Z is numpy array of shape (n, m) where n is number of neurons in the layer and m is the number of samples 
    # sigmoid_memory is stored as it is used later on in backpropagation
    
    H = 1/(1+np.exp(-Z))
    sigmoid_memory = Z
    
    return H, sigmoid_memory

def relu(Z):
    # Z is numpy array of shape (n, m) where n is number of neurons in the layer and m is the number of samples 
    # relu_memory is stored as it is used later on in backpropagation
    
    H = np.maximum(0,Z)
    
    assert(H.shape == Z.shape)
    
    relu_memory = Z 
    return H, relu_memory

def softmax(Z):
    # Z is numpy array of shape (n, m) where n is number of neurons in the layer and m is the number of samples 
    # softmax_memory is stored as it is used later on in backpropagation
   
    Z_exp = np.exp(Z)

    Z_sum = np.sum(Z_exp,axis = 0, keepdims = True)
    
    H = Z_exp/Z_sum  #normalising step
    softmax_memory = Z
    
    return H, softmax_memory

def initialize_parameters(dimensions):

    # dimensions is a list containing the number of neuron in each layer in the network
    # It returns parameters which is a python dictionary containing the parameters "W1", "A1", ..., "WL", "AL":

    np.random.seed(2)
    parameters = {}
    L = len(dimensions)            # number of layers in the network + 1

    for l in range(1, L): 
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l-1]) * 0.1
        parameters['A' + str(l)] = np.random.randn(dimensions[l], 3) * 0.1        
        assert(parameters['W' + str(l)].shape == (dimensions[l], dimensions[l-1]))
        assert(parameters['A' + str(l)].shape == (dimensions[l], 3))

        
    return parameters


def layer_forward(H_prev, W, A, activation = 'relu'):

    # H_prev is of shape (size of previous layer, number of examples)
    # W is weights matrix of shape (size of current layer, size of previous layer)
    # A is Coefficient matrix of shape (size of the current layer, 3)
    # activation is the activation to be used for forward propagation : "softmax", "relu", "sigmoid"

    # H is the output of the activation function 
    # memory is a python dictionary containing "linear_memory" and "activation_memory"
    
    if activation == "sigmoid":
        z = np.dot(W,H_prev)
        linear_memory = (H_prev, W, A)
        zz = np.multiply(z,z)
        A_2 = np.array(A[:,2])
        A_1 = np.array(A[:,1])
        A_0 = np.array(A[:,0])
        Z = (zz.T*A_2).T + (z.T*A_1).T + (np.ones(z.shape).T*A_0).T 
        H, activation_memory = sigmoid(Z)
 
    elif activation == "softmax":
        z = np.dot(W,H_prev)
        linear_memory = (H_prev, W, A)
        zz = np.multiply(z,z)
        A_2 = np.array(A[:,2])
        A_1 = np.array(A[:,1])
        A_0 = np.array(A[:,0])
        Z = (zz.T*A_2).T + (z.T*A_1).T + (np.ones(z.shape).T*A_0).T 
        H, activation_memory = softmax(Z)
    
    elif activation == "relu":
        z = np.dot(W,H_prev)
        linear_memory = (H_prev, W, A)
        zz = np.multiply(z,z)
        A_2 = np.array(A[:,2])
        A_1 = np.array(A[:,1])
        A_0 = np.array(A[:,0])
        Z = 1/2 * (zz.T*A_2).T + (z.T*A_1).T + (np.ones(z.shape).T*A_0).T 
        H, activation_memory = relu(Z)
        
    assert (H.shape == (W.shape[0], H_prev.shape[1]))
    memory = (linear_memory, activation_memory)

    return H, memory

def L_layer_forward(X, parameters):

    # X is input data of shape (input size, number of examples)
    # parameters is output of initialize_parameters()
    
    # HL is the last layer's post-activation value
    # memories is the list of memory containing (for a relu activation, for example):
    # - every memory of relu forward (there are L-1 of them, indexed from 1 to L-1), 
    # - the memory of softmax forward (there is one, indexed L) 

    memories = []
    H = X
    L = len(parameters) // 2                  # number of layers in the neural network
    Al=parameters['A'+str(L)]
    wl=parameters['W'+str(L)]
    # Implement relu layer (L-1) times as the Lth layer is the softmax layer
    for l in range(1, L):
        wn=parameters['W'+str(l)]
        An=parameters['A'+str(l)]
        H_prev=H
        H,memory=layer_forward(H_prev,wn,An,activation="relu")
        memories.append(memory)
    # Implement the final softmax layer
    # HL here is the final prediction P as specified in the lectures
    HL, memory = layer_forward(H,wl,Al,activation="softmax")
    
    memories.append(memory)

    assert(HL.shape == (10, X.shape[1]))
            
    return HL, memories

def compute_loss(HL, Y):


    # HL is probability matrix of shape (10, number of examples)
    # Y is true "label" vector shape (10, number of examples)

    # loss is the cross-entropy loss

    m = Y.shape[1]
    loss=-(1/m)*np.sum((Y*np.log(HL)))
    loss = np.squeeze(loss)      # To make sure that the loss's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())
    
    return loss


def sigmoid_backward(dH, sigmoid_memory):
    
    # Implement the backpropagation of a sigmoid function
    # dH is gradient of the sigmoid activated activation of shape same as H or Z in the same layer    
    # sigmoid_memory is the memory stored in the sigmoid(Z) calculation
    
    Z = sigmoid_memory
    
    H = 1/(1+np.exp(-Z))
    dZ = dH * H * (1-H)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dH, relu_memory):
    
    # Implement the backpropagation of a relu function
    # dH is gradient of the relu activated activation of shape same as H or Z in the same layer    
    # relu_memory is the memory stored in the relu(Z) calculation
    
    Z = relu_memory
    dZ = np.array(dH, copy=True) # dZ will be the same as dH wherever the elements of H weren't 0
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def layer_backward(dH, memory, activation = 'relu'):
    
    
    # takes dH and the memory calculated in layer_forward and activation as input to calculate the dH_prev, dW, dA
    # performs the backprop depending upon the activation function
    

    linear_memory, activation_memory = memory
    
    if activation == "relu":
        dZ = relu_backward(dH,activation_memory)
        H_prev, W, A = linear_memory
        m = H_prev.shape[1]
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dH,activation_memory)
        H_prev, W, A = linear_memory
        m = H_prev.shape[1]
     
        
    z = np.dot(W,H_prev)
    zz = np.multiply(z,z)
    A_2 = np.array(A[:,2])
    A_1 = np.array(A[:,1])
    K = (z.T*A_2).T + (np.ones(z.shape).T*A_1).T 
    dW1 = K * dZ
    dW = (1./m) * np.dot(dW1,H_prev.T)
    dA_2 = (1./m)* np.sum( 1/2 *(zz * dZ ),axis=1,keepdims=True )
    dA_1 = (1./m)* np.sum(z * dZ,axis=1,keepdims=True )
    dA_0 = (1./m)* np.sum( dZ,axis=1,keepdims=True )
    dA = np.column_stack((dA_0,dA_1,dA_2))
    dH_prev1 = dZ * K
    dH_prev = np.dot(W.T,dH_prev1)
    
    return dH_prev, dW, dA


def L_layer_backward(HL, Y, memories):
    
    # Takes the predicted value HL and the true target value Y and the 
    # memories calculated by L_layer_forward as input
    
    # returns the gradients calulated for all the layers as a dict

    gradients = {}
    L = len(memories) # the number of layers
    m = HL.shape[1]
    Y = Y.reshape(HL.shape) # after this line, Y is the same shape as HL
    # Perform the backprop for the last layer that is the softmax layer
    current_memory = memories[L-1]
    linear_memory, activation_memory = current_memory
    dZ = HL - Y
    H_prev, W, A = linear_memory
    z = np.dot(W,H_prev)
    zz = np.multiply(z,z)
    A_2 = np.array(A[:,2])
    A_1 = np.array(A[:,1])
    K = (z.T*A_2).T + (np.ones(z.shape).T*A_1).T 
    dW1 = K * dZ
    dA_2 = (1./m)* np.sum( 1/2 *(zz * dZ ),axis=1,keepdims=True )
    dA_1 = (1./m)* np.sum(z * dZ,axis=1,keepdims=True )
    dA_0 = (1./m)* np.sum( dZ,axis=1,keepdims=True )
    dH_prev1 = dZ * K   
    gradients["dW" + str(L)] = (1./m) * np.dot(dW1,H_prev.T)
    gradients["dA" + str(L)] = np.column_stack((dA_0,dA_1,dA_2)) 
    gradients["dH" + str(L-1)] = np.dot(W.T,dH_prev1) 
    # Perform the backpropagation l-1 times
    for l in reversed(range(L-1)):
        # Lth layer gradients: "gradients["dH" + str(l + 1)] ", gradients["dW" + str(l + 2)] , gradients["dA" + str(l + 2)]
        current_memory = memories[l]
        
        dH_prev_temp, dW_temp, dA_temp = layer_backward(gradients["dH"+str(l+1)],current_memory,"relu")
        gradients["dH" + str(l)] = dH_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["dA" + str(l + 1)] = dA_temp


    return gradients

def update_parameters(parameters, gradients, learning_rate):

    # parameters is the python dictionary containing the parameters W and A for all the layers
    # gradients is the python dictionary containing your gradients, output of L_model_backward
    
    # returns updated weights after applying the gradient descent update

    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * gradients["dW"+ str(l+1)]
        parameters["A" + str(l+1)] = parameters["A" + str(l+1)] - learning_rate * gradients["dA"+ str(l+1)]

        
    return parameters

dimensions = [784,10] 

def L_layer_model(X, Y, dimensions, learning_rate = 0.1, num_iterations = 3000, print_loss=False):
    
    # X and Y are the input training datasets
    # learning_rate, num_iterations are gradient descent optimization parameters
    # returns updated parameters

    np.random.seed(2)
    losses = []                         # keep track of loss
    
    # Parameters initialization
    parameters = initialize_parameters(dimensions)
 
    for i in range(0, num_iterations):

        # Forward propagation
        HL, memories = L_layer_forward(X,parameters)
        
        # Compute loss
        loss = compute_loss(HL,Y)
    
        # Backward propagation
        gradients = L_layer_backward(HL,Y, memories)
 
        # Update parameters.
        parameters = update_parameters(parameters, gradients, learning_rate)
        # Printing the loss every 100 training example
        if print_loss and i % 100 == 0:
            print ("Loss after iteration %i: %f" %(i, loss))
            losses.append(loss)
    # plotting the loss
    plt.plot(np.squeeze(losses))
    plt.ylabel('loss')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X, y, parameters):
    
    # Performs forward propogation using the trained parameters and calculates the accuracy
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    
    # Forward propagation
    probas, caches = L_layer_forward(X, parameters)
    
    p = np.argmax(probas, axis = 0)
    act = np.argmax(y, axis = 0)

    print("Accuracy: "  + str(round(np.sum((p == act)/m),3)))
        
    return p

train_set_x_new = train_set_x[:,0:50000]
train_set_y_new = train_set_y[:,0:50000]
train_set_x_new.shape

parameters = L_layer_model(train_set_x_new, train_set_y_new, dimensions, num_iterations = 10000, print_loss = True)
pred_train = predict(train_set_x_new, train_set_y_new, parameters)
pred_test = predict(test_set_x, test_set_y, parameters)



