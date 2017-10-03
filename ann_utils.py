
import numpy as np


def relu(Z):
    """
    Computes element wise ReLU of the input, where
    relu(x) = max(0, x)

    Input:
    Z -- np.array of shape (layer_size, n_samples)

    Output:
    A -- np.array with elementwise ReLU's of Z
    cache -- Z, stored for later use in backprop
    """

    A = np.maximum(0, Z)
    cache = Z
    
    return A, cache


def sigmoid(Z):
    """
    Computes elementwise Sigmoid of the imput

    Input:
    Z -- np.array of shape (layer_size, n_samples)

    Output:
    A -- np.array with elementwise sigmoids of Z
    cache -- Z, stored for later use in backprop
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z
    assert(A.shape == Z.shape)

    return A, cache


def softmax(Z):
    """
    Computes softmax of Z
    
    Inputs:
    Z -- np.array of shape (layer_size, n_samples)
    
    Outputs:
    y -- np.array of shape (n_classes, n_samples)
    """
    
    z_exp = np.exp(Z)
    sum_z_exp = np.sum(z_exp, axis=0, keepdims=True)

    assert(sum_z_exp.shape == (1, Z.shape[1]))
    
    A = z_exp / sum_z_exp

    cache = Z
    assert(A.shape == Z.shape)
    
    return A, cache


def relu_backward(dA, cache):
    """
    Computes the deriative of ReLU.

    Inputs:
    dA -- np.array with the derivative of the cost w/respect to the output A of a layer
    cache = np.array Z from forward prop

    dZ -- np.array with the derivative of the cost w/respect to Z of the layer
    """

    Z = cache

    assert(dA.shape == Z.shape)
    
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    Computes the deriative of sigmoid.

    Inputs:
    dA -- np.array with the derivative of the cost w/respect to the output A of a layer
    cache -- np.array Z from forward prop

    dZ -- np.array with the derivative of the cost w/respect to Z of the layer
    """
    
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ


def softmax_backward(dA, cache):
    """
    Computes the deriative of softmax.

    Inputs:
    dA -- np.array with the derivative of the cost w/respect to the output A of a layer
    cache -- np.array Z from forward prop

    Outputs:
    dZ -- np.array with the derivative of the cost w/respect to Z of the layer
    """
    
    Z = cache
    
    A, cache = softmax(Z)
    
    dZ = dA * A * (1 - A)
    
    return dZ


def cost_total(AL, Y):
    """
    Computes the cost function associated with a set of predictions AL and ground truth labels Y.

    Inputs:
    AL -- np.array with predictions, AL.shape = (output_size, n_samples)
    Y -- np.array with ground truth labels, Y.shape = (output_size, n_samples)

    Outputs:
    cost -- float, logLoss cost associated with the predictions and for the correct labels.
    """
    
    assert(AL.shape == Y.shape)
    
    m = AL.shape[1]
    
    loss = np.log(np.sum(AL*Y, axis=0, keepdims=True))
    cost = (-1 / m) * np.sum(loss)
    
    return cost
