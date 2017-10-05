
from ann_utils import *
import numpy as np


def init_params(layer_dims):
    """
    Initializess parameters according to He et al initialization.
    """
    
    params = {}
    L = len(layer_dims)

    for i in range(1, L):
        params['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01  # * np.sqrt(2 / layer_dims[i-1])
        params['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return params


def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2

    for i in range(L):
        parameters['W' + str(i + 1)] = parameters['W' + str(i + 1)] - learning_rate * grads['dW' + str(i + 1)]
        parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - learning_rate * grads['db' + str(i + 1)]

    return parameters


def linear_forward(A_prev, W, b):
    """
    Computes linear function of the parameters W and b, and previous
    activation A
    
    Inputs:
    A_prev = activation from the previous layer, A.shape = (n_(l-1), m)
    W = parameters of layer l, W.shape = (n_l, n_(l-1))
    b = bias of layer l, b.shape = (n_l, 1)
    
    Outputs:
    Z = computed linear function of the node, Z.shape = (n_l, m)
    cache = variable that stores parameters A_prev, Wl, bl to be used in back propargation
    """

    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache


def linear_backward(dZ, cache):
    
    A_prev, W, b = cache

    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def activation_forward(A_prev, W, b, activation='relu'):
    
    Z, lin_cache = linear_forward(A_prev, W, b)

    if activation == 'relu':
        A, activ_cache = relu(Z)

    if activation == 'sigmoid':
        A, activ_cache = sigmoid(Z)

    if activation == 'softmax':
        A, activ_cache = softmax(Z)

    cache = lin_cache, activ_cache  # cache = ((A_prev, W, b), Z)

    return A, cache


def activation_backward(dA, cache, activation):
    
    lin_cache, activ_cache = cache  # lin_cache = (A, W, b), activ_cache = Z

    if activation == 'relu':
        dZ = relu_backward(dA, activ_cache)

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activ_cache)

    if activation == 'softmax':
        dZ = softmax_backward(dA, activ_cache)

    dA_prev, dW, db = linear_backward(dZ, lin_cache)

    return dA_prev, dW, db


def L_forward(X, parameters):
    
    caches = []
    n_layers = len(parameters) // 2

    A_prev = X

    for layer in range(1, n_layers):
        W = parameters['W' + str(layer)]
        b = parameters['b' + str(layer)]
        Z, lin_cache = linear_forward(A_prev, W, b)
        A, activ_cache = relu(Z)

        A_prev = A

        caches.append((lin_cache, activ_cache))

    WL = parameters['W' + str(n_layers)]
    bL = parameters['b' + str(n_layers)]
    ZL, lin_cache = linear_forward(A_prev, WL, bL)
    AL, activ_cache = softmax(ZL)  #

    caches.append((lin_cache, activ_cache))

    return AL, caches


def L_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches)  # number of layers
    m = Y.shape[1]  # number of samples

    # initialize back propagation:
    dAL = (-1 / m) * np.divide(Y, AL)  # * S * (1 - S)

    current_cache = caches[L - 1]
    dA_prev, dW, db = activation_backward(dAL, current_cache, activation='softmax')
    grads['dW' + str(L)] = dW
    grads['db' + str(L)] = db

    for i in reversed(range(L - 1)):
        current_cache = caches[i]
        dA_prev, dW, db = activation_backward(dA_prev, current_cache, activation='relu')
        grads['dW' + str(i + 1)] = dW
        grads['db' + str(i + 1)] = db

    return grads
