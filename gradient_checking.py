import numpy as np
from ann_structure import L_forward
from ann_utils import cost_total


def dict_to_vector(d):
    """
    Converts a dictionary with numpy array values into a vector with
    concatenated flattened arrays.
    
    Inputs:
    d -- dictionary with numpy array keys.
    
    Outputs:
    vector -- vector array created from the dictionary values
    struct -- dictionary with the structure of the original dictionary d;
        struct.keys() = d.keys()
        struct[key] = d[key].shape
    """
    
    vector = np.array([])
    struct = {}
    
    keys = set(d.keys())
    
    for key in keys:
        struct[key] = d[key].shape
        temp = d[key].flatten()
        vector = np.append(vector, temp)
    
    return vector, struct


def vector_to_dict(vector, struct):
    """
    Converts a vector to a dictionary according to a structure defined in struct.
    Inputs:
    vector -- vector array
    struct -- dictionary defining the structure of the output dictionary;
        struct[key] = d.shape
        
    Outputs:
    d -- dictionary created from vector according to the structure struct
    """
    
    d = {}
    
    for key in set(struct.keys()):
        length = np.prod(struct[key])
        
        val = vector[:length].reshape(struct[key])
        d[key] = val
        
        vector = vector[length:]
    
    return d


def approximate_gradient(parameters, X, Y, epsilon=1e-7):
    """
    Approximates the gradient of logLoss function with respect to parameters
    
    Inputs:
    parameters -- dictionary with parameters W and b for all layers
    X -- input data
    Y -- ground truth labels of the data
    epsilon (optional) -- small number to approximate the derivative, defaults to 1e-7.
    
    Outputs:
    grad_approx -- vector array with approximate gradient.
    """
    
    p, p_struct = dict_to_vector(parameters)
    n_params = len(p)
    Jp = np.zeros((n_params, 1))
    Jm = np.zeros((n_params, 1))
    grad_approx = np.zeros((n_params, 1))
    
    for i in range(n_params):
        pp = np.copy(p)
        pp[i] += epsilon
        AL, _ = L_forward(X, vector_to_dict(pp, p_struct))
        Jp[i] = cost_total(AL, Y)
        
        pm = np.copy(p)
        pm[i] -= epsilon
        AL, _ = L_forward(X, vector_to_dict(pm, p_struct))
        Jm[i] = cost_total(AL, Y)
        
        grad_approx[i] = (Jp[i] - Jm[i]) / (2 * epsilon)
    
    return grad_approx


def gradient_difference(grad_approx, grad_exact):
    """
    Computes the difference between the approximate and exact gradient
    
    Inputs:
    grad_approx:ndarray -- approximate gradient of logLoss w/r to parameters W, b
    grad_exact:ndarray -- gradient of logLoss w/r to parameters W, b
    
    Outputs:
    difference -- e measure of how close the input parameters are.
    """
    
    assert(grad_approx.shape == grad_exact.shape)
    
    difference = np.linalg.norm(grad_exact - grad_approx) / (
    np.linalg.norm(grad_exact) + np.linalg.norm(grad_approx))
    
    if difference > 1e-7:
        print('Possibility of Error: ' + str(difference))
    else:
        print('Like Correct ' + str(difference))
    
    return difference
