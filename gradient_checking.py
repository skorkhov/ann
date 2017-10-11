import numpy as np
from ann_structure import L_forward
from ann_utils import cost_total


def dict_to_vector(d): # correct
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
    
    vector = np.array([], dtype=np.float128)
    vector = vector.reshape(-1, 1)
    struct = {}
    
    keys = sorted(d.keys())
    
    for key in keys:
        struct[key] = d[key].shape
        temp = d[key].reshape(-1, 1)
        vector = np.append(vector, temp, axis=0)
    
    return vector, struct


def vector_to_dict(vector, struct): # correct
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
    keys = sorted(struct.keys())
    
    for key in keys:
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
    
    param_vector, param_struct = dict_to_vector(parameters)
    n_params = len(param_vector)
    cost_plus = np.zeros((n_params, 1))
    cost_minus = np.zeros((n_params, 1))
    grad_approx = np.zeros((n_params, 1))
    
    for i in range(n_params):
        p_plus = np.copy(param_vector)
        p_plus[i][0] += epsilon
        params_plus = vector_to_dict(p_plus, param_struct)
        AL_plus, _ = L_forward(X, params_plus)
        cost_plus[i][0] = cost_total(AL_plus, Y)
        
        p_minus = np.copy(param_vector)
        p_minus[i][0] -= epsilon
        params_minus = vector_to_dict(p_minus, param_struct)
        AL_minus, _ = L_forward(X, params_minus)
        cost_minus[i][0] = cost_total(AL_minus, Y)
        
        grad_approx[i][0] = (cost_plus[i][0] - cost_minus[i][0]) / (2 * epsilon)
    
    return grad_approx


def gradient_difference(grad_approx, grad_exact, epsilon=1e-7): # correct
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
    
    if difference > epsilon:
        print('Possibility of Error: ' + str(difference))
    else:
        print('Likely Correct ' + str(difference))
    
    return difference
