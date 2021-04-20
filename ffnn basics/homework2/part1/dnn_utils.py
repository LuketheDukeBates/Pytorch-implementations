import numpy as np
import torch


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+torch.exp(-Z))
    #A = torch.tensor(A, dtype=torch.double)
    A = A.double()
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
#     A = torch.max(torch.zeros(1,1, dtype=torch.float), torch.tensor(Z, dtype=torch.float))
#     A = torch.tensor(A, dtype=torch.float)
    
    A = torch.max(torch.zeros(1,1, dtype=torch.float), Z.float().clone().detach())
    A = A.double()
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    #dZ = torch.tensor(dA) # just converting dz to a correct object.
    dZ = dA.clone().detach()
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+torch.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

# def sigmoid(Z):
#     """
#     Implements the sigmoid activation in numpy
    
#     Arguments:
#     Z -- numpy array of any shape
    
#     Returns:
#     A -- output of sigmoid(z), same shape as Z
#     cache -- returns Z as well, useful during backpropagation
#     """
    
#     A = 1/(1+torch.exp(-Z))
#     A = torch.tensor(A, dtype=torch.double)
#     cache = Z
    
#     return A, cache

# def relu(Z):
#     """
#     Implement the RELU function.
#     Arguments:
#     Z -- Output of the linear layer, of any shape
#     Returns:
#     A -- Post-activation parameter, of the same shape as Z
#     cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
#     """
    
#     A = torch.max(torch.zeros(1,1, dtype=torch.float), torch.tensor(Z, dtype=torch.float))
#     A = torch.tensor(A, dtype=torch.float)
    
#     assert(A.shape == Z.shape)
    
#     cache = Z 
#     return A, cache


# def relu_backward(dA, cache):
#     """
#     Implement the backward propagation for a single RELU unit.
#     Arguments:
#     dA -- post-activation gradient, of any shape
#     cache -- 'Z' where we store for computing backward propagation efficiently
#     Returns:
#     dZ -- Gradient of the cost with respect to Z
#     """
    
#     Z = cache
#     dZ = torch.tensor(dA) # just converting dz to a correct object.
    
#     # When z <= 0, you should set dz to 0 as well. 
#     dZ[Z <= 0] = 0
    
#     assert (dZ.shape == Z.shape)
    
#     return dZ

# def sigmoid_backward(dA, cache):
#     """
#     Implement the backward propagation for a single SIGMOID unit.
#     Arguments:
#     dA -- post-activation gradient, of any shape
#     cache -- 'Z' where we store for computing backward propagation efficiently
#     Returns:
#     dZ -- Gradient of the cost with respect to Z
#     """
    
#     Z = cache
    
#     s = 1/(1+torch.exp(-Z))
#     dZ = dA * s * (1-s)
    
#     assert (dZ.shape == Z.shape)
    
#     return dZ
