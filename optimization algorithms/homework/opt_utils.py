import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
import scipy.io
import sklearn
import sklearn.datasets

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def load_params_and_grads(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    W1 = torch.randn(2,3, dtype=torch.double)
    b1 = torch.randn(2,1, dtype=torch.double)
    W2 = torch.randn(3,3, dtype=torch.double)
    b2 = torch.randn(3,1, dtype=torch.double)

    dW1 = torch.randn(2,3, dtype=torch.double)
    db1 = torch.randn(2,1, dtype=torch.double)
    dW2 = torch.randn(3,3, dtype=torch.double)
    db2 = torch.randn(3,1, dtype=torch.double)
    
    return W1, b1, W2, b2, dW1, db1, dW2, db2


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])
                    
    Tips:
    - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 
    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
    """
    
    np.random.seed(3)
    torch.manual_seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
#         parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*  np.sqrt(2 / layer_dims[l-1])
#         parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        factor = torch.sqrt(torch.tensor(2.0/layer_dims[l-1], dtype=torch.double))
        parameters['W' + str(l)] = torch.randn((layer_dims[l], layer_dims[l-1])) * factor
        parameters['b' + str(l)] = torch.zeros(layer_dims[l], 1)
        
#         assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
#         assert(parameters['W' + str(l)].shape == layer_dims[l], 1)
        
    return parameters


def compute_cost(A3, Y):
    
    """
    Implement the cost function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]
    
#     logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
#     cost = 1./m * np.sum(logprobs)
    logprobs = torch.mul(-torch.log(A3),Y) + torch.mul(-torch.log(1 - A3), 1 - Y)
    cost = 1./m * torch.sum(logprobs)
    
    return cost

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()
    
    Returns:
    loss -- the loss function (vanilla logistic loss)
    """
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
#     z1 = np.dot(W1, X) + b1
#     a1 = relu(z1)
#     z2 = np.dot(W2, a1) + b2
#     a2 = relu(z2)
#     z3 = np.dot(W3, a2) + b3
#     a3 = sigmoid(z3)
    
#     cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    X = torch.tensor(X, dtype=torch.float)
    Z1 = torch.matmul(W1, X) + b1
    A1 = torch.relu(Z1)
    Z2 = torch.matmul(W2, A1) + b2
    A2 = torch.relu(Z2)
    Z3 = torch.matmul(W3, A2) + b3
    A3 = torch.sigmoid(Z3)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
#     dz3 = 1./m * (a3 - Y)
#     dW3 = np.dot(dz3, a2.T)
#     db3 = np.sum(dz3, axis=1, keepdims = True)
    
#     da2 = np.dot(W3.T, dz3)
#     dz2 = np.multiply(da2, np.int64(a2 > 0))
#     dW2 = np.dot(dz2, a1.T)
#     db2 = np.sum(dz2, axis=1, keepdims = True)
    
#     da1 = np.dot(W2.T, dz2)
#     dz1 = np.multiply(da1, np.int64(a1 > 0))
#     dW1 = np.dot(dz1, X.T)
#     db1 = np.sum(dz1, axis=1, keepdims = True)
    
#     gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
#                  "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
#                  "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    dZ3 = (A3 - Y) / m
    dW3 = torch.matmul(dZ3, A2.T) / m
    db3 = torch.sum(dZ3, dim=1, keepdims = True) / m
    
    dA2 = torch.matmul(W3.T, dZ3)
    dZ2 = torch.mul(dA2, (A2 > 0).int())
    dW2 = torch.matmul(dZ2, A1.T) / m # 1./m * np.dot(dZ2, A1.T)
    db2 = torch.sum(dZ2, dim=1, keepdim = True) / m
    
    dA1 = torch.matmul(W2.T, dZ2)
    dZ1 = torch.mul(dA1, (A1 > 0).int())
    dW1 = torch.matmul(dZ1, X.T.float()) / m
    db1 = torch.sum(dZ1, dim=1, keepdim = True) / m
    
    gradients = {"dz3": dZ3, "dW3": dW3, "db3": db3,
                 "da2": dA2, "dz2": dZ2, "dW2": dW2, "db2": db2,
                 "da1": dA1, "dz1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

def _predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    # Forward propagation
    a3, caches = forward_propagation(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results

    #print ("predictions: " + str(p[0,:]))
    #print ("true labels: " + str(y[0,:]))
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

def binary_acc(y_pred, y_test):
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[1]
    acc = torch.round(acc * 100)
    
    return acc

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = torch.zeros((1,m))
    
    # Forward propagation
    probas, caches = forward_propagation(X, parameters)
    p[probas >0.5]=1

    
    # convert probas to 0/1 predictions
#     for i in range(0, probas.shape[1]):
#         if probas[0,i] > 0.5:
#             p[0,i] = 1
#         else:
#             p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(binary_acc(p, y).item()))
        
    return p

def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    
    return torch.tensor(train_X), torch.tensor(train_Y), torch.tensor(test_X), torch.tensor(test_Y)

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    X = np.array(X); y = np.array(y).ravel()
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions

def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return torch.tensor(train_X), torch.tensor(train_Y)