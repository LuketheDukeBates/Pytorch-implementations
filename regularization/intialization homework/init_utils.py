import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
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

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
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
    #(z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
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
    dZ3 = (A3 - Y) / m
    dW3 = torch.matmul(dZ3, A2.T) / m
    db3 = torch.sum(dZ3, dim=1, keepdims = True) / m
    
    dA2 = torch.matmul(W3.T, dZ3)
    dZ2 = torch.mul(dA2, (A2 > 0).int())
    dW2 = torch.matmul(dZ2, A1.T) / m # 1./m * np.dot(dZ2, A1.T)
    db2 = torch.sum(dZ2, dim=1, keepdim = True) / m
    
    dA1 = torch.matmul(W2.T, dZ2)
    dZ1 = torch.mul(dA1, (A1 > 0).int())
    dW1 = torch.matmul(dZ1, X.T) / m
    db1 = torch.sum(dZ1, dim=1, keepdim = True) / m
    
    gradients = {"dz3": dZ3, "dW3": dW3, "db3": db3,
                 "da2": dA2, "dz2": dZ2, "dW2": dW2, "db2": db2,
                 "da1": dA1, "dz1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of n_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters['W' + str(i)] = ... 
                  parameters['b' + str(i)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for k in range(L):
        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
        
    return parameters

def compute_loss(A3, Y):
    
    """
    Implement the loss function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    loss - value of the loss function
    """
    
    m = Y.shape[1]
#     logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
#     loss = 1./m * np.nansum(logprobs)
    logprobs = torch.mul(-torch.log(A3),Y) + torch.mul(-torch.log(1 - A3), 1 - Y)
    cost = 1./m * torch.sum(logprobs)
    
    return cost

def load_cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = torch.tensor(np.array(train_dataset["train_set_x"][:]), dtype=torch.float) # your train set features
    train_set_y_orig = torch.tensor(train_dataset["train_set_y"][:], dtype=torch.float) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = torch.tensor(np.array(test_dataset["test_set_x"][:]), dtype=torch.float) # your test set features
    test_set_y_orig = torch.tensor(np.array(test_dataset["test_set_y"][:]), dtype=torch.float) # your test set labels

    classes = list(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def _load_cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    train_set_x = train_set_x_orig/255
    test_set_x = test_set_x_orig/255

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


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

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    X = np.array(X)
    y = np.array(y).ravel()
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
    predictions = (a3>0.5)
    return predictions

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = torch.tensor(train_X.T,dtype=torch.float)
    train_Y = torch.tensor(train_Y.reshape((1, train_Y.shape[0])),dtype=torch.float)
    test_X = torch.tensor(test_X.T,dtype=torch.float)
    test_Y = torch.tensor(test_Y.reshape((1, test_Y.shape[0])),dtype=torch.float)
    return train_X, train_Y, test_X, test_Y