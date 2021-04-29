import torch
import numpy as np

def gradient_check_n_test_case(): 
    np.random.seed(1)
    x = np.random.randn(4,3)
    y = np.array([1, 1, 0])
    W1 = np.random.randn(5,4) 
    b1 = np.random.randn(5,1) 
    W2 = np.random.randn(3,5) 
    b2 = np.random.randn(3,1) 
    W3 = np.random.randn(1,3) 
    b3 = np.random.randn(1,1) 
    parameters = {"W1": torch.tensor(W1),
                  "b1": torch.tensor(b1),
                  "W2": torch.tensor(W2),
                  "b2": torch.tensor(b2),
                  "W3": torch.tensor(W3),
                  "b3": torch.tensor(b3)}

    
    return torch.tensor(x), torch.tensor(y), parameters