import h5py
import numpy as np
import cv2
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def show_signs():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    X_train = np.array(train_dataset["train_set_x"][:]) # your train set features
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    Y_train = np.array(train_dataset["train_set_y"][:]) # your train set labels
    Y_train = Y_train.reshape((1, Y_train.shape[0])).T

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    X_test = np.array(test_dataset["test_set_x"][:]) # your test set features
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    Y_test = np.array(test_dataset["test_set_y"][:]) # your test set labels
    Y_test = Y_test.reshape((1, Y_test.shape[0])).T

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    
    return X_train, Y_train, X_test, Y_test, classes



def predict_image(image_path, input_size, device):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_size, input_size))                  #Resize
    img = img[..., ::-1].transpose((2, 0, 1))                        #BGR -> RGB and HxWxC -> CxHxW
    img = img[np.newaxis, ...] / 255.0                               #Add a channel at 0, thus making it a batch
    img = torch.tensor(img, dtype=torch.float, device=device)        #Convert to Tensor
    return img

def get_accuracy(model, loader):
    model.eval()
    num_samples = 0
    num_correct = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, preds = y_pred.data.max(1)
            num_samples += preds.size(0)
            num_correct += (y.view(-1) == preds).sum()
        
    return num_correct.item() / num_samples

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']