import numpy as np
from beras.onehot import OneHotEncoder
from beras.core import Tensor
from tensorflow.keras import datasets

def load_and_preprocess_data() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    '''This is where we load in and preprocess our data! We load in the data 
        for you but you'll need to flatten the images, normalize the values and 
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!'''
    
    #Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs, test_labels) = datasets.mnist.load_data()

    n_train_inputs = nor(train_inputs)
    n_test_inputs = nor(test_inputs)
   
    r_train = n_train_inputs.reshape(n_train_inputs.shape[0], 28*28)
    r_test = n_test_inputs.reshape(n_test_inputs.shape[0], 28*28)

    # Convert the arrays to Tensors and return the train inputs, train labels, test inputs and test labels in that order.
    # print(len(train_labels)) 60000
    ftrain_i = Tensor(r_train)
    ftrain_l = Tensor(train_labels)
    ftest_i = Tensor(r_test)
    ftest_l = Tensor(test_labels)

    return [ftrain_i, ftrain_l , ftest_i , ftest_l]


def nor(x: np.ndarray) -> np.ndarray:
    
    return (x - x.min()) / (x.max() - x.min())
   