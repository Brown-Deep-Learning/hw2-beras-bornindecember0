from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

def get_model():
    model = SequentialModel(
        [
           # Add in your layers here as elements of the list!
           # e.g. Dense(10, 10),
           Dense(784, 200, initializer="kaiming"),  
           LeakyReLU(),                                  
           Dense(200, 10, initializer="kaiming"),  
           Softmax()  
        ]
    )
    return model

def get_optimizer():
    # choose an optimizer, initialize it and return it!
    return Adam(learning_rate=0.001)
    

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    return CategoricalCrossEntropy()

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    return CategoricalAccuracy()

if __name__ == '__main__':

    ### Use this area to test your implementation!

    # 1. Create a SequentialModel using get_model
    model = get_model()

    # 2. Compile the model with optimizer, loss function, and accuracy metric

    model.compile(get_optimizer(), get_loss_fn(), get_acc_fn())
    
    # 3. Load and preprocess the data
    ftrain_i, ftrain_l , ftest_i , ftest_l = load_and_preprocess_data()
    
    encoder = OneHotEncoder()
    encoder.fit(ftrain_l)
    encoded_train_labels = encoder.forward(ftrain_l)
    encoded_test_labels = encoder(ftest_l)
   

    # 4. Train the model
    epochs = 10
    batch_size = 40
    train_m_dict = model.fit(ftrain_i, encoded_train_labels, epochs, batch_size)
    # 5. Evaluate the model
    evaluate_m_dict, prediction = model.evaluate(ftest_i, encoded_test_labels, batch_size)
    # print('-------------------------------------------------------')
    # print('current param: Adam(learning_rate=0.001) 784 200 / relu/ 200 10/ sm ')
    # print('current param: epochs', epochs,'batch_size', batch_size)
   
    predicted_classes = np.argmax(prediction, axis=1)
    np.save('predictions.npy', predicted_classes)
   
    
                                                                                                    
    
