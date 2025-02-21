import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # per batch
        
        squared_diff = (y_true - y_pred)**2
        return np.mean(squared_diff)
        
    def get_input_gradients(self) -> list[Tensor]:
        y_pred = self.inputs[0] 
        y_true = self.inputs[1]
        batch_size, output_size = y_true.shape
       

        grad_p = 2 * (y_pred - y_true) / (batch_size*output_size)
        grad_t = np.zeros_like(y_true) 
 
        
        return [grad_p, grad_t]
    

class CategoricalCrossEntropy(Loss):
   
   
    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        
        # per batch
        y_pred = np.clip(y_pred, 1e-11, 1 - 1e-11)
        per_sample = -np.sum(y_true * np.log(y_pred), axis=-1)
    
        return np.mean(per_sample)

       
        
        

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred = self.inputs[0] 
        y_true = self.inputs[1]

        y_pred_safe = np.clip(y_pred, 1e-10, 1.0)
         
        grad_p = -(y_true / y_pred_safe ) / y_pred.shape[0]  
        grad_t = np.zeros_like(y_true)

        # Current shape: (batch_size, num_classes)
       
        return [grad_p, grad_t]
        
