from collections import defaultdict
import numpy as np

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def apply_gradients(self, trainable_params, grads):
        
        trainable_params.assign(trainable_params - np.array(grads) * self.learning_rate)
         
       

class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params, grads):

        for pid in range(len(trainable_params)):
            self.v[pid] = self.beta * self.v[pid] + (1 - self.beta) * (grads[pid] ** 2)
            up =  (self.learning_rate / np.sqrt(self.v[pid] + self.epsilon)) * grads[pid]
            trainable_params[pid].assign(trainable_params[pid]-up)
    

        # self.v = self.beta*self.v+ (1-self.beta)*(grads**2)
        # trainable_params.assign(trainable_params - ((self.learning_rate/((np.sqrt(self.v))+self.epsilon))*grads))
        


class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):


        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.t = 0                              # Time counter

    def apply_gradients(self, trainable_params, grads):
        self.t += 1

        for pid in range(len(trainable_params)):
             self.m[pid] = self.beta_1*self.m[pid] + (1-self.beta_1)*grads[pid]
             self.v[pid] = self.beta_2*self.v[pid] + (1-self.beta_2)*(grads[pid] ** 2)
    
             m_hat = self.m[pid] / (1 - self.beta_1**self.t)
             v_hat = self.v[pid] / (1 - self.beta_2**self.t)

             update = self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))
             trainable_params[pid].assign(trainable_params[pid] - update)
           
        # self.t += 1

        # self.m = self.beta_1*self.m + (1-self.beta_1)*grads
        # self.v = self.beta_2*self.v + (1-self.beta_2)*(grads**2)
        
        # self.m_hat = self.m/(1 - self.beta_1**self.t)
        # self.v_hat = self.v/(1 - self.beta_2**self.t)
        # trainable_params.assign(trainable_params - self.learning_rate *(self.m_hat/(np.sqrt(self.v_hat)) + self.epsilon))
       