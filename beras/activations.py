import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""

        return np.where(x <= 0, self.alpha * x, x)
        
        

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        x = self.inputs[0]
        mask = np.where(x > 0, 1.0, self.alpha)  # 1.0 -> >0
        return [mask]
        

    def compose_input_gradients(self, J):
       
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def forward(self, x) -> Tensor:
        # y = 1/(1+e^(-x))
        
        return  1 / (1 + np.exp(-x))

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        y = self.forward(self.inputs[0])  
        return [Tensor(y * (1 - y))]
       

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def forward(self, x):
        """Softmax forward propagation!"""
        # Not stable version
        # exps = np.exp(inputs)
        # outs = exps / np.sum(exps, axis=-1, keepdims=True)

        # HINT: Use stable softmax, which subtracts maximum from
        # all entries to prevent overflow/underflow issues

        # 1D 
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x_shifted = np.exp(x - x_max)
        output = exp_x_shifted / np.sum(exp_x_shifted, axis=-1, keepdims=True)
        self.outputs = output


        return output
        
        
        

    def get_input_gradients(self):
        """Softmax input gradients!"""
        x, y = self.inputs + self.outputs
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)
        
        # TODO: Implement softmax gradient

        for b in range(bn):
            for i in range(n):
                for j in range(n):
                    grad[b, i, j] = y[b, i] * ((i == j) - y[b, j])

        return [grad]
        