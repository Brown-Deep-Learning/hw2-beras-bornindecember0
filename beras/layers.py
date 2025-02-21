import numpy as np

from typing import Literal
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):

    def __init__(self, input_size, output_size, initializer: DENSE_INITIALIZERS = "normal"):
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> list[Tensor]:
        return self.w, self.b

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a dense layer! Refer to lecture slides for how this is computed.
        """
        w,b = self.weights
        return x@w + b

    def get_input_gradients(self) -> list[Tensor]:

         w, b = self.weights
         return [w]
       
       

    def get_weight_gradients(self) -> list[Tensor]:
        x = self.inputs[0]  
        batch_size = x.shape[0]
        input_size = x.shape[1]
        w, b = self.weights
        output_size = b.shape[0]  


        grad_w = x.reshape((batch_size, input_size, 1))
        grad_w = np.tile(grad_w, (1, 1, output_size))
        grad_b = np.ones(output_size)

        return [grad_w, grad_b]

        
        
      
    

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Variable, Variable]:
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        bias = Variable(np.zeros(output_size))

        if initializer == "zero":
            weights = Variable(np.zeros((input_size, output_size)))

        elif initializer == "normal":
            weights = Variable(np.random.normal(0, 1, (input_size,output_size)))
            # weights = weights - np.mean(weights)
            # weights = weights / np.std(weights) 
           

        elif initializer == "xavier":
            std = np.sqrt(2.0 / (input_size + output_size))
            weights = Variable(np.random.normal(0, std, (input_size, output_size)))
            # weights = weights - np.mean(weights)
            # weights = weights / np.std(weights) * std 
        
        elif initializer == "kaiming":
             std = np.sqrt(2.0 / input_size)
             weights = Variable(np.random.normal(0, std, (input_size, output_size)))
            #  weights = weights - np.mean(weights)
            #  weights = weights / np.std(weights) * std 
        
       
        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        return [Variable(weights), Variable(bias)]
