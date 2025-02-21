from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}
        
        # print("Startg")
        up_gradient = []  
    
        while queue:
            o = queue.pop(0)
            
            if len(up_gradient) == 0:
                J = None
            else:
                J = [up_gradient.pop(0)]
                
            l_layer = self.previous_layers[id(o)]
            if l_layer is None:
                continue
                
            weights_list = l_layer.weights # get w b

            if len(weights_list) != 0:
                weights_gradient = l_layer.compose_weight_gradients(J)

                for weight, weight_grad in zip(weights_list, weights_gradient):
                   
                    if grads[id(weight)] is None:
                        grads[id(weight)] = weight_grad
                    else:
                        grads[id(weight)] = grads[id(weight)]+weight_grad
                        
            # print("Processing layer:", type(prev_layer))
            inputs_list = l_layer.inputs
            input_gradient = l_layer.compose_input_gradients(J)

            # print(J)
            
            # print("Processing layer:", type(prev_layer))
            # print("Input list:", [id(inp) for inp in inputs_list])
            # print("Input gradients shapes:", [g.shape for g in input_gradients])
           
            queue.extend(inputs_list)
            
            for input, input_grad in zip(inputs_list, input_gradient):
                if grads[id(input)] is None:
                    grads[id(input)] = input_grad
                else:
                    grads[id(input)] = grads[id(input)]+input_grad
            
            up_gradient.extend(input_gradient)
           
            # print(" nnnnnnnext layer")
        
        result = [grads[id(source)] for source in sources]
        
        return result


          