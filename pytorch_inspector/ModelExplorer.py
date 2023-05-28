import inspect
import traceback
from typing import Optional, Any
import torch

from pytorch_inspector.utils.Decorators import *

#__all__ = ["ModelExplorer"]

class ModelExplorer():
    """
    Collection of static functions to explore a model content.
    """    
    @staticmethod
    def get_layers(model: torch.nn.Module, names = None, model_types = None):
        """
        Get all the possible layers from a torch nn Module.
        Args:
        - **model**: Model to target.
        - **names**: List of layer names (passed recursively).
        - **model_types**: List of layer types passed through (passed recursively).

        Returns:
        list of layers, associated layer names separated by a dot, layer name, 
        layer parent name, associated layer types separated by a dot.
        i.e. from AlexNet
        Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)), 'features.0', 'Conv2d', 'features', 'AlexNet.Sequential'
        """
        if names is None:
            names = []
        if model_types is None:
            model_types = []
        children = list(model.named_children())
        if len(children) == 0:
            return [(model, ".".join(names), type(model).__name__, ".".join(names[:-1]), ".".join(model_types))]
        else:
            return [ci for name, c in children for ci in ModelExplorer.get_layers(c, names + [name], model_types + [type(model).__name__])]


    @staticmethod
    def forward_hook_dummy(module, input, output):
        """
        Test forward hook. Nothing done.
        The function firm is from the pytorch documentation. Please refer to the original documentation
        for more information.
        """
        try:
            #global model_output
            #model_output = None
            #print("forward_hook called")
            #global model_output
            #model_output = output.clone().detach()
            pass
        except Exception as e:
            print(f'forward_hook ex:{e}')
            traceback.print_exc()
            raise e
        finally:
            return None

    # backpropagation hook
    @staticmethod
    def backward_hook_dummy(module, grad_input, grad_output) -> torch.Tensor or None:
        """
        Test backward hook. Nothing done.
        The function firm is from the pytorch documentation. Please refer to the original documentation
        for more information.
        """
        try:
            #global grads
            #grads = None
            #print("backward_hook called")
            #global grads
            #grads = grad_output[0].clone().detach()
            pass
        except Exception as e:
            print(f'backward_hook ex:{e}')
            traceback.print_exc()
            raise e
        finally:
            return None

    @staticmethod
    @exception_decorator
    def get_hook_layers(model: torch.nn.Module, input_to_test : list, get_out_tensor : Optional[Any] = None) -> list:
        """
        Get the layers that can be used for a forward/backward hook from a model.
        Args:
        - **model**: Model to target.
        - **input_to_test**: List of inputs to pass to the model.
        - **get_out_tensor**: Function applied to the output of the model to get the output tensor for backward operation.
                              Required for models which output is not a tensor but, for example, a tuple.
        Returns:
        - List with valid layers for the forward and backward hook. The list contains pairs of
          key, layer
        """
        print('ModelExplorer.get_hook_layers')
        if isinstance(input_to_test, list) is False:
            raise "input_to_test is expected to be a list"
        skip_layers = []#torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.MaxPool2d]
        # Get all the possible hooks
        list_candidate = []
        layers = ModelExplorer.get_layers(model)
        #print(f'ModelExplorer.layers:{layers}')
        for layer in layers:

            # Get the layer information, type, and name
            module = layer[0]
            layer_type = layer[2]
            name = layer[4] + '.' + layer[1]

            # Add potential valid layers to a list of candidates
            add_to_list = False
            # a layer can be added if no inplace or inplace is false
            if 'inplace' in inspect.getfullargspec(module.__init__).args or 'inplace' in inspect.getfullargspec(module.__init__).kwonlyargs:
                if module.inplace is False:
                    add_to_list = True
            else:
                if type(module) in skip_layers or layer_type in skip_layers:
                    pass
                else:
                    add_to_list = True
            if add_to_list:
                elem = (name , module)
                list_candidate.append(elem)

        print(f'list_candidate:{list_candidate}')

        # Create a a list of valid backward layers
        list_valid_backward = dict()
        for candidate in list_candidate:
            print(f'candidate:{candidate}')
            # Backpropagation
            try:
                handle = candidate[1].register_full_backward_hook(ModelExplorer.backward_hook_dummy)
                out = model(*input_to_test)
                if get_out_tensor is not None:
                    out = get_out_tensor(out)
                out.mean().backward()
                list_valid_backward[candidate[0]] = candidate[1]
                # Hook called
                handle.remove()
            except Exception as e:
                print(f'Exception get_hook_layers backward:{e}')
                # use print_exc to print the formatted traceback
                traceback.print_exc()
                # Hook called
                handle.remove()

        # Create a a list of valid forward layers
        list_valid_forward = dict()
        for candidate in list_candidate:
            print(f'candidate:{candidate}')
            # Backpropagation
            try:
                handle = candidate[1].register_forward_hook(ModelExplorer.forward_hook_dummy)
                out = model(*input_to_test)
                if get_out_tensor is not None:
                    out = get_out_tensor(out)
                out.mean().backward()
                list_valid_forward[candidate[0]] = candidate[1]
                # Hook called
                handle.remove()
            except Exception as e:
                print(f'Exception get_hook_layers forward:{e}')
                # use print_exc to print the formatted traceback
                traceback.print_exc()
                # Hook called
                handle.remove()

        return list_valid_forward, list_valid_backward

    @staticmethod
    def get_hook_layers_fromlist(model: torch.nn.Module, input_layers : list) -> list:
        """
        Get the layers that can be used for a forward/backward hook from a model.
        Args:
        - **model**: Model to target.
        - **input_layers**: List of input layers.
        Returns:
        - List with valid layers for the forward and backward hook. The list contains pairs of
          key, layer
        """
        print('ModelExplorer.get_hook_layers_fromlist')
        # Get all the possible hooks
        list_valid = dict()
        layers = ModelExplorer.get_layers(model)
        #print(f'ModelExplorer.layers:{layers}')
        for layer in layers:
            # Get the layer information, type, and name
            module = layer[0]
            name = layer[4] + '.' + layer[1]

            if name in input_layers:
                list_valid[name] = module
        return list_valid

    @staticmethod
    def save_list_layers_name(list_valid : list, fname : str) -> None:
        """
        Save a list with layers name in a file
        Args:
        - **list_valid**: List container with the valid layers to save (each element is a string).
        - **fname**: Output filename.
        """
        with open(fname, 'w') as f:
            for line in list_valid:
                f.write(f"{line}\n")        

    @staticmethod
    def load_list_layers_name(fname : str) -> list:
        """
        Load a list with layers name from a file
        Args:
        - **fname**: Output filename.
        Return:
        List container with the valid layers.
        """
        list_valid = []
        with open(fname) as file:
            while line := file.readline():
                list_valid.append(line.rstrip())        
        return list_valid