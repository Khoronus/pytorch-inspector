# coding: utf-8
import inspect
import traceback
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.append('.')
from pytorch_inspector import ParrallelHandler
from pytorch_inspector import ModelExplorer

def get_layers_from_model():
    print('>>>>> get_layers_from_model <<<<<')

    # forward propagation hook
    #global model_output
    #model_output = None
    def forward_hook(module, input, output):
        try:
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
    #global grads
    #grads = None
    def backward_hook(module, grad_input, grad_output) -> torch.Tensor or None:
        try:
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


    class module_list_model(nn.Module):
        def __init__(self):
            super(module_list_model, self).__init__()

            self.fc = nn.ModuleList(
                [nn.Linear(d*100, (d-1)*100) for d in range(2, 8).__reversed__()]
            )

            self.fc_final = nn.Linear(100, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, inputs):
            for fc in self.fc:
                inputs = fc(inputs)

            outputs = self.fc_final(inputs)

            return self.sigmoid(outputs)

    model = module_list_model()
    source = '_modules'
    if True:
        #print(f'model:{model}')
        print(f'vars:{vars(model)}')

        # It expects to know the model vars name...
        #for layer in model.fc:
        #    print(f'layer:{layer}')
        print('=============================')
        for layer in model.children():
            print(f'children layer: {layer}')
        print('=============================')
        for name, layer in model.named_children():
            print(f'named_children name:{name} layer:{layer}')

        print('=============================')
        def get_layer(model: torch.nn.Module):
            print(f'model:{type(model)}')
            children = list(model.children())
            return [model] if len(children) == 0 else [ci for c in children for ci in get_layer(c)]
        
        list_layers = get_layer(model)
        print(f'recursive list_layers: {list_layers}')

        print('=============================')
        layers = ModelExplorer.get_layers(model)
        print(f'recursive list_layers: {layers}')

    # Input source to pass to the model, used to test the hook
    input_to_test = torch.rand([700])
    list_valid_forward, list_valid_backward = ModelExplorer.get_hook_layers(model, input_to_test)
    print('##################')
    print(f'list_valid list_valid_backward:{list_valid_backward}')
    print('##################')
    print(f'list_valid list_valid_forward:{list_valid_forward}')

    # Backpropagation
    try:
        ph = ParrallelHandler(callback_onrun=None, callback_onclosing=None, frequency=20.0, max_elapsed_time=30.0)
        ph.track_layer(0, list_valid_backward)
        outputs = model(input_to_test)
        print(model)
        print(outputs.shape)
        outputs.mean().backward()

    except Exception as e:
        print(f'Ex:{e}')

if __name__ == "__main__":
  get_layers_from_model()
