import inspect
import traceback
import numpy as np
import torch
from torchvision import models

import sys
sys.path.append('.')
from pytorch_inspector import ModelExplorer
from pytorch_inspector import ParrallelHandler


def get_layers_from_model():
    print('>>>>> get_layers_from_model <<<<<')

    model = models.alexnet()
    source = '_modules'
    print(f'model:{model}')
    if True:
        #print(f'model:{model}')
        print(f'vars:{vars(model)}')
        #source = '_modules'
        #print(f'source:{vars(model)[source]}')
        #print(f'model.features:{model.features}')
        #print(f'model.classifier:{model.classifier}')

        print('=============================')
        for layer in model.children():
            print(f'children layer: {layer}')
        print('=============================')
        for name, layer in model.named_children():
            print(f'named_children name:{name} layer:{layer}')

        print('=============================')
        def get_layer(model: torch.nn.Module):
            children = list(model.children())
            return [model] if len(children) == 0 else [ci for c in children for ci in get_layer(c)]
        list_layers = get_layer(model)
        print(f'recursive list_layers: {list_layers}')

        print('=============================')
        layers = ModelExplorer.get_layers(model)
        print(f'recursive list_layers: {layers}')

    # Input source to pass to the model, used to test the hook
    input_to_test = torch.randn(1, 3, 224, 224)
    list_valid_forward, list_valid_backward = ModelExplorer.get_hook_layers(model, input_to_test)
    print('##################')
    print(f'list_valid list_valid_backward:{list_valid_backward}')
    print('##################')
    print(f'list_valid list_valid_forward:{list_valid_forward}')

    # Backpropagation
    try:
        ph = ParrallelHandler(callback_onrun=None, callback_onclosing=None, frequency=20.0, timeout=30.0, target_method=None)
        ph.track_layer(0, list_valid_backward, callback_transform=None)
        out = model(input_to_test)
        print(f'out shape:{out.shape}')
        out.mean().backward()
    except Exception as e:
        print(f'Ex:{e}')

if __name__ == "__main__":
  get_layers_from_model()
