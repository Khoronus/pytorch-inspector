import inspect
import traceback
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


import sys
sys.path.append('.')
from pytorch_inspector import ParrallelHandler, ModelExplorer, DataRecorder, DataPlot

def test_hook():
    # Define the model and load the pretrained weights
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT) # load the AlexNet model from torchvision.models
    model.eval() # set the model to evaluation mode

    # Input source to pass to the model, used to test the hook
    input_to_test = torch.randn(1, 3, 224, 224)
    list_valid_forward, list_valid_backward = ModelExplorer.get_hook_layers(model, [input_to_test])
    print('##################')
    print(f'list_valid list_valid_forward:{list_valid_forward}')
    print('##################')
    print(f'list_valid list_valid_backward:{list_valid_backward}')

if __name__ == "__main__":
    test_hook()

