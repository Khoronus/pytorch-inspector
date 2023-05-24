import numpy as np
import torch
from torchvision import models

def main():

  # backpropagation hook
  global grads
  grads = None
  def grad_hook(module, grad_input, grad_output):
      print("grad_hook called")
      global grads
      grads = grad_output[0].clone().detach()

  # forward hook
  global model_output
  model_output = None
  def forward_hook(module, input, output):
      print("forward_hook called")
      global model_output
      model_output = output.clone().detach()

  # 1
  model = models.alexnet()
  print(f'model:{model}')
  print(f'vars:{vars(model)}')
  source = '_modules'
  print(f'source:{vars(model)[source]}')
  print(f'model.features:{model.features}')
  print(f'model.classifier:{model.classifier}')

  print('register_backward_hook')
  model.features.register_backward_hook(grad_hook)
  x = torch.randn(1, 3, 224, 224)
  out = model(x)
  out.mean().backward() # prints "grad_hook called"
  print(grads.size()) # prints torch.Size([1, 256, 6, 6])
  print(grads.abs().sum()) # prints tensor(0.3186)

  # 2
  print('register_full_backward_hook')
  model = models.alexnet()
  model.features.register_full_backward_hook(grad_hook)
  x = torch.randn(1, 3, 224, 224)
  out = model(x)
  out.mean().backward()
  if grads is not None:
    print(grads.size())
    print(grads.abs().sum())

  # 3
  print('register_full_backward_hook')
  model = models.alexnet()
  model.register_forward_hook(forward_hook)
  model.classifier[6].register_full_backward_hook(grad_hook)  # select 0,3,6
  # warning: other layers will give the error:
  # RuntimeError: Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace. 
  # This view was created inside a custom Function (or because an input was returned as-is) and 
  # the autograd logic to handle view+inplace would override the custom backward associated with 
  # the custom Function, leading to incorrect gradients. 
  # This behavior is forbidden. You can fix this by cloning the output of the custom Function.
  x = torch.randn(1, 3, 224, 224)
  out = model(x)
  out.mean().backward()
  if grads is not None:
    print(grads.size())
    print(grads.abs().sum())

  # 4
  print('register_forward_hook')
  model = models.alexnet()
  model.register_forward_hook(forward_hook)
  x = torch.randn(1, 3, 224, 224)
  out = model(x)
  out.mean().backward()
  if model_output is not None:
    print(f'out:{out}')
    print(f'model_output:{model_output}') # same as out


if __name__ == "__main__":
  main()
