import numpy as np
import torch
from torchvision import models

def main():

  # Simple sequential model
  model = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
    torch.nn.Flatten(0,1)
  )

  global grads
  grads = None
  # backpropagation hook
  def grad_hook(module, grad_input, grad_output):
    print('grad_hook')
    global grads 
    grads = grad_output[0]

  # wrapper around a backpropagation hook
  def my_hook(name):
      def grad_hook(module, grad_input, grad_output):
          print(f'my_hook.grad_hook ok:{name} module:{module}')
          global grads 
          grads = grad_output[0]
          return
      return grad_hook

  print('register_full_backward_hook')
  model[2].register_full_backward_hook(grad_hook)
  model[2].register_full_backward_hook(my_hook('layer2'))
  x = torch.randn(1, 4, 3)
  out = model(x)
  out.mean().backward()
  if grads is not None:
    print(f'grads:{grads.size()}')
    print(f'grads abs:{grads.abs().sum()}')

if __name__ == "__main__":
  main()
